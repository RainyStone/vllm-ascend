"""
ShmemRuntime: module-level wrapper over the shmem symmetric-heap lifecycle.

[方案 B] init/finalize 复用 ``ascend_moonep.buffer_c`` 的 C 接口封装
(``shmem_init`` / ``shmem_finalize``)，使 ShmemRuntime 与 ``_C`` 共享同一个
libshmem.so 实例（不再依赖 shmem Python 包）。rank/world_size/barrier 改用
``torch.distributed``。原 shmem Python 包实现保留在注释里，便于回退。

buffer-backed 初始化支持两种 bootstrap：
  - ``init_with_buffer``（默认）—— UID bootstrap（单机，rank0 生成 UID 经 dist
    广播；对称空间作用域由广播范围决定）。
  - ``init_with_ip_port``（或 ``init_with_buffer(ip_port=...)``）—— ip_port /
    DEFAULT bootstrap（多机，全体 rank 传同一 ``tcp://<rank0_ip>:<port>``，
    rank0 建 config store server、其余连 client，无需 UID 广播）。

原方案 A 说明：shmem Python 包加载自己的 libshmem.so 实例，与 _C 用的副本
独立，两条路径不能混用（Do not mix）。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ._common import (
    ACLSHMEM_DATA_OP_MTE,
    ACLSHMEM_DATA_OP_UDMA,
    SHMEM_LARGE_PAGE,
    compute_shmem_mem_size,
)
from .buffer_c import (
    shmem_finalize,
    shmem_init_with_ip_port,
    shmem_init_with_buffer,
    symmetric_descriptors,
)
from ._C import acl_shmem_barrier_all, acl_shmem_get_unique_id
from .runtime_audit import audit_loaded_shmem_library


def _env_udma_enabled() -> bool:
    return os.environ.get("ASCEND_MOONEP_USE_UDMA", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def default_shmem_ip_port(master_addr: str, master_port: int) -> str:
    """从 torchrun 的 rendezvous 地址推导 shmem DEFAULT-bootstrap 的 ip_port URL。

    shmem 的 config store 是**独立的** TCP server，不能复用 torch 占用的
    ``MASTER_PORT``，因此端口 +1（避免与 torch rendezvous 冲突）。``ip_port``
    必须对所有 rank 一致：``MASTER_ADDR``/``MASTER_PORT`` 在 torchrun 多机启动下
    是全体 rank 相同的环境变量，据此推导天然一致。
    """
    return f"tcp://{master_addr}:{int(master_port) + 1}"


def _multi_node_launch() -> bool:
    """torchrun 环境下是否多机（``WORLD_SIZE`` > 每节点进程数）。"""
    world = os.environ.get("WORLD_SIZE")
    local = os.environ.get("LOCAL_WORLD_SIZE")
    if world is None or local is None:
        return False
    return int(world) > int(local)


def _env_shmem_ip_port() -> str | None:
    """从 torchrun 环境推导 shmem ip_port；非多机或缺环境时返回 None。"""
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if not _multi_node_launch() or not master_addr or not master_port:
        return None
    return default_shmem_ip_port(master_addr, int(master_port))

# --- 原实现（方案 A：shmem Python 包）---
# import shmem.core as core
# from shmem.core.direct import my_pe, n_pes
# from shmem.core.utils import Buffer as SymBuffer


class ShmemRuntime:
    """Thin facade over the shmem symmetric-memory runtime lifecycle."""

    _initialized = False
    _poisoned = False
    _runtime_library_record = None

    @staticmethod
    def compute_mem_size(
        S: int,
        H: int,
        K: int,
        E: int,
        H_prime: int,
        num_ep_ranks: int,
        **kwargs,
    ) -> int:
        """Per-PE symmetric heap bytes needed for a given MoonEP config.

        Convenience over ``_common.compute_shmem_mem_size``: pass the model
        config (S seqlen, K topk count, E expert count, H / H' hidden sizes,
        num_ep_ranks) and get the 2 MiB-aligned ``mem_size`` to feed to
        ``ShmemRuntime.init``. The calculation covers the Buffer's symmetric
        chunks, the RMA op workspaces, the grad-reduce buffers and the prefetch
        weight slots; extra framework-side symmetric tensors can be added via
        ``owner_weight_bytes`` / the ``*_buffers`` / ``*_slots`` counts.
        """
        return compute_shmem_mem_size(S, H, K, E, H_prime, num_ep_ranks, **kwargs)

    @classmethod
    def init(cls, group=None, mem_size: int = None, uid=None) -> None:
        """[已移除] 遗留的 UID+``local_mem_size`` 对称堆初始化。

        当前只支持 buffer-backed 初始化：先构造 Buffer / ``create_vmm_physical``
        登记 descriptor，再调 ``ShmemRuntime.init_with_buffer``。遗留路径与本进程
        的 buffer-backed 堆互斥，保留符号但显式报错，避免误用被 C 侧幂等检查静默吞掉。
        """
        raise RuntimeError(
            "ShmemRuntime.init(local_mem_size) 已移除：仅支持 buffer-backed "
            "ShmemRuntime.init_with_buffer（PR-653）")

    @classmethod
    def init_with_buffer(cls, group=None, descriptors=None, uid_bytes=None,
                         tail_bytes: int = 0, ip_port: str | None = None,
                         enable_udma: bool | None = None, qp_count: int = 0) -> None:
        """Initialize the buffer-backed shmem symmetric heap (PR-653, collective).

        Uses caller-mapped CANN VMM buffers instead of ``local_mem_size``. Two
        bootstraps:

          - ``ip_port is None``（默认）：UID bootstrap（单机）。rank0 生成 UID 并
            在 ``group`` 内广播；全体 rank 用同一 UID 调
            ``buffer_c.shmem_init_with_buffer``，对称空间作用域由广播范围决定。
          - ``ip_port`` 非 None：ip_port/DEFAULT bootstrap（多机）。全体 rank 传
            同一 ``ip_port``（``tcp://<rank0_ip>:<port>``），rank0 建 config
            store server、其余连 client，无需 UID 广播 —— 见
            ``ShmemRuntime.init_with_ip_port``。

        ``descriptors`` defaults to the process-wide registered VMM allocations
        (``buffer_c.symmetric_descriptors()``, populated by ``Buffer`` /
        ``create_vmm_physical`` at construction). Must run AFTER every Buffer is
        constructed and BEFORE the first dispatch/combine/prefetch/reduce_grad.
        """
        if cls._poisoned:
            raise RuntimeError(
                "shmem runtime is poisoned by a prior finalize failure; "
                "reinitialization is forbidden")
        if enable_udma is None:
            enable_udma = _env_udma_enabled()
        if cls._initialized:
            return
        if enable_udma and not qp_count:
            qp_count = 2
        if ip_port is not None:
            cls.init_with_ip_port(group=group, descriptors=descriptors,
                                  tail_bytes=tail_bytes, ip_port=ip_port,
                                  enable_udma=enable_udma, qp_count=qp_count)
            return
        # Fail closed before any SHMEM-native call (including UID generation).
        # Loading two libshmem images creates independent process-global
        # PE/heap state.
        # cls._runtime_library_record = audit_loaded_shmem_library()
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        if uid_bytes is None:
            uid = acl_shmem_get_unique_id() if rank == 0 else None
            # ``src`` is always a global rank even when ``group`` is an EP
            # subgroup.  EP rank 0 is not necessarily global rank 0.
            uid_root = 0 if group is None else dist.get_global_rank(group, 0)
            # HCCL process groups do not provide a CPU/object transport.  Send
            # the binary UID as NPU tensors, including its ABI-derived length,
            # so embedded NUL bytes are preserved without pickle or Gloo.
            device = torch.device("npu", torch.npu.current_device())
            uid_length = torch.tensor(
                [len(uid) if uid is not None else 0],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(uid_length, src=uid_root, group=group)
            length = int(uid_length.item())
            if length <= 0:
                raise RuntimeError("aclshmem UID broadcast produced an invalid byte length")
            uid_tensor = (
                torch.tensor(list(uid), dtype=torch.uint8, device=device)
                if uid is not None
                else torch.empty(length, dtype=torch.uint8, device=device)
            )
            if int(uid_tensor.numel()) != length:
                raise RuntimeError(
                    "aclshmem UID length changed while preparing subgroup broadcast"
                )
            dist.broadcast(uid_tensor, src=uid_root, group=group)
            uid_bytes = bytes(uid_tensor.cpu().tolist())
        if descriptors is None:
            descriptors = symmetric_descriptors()
        shmem_init_with_buffer(
            descriptors, int(rank), int(world_size), uid_bytes,
            tail_bytes=int(tail_bytes),
            data_op_engine_type=(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_MTE if enable_udma
                                 else ACLSHMEM_DATA_OP_MTE),
            device_id=int(torch.npu.current_device()),
            enable_udma=enable_udma, qp_count=qp_count)
        cls._initialized = True

    @classmethod
    def init_with_ip_port(cls, group=None, ip_port=None, descriptors=None,
                          tail_bytes: int = 0, rank: int | None = None,
                          world_size: int | None = None,
                          enable_udma: bool | None = None, qp_count: int = 0) -> None:
        """Initialize the buffer-backed shmem heap via ip_port/DEFAULT bootstrap
        （多机，collective）。

        默认从 ``dist`` 获取 rank/world_size。未创建 process group 的调用方可显式
        传入 ``rank`` 和 ``world_size``，此时本方法完全不调用 dist。全体 rank 传
        同一个 ``ip_port``
        （``"tcp://<rank0_ip>:<port>"`` 或 ``"tcp6://[ipv6]:<port>"``）：
        rank0（my_pe=0）起 config store server、其余连 client，shmem 对称空间跨
        多个节点，无需 UID 广播（等价 shmem Python 包的
        ``aclshmem_init(InitAttr(ip_port=...))``）。

        ``ip_port`` 缺省时从 torchrun 环境推导（``MASTER_ADDR`` / ``MASTER_PORT``，
        端口 +1 避开 torch rendezvous，见 ``default_shmem_ip_port``）；多机 torchrun
        启动下也可不传。``descriptors`` 缺省为进程内已登记的 VMM 分配
        （``buffer_c.symmetric_descriptors()``）。调用时机同 ``init_with_buffer``。
        """
        if cls._poisoned:
            raise RuntimeError(
                "shmem runtime is poisoned by a prior finalize failure; "
                "reinitialization is forbidden")
        if enable_udma is None:
            enable_udma = _env_udma_enabled()
        if cls._initialized:
            return
        # The IP/port path owns the same process-global runtime and backing as
        # the UID path, so it must pass the identical pre-native audit.
        # cls._runtime_library_record = audit_loaded_shmem_library()
        if ip_port is None:
            ip_port = _env_shmem_ip_port()
        if ip_port is None:
            raise RuntimeError(
                "ShmemRuntime.init_with_ip_port requires ip_port "
                "('tcp://<rank0_ip>:<port>' or 'tcp6://[ipv6]:<port>'); pass it "
                "explicitly, or set MASTER_ADDR/MASTER_PORT for a multi-node "
                "torchrun launch")
        if rank is None and world_size is None:
            rank = dist.get_rank(group=group)
            world_size = dist.get_world_size(group=group)
        else:
            if rank is None or world_size is None:
                raise ValueError("rank and world_size must be provided together")
            if group is not None:
                raise ValueError("group must be None when rank is provided explicitly")
            if not isinstance(rank, int) or not isinstance(world_size, int):
                raise TypeError("rank and world_size must be integers")
            if world_size <= 0 or rank < 0 or rank >= world_size:
                raise ValueError(
                    f"invalid rank/world_size: rank={rank}, world_size={world_size}"
                )
        if descriptors is None:
            descriptors = symmetric_descriptors()
        if enable_udma and not qp_count:
            qp_count = 2
        shmem_init_with_ip_port(
            int(rank), int(world_size), ip_port, descriptors,
            tail_bytes=int(tail_bytes),
            data_op_engine_type=(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_MTE if enable_udma
                                 else ACLSHMEM_DATA_OP_MTE),
            device_id=int(torch.npu.current_device()),
            enable_udma=enable_udma, qp_count=qp_count)
        cls._initialized = True

    @classmethod
    def finalize(cls) -> int:
        """Finalize collectively and return the native status code.

        On failure ``_initialized`` deliberately remains true, which prevents
        framework teardown from treating registered caller-owned VMM backing as
        releasable.  The process should be considered poisoned and terminated
        only after diagnostics are captured.
        """
        if cls._initialized:
            # [方案 B] 复用 buffer_c 封装
            rc = int(shmem_finalize())
            if rc != 0:
                cls._poisoned = True
                raise RuntimeError(
                    f"aclshmem_finalize failed with {rc}; runtime state and "
                    "caller-owned VMM backing remain live")
            # --- 原实现（方案 A）---
            # core.finalize()
            cls._initialized = False
            cls._poisoned = False
            return rc
        return 0

    @classmethod
    def is_initialized(cls) -> bool:
        # [方案 B] C 接口未暴露 init_status，用内部标志
        return cls._initialized
        # --- 原实现（方案 A）---
        # try:
        #     from shmem.core.direct import InitStatus, init_status
        #     return init_status() == InitStatus.INITIALIZED
        # except Exception:
        #     return cls._initialized

    @classmethod
    def runtime_library_info(cls) -> dict[str, object] | None:
        """Return a copy of the last successful pre-init runtime identity audit."""
        if cls._runtime_library_record is None:
            return None
        return dict(cls._runtime_library_record)

    @staticmethod
    def rank() -> int:
        # [方案 B] C 接口路径下 Python 包实例未 init，改用 dist
        return int(dist.get_rank())
        # --- 原实现（方案 A）---
        # return int(my_pe())

    @staticmethod
    def world_size() -> int:
        return int(dist.get_world_size())
        # --- 原实现（方案 A）---
        # return int(n_pes())

    @staticmethod
    def barrier(group=None) -> None:
        # [方案 B] v1.3 aclshmem_barrier_all：全局 barrier + 等 pending RMA 完成
        acl_shmem_barrier_all()
        # --- 原实现（方案 A）---
        # """Best-effort shmem barrier; falls back to ``dist.barrier`` when the
        # python binding does not expose one."""
        # barrier = getattr(core, "barrier", None)
        # if callable(barrier):
        #     barrier()
        # else:
        #     dist.barrier(group=group)


def runtime_library_info() -> dict[str, object] | None:
    """Return the cached ``libshmem.so`` realpath/dev/inode/SHA256 evidence."""
    return ShmemRuntime.runtime_library_info()
