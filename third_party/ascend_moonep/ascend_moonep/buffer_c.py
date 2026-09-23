"""C API 封装的对称内存 buffer 方法（等价替换 buffer.py 的 shmem Python API 版）。

底层用 ``ascend_moonep._C`` 的 ``acl_*`` 接口（自编 C 扩展，与 _C 共享同一个
libshmem.so 实例），替换掉 shmem Python 包（``shmem.aclshmem_create_tensor`` /
``core.*``）的调用。方法与 ``buffer.py`` 同名、同签名，调用方只改 import 即可
无痛迁移。

  - ``shmem_init`` / ``shmem_finalize``：UID bootstrap 初始化/终结
  - ``shmem_init_with_ip_port``：ip_port/DEFAULT bootstrap 初始化
    （等价 shmem Python 包的 ``aclshmem_init(InitAttr)``；buffer-backed）
  - ``create_sym_dist_tensor``：等价 buffer.py 同名函数（分配对称 chunk）
  - ``create_sym_peer_addr``：等价 buffer.py 同名函数（本地地址 → pe 地址）
  - ``create_sym_single_owner_tensor``：等价 buffer.py 同名函数
    （shmem 无单 owner 概念，降级为全员对称分配）
  - ``create_sym_full_tensor``：新增 —— alloc + map，返回跨 rank 连续视图
    （等价 CUDA 版 ``create_nvl_dist_tensor``）

与 Python 包版的差异：RAII —— tensor 销毁时 deleter 自动 ``aclshmem_free``，
无需手动 ``core.free``。
"""

from __future__ import annotations

import math
import os

import torch

from ._common import ACLSHMEM_DATA_OP_MTE

from ._C import (
    acl_dist_alloc,
    acl_dist_map,
    acl_shmem_ptr,
    acl_shmem_init,
    acl_shmem_get_unique_id,
    acl_shmem_init_uid,
    acl_shmem_finalize,
    # CANN VMM（aclrt* 物理分配 + shareable VA）
    acl_vmm_granularity,
    acl_vmm_malloc_physical,
    acl_vmm_free_physical,
    acl_vmm_reserve,
    acl_vmm_release,
    acl_vmm_map,
    acl_vmm_unmap,
    acl_vmm_set_access,
    acl_vmm_export_fabric,
    acl_vmm_export_shareable,
    acl_vmm_import_shareable,
    acl_vmm_alloc_source,
    acl_vmm_free_source,
    # shmem buffer-heap（PR-653）
    acl_shmem_get_heap_base,
    acl_shmem_get_buffer_ptr,
    acl_shmem_init_with_buffers,
    acl_shmem_init_with_buffers_ip_port,
    acl_shmem_barrier_all_on_stream,
)

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.uint8: 1,
}

# Caller-owned descriptor backing is pinned from successful buffer-backed init
# until successful collective finalize.  A failed finalize keeps this latch set
# so Python teardown cannot invalidate handles still referenced by ACLSHMEM.
_SHMEM_BACKING_LOCKED = False


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    return a // math.gcd(a, b) * b


def _chunk_nbytes(chunk_shape, dtype: torch.dtype) -> int:
    nbytes = _numel(chunk_shape) * _ELEM_SIZE[dtype]
    assert nbytes > 0, "chunk must have at least one element"
    return nbytes


# ---------------------------------------------------------------------------
# shmem 初始化 / 终结（UID bootstrap）
# ---------------------------------------------------------------------------

def shmem_init(group=None, mem_size: int = None) -> None:
    """[已移除] 遗留的 UID+``local_mem_size`` 对称堆初始化。

    当前只支持 buffer-backed 初始化（``ShmemRuntime.init_with_buffer`` →
    ``shmem_init_with_buffer``）。遗留路径与本进程的 buffer-backed 堆互斥
    （会触发二次 init），因此保留符号但显式报错，避免误用被 _C 的幂等检查静默吞掉。
    """
    raise RuntimeError(
        "legacy UID bootstrap shmem init (local_mem_size) 已移除：仅支持 "
        "buffer-backed ShmemRuntime.init_with_buffer（PR-653）")


def shmem_finalize() -> int:
    """Finalize the shmem symmetric heap and return the native status code.

    A non-zero return means native teardown did not complete.  Callers must
    retain every registered VMM allocation/handle and treat the process as
    still initialized; freeing caller-owned backing after such a failure is a
    use-after-free hazard for ACLSHMEM.
    """
    global _SHMEM_BACKING_LOCKED
    rc = int(acl_shmem_finalize())
    if rc == 0:
        _SHMEM_BACKING_LOCKED = False
    return rc


def shmem_init_with_ip_port(my_pe: int, n_pes: int, ip_port: str,
                            descriptors, tail_bytes: int = 0,
                            data_op_engine_type: int = ACLSHMEM_DATA_OP_MTE,
                            device_id: int | None = None,
                            enable_udma: bool | None = None, qp_count: int = 0) -> None:
    """ip_port/DEFAULT bootstrap 的 buffer-backed shmem 对称堆初始化（collective）。

    等价 shmem Python 包的 ``aclshmem_init(InitAttr(ip_port=...))``，但走
    buffer-backed（PR-653 ``aclshmemx_init_attr_with_buffers``，descriptor 定义见
    ``shmem_init_with_buffer``）。这是**多机**（多节点）的运行方式：全体 rank 传
    同一个 ``ip_port``（``"tcp://<rank0_ip>:<port>"`` 或 ``"tcp6://[ipv6]:<port>"``），
    rank0（my_pe=0）建 config store server、其余 rank 连 client，不依赖 dist 广播
    UID —— 对称空间可跨节点。``tail_bytes`` 非零时 shmem 保留一段自家 malloc 堆。

    ``my_pe`` / ``n_pes`` 必须与 ``aclshmem_my_pe()`` / ``aclshmem_n_pes()`` 一致
    （调用方从 dist 取 rank/world_size 即可）。
    """
    global _SHMEM_BACKING_LOCKED
    if device_id is None:
        device_id = int(torch.npu.current_device())
    # Match the UID-backed path's ownership transaction: seal the registry
    # before descriptor conversion/native collectives and roll the seal back
    # only when initialization did not succeed.
    was_locked = _SHMEM_BACKING_LOCKED
    _SHMEM_BACKING_LOCKED = True
    try:
        native_descriptors = [
            (int(addr), int(size), int(mh),
             fb if isinstance(fb, bytes) else bytes(fb or b""))
            for (addr, size, mh, fb) in descriptors
        ]
        rc = acl_shmem_init_with_buffers_ip_port(
            int(my_pe), int(n_pes), int(tail_bytes),
            int(data_op_engine_type), ip_port, native_descriptors,
            int(device_id), bool(enable_udma), int(qp_count))
    except Exception:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise
    if rc != 0:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise RuntimeError(
            f"acl_shmem_init_with_buffers_ip_port failed with {rc}")


# ---------------------------------------------------------------------------
# 对称 chunk 分配（与 buffer.py 同名、同签名）
# ---------------------------------------------------------------------------

def create_sym_dist_tensor(chunk_shape, dtype: torch.dtype, device: int) -> torch.Tensor:
    """Allocate one symmetric chunk and wrap it as an NPU tensor.

    等价于 buffer.py 的同名函数。``tensor.data_ptr()`` 即本地对称基址（peer
    寻址 / 跨 rank 访问的基址）。

    ``device`` 仅作签名兼容（``acl_dist_alloc`` 的 from_blob 用当前 NPU 设备，
    调用方应已 ``torch.npu.set_device``）。

    RAII：tensor 销毁时自动 ``aclshmem_free``，无需手动释放。分配是
    collective 的：所有 rank 按相同顺序、相同大小分配。
    """
    _chunk_nbytes(chunk_shape, dtype)
    keepalive, _, _ = acl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
    return keepalive


def create_sym_peer_addr(local_tensor: torch.Tensor, pe: int) -> int:
    """Raw peer symmetric address of ``local_tensor`` on PE ``pe`` (host side).

    等价于 buffer.py 的同名函数（用 C 接口 ``acl_shmem_ptr`` 实现）。
    """
    return acl_shmem_ptr(local_tensor.data_ptr(), pe)


def create_sym_single_owner_tensor(
    shape: list[int],
    dtype: torch.dtype,
    device: int,
    owner_rank: int,
    local_rank: int,
) -> torch.Tensor:
    """Allocate a symmetric tensor on one rank, visible to all via peer addr.

    等价于 buffer.py 的同名函数。shmem 无单 owner 概念（对称内存所有 rank 都
    分配），降级为全员对称分配；仅 owner 的 chunk 携带数据。RAII：tensor 销毁
    时自动 ``aclshmem_free``。
    """
    keepalive, _, _ = acl_dist_alloc(shape=list(shape), dtype=dtype)
    return keepalive


# ---------------------------------------------------------------------------
# 跨 rank 视图（新增，等价 CUDA 版 create_nvl_dist_tensor）
# ---------------------------------------------------------------------------

def create_sym_full_tensor(chunk_shape, dtype: torch.dtype, local_rank: int,
                           world_size: int, device: int = 0) -> torch.Tensor:
    """alloc + map：分配对称 chunk 并构造跨 rank 连续视图。

    返回 ``[world_size] + chunk_shape`` 的视图，stride[0] = 1GB/elemsize
    （对称堆 1GB 对齐步长）；``full[r]`` 即第 r 个 rank 的 chunk，任意 rank
    可读写任意段（远程走硬件互连，零拷贝）。

    视图不拥有内存；本地 chunk 由 keepalive 持有，通过 ``full._keepalive``
    绑定生命周期（视图先亡、keepalive 后亡）。

    ``local_rank`` / ``world_size`` 必须与 shmem 的 my_pe / n_pes 一致。
    """
    keepalive, _, _ = acl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
    full = acl_dist_map(
        chunk_shape=list(chunk_shape), dtype=dtype,
        local_ptr=keepalive.data_ptr(),
        local_rank=local_rank, world_size=world_size,
    )
    full._keepalive = keepalive
    return full


# ---------------------------------------------------------------------------
# CANN VMM（aclrt* 物理分配 + shareable VA）—— buffer-backed shmem init 的前置
# ---------------------------------------------------------------------------

class VmmAllocation:
    """一个 caller 拥有的 CANN VMM 物理分配，映射在唯一的 source VA。

    对照 ``moonep.ascend_vmm.AscendPhysicalAllocation``：
      - ``source_va``        : 登记进对称 arena 的映射 VA（注册一次）
      - ``mem_handle``       : 原始 ``aclrtMallocPhysical`` 句柄（即 ``owned_handle``）
      - ``shareable_handle`` : 为 descriptor 导出的 legacy V1 shareable 句柄
      - ``fabric_handle``    : Fabric V2 句柄（legacy-V1 路径下为 None）
      - ``size``             : 字节数
    ``destroy()`` 按依赖顺序释放：unmap -> release -> free_physical（VMM 逆操作，
    不是 ``aclshmem_free``）。
    """

    def __init__(self, source_va, mem_handle, shareable_handle, size,
                 fabric_handle_bytes=b"", device_id=0):
        self.source_va = int(source_va)
        self.mem_handle = mem_handle
        self.shareable_handle = int(shareable_handle) if shareable_handle else None
        # 64B aclrtMemFabricHandle 原始字节（Ascend950）；shmem init_with_buffers
        # 收到后直接用它，不再 re-export。910/legacy 为空。
        self.fabric_handle_bytes = bytes(fabric_handle_bytes or b"")
        self.size = int(size)
        self.device_id = int(device_id)
        self._destroyed = False
        self._registered = False

    @property
    def owned_handle(self):
        """arena descriptor 用的别名（同 ``mem_handle``）。"""
        return self.mem_handle

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self._registered and _SHMEM_BACKING_LOCKED:
            raise RuntimeError(
                "cannot release VMM descriptor backing before a successful "
                "collective shmem finalize")
        acl_vmm_free_source(self.source_va, self.mem_handle)
        self._destroyed = True


class SegmentedVAMapping:
    """一段保留的连续 VA，由多个分开映射的 segment 拼成。

    对照 ``moonep.ascend_vmm.SegmentedVAMapping``：把多个物理 handle（或经
    ``offsets`` 取 handle 子区间）映射进一段 ``acl_vmm_reserve`` 的连续 VA ——
    这是构造跨卡连续 ``[E+B]`` 投影视图的基础。
    """

    def __init__(self, handles, sizes, device_id, offsets=None):
        self.device_id = int(device_id)
        handles = tuple(handles)
        sizes = tuple(int(s) for s in sizes)
        if not handles or len(handles) != len(sizes):
            raise ValueError("VMM handles/sizes must be non-empty and paired")
        offsets = (0,) * len(handles) if offsets is None \
            else tuple(int(o) for o in offsets)
        if len(offsets) != len(handles):
            raise ValueError("VMM handles/sizes/offsets must be paired")
        gran = int(acl_vmm_granularity(self.device_id))
        for size, offset in zip(sizes, offsets):
            if size <= 0 or size % gran or offset < 0 or offset % gran:
                raise ValueError(
                    f"VMM segment size/offset must be non-negative multiples of "
                    f"granularity {gran}, got size={size}, offset={offset}")
        self.base = 0
        self.size = sum(sizes)
        self._segments = []
        self.base = int(acl_vmm_reserve(self.size))
        try:
            cursor = self.base
            for handle, size, offset in zip(handles, sizes, offsets):
                acl_vmm_map(cursor, size, handle, offset)
                acl_vmm_set_access(
                    cursor, size, acl_vmm_access_device_id(self.device_id))
                self._segments.append((cursor, size, handle, offset))
                cursor += size
        except Exception:
            self.destroy()
            raise

    def destroy(self) -> None:
        while self._segments:
            addr, _, _, _ = self._segments.pop()
            acl_vmm_unmap(addr)
        if self.base:
            acl_vmm_release(self.base)
            self.base = 0


# 全局 VMM descriptor 注册表：框架做一次 ``init_with_buffer`` 前收集本进程所有
# 需要进对称空间的 VMM 区间（Buffer 通信 chunk + op workspace + 框架 full 张量）。
_VMM_REGISTRY: list[VmmAllocation] = []

# ``create_sym_tensor_from_source`` 的结果缓存，键见该函数。值里同时存了
# allocation 引用，命中时用它做一次 ``_destroyed`` 存活检查 —— 这样既省掉
# 线性扫描 + 两次 C++ 调用，又不会把已销毁 allocation 的悬垂 view 交出去。
# 随 ``reset_symmetric_descriptors()`` 一起清空。
_SYM_TENSOR_CACHE: dict = {}


def register_symmetric_descriptor(allocation: VmmAllocation) -> None:
    """登记一个 VMM 区间，供框架收集进 buffer-backed 对称空间初始化。"""
    if _SHMEM_BACKING_LOCKED:
        raise RuntimeError(
            "cannot register a symmetric descriptor after buffer-backed "
            "shmem initialization; it would not belong to the active heap")
    _VMM_REGISTRY.append(allocation)
    allocation._registered = True


def symmetric_descriptors():
    """返回所有**未销毁** VMM 区间的 ``(addr, size, mem_handle, fabric_bytes)``。

    与 ``acl_shmem_init_with_buffers`` 的 descriptor 格式一致：mem_handle 传 0 且
    fabric_bytes 为空表示 VA-only（shmem 从 addr 保留分配句柄）；fabric_bytes 非空
    时 shmem 直接用它（不再 re-export）。已 ``destroy()`` 的分配会被跳过。
    """
    return tuple(
        (a.source_va, a.size,
         int(a.mem_handle) if a.mem_handle else 0,
         a.fabric_handle_bytes)
        for a in _VMM_REGISTRY
        if not getattr(a, '_destroyed', False)
    )


def reset_symmetric_descriptors() -> None:
    """清空全局 VMM descriptor 注册表（测试用例间重置用）。

    进程内 shmem init 是一次性的；多 case 复用同一进程时（pytest 模块），
    ``destroy_active_buffers`` 在 finalize 后调用本函数，下一个 case 才能用
    新 Buffer 的 VMM chunk 重新 init。
    """
    if _SHMEM_BACKING_LOCKED:
        raise RuntimeError(
            "cannot reset symmetric descriptors before a successful "
            "collective shmem finalize")
    _VMM_REGISTRY.clear()
    # 注册表清空 ⇒ 所有 source VA 都不再有效，缓存的 view 必须一起丢弃
    _SYM_TENSOR_CACHE.clear()


def _contiguous_strides(shape):
    if len(shape) == 0:
        return ()
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, strides[0] * d)
    return tuple(strides)


def create_sym_tensor_from_ptr(ptr, shape, dtype: torch.dtype, device,
                               stride=None, storage_offset=None) -> torch.Tensor:
    """用已有设备地址构造一个非 owning 的 NPU tensor（``data_ptr()==ptr``）。

    对照 ``shmem.construct_tensor_from_ptr``（torch_npu._C storage 构造）。
    返回的 tensor 不拥有 ``ptr`` 的内存：地址生命周期由调用方持有（VMM
    ``source_va``，或经 ``map_to_sym_ptr`` 得到的本地对称地址）。
    """
    import torch_npu  # noqa: F401

    if not isinstance(ptr, int):
        raise TypeError(f"ptr must be an int address, got {type(ptr).__name__}")
    if ptr <= 0:
        raise ValueError(f"ptr must be positive, got {ptr}")
    shape = tuple(int(d) for d in shape)
    if any(d < 0 for d in shape):
        raise ValueError(f"shape dims must be non-negative, got {shape}")
    if stride is None:
        stride = _contiguous_strides(shape)
    if storage_offset is None:
        storage_offset = 0
    nbytes = _numel(shape) * torch.empty((), dtype=dtype).element_size()
    # torch_npu._C._construct_storage_from_data_pointer 需要 torch.device，
    # 归一化字符串（如 'npu:1'）为 torch.device。
    if not isinstance(device, torch.device):
        device = torch.device(device)
    metadata = {
        "data_ptr": int(ptr),
        "device": device,
        "nbytes": nbytes,
        "dtype": dtype,
        "size": list(shape),
        "stride": [int(s) for s in stride],
        "storage_offset": int(storage_offset),
    }
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"])
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        metadata, storage)


def acl_vmm_access_device_id(device_id: int) -> int:
    """把进程内逻辑 device id 翻译成 ``aclrtMemSetAccess`` 需要的物理号。

    CANN 9.2（Ascend950）实测两个 VMM 接口的 device id 语义不同：
    ``aclrtMallocPhysical`` 收 ASCEND_RT_VISIBLE_DEVICES 列表内的**逻辑下标**，
    而 ``aclrtMemSetAccess`` 的 location.id 按**全局物理号**解释
    （visible='1' 时 set_access(0)→507899、set_access(1)→OK）。
    vLLM DP（dp>1, tp=1）会把每个 rank 进程收窄成单卡，逻辑 0 ≠ 物理卡号即触发；
    未设该环境变量时两者相同，行为不变。
    """
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        entries = [p.strip() for p in visible.split(",") if p.strip()]
        if 0 <= int(device_id) < len(entries):
            try:
                return int(entries[int(device_id)])
            except ValueError:
                pass
    return int(device_id)


def _alloc_vmm_source_split_id(alloc_size: int, malloc_device: int,
                               access_device: int):
    """``acl_vmm_alloc_source`` 的等价实现，malloc/set_access 的 id 分开传。

    仅当逻辑号 ≠ 物理号（进程可见设备被收窄）时使用；否则仍走 C++ 实现，
    不为常见路径引入分支差异。
    """
    import torch_npu  # noqa: F401

    handle = acl_vmm_malloc_physical(alloc_size, malloc_device)
    va = 0
    mapped = False
    try:
        va = acl_vmm_reserve(alloc_size)
        acl_vmm_map(va, alloc_size, handle, 0)
        mapped = True
        acl_vmm_set_access(va, alloc_size, access_device)
        soc = torch.npu.get_device_name(malloc_device) or ""
        if "950" in soc:
            return va, handle, 0, alloc_size, acl_vmm_export_fabric(handle)
        return va, handle, acl_vmm_export_shareable(handle), alloc_size, b""
    except Exception:
        if mapped:
            try:
                acl_vmm_unmap(va)
            except Exception:  # noqa: BLE001
                pass
        if va:
            try:
                acl_vmm_release(va)
            except Exception:  # noqa: BLE001
                pass
        if handle:
            try:
                acl_vmm_free_physical(handle)
            except Exception:  # noqa: BLE001
                pass
        raise


def create_vmm_physical(shape, dtype: torch.dtype, device: int):
    """分配一个 CANN VMM 物理 chunk，并把 source_va 包成 NPU tensor。

    返回 ``(tensor, mem_handle, shareable_va)``；``tensor.data_ptr()`` 即已登记的
    source VA。CANN VMM 物理分配必须按分配粒度对齐（2MiB），因此**实际分配大小会
    round up 到粒度**（多余的 padding 只是物理/对称 VA 余量；tensor 仍只覆盖请求的
    shape）。分配会：
      - 构造 ``VmmAllocation`` 并附加为 ``tensor._vmm_allocation``（destroy/GC 走
        VMM 逆操作释放，不是 aclshmem_free）；
      - 登记进全局 descriptor 注册表（``register_symmetric_descriptor``），供框架
        收集后做一次 ``shmem_init_with_buffer``。
    """
    nbytes = _chunk_nbytes(shape, dtype)
    device_id = int(device)
    gran = int(acl_vmm_granularity(device_id))
    alloc_size = ((nbytes + gran - 1) // gran) * gran
    # Ascend950 上 fabric_bytes 非空（acl_vmm_alloc_source 内部导出 Fabric V2）。
    # 进程可见设备被收窄（vLLM DP）时，set_access 需要物理号而非逻辑号。
    access_device_id = acl_vmm_access_device_id(device_id)
    if access_device_id == device_id:
        va, handle, shareable, size, fabric_bytes = acl_vmm_alloc_source(
            alloc_size, device_id)
    else:
        va, handle, shareable, size, fabric_bytes = _alloc_vmm_source_split_id(
            alloc_size, device_id, access_device_id)
    allocation = VmmAllocation(
        va, handle, shareable, size, fabric_handle_bytes=fabric_bytes,
        device_id=device_id)
    try:
        tensor = create_sym_tensor_from_ptr(
            va, list(shape), dtype, torch.device(f"npu:{device_id}"))
        tensor._vmm_allocation = allocation
        register_symmetric_descriptor(allocation)
    except Exception:
        # Registration can fail when the active heap has already frozen its
        # descriptor set.  This allocation was never registered/referenced by
        # SHMEM, so release its source mapping/handle transactionally.
        allocation.destroy()
        raise
    return tensor, handle, shareable


def map_to_sym_ptr(va, name: str) -> int:
    """把已注册的本地 VA 翻译成本地 shmem 对称地址（PR-653 ``get_buffer_ptr``）。

    要求 buffer-backed 对称堆已初始化（``shmem_init_with_buffer``）且 ``va`` 已登记。
    未注册/未初始化时报错。
    """
    sym = int(acl_shmem_get_buffer_ptr(int(va)))
    if sym == 0:
        raise RuntimeError(
            f"{name}: VA 0x{int(va):x} is not registered in the buffer-backed "
            f"shmem heap; call shmem_init_with_buffer with its descriptor first")
    return sym


# 框架侧约定的本地 source VA 属性名（对应 MindSpeed 的
# ``symmetric_owner_weight_source_vas`` / ``symmetric_weight_slots`` 映射）：
#   ``_moonep_owner_source_va`` —— 本 rank 本地 home chunk 的 source VA
#   ``_moonep_slot_source_va``  —— 本 rank 本地 slot chunk 的 source VA
# Buffer 在 prefetch/reduce_grad 的 VMM 路径里读这两个属性，用
# ``map_to_sym_ptr`` 翻译成本地对称基址。框架构造 full weight 时用
# ``attach_local_source_vas`` 挂上即可。
_OWNER_SOURCE_VA_ATTR = "_moonep_owner_source_va"
_SLOT_SOURCE_VA_ATTR = "_moonep_slot_source_va"


def attach_local_source_vas(tensor, owner_source_va, slot_source_va):
    """把本 rank 本地 home/slot chunk 的 source VA 附加到 full weight 张量上。

    供框架/测试在构造 full weight 后调用；Buffer 的 prefetch/reduce_grad VMM
    路径据此用 ``get_buffer_ptr`` 构造本地 shmem 张量。
    """
    setattr(tensor, _OWNER_SOURCE_VA_ATTR, int(owner_source_va))
    setattr(tensor, _SLOT_SOURCE_VA_ATTR, int(slot_source_va))
    return tensor


def create_sym_tensor_from_source(source_va, shape, dtype: torch.dtype, device,
                                  name: str, *, allow_logical_extent: bool = False
                                  ) -> torch.Tensor:
    """从已登记的本地 source VA 构造本地 shmem 张量（get_buffer_ptr 翻译）。

    ``source_va`` 必须精确对应一个已登记且未销毁的 VMM allocation。默认请求
    view 不能大于该 allocation 的物理字节数；否则非 owning tensor metadata 会把
    kernel 可访问范围伪装得比 caller-owned backing 更大，并在异步 MTE 中越界。

    ``allow_logical_extent`` 仅用于 prefetch home 的既有 ABI：它以逻辑 ``E``
    推导 ``epn=E/R``，但 kernel 实际只访问本地 ``epn`` 行。GradReduce 不得使用
    该豁免，因为其 native ABI 直接把第一维解释为 ``epn``。
    返回的张量 ``data_ptr()`` 即本地对称基址（统一对称槽位的本地部分）。

    **结果按 ``(source_va, shape, dtype, device, allow_logical_extent)`` 缓存。**
    prefetch 路径每层要构造 4 个（2 个 projection × home/slots），而 source VA
    在整轮推理里不变 —— 原先每次都要做一遍「线性扫 ``_VMM_REGISTRY`` +
    ``acl_shmem_get_buffer_ptr``（C++）+ 构造 storage」，实测约 85 us/次。
    命中时只多一次 dict 查找和一次 ``_destroyed`` 属性检查，不做线性扫描。
    """
    key = (int(source_va), tuple(int(d) for d in shape), dtype,
           str(device), bool(allow_logical_extent))
    hit = _SYM_TENSOR_CACHE.get(key)
    if hit is not None:
        cached_alloc, cached_tensor = hit
        # 存活检查：allocation 被 destroy 之后缓存的 view 就是悬垂指针
        if not getattr(cached_alloc, '_destroyed', False):
            return cached_tensor
        del _SYM_TENSOR_CACHE[key]

    source_va = int(source_va)
    allocation = next(
        (
            item for item in _VMM_REGISTRY
            if not getattr(item, '_destroyed', False)
            and int(item.source_va) == source_va
        ),
        None,
    )
    if allocation is None:
        raise RuntimeError(
            f"{name}: source VA 0x{source_va:x} does not identify an active "
            "registered VMM allocation")
    requested_bytes = _chunk_nbytes(shape, dtype)
    if requested_bytes > int(allocation.size) and not allow_logical_extent:
        raise RuntimeError(
            f"{name}: requested symmetric view requires {requested_bytes} bytes, "
            f"but source allocation 0x{source_va:x} owns only "
            f"{int(allocation.size)} bytes")
    sym = map_to_sym_ptr(source_va, name)
    tensor = create_sym_tensor_from_ptr(sym, shape, dtype, device)
    _SYM_TENSOR_CACHE[key] = (allocation, tensor)
    return tensor


class VmmProjection:
    """一段保留 VA + 多个物理 handle 段的投影视图（§6.4 的 ``[E+B]`` full_weight）。

    与 ``SegmentedVAMapping`` 的差异：允许在**更大**的保留 VA 里按任意 gran
    对齐偏移映射各段（``[E+B]`` 视图只需映射本 rank 的 home/slot 段，其余行是
    保留未映射区）。``destroy()`` 只 unmap+release 保留 VA，不释放物理内存
    （物理归各段所属的 ``_vmm_allocation``）。
    """

    def __init__(self, base, segments):
        self.base = int(base)
        self._segments = list(segments)  # (va, size, handle)
        self._destroyed = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        for va, _, _ in self._segments:
            acl_vmm_unmap(va)
        if self.base:
            acl_vmm_release(self.base)
            self.base = 0
        self._destroyed = True


def create_full_projection_tensor(home, slot, E, B, rank, device):
    """构造 §6.4 的真实 ``[E+B, H, H']`` VMM 投影视图（替代 ``torch.empty``）。

    ``home`` / ``slot`` 必须是已登记的 VMM chunk（``create_vmm_physical``），各自的
    ``_vmm_allocation`` 提供 gran 对齐的物理段。映射：
      - ``epn*row % gran == 0`` 时保持"全局专家行"布局：home 段 → 逻辑全局专家行
        ``[rank*epn, (rank+1)*epn)``（``aclrtMapMem`` 要求映射 VA 按 gran 对齐，
        ``epn*row`` 为 gran 倍数即保证所有 rank 的 ``rank*epn*row`` 都对齐）；
      - 否则回退布局：home 段从投影基址起（``[0, epn*row)``），slot 段放在
        ``ALIGN_UP(epn*row, lcm(gran, row))`` 起——``lcm(gran, row)`` 同时满足
        aclrtMapMem 的 gran 对齐与张量行边界（``row``）对齐，``slot_start`` 才能
        精确等于 ``slot_off // row``（否则 slot_off 落在行中间，别名断裂）。若
        ``slot_start > E``（等价 ``lcm(gran,row)/row > E``），该配置无法表示为
        ``[E+B]`` 行对齐投影，抛 ValueError。
    slot 段在"全局"布局下从 ``ALIGN_UP(E*row, gran)`` 起（``E*row`` 为 gran 倍数
    时即逻辑 ``[E, E+B)`` 行）。保留 VA 只覆盖实际映射的两个段，其余行是保留未映射
    区——kernel 不读投影视图（远端行走对称寻址，见 §6.3/§6.4）。

    返回张量 ``data_ptr()`` 即投影 VA 基址；该基址**未登记**进 buffer-backed shmem
    init，不能 ``map_to_sym_ptr``。``tensor._projection`` 持有可销毁句柄
    （``home_start`` / ``slot_start`` 给出 home/slot 段在 ``[E+B]`` 逻辑行号，
    供测试校验别名）。同一物理 handle 被映射进两个 VA（source_va 与投影基址），
    物理内存仍归 home/slot 的 ``_vmm_allocation``。
    """
    H, Hp = int(home.shape[1]), int(home.shape[2])
    row = H * Hp * int(home.element_size())
    gran = int(acl_vmm_granularity(int(device)))
    epn = int(home.shape[0])
    home_alloc = getattr(home, '_vmm_allocation', None)
    slot_alloc = getattr(slot, '_vmm_allocation', None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError(
            "home/slot must be VMM chunks (create_vmm_physical)")
    home_size = int(home_alloc.size)
    slot_size = int(slot_alloc.size)
    step = _lcm(gran, row)
    if (epn * row) % gran == 0:
        # 全局专家行布局（原语义；``aclrtMapMem`` 要求映射 VA 按 gran 对齐，
        # ``epn*row`` 为 gran 倍数即保证所有 rank 的 ``rank*epn*row`` 都对齐）。
        home_off = int(rank) * epn * row
        slot_off = _align_up(E * row, gran)
    else:
        # 回退布局：home 从投影基址起、slot 段放在 lcm(gran, row) 对齐的偏移
        # （同时满足 aclrtMapMem 的 gran 对齐与张量行边界对齐，slot_start 才
        # 精确等于 slot_off // row；否则 slot_off 落在行中间，别名断裂）。
        home_off = 0
        slot_off = _align_up(epn * row, step)
    home_start = home_off // row
    slot_start = slot_off // row
    if slot_start + B > E + B:
        raise ValueError(
            f"config not representable as a [E+B] row-aligned projection: "
            f"lcm(gran,row)/row = {step // row} forces slot_start={slot_start} "
            f"> E={E}; choose H*Hp so that lcm(2MiB, H*Hp*elemsize) <= E*row")
    # 保留 VA 只覆盖实际映射的两个段（``ALIGN_UP(a)+ALIGN_UP(b)`` 可能超过
    # ``ALIGN_UP((E+B)*row)``，不能用逻辑总大小估算）。
    total = _align_up(max(home_off + home_size, slot_off + slot_size), gran)
    base = int(acl_vmm_reserve(total))
    segments = []
    try:
        for alloc, off in ((home_alloc, home_off), (slot_alloc, slot_off)):
            va = base + off
            acl_vmm_map(va, int(alloc.size), alloc.mem_handle, 0)
            acl_vmm_set_access(
                va, int(alloc.size), acl_vmm_access_device_id(int(device)))
            segments.append((va, int(alloc.size), alloc.mem_handle))
    except Exception:
        for va, _, _ in segments:
            try:
                acl_vmm_unmap(va)
            except Exception:
                pass
        try:
            acl_vmm_release(base)
        except Exception:
            pass
        raise
    tensor = create_sym_tensor_from_ptr(
        base, [E + B, H, Hp], home.dtype, torch.device(f"npu:{int(device)}"))
    proj = VmmProjection(base, segments)
    proj.home_start = home_start
    proj.slot_start = slot_start
    tensor._projection = proj
    return tensor


def sym_heap_base(mem_type: int = 1) -> int:
    """shmem 对称堆基址（mem_type: 0=HOST_SIDE, 1=DEVICE_SIDE）。"""
    return int(acl_shmem_get_heap_base(int(mem_type)))


def shmem_init_with_buffer(descriptors, my_pe: int, n_pes: int, uid_bytes,
                           tail_bytes: int = 0,
                           data_op_engine_type: int = ACLSHMEM_DATA_OP_MTE,
                           device_id: int | None = None,
                           enable_udma: bool | None = None, qp_count: int = 0) -> None:
    """用 caller 映射的 VMM buffer 初始化 buffer-backed shmem 对称堆（collective）。

    ``descriptors`` = ``(addr, size, mem_handle, fabric_handle)`` 列表，每项对应一个
    已登记 VMM buffer（见 ``symmetric_descriptors()``）。所有 rank 传相同的
    descriptor 数量/大小。``uid_bytes`` 是 rank0 的 ``acl_shmem_get_unique_id()``
    经 dist 广播后的 UID。``tail_bytes`` 非零时 shmem 会保留一段自家 malloc 堆
    （``aclshmem_malloc`` 可用）。
    """
    global _SHMEM_BACKING_LOCKED
    if device_id is None:
        device_id = int(torch.npu.current_device())
    # Freeze the registry before taking/converting the descriptor snapshot and
    # entering the collective.  Otherwise a second Python thread could append
    # an allocation after the snapshot but before native init returns.
    was_locked = _SHMEM_BACKING_LOCKED
    _SHMEM_BACKING_LOCKED = True
    try:
        native_descriptors = [
            (int(addr), int(size), int(mh),
             fb if isinstance(fb, bytes) else bytes(fb or b""))
            for (addr, size, mh, fb) in descriptors
        ]
        rc = acl_shmem_init_with_buffers(
            int(my_pe), int(n_pes), int(tail_bytes), int(data_op_engine_type),
            uid_bytes, native_descriptors, int(device_id), bool(enable_udma), int(qp_count))
    except Exception:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise
    if rc != 0:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise RuntimeError(f"acl_shmem_init_with_buffers failed with {rc}")


def alloc_sym_chunk(shape, dtype: torch.dtype, device: int) -> torch.Tensor:
    """分配一个登记进 buffer-backed init 的 VMM chunk，返回 source 张量。

    供不经 Buffer/ShmemRuntime 的低层测试使用：分配后（已登记进全局 descriptor
    注册表），调用方做 init，再用 ``sym_tensor()`` 把该张量翻译成内核要的本地
    对称张量。
    """
    return create_vmm_physical(shape, dtype, device)[0]


def sym_tensor(source_tensor: torch.Tensor) -> torch.Tensor:
    """把已登记的 source 张量翻译成内核要的本地对称张量（get_buffer_ptr）。

    ``source_tensor.data_ptr()`` 必须是已登记进 buffer-backed heap 的 source VA；
    返回张量的 ``data_ptr()`` 即本地对称基址（统一对称槽位的本地部分）。
    """
    return create_sym_tensor_from_ptr(
        map_to_sym_ptr(source_tensor.data_ptr(), 'sym_tensor'),
        source_tensor.shape, source_tensor.dtype, source_tensor.device)


def init_buffer_backed(rank: int, n_pes: int, uid_bytes,
                       tail_bytes: int = 0,
                       data_op_engine_type: int = ACLSHMEM_DATA_OP_MTE) -> None:
    """Collective buffer-backed shmem init，用全局已登记的 VMM descriptor。

    低层测试不经 ShmemRuntime 时直接用：调用前需先用 ``alloc_sym_chunk`` /
    ``create_vmm_physical`` 分配并登记所有对称 chunk，``uid_bytes`` 由调用方在
    rank0 生成（``acl_shmem_get_unique_id``）并经 dist 广播。``tail_bytes`` 非零
    时保留 shmem-owned malloc 堆（``aclshmem_malloc`` 可用）。
    """
    shmem_init_with_buffer(
        symmetric_descriptors(), int(rank), int(n_pes), uid_bytes,
        tail_bytes=int(tail_bytes), data_op_engine_type=data_op_engine_type)
