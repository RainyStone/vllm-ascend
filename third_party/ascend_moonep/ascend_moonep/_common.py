"""
Ascend port shared helpers for ``ascend_moonep``.

Replaces the VMM/multicast alignment and IPC-fd machinery of ``moonep/buffer.py``
with shmem (cann-shmem) symmetric-memory primitives. Everything here is
host-side; device-side symmetric addressing lives in the migrated kernels
(``aclshmem_ptr`` at the kernel seam).

Import notes (verified against the vendored shmem source):
  - ``shmem.core`` re-exports only ``init_final``/``rma``/``memory`` names, so
    ``Buffer`` is NOT ``core.Buffer`` — import it from ``shmem.core.utils``.
  - ``my_pe``/``n_pes``/``SignalOp``/``ComparisonType``/``init_status`` live in
    ``shmem.core.direct`` (not re-exported at ``core`` level).
  - ``construct_tensor_from_ptr`` is NOT in ``shmem.__all__`` (and the module's
    ``__getattr__`` only loads natives for names in ``__all__``), so it is not
    a reliable ``shmem.construct_tensor_from_ptr`` accessor. Use the public
    ``shmem.aclshmem_create_tensor`` instead and recover the ``Buffer`` handle
    from the returned tensor's ``data_ptr()``.
"""

from __future__ import annotations

import torch

from ._C import acl_shmem_putmem, acl_shmem_getmem


ACLSHMEM_DATA_OP_MTE = 0x01
ACLSHMEM_DATA_OP_UDMA = 0x08


class SymBuffer:
    """对称内存句柄（addr + length），替代 shmem.core.utils.Buffer。"""

    __slots__ = ("addr", "length")

    def __init__(self, addr: int, length: int):
        self.addr = int(addr)
        self.length = int(length)


# --- 原实现（方案 A：shmem Python 包）---
# import shmem.core as core
# from shmem.core.utils import Buffer as SymBuffer

# shmem symmetric-heap page size (DEVICE_LARGE_PAGE_SIZE). Only the per-PE heap
# ``mem_size`` passed to ``core.init`` must be a multiple of this (see
# Troubleshooting_FAQs: non-2MiB-aligned mem_size fails). Individual allocations
# (``aclshmem_create_tensor`` / ``core.buffer``) need NO 2 MiB padding: the
# symmetric allocator aligns each allocation internally, so chunk sizes are used
# exactly as requested on Ascend.
SHMEM_LARGE_PAGE = 2 * 1024 * 1024

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.uint8: 1,
}


# ================================================================
# Symmetric-heap sizing.
#
# The per-PE ``mem_size`` passed to ``core.init`` must cover every symmetric
# allocation made in the process. Symmetric memory is collective: every rank
# reserves the same footprint on its own heap, so the required size depends only
# on the model config — seqlen (S), topk count (K), expert count (E), the hidden
# sizes (H, H') and the number of EP ranks (R). All sizes below are per rank.
# ================================================================

# Barrier signal words per rank (dedicated symmetric ``barrier_buf`` chunk).
BARRIER_SLOTS = 3
# Default per-op symmetric workspace for the RMA-capable ops (dispatch/combine).
OP_WS_BYTES_DEFAULT = 16 * 1024 * 1024


def planning_symmetric_workspace_nbytes(E: int, NvS: int, R: int) -> int:
    """Bytes of Planning's per-rank symmetric communication workspace.

    The regions hold local TPE, source-side duplicate payload metadata and the
    compact duplicate results exchanged between ranks. All offsets are aligned
    to 32 bytes, matching the host launcher and device kernel layout.
    """
    align_elems = 8

    def align_up(value: int) -> int:
        return ((value + align_elems - 1) // align_elems) * align_elems

    dup_rank_counts_off = align_up(E)
    # Count and prefix exchanges each reserve one 32-byte slot per peer.
    # Payload completion reserves one flag per source rank.
    dup_payload_done_off = dup_rank_counts_off + 2 * R * align_elems
    dup_groups_off = dup_payload_done_off + align_up(R)
    dup_loffs_off = dup_groups_off + align_up(NvS * 3)
    return (dup_loffs_off + align_up(NvS)) * 4


def compute_shmem_mem_size(
    S: int,
    H: int,
    K: int,
    E: int,
    H_prime: int,
    num_ep_ranks: int,
    *,
    op_ws_bytes: int = OP_WS_BYTES_DEFAULT,
    num_reduce_buffers: int = 3,
    num_prefetch_slots: int = 3,
    owner_weight_bytes: int = 0,
) -> int:
    """Per-PE symmetric heap bytes required to run MoonEP with the given config.

    Sums every symmetric allocation the Buffer and the framework make, then
    rounds up to the 2 MiB large page (``core.init`` requires ``mem_size`` to be
    2 MiB-aligned). With ``B = E // num_ep_ranks`` (this rank's expert share)
    and ``NvS = S * K``, the itemized calculation is:

      Buffer symmetric chunks (per rank):
        hidden_buf  [NvS, H]           bf16 : NvS * H * 2
        weights_buf [NvS]              int32: NvS * 4
        barrier_buf [BARRIER_SLOTS]    int32: BARRIER_SLOTS * 4
        dispatch_sync_buffer                 : Token control + local Token AIV sync
      RMA op workspaces:
        Planning exact communication workspace
        dispatch + combine: 2 * op_ws_bytes
      Grad-reduce buffers [B, H, H']   fp32 : num_reduce_buffers * B * H * H' * 4
      Prefetch weight slots [B, H, H'] bf16 : num_prefetch_slots * B * H * H' * 2
      owner_weight_bytes: caller-supplied extra (e.g. single-owner weight tensors
        the framework allocates outside the Buffer).

    The result is a lower bound for the default counts; a framework that
    allocates more or fewer extra symmetric tensors passes the matching
    parameters. Equal on every rank (B/R are global). 注意：buffer-backed 路径下
    对称堆大小由注册进 init_with_buffer 的 VMM descriptor 决定，本函数仅作
    预算参考（legacy ``ShmemRuntime.init(mem_size=...)`` 已移除）。
    """
    R = num_ep_ranks
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
    B = E // R
    NvS = S * K

    buffer_chunks = (
        NvS * H * 2                # hidden_buf (bf16)
        + NvS * 4                  # weights_buf (int32)
        + BARRIER_SLOTS * 4        # barrier_buf (int32)
        + R * 16 * 6 * 512         # dispatch_sync_buffer 的 Token 控制区
        + 16 * 4 * 512             # 16 个 Token AIV 的四轮本地同步区
    )
    op_workspaces = planning_symmetric_workspace_nbytes(E, NvS, R) + 2 * op_ws_bytes
    grad_reduce = num_reduce_buffers * B * H * H_prime * _ELEM_SIZE[torch.float32]
    prefetch = num_prefetch_slots * B * H * H_prime * _ELEM_SIZE[torch.bfloat16]

    total = (
        buffer_chunks + op_workspaces + grad_reduce + prefetch
        + int(owner_weight_bytes)
    )
    # mem_size must be 2 MiB-aligned (only the heap itself needs this on Ascend).
    return ((total + SHMEM_LARGE_PAGE - 1) // SHMEM_LARGE_PAGE) * SHMEM_LARGE_PAGE


def sym_view(buf: SymBuffer, byte_off: int, nbytes: int) -> SymBuffer:
    """Return a sub-range of a local symmetric allocation as an shmem Buffer.

    The address is the *local symmetric* address; ``put``/``get`` translate it
    to the target PE internally (pass ``pe``). Byte offsets follow the ``ctx``
    meta-layout constants, so the CUDA meta layout carries over unchanged.
    """
    return SymBuffer(buf.addr + byte_off, nbytes)


def tensor_buffer(t: torch.Tensor) -> SymBuffer:
    """Transient shmem Buffer view over a tensor.

    For a tensor allocated from the shmem heap (``create_sym_dist_tensor`` /
    ``aclshmem_create_tensor``), ``t.data_ptr()`` is the local symmetric base;
    this wraps it in the ``Buffer`` handle the shmem Python API expects for
    put/get/signal and for ``core.free``. The handle is transient — the tensor
    remains the single owner of the address.
    """
    return SymBuffer(t.data_ptr(), t.numel() * t.element_size())


def sym_put(local_chunk: torch.Tensor, byte_off: int, src: SymBuffer, pe: int, stream: int) -> None:
    """Copy local ``src`` to ``pe``'s symmetric region at ``byte_off`` of ``local_chunk``.

    [方案 B] C 接口 on_stream 非阻塞 RMA（与 core.put 语义对齐，内部无 quiet）；
    stream 参数保留，同步由调用方的 barrier/synchronize 负责。
    """
    acl_shmem_putmem(local_chunk.data_ptr() + byte_off, src.addr, src.length,
                     pe, stream)
    # --- 原实现（方案 A：core.put）---
    # core.put(sym_view(tensor_buffer(local_chunk), byte_off, src.length), src, pe, stream)


def sym_get(local_chunk: torch.Tensor, byte_off: int, dst: SymBuffer, pe: int, stream: int) -> None:
    """Copy ``pe``'s symmetric region at ``byte_off`` of ``local_chunk`` into local ``dst``.

    [方案 B] C 接口 on_stream 非阻塞 RMA（与 core.put 语义对齐，内部无 quiet）；
    stream 参数保留，同步由调用方的 barrier/synchronize 负责。
    """
    acl_shmem_getmem(dst.addr, local_chunk.data_ptr() + byte_off, dst.length,
                     pe, stream)
    # --- 原实现（方案 A：core.get）---
    # core.get(dst, sym_view(tensor_buffer(local_chunk), byte_off, dst.length), pe, stream)


def sym_get_into(
    local_chunk: torch.Tensor, byte_off: int, dst_tensor: torch.Tensor, pe: int, stream: int
) -> None:
    """Materialize a remote rank's symmetric region into a local NPU tensor.

    This is the Ascend replacement for CUDA's VMM "single-owner tensor visible
    everywhere": a remote rank's chunk (or chunk slice) is fetched into a local
    tensor via host RMA. [方案 B] C 接口 on_stream 非阻塞实现；stream 参数保留，
    同步由调用方负责。
    """
    nbytes = dst_tensor.numel() * dst_tensor.element_size()
    acl_shmem_getmem(dst_tensor.data_ptr(), local_chunk.data_ptr() + byte_off,
                     nbytes, pe, stream)
    # --- 原实现（方案 A：core.get）---
    # core.get(
    #     tensor_buffer(dst_tensor),
    #     sym_view(tensor_buffer(local_chunk), byte_off, dst_tensor.numel() * dst_tensor.element_size()),
    #     pe, stream,
    # )


def sym_put_from(
    local_chunk: torch.Tensor, byte_off: int, src_tensor: torch.Tensor, pe: int, stream: int
) -> None:
    """Push a local NPU tensor into a remote rank's symmetric region.

    [方案 B] C 接口 on_stream 非阻塞实现；stream 参数保留，同步由调用方负责。
    """
    nbytes = src_tensor.numel() * src_tensor.element_size()
    acl_shmem_putmem(local_chunk.data_ptr() + byte_off, src_tensor.data_ptr(),
                     nbytes, pe, stream)
    # --- 原实现（方案 A：core.put）---
    # core.put(
    #     sym_view(tensor_buffer(local_chunk), byte_off, src_tensor.numel() * src_tensor.element_size()),
    #     tensor_buffer(src_tensor),
    #     pe,
    #     stream,
    # )


def record_streams(tensors, stream: torch.Stream) -> None:
    """Record ``stream`` against tensors so torch does not free them early."""
    for t in tensors:
        if t is not None:
            t.record_stream(stream)


def cross_rank_barrier(ctx: dict, stream: int) -> None:
    """Host-side cross-rank barrier（dispatch/combine 的 inter_rank_sync）。

    **当前为 no-op**：host-blocking 跨 rank 集体（无论 shmem ``acl_shmem_barrier_all``
    还是 ``torch.distributed.barrier``）在 buffer-backed + 异步流（comm stream）场景
    下都容易卡死（等待 pending RMA / 非默认 stream 上的 HCCL 集体互相等待）。跨卡
    RMA 的顺序由 kernel 内 grid_sync 与调用方的 ``torch.npu.synchronize()`` /
    ``dist.barrier`` 保证。
    ``stream`` / ``_barrier_round`` 保留兼容。
    """
    ctx['_barrier_round'] = int(ctx.get('_barrier_round', 0)) + 1
