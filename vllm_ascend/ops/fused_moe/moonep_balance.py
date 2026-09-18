# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""MoonEP load-balance path (MindSpeed-MM style): per-rank prefetch slots.

Owner-only dispatch (the default MoonEP comm path) sends every row to the rank
that owns its expert. Under skewed routing that concentrates rows on the hot
expert's rank. The MoonEP planning kernel can instead keep some rows *local*
and mark their (remote) expert in ``plan.experts_to_copy``; the rank then
prefetches that expert's weights into one of its B slots (SHMEM RMA) and
computes the rows locally against the [epn+B] weight alias. Loads stay
balanced across ranks without vLLM's dynamic EPLB layer — the same trade-off
MindSpeed-MM_MoonEP's ``shmem_ep_dispatcher`` makes ("perfect load balance").

Per layer, per *payload* (bf16: w13 [epn,H,2F] and w2 [epn,F,H]; W4A8MXFP adds
the two e8m0 scales, four payloads in total — a remote expert is only usable if
its scales travel with it):
  home [epn, R, C]  VMM chunk, bf16 *byte carrier* mirroring this rank's local
                    weight rows (published once at init; inference weights are
                    static). ``R*C*2`` == the payload's per-expert byte count.
  slot [B,  R, C]   VMM chunk shared across layers of one shape (B = epn);
                    filled per step by prefetch_weight with the remote rows of
                    plan.experts_to_copy
  full [E+B, R, C]  projection view (home at [rank*epn, ..)) used by the
                    prefetch kernel (via attached source VAs)
  alias [epn+B,..]  zero-copy packed view (home rows 0..epn, slot rows epn..)
                    used as the grouped-matmul weights; shape/stride/dtype/
                    npu_format are read off the weight tensor at runtime

The prefetch chain only reads ``shape[1:]`` plus the attached source VAs and
copies ``H*Hp*2`` bytes per expert as flat bytes
(ascend-moonep/kernels/prefetch.cpp:97-141), so the bf16 carrier is transparent
for quantized payloads — no library, csrc or kernel change is involved.

Scope: bf16 (AscendUnquantizedFusedMoEMethod) plus quantized layouts whose
per-expert block is dense (``stride(0)*element_size == numel/E``) — verified
on-card for W4A8MXFP (uint8 NZ weights + e8m0 ND scales). Anything else raises.
Activated with ``MOONEP_SLOT_MODE=1`` (default off: owner-only mode);
``MOONEP_SLOT_QUANT=0`` additionally forces the pre-quant scope even when the
mode is on.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from dataclasses import dataclass

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEWeights

_slot_mode = os.environ.get("MOONEP_SLOT_MODE", "0") == "1"
_slot_quant = os.environ.get("MOONEP_SLOT_QUANT", "1") == "1"

# prefetch kernel 的 UB 约束（ascend-moonep/kernels/prefetch.cpp:100-105）：
# ``elements_per_row`` 就是载体的 Hp，且要求 ub_bytes / sizeof(bfloat16) / Hp >= 1
# （UB 内部暂存区 < 188 KiB），所以 Hp 最多 32768 个 2 字节元素 = 64 KiB 字节。
_MAX_CARRIER_ROW_ELEMS = 32768


def slot_mode_enabled() -> bool:
    return _slot_mode


def slot_quant_enabled() -> bool:
    return _slot_quant


# ---------------------------------------------------------------------------
# Payload layout: bf16 byte carriers + runtime-read weight metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PayloadSpec:
    """One weight payload's mirror/prefetch/alias geometry.

    Everything layout-ish is read off ``ref`` at runtime (shape, stride, dtype,
    element_size, npu_format) and never hardcoded: on-card measurement shows the
    NZ w13/w2 payloads come back as ``dtype=uint8`` with
    ``get_npu_format() == 50`` (the ``29`` passed to ``npu_format_cast`` is an
    input, not the result), and the two e8m0 scales are 4-D non-contiguous ND
    views (format 2).
    """

    name: str
    ref: torch.Tensor
    is_nz: bool  # NZ transposed view (w13/w2): alias needs cast-shape + transpose
    row_bytes: int  # per-expert bytes == stride(0) * element_size()
    nbytes: int  # per-expert bytes * local expert count
    carrier: tuple[int, int]  # bf16 (R, C) with R*C*2 == row_bytes
    npu_format: int


def _flat_bytes(t: torch.Tensor, nbytes: int) -> torch.Tensor:
    """View the first ``nbytes`` bytes of ``t``'s payload as flat uint8.

    Starts at ``data_ptr()``, so a weight that is a slice of a larger storage
    (``full[rank*epn:(rank+1)*epn]`` in the harnesses) mirrors its own rows and
    not the head of the table. Non-contiguous NZ / 4-D scale views cannot be
    reshaped and ``copy_`` between two NZ tensors is not a path we rely on, so
    mirrors move raw bytes; ``_make_spec`` has already established that every
    expert owns one dense ``row_bytes`` block, which makes the payload a dense
    range starting at ``data_ptr()``.
    """
    return _sym_tensor(t.data_ptr(), (int(nbytes),), torch.uint8, t.device,
                       nbytes=int(nbytes))


def _sym_tensor(ptr, shape, dtype, device, *, stride=None, storage_offset=0,
                nbytes=None, npu_format=None) -> torch.Tensor:
    """Build a non-owning tensor over an existing device address.

    Counterpart of the library's ``buffer_c.create_sym_tensor_from_ptr`` (pure
    Python, no C++ behind it) with one extra ``npu_format`` metadata key — the
    measured way to hand a VMM alias the NZ format without any transdata
    (shape/stride/storage stay untouched; verified on card). Phase 2 can fold
    this parameter into the library.
    """
    import torch_npu

    shape = tuple(int(d) for d in shape)
    if stride is None:
        stride = _contiguous_strides(shape)
    if nbytes is None:
        nbytes = _numel(shape) * torch.empty((), dtype=dtype).element_size()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    metadata = {
        "data_ptr": int(ptr),
        "device": device,
        "nbytes": int(nbytes),
        "dtype": dtype,
        "size": list(shape),
        "stride": [int(s) for s in stride],
        "storage_offset": int(storage_offset),
    }
    if npu_format is not None:
        metadata["npu_format"] = int(npu_format)
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"])
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        metadata, storage)


def _contiguous_strides(shape) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.insert(0, strides[0] * dim)
    return tuple(strides)


def _numel(shape) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _carrier_shape(row_bytes: int, natural: tuple[int, int] | None = None) -> tuple[int, int]:
    """Factor the per-expert byte count into a bf16 carrier (R, C), R*C*2 == row_bytes.

    Any factorization is equivalent for the prefetch chain (it copies flat bytes),
    but bf16 weights keep their own natural row split so the unquantized path
    stays byte-for-byte what it was before. Quantized payloads have no bf16 row
    split to reuse, so they take the widest C within the kernel's UB bound.
    """
    elems = row_bytes // 2
    if elems <= 0 or elems * 2 != row_bytes:
        raise ValueError(f"row_bytes={row_bytes} must be a positive even number.")
    if natural is not None and natural[0] * natural[1] == elems and natural[1] <= _MAX_CARRIER_ROW_ELEMS:
        return (int(natural[0]), int(natural[1]))
    cols = min(elems, _MAX_CARRIER_ROW_ELEMS)
    while cols > 1 and elems % cols:
        cols //= 2
    return (elems // cols, cols)


def _make_spec(name: str, ref: torch.Tensor, *, is_nz: bool,
               natural: tuple[int, int] | None = None) -> _PayloadSpec:
    import torch_npu

    experts = int(ref.shape[0])
    row_bytes = int(ref.stride(0)) * int(ref.element_size())
    if experts <= 0 or row_bytes <= 0:
        raise NotImplementedError(
            f"MoonEP slot mode: {name} has an empty/unstrided layout "
            f"(shape={tuple(ref.shape)} stride={tuple(ref.stride())}).")
    # The packed [epn+B] alias assumes one dense block per expert; NZ padding
    # would silently shift every row after the first. Both sides in bytes --
    # for 1-byte payloads the element count happens to match, for bf16 it does not.
    numel_per_expert = ref.numel() // experts * int(ref.element_size())
    if numel_per_expert != row_bytes:
        raise NotImplementedError(
            f"MoonEP slot mode: {name} has per-expert padding "
            f"(numel/experts={numel_per_expert} != stride(0)*element_size={row_bytes}); "
            "the packed [epn+B] alias needs one contiguous block per expert.")
    nbytes = experts * row_bytes
    # The weight may be a slice of a larger table (the harnesses build ranks from
    # `full[rank*epn:(rank+1)*epn]`), so it need not start at the storage head --
    # it only needs `nbytes` to fit from its own offset. `_flat_bytes` starts at
    # `data_ptr()`, which already accounts for the offset.
    offset_bytes = int(ref.storage_offset()) * int(ref.element_size())
    storage = ref.untyped_storage()
    if offset_bytes + nbytes > int(storage.nbytes()):
        raise NotImplementedError(
            f"MoonEP slot mode: {name} payload runs past its storage "
            f"(storage_offset*element_size={offset_bytes} + nbytes={nbytes} > "
            f"storage_nbytes={int(storage.nbytes())}).")
    return _PayloadSpec(
        name=name,
        ref=ref,
        is_nz=is_nz,
        row_bytes=row_bytes,
        nbytes=nbytes,
        carrier=_carrier_shape(row_bytes, natural),
        npu_format=int(torch_npu.get_npu_format(ref)),
    )


def _infer_F(layer, H: int) -> int:
    """Derive the per-partition ffn size from the layer's own weight layout.

    Both layouts keep ``2F`` on the last axis, so only the hidden axis differs:
    bf16 leaves w13 as ``[epn, H, 2F]``, while W4A8MXFP packs 2 values per byte
    along hidden and transposes to ``[epn, H/2, 2F]`` (measured on card, see
    docs/moonep_slot_quant.md).
    """
    w13 = layer.w13_weight
    if w13.dim() != 3 or int(w13.shape[2]) % 2:
        raise NotImplementedError(
            f"MoonEP slot mode: unexpected w13 shape {tuple(w13.shape)}.")
    F = int(w13.shape[2]) // 2
    if w13.dtype == torch.bfloat16:
        if int(w13.shape[1]) != H:
            raise NotImplementedError(
                f"MoonEP slot mode: bf16 w13 should be [epn, H={H}, 2F], got "
                f"{tuple(w13.shape)}."
            )
        if tuple(layer.w2_weight.shape) != (int(w13.shape[0]), F, H):
            raise NotImplementedError(
                f"MoonEP slot mode: bf16 w2 should be [epn, F={F}, H={H}], got "
                f"{tuple(layer.w2_weight.shape)}."
            )
        return F
    if int(w13.shape[1]) * 2 != H:
        raise NotImplementedError(
            f"MoonEP slot mode: quantized w13 should be [epn, H/2={H // 2}, 2F], "
            f"got {tuple(w13.shape)} (2 values per byte along hidden)."
        )
    scale13 = getattr(layer, "w13_weight_scale", None)
    if scale13 is None:
        raise NotImplementedError(
            "MoonEP slot mode with quantized weights requires w13_weight_scale."
        )
    # DS leaves the scale 4-D after process_weights_after_loading:
    # [epn, H/(2*group), 2F, 2] — cross-check 2F on its own axis.
    if scale13.dim() == 4 and int(scale13.shape[2]) != 2 * F:
        raise NotImplementedError(
            f"MoonEP slot mode: w13_weight_scale {tuple(scale13.shape)} disagrees "
            f"with w13's 2F={2 * F}."
        )
    return F


def _payload_specs(layer, H: int, F: int) -> list[_PayloadSpec]:
    """Every payload this layer needs mirrored, prefetched and aliased."""
    w13, w2 = layer.w13_weight, layer.w2_weight
    if w13.dtype == torch.bfloat16:
        return [
            _make_spec("w13", w13, is_nz=False, natural=(H, 2 * F)),
            _make_spec("w2", w2, is_nz=False, natural=(F, H)),
        ]
    if not _slot_quant:
        raise NotImplementedError(
            "MoonEP slot mode with quantized MoE weights is disabled "
            "(MOONEP_SLOT_QUANT=0)."
        )
    scale13 = getattr(layer, "w13_weight_scale", None)
    scale2 = getattr(layer, "w2_weight_scale", None)
    if scale13 is None or scale2 is None:
        raise NotImplementedError(
            "MoonEP slot mode with quantized weights requires both "
            "w13_weight_scale and w2_weight_scale on the layer."
        )
    for name, weight in (("w13", w13), ("w2", w2)):
        if int(weight.element_size()) != 1:
            raise NotImplementedError(
                f"MoonEP slot mode: unsupported {name} layout dtype={weight.dtype} "
                f"element_size={weight.element_size()}."
            )
    return [
        _make_spec("w13", w13, is_nz=True),
        _make_spec("w13_scale", scale13, is_nz=False),
        _make_spec("w2", w2, is_nz=True),
        _make_spec("w2_scale", scale2, is_nz=False),
    ]


# ---------------------------------------------------------------------------
# Per-(E, H, F, payload geometry) shared slot pool + per-layer home mirrors
# ---------------------------------------------------------------------------
class _SharedSlots:
    """B prefetch slot chunks per payload, shared across layers of one shape.

    Safe: one layer's dispatch->prefetch->GEMM->combine fully contains its slot
    use, and MoE layers run serially on the stream."""

    _lock = threading.Lock()
    _pools: dict[tuple, dict] = {}

    @classmethod
    def get(cls, E: int, H: int, F: int, B: int, rank: int, dev: int,
            specs: list[_PayloadSpec]) -> dict:
        key = (E, H, F, B, rank, dev,
               tuple((spec.name, *spec.carrier) for spec in specs))
        with cls._lock:
            pool = cls._pools.get(key)
            if pool is None:
                from ascend_moonep.buffer_c import create_vmm_physical

                # 每个 payload 单独一块槽：四个 payload 的逐专家字节数互不相同。
                slots = {}
                for spec in specs:
                    rows, cols = spec.carrier
                    slot, _, _ = create_vmm_physical([B, rows, cols], torch.bfloat16, dev)
                    slot.zero_()
                    slots[spec.name] = slot
                pool = {"slots": slots, "B": B}
                cls._pools[key] = pool
        return pool


class _LayerWeightState:
    """Home mirrors + full projections + packed [epn+B] aliases for ONE MoE layer."""

    def __init__(self, layer, E: int, H: int, rank: int, dev: int):
        from ascend_moonep.buffer_c import (
            attach_local_source_vas,
            create_full_projection_tensor,
            create_vmm_physical,
        )

        w13 = layer.w13_weight
        epn = int(w13.shape[0])
        B = int(os.environ.get("MOONEP_B", epn))
        if B < 1 or B > epn:
            raise ValueError(f"MOONEP_B must be in [1, {epn}], got {B}")
        F = _infer_F(layer, H)
        self.E, self.H, self.F, self.epn, self.B = E, H, F, epn, B
        self._specs = _payload_specs(layer, H, F)

        shared = _SharedSlots.get(E, H, F, B, rank, dev, self._specs)
        self.slots: dict[str, torch.Tensor] = {}
        self.homes: dict[str, torch.Tensor] = {}
        self.fulls: dict[str, torch.Tensor] = {}
        self.aliases: dict[str, torch.Tensor] = {}
        for spec in self._specs:
            rows, cols = spec.carrier
            # per-layer home chunk (mirror of the local vllm weight rows)
            home, _, _ = create_vmm_physical([epn, rows, cols], torch.bfloat16, dev)
            home.zero_()
            slot = shared["slots"][spec.name]
            full = create_full_projection_tensor(home, slot, E, B, rank, dev)
            attach_local_source_vas(full, home.data_ptr(), slot.data_ptr())
            self.slots[spec.name] = slot
            self.homes[spec.name] = home
            self.fulls[spec.name] = full
            self.aliases[spec.name] = _packed_alias(home, slot, dev, spec)

        self._w13_src = w13  # key of the runtime's _state_by_w13 index
        self._published = False
        self._vmm_tensors = list(self.homes.values())

    def publish(self) -> None:
        """One-time mirror of the local vllm weight rows into the home chunks.

        Must run after ShmemRuntime init (copies only; no new descriptors).
        Quantized payloads are NZ / 4-D non-contiguous views, so every mirror is
        a raw byte copy of the payload range (``_flat_bytes``).
        """
        if self._published:
            return
        with torch.no_grad():
            for spec in self._specs:
                dst = _flat_bytes(self.homes[spec.name], spec.nbytes)
                src = _flat_bytes(spec.ref, spec.nbytes)
                dst.copy_(src)
        self._published = True

    def prefetch_views(self) -> list[torch.Tensor]:
        """The [E+B] projection descriptors handed to the prefetch kernel."""
        return [self.fulls[spec.name] for spec in self._specs]

    def compute_weights(self) -> MoEWeights:
        """Grouped-matmul weights: [epn+B] aliases, scales included."""
        return MoEWeights(
            w1=self.aliases["w13"],
            w2=self.aliases["w2"],
            w1_scale=self.aliases.get("w13_scale"),
            w2_scale=self.aliases.get("w2_scale"),
        )

    def destroy(self):
        for tensor in self._vmm_tensors:
            alloc = getattr(tensor, "_vmm_allocation", None)
            if alloc is not None:
                try:
                    alloc.destroy()
                except Exception:
                    pass


def _packed_alias(home: torch.Tensor, slot: torch.Tensor, dev: int,
                  spec: _PayloadSpec) -> torch.Tensor:
    """Zero-copy [epn+B] alias: home at rows 0..epn, slot after.

    The alias mirrors ``spec.ref`` field by field — NZ payloads are built with
    the *cast* shape plus their format and then transposed (exactly how
    ``process_weights_after_loading`` produced the reference), ND payloads copy
    shape[1:]/stride/storage_offset/format verbatim. On-card spike M4 shows both
    variants are bit-identical to the real weights on a W4A8MXFP gmm1.
    """
    from ascend_moonep.buffer_c import (
        VmmProjection,
        acl_vmm_granularity,
        acl_vmm_map,
        acl_vmm_release,
        acl_vmm_reserve,
        acl_vmm_set_access,
        acl_vmm_unmap,
    )

    ref = spec.ref
    epn = int(home.shape[0])
    B = int(slot.shape[0])
    row = spec.row_bytes
    gran = int(acl_vmm_granularity(int(dev)))
    slot_off = epn * row
    if slot_off % gran != 0:
        raise RuntimeError(
            f"packed [epn+B] alias requires epn*row aligned to VMM gran: "
            f"epn*row={slot_off} gran={gran} (payload={spec.name})."
        )
    home_alloc = getattr(home, "_vmm_allocation", None)
    slot_alloc = getattr(slot, "_vmm_allocation", None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError("home/slot must be VMM chunks (create_vmm_physical)")
    n_rows = epn + B
    total = slot_off + int(slot_alloc.size)
    base = int(acl_vmm_reserve(total))
    segments = []
    try:
        for alloc, off in ((home_alloc, 0), (slot_alloc, slot_off)):
            va = base + off
            acl_vmm_map(va, int(alloc.size), alloc.mem_handle, 0)
            acl_vmm_set_access(va, int(alloc.size), dev)
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
    device = torch.device(f"npu:{int(dev)}")
    if spec.is_nz:
        cast = _sym_tensor(base, (n_rows, int(ref.shape[2]), int(ref.shape[1])),
                           ref.dtype, device, nbytes=n_rows * row,
                           npu_format=spec.npu_format)
        tensor = cast.transpose(1, 2)
    else:
        # Within-row strides come from the reference, but the row base is the
        # alias head: the reference's own storage_offset (a `full[rank*epn:...]`
        # slice carries one) must not shift the alias into the slot segment.
        stride = (int(ref.stride(0)),) + tuple(int(s) for s in ref.stride()[1:])
        tensor = _sym_tensor(base, (n_rows,) + tuple(int(d) for d in ref.shape[1:]),
                             ref.dtype, device, stride=stride,
                             nbytes=n_rows * row, npu_format=spec.npu_format)
    expected_shape = (n_rows,) + tuple(int(d) for d in ref.shape[1:])
    # stride(0) is in elements, row_bytes in bytes: only equal for 1-byte payloads.
    stride0_bytes = int(tensor.stride(0)) * int(ref.element_size())
    if (tensor.shape != expected_shape or stride0_bytes != row
            or int(tensor.data_ptr()) != base):
        raise RuntimeError(
            f"MoonEP slot mode: {spec.name} alias mismatch shape={tuple(tensor.shape)} "
            f"(want {expected_shape}) stride0={stride0_bytes}B (want {row}B) "
            f"ptr={int(tensor.data_ptr())} (want {base})."
        )
    tensor._projection = VmmProjection(base, segments)  # noqa: SLF001
    return tensor


# ---------------------------------------------------------------------------
# Runtime forward: dispatch -> prefetch -> grouped GEMM on [epn+B] aliases
# ---------------------------------------------------------------------------
def run_slot_forward(
    *,
    state: _LayerWeightState,
    buffer,
    plan,
    token_dispatch_output,
    fused_experts_input: MoEFusedExpertsInput,
    use_fusion_ops: bool,
):
    """Prefetch remote slot weights, then run the grouped GEMMs on the aliases.

    Mirrors MindSpeed shmem_ep_forward's middle section (weights publish is done
    once at init for inference). Goes through the standard
    ``build_mlp_compute_input`` + ``unified_apply_mlp`` entry points so bf16 and
    quantized layers share one operator call with the owner path, and only the
    weight payloads are swapped for the aliases. Returns the MLP output rows
    (same order as the dispatched rows) for the caller's combine.
    """
    try:
        capturing = _EXTRA_CTX.capturing
    except Exception:
        capturing = False  # outside an engine forward context (harness use)
    if capturing:
        raise RuntimeError("MoonEP slot mode cannot run inside captured graphs.")
    buffer.prefetch_projections(
        plan=plan,
        projections=state.prefetch_views(),
    )
    mlp_compute_input = build_mlp_compute_input(
        fused_experts_input=fused_experts_input,
        token_dispatch_output=token_dispatch_output,
        use_fusion_ops=use_fusion_ops,
    )
    # Both branches of unified_apply_mlp return (hidden_states, before_gmm2_evt);
    # slot mode never runs under capture, so the event is not needed.
    rows, _ = unified_apply_mlp(
        mlp_compute_input=dataclasses.replace(
            mlp_compute_input, weights=state.compute_weights())
    )
    return rows


__all__ = [
    "slot_mode_enabled",
    "slot_quant_enabled",
    "_LayerWeightState",
    "_SharedSlots",
    "run_slot_forward",
]
