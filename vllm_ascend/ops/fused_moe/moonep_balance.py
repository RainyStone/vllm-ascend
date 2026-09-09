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

Per layer, per projection (w13 = gate+up fused [E, H, 2F], w2 = [E, F, H]):
  home [epn, ...]  VMM chunk, mirror of this rank's local vllm weight rows
                    (published once at init; inference weights are static)
  slot [B,  ...]   VMM chunk shared across layers of one shape (B = epn);
                    filled per step by prefetch_weight with the remote rows of
                    plan.experts_to_copy
  full [E+B, ...]  projection view (home at [rank*epn, ...)) used by the
                    prefetch kernel (via attached source VAs)
  alias [epn+B,..] zero-copy packed view (home rows 0..epn, slot rows epn..)
                    used as the grouped-matmul weights

Scope (POC): unquantized bf16 MoE only (AscendUnquantizedFusedMoEMethod with
plain contiguous w13/w2 tensors); quantized and NZ/fusion layouts raise.
Activated with ``MOONEP_SLOT_MODE=1`` (default off: owner-only mode).
"""

from __future__ import annotations

import os
import threading

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput

_slot_mode = os.environ.get("MOONEP_SLOT_MODE", "0") == "1"


def slot_mode_enabled() -> bool:
    return _slot_mode


# ---------------------------------------------------------------------------
# Per-(E, H, F) shared slot pool + per-layer home mirrors
# ---------------------------------------------------------------------------
class _SharedSlots:
    """B prefetch slot chunks + [E+B] projections, shared across layers of one
    weight shape. Safe: one layer's dispatch->prefetch->GEMM->combine fully
    contains its slot use, and MoE layers run serially on the stream."""

    _lock = threading.Lock()
    _pools: dict[tuple, dict] = {}

    @classmethod
    def get(cls, E: int, H: int, F: int, B: int, rank: int, dev: int) -> dict:
        key = (E, H, F, B, rank, dev)
        with cls._lock:
            pool = cls._pools.get(key)
            if pool is None:
                from ascend_moonep.buffer_c import (
                    attach_local_source_vas,
                    create_full_projection_tensor,
                    create_vmm_physical,
                )

                slots = {}
                fulls = {}
                for name, (rows, cols) in (("w13", (H, 2 * F)), ("w2", (F, H))):
                    slot, _, _ = create_vmm_physical([B, rows, cols], torch.bfloat16, dev)
                    slot.zero_()
                    slots[name] = slot
                # full [E+B] projections need home chunks; built per layer, so
                # keep slot chunks here and let _LayerState build the fulls.
                pool = {"slots": slots, "B": B}
                cls._pools[key] = pool
        return pool


class _LayerWeightState:
    """Home mirror + full projection + packed [epn+B] alias for ONE MoE layer."""

    def __init__(self, layer, E: int, H: int, F: int, rank: int, dev: int):
        from ascend_moonep.buffer_c import (
            VmmProjection,
            acl_vmm_granularity,
            acl_vmm_map,
            acl_vmm_release,
            acl_vmm_reserve,
            acl_vmm_set_access,
            attach_local_source_vas,
            create_full_projection_tensor,
            create_sym_tensor_from_ptr,
            create_vmm_physical,
        )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            raise NotImplementedError("MoonEP slot mode requires bf16 MoE weights (POC).")
        # unquant path keeps need_trans=False layouts: w13 [E,H,2F], w2 [E,F,H]
        epn = w13.shape[0]
        assert w13.shape[1] == H and w13.shape[2] == 2 * F, w13.shape
        assert w2.shape == (epn, F, H), w2.shape
        B = int(os.environ.get("MOONEP_B", epn))
        if B < 1 or B > epn:
            raise ValueError(f"MOONEP_B must be in [1, {epn}], got {B}")
        self.E, self.H, self.F, self.epn, self.B = E, H, F, epn, B

        shared = _SharedSlots.get(E, H, F, B, rank, dev)
        self.slot_w13 = shared["slots"]["w13"]
        self.slot_w2 = shared["slots"]["w2"]

        # per-layer home chunks (mirror of the local vllm weight rows)
        self.home_w13, _, _ = create_vmm_physical([epn, H, 2 * F], torch.bfloat16, dev)
        self.home_w2, _, _ = create_vmm_physical([epn, F, H], torch.bfloat16, dev)
        self.home_w13.zero_()
        self.home_w2.zero_()

        # [E+B] projections for the prefetch kernel + source-VA tags
        self.full_w13 = create_full_projection_tensor(
            self.home_w13, self.slot_w13, E, B, rank, dev)
        attach_local_source_vas(self.full_w13, self.home_w13.data_ptr(),
                                self.slot_w13.data_ptr())
        self.full_w2 = create_full_projection_tensor(
            self.home_w2, self.slot_w2, E, B, rank, dev)
        attach_local_source_vas(self.full_w2, self.home_w2.data_ptr(),
                                self.slot_w2.data_ptr())

        # zero-copy [epn+B] packed compute aliases (MindSpeed
        # _create_packed_projection): home rows 0..epn, slot rows epn..epn+B
        self.alias_w13 = _packed_alias(self.home_w13, self.slot_w13, dev)
        self.alias_w2 = _packed_alias(self.home_w2, self.slot_w2, dev)

        self._w13_src = w13
        self._w2_src = w2
        self._published = False
        self._vmm_tensors = [self.home_w13, self.home_w2]

    def publish(self) -> None:
        """One-time mirror of the local vllm weight rows into the home chunks.

        Must run after ShmemRuntime init (copies only; no new descriptors).
        """
        if self._published:
            return
        with torch.no_grad():
            self.home_w13.copy_(self._w13_src)
            self.home_w2.copy_(self._w2_src)
        self._published = True

    def destroy(self):
        for t in self._vmm_tensors:
            alloc = getattr(t, "_vmm_allocation", None)
            if alloc is not None:
                try:
                    alloc.destroy()
                except Exception:
                    pass


def _packed_alias(home: torch.Tensor, slot: torch.Tensor, dev: int) -> torch.Tensor:
    """Zero-copy [epn+B, rows, cols] alias: home at rows 0..epn, slot after."""
    from ascend_moonep.buffer_c import (
        VmmProjection,
        acl_vmm_granularity,
        acl_vmm_map,
        acl_vmm_release,
        acl_vmm_reserve,
        acl_vmm_set_access,
        create_sym_tensor_from_ptr,
    )

    epn = int(home.shape[0])
    B = int(slot.shape[0])
    row = int(home.shape[1]) * int(home.shape[2]) * int(home.element_size())
    gran = int(acl_vmm_granularity(dev))
    slot_off = epn * row
    if slot_off % gran != 0:
        raise RuntimeError(
            f"packed [epn+B] alias requires epn*row aligned to VMM gran: "
            f"epn*row={slot_off} gran={gran} (epn={epn})."
        )
    home_alloc = getattr(home, "_vmm_allocation", None)
    slot_alloc = getattr(slot, "_vmm_allocation", None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError("home/slot must be VMM chunks (create_vmm_physical)")
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
                acl_vmm_unmap(va)  # noqa: F821 - imported below in except path
            except Exception:
                pass
        try:
            acl_vmm_release(base)
        except Exception:
            pass
        raise
    tensor = create_sym_tensor_from_ptr(
        base, [epn + B, int(home.shape[1]), int(home.shape[2])],
        home.dtype, torch.device(f"npu:{dev}"))
    tensor._projection = VmmProjection(base, segments)  # noqa: SLF001
    return tensor


# ---------------------------------------------------------------------------
# Runtime forward: dispatch -> prefetch -> grouped GEMM on [epn+B] aliases
# ---------------------------------------------------------------------------
def run_slot_forward(
    state: _LayerWeightState,
    buffer,
    plan,
    hidden_trim: torch.Tensor,   # shard rows (prefix), home then slot groups
    group_list: torch.Tensor,    # [epn+B] int64 counts, local-expert order
    fused_experts_input: MoEFusedExpertsInput,
):
    """prefetch remote slot weights, then run the grouped GEMMs on the aliases.

    Mirrors MindSpeed shmem_ep_forward's middle section (weights publish is
    done once at init for inference). Returns the MLP output rows (same order
    as hidden_trim) for the caller's combine.
    """
    try:
        capturing = _EXTRA_CTX.capturing
    except Exception:
        capturing = False  # outside an engine forward context (harness use)
    if capturing:
        raise RuntimeError("MoonEP slot mode cannot run inside captured graphs.")
    buffer.prefetch_projections(
        plan=plan,
        projections=[state.full_w13, state.full_w2],
    )
    swiglu_limit = fused_experts_input.swiglu_limit
    rows, _ = unquant_apply_mlp(
        hidden_states=hidden_trim,
        w1=state.alias_w13,
        w2=state.alias_w2,
        group_list=group_list,
        group_list_type=1,  # count mode
        activation=fused_experts_input.activation,
        need_trans=False,
        swiglu_limit=swiglu_limit,
    )
    return rows


__all__ = ["slot_mode_enabled", "_LayerWeightState", "_SharedSlots", "run_slot_forward"]
