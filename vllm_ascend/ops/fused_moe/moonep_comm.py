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
"""MoonEP (SHMEM symmetric-memory) MoE communication method.

Routes expert-parallel MoE steps through the ``ascend-moonep`` Buffer instead
of HCCL all-to-all / MC2 token exchange:

  * ``buffer.dispatch`` scatters each token's top-K hidden rows to the rank
    that owns the selected expert (CANN VMM symmetric memory + SHMEM RMA).
  * The owner rank computes the per-expert GEMMs with its *local* expert
    weight tables (the standard vLLM-Ascend grouped-matmul MLP, untouched).
  * ``buffer.combine`` gathers the K per-token rows back and sums them.

Forward-only subset of the MindSpeed-MM shmem EP dispatcher
(``shmem_ep_dispatcher.py``): inference never touches prefetch_weight or
reduce_grad (those serve remote-expert weight preloading / training
gradients), so no symmetric weight mirrors are created at all.

Integration contract
--------------------
The MoonEP planning kernel distributes rows deterministically: rank ``r``
receives the rows of global experts ``[r*epn, (r+1)*epn)`` from *all* ranks,
prefix-packed in ascending owned-expert order. ``cu_seqlens`` (per rank) is a
cumulative count over global groups, so per-local-expert counts are the diff
over the owned range. This matches the MLP contract of the all-to-all token
dispatcher (rows grouped by local expert, count-mode group list), which is why
this comm method reuses ``PrepareAndFinalizeWithAll2All`` and the standard
``unified_apply_mlp``.

Limitations (POC scope, all fail fast):
  * no dynamic EPLB / expert maps / mix-placement (plain contiguous expert
    sharding), and no LoRA,
  * the MoE layer cannot run inside captured CUDA graphs (collective device
    kernels): run with ``--enforce-eager`` when ``enable_moonep`` is on,
  * per-step token count must fit the maximum capacity S
    (``max_num_batched_tokens``). Steps are dispatched against a pooled
    capacity (pow2 ladder, min 128) chosen from the rank-symmetric step token
    count, so short steps do not pay for the full S*K row space.
  * Capacity envelope (empirical, this ascend-moonep build): the planning
    kernel silently drops dispatched rows once a rank's receive share exceeds
    its S*K symmetric chunk (measured at R=4 with heavy per-expert skew and
    with every rank padding to expert 0 — combine then misplaces rows for the
    other ranks). Pad rows therefore route to each rank's own first expert
    (zero-weight; MOONEP_PAD_EXPERT0=1 restores the expert-0 behavior), and
    expert loads must stay roughly balanced across ranks (uniform routing
    passes bit-exact comm checks at R=4; MindSpeed-MM's slot/prefetch scheme
    exists precisely to enforce that balance, and is out of POC scope here).
    Models with E >> R and smooth routers (e.g. E=256 / R=4: per-rank receive
    is the sum over its 64 owned experts) sit well inside the envelope.

The ascend-moonep library is imported lazily so a stock vllm-ascend install
(without the package or with ``enable_moonep`` off) is unaffected.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context

import atexit

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_moonep_tokens_capacity
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, MoECommMethod
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.moe_stage_contracts import TMoECombineMetadata
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatcher

@dataclass
class MoonEPCombineMetadata:
    """Carry-over state from token_dispatch to token_combine.

    ``w_trim`` is a *clone* of the per-row route weights: the symmetric
    dispatch buffers are reused by every MoE layer of the same shape, so the
    row-aligned weights must be snapshotted at dispatch time.
    """

    plan: object  # ascend_moonep MoonEPCommPlan
    buffer: object  # ascend_moonep Buffer used for this call
    w_trim: torch.Tensor  # [rows_total] fp32, row-aligned with dispatched rows
    num_tokens: int  # actual (pre-pad) token count of this MoE call
    hidden_shape: torch.Size  # original hidden_states shape (for restore)
    state: object | None = None  # _LayerWeightState when slot mode active
    group_list: torch.Tensor | None = None  # [epn+B] int64 counts (slot mode)
    tail_masked: bool = False  # dispatch used the no-sync full-shard path


# ---------------------------------------------------------------------------
# Per-(H, K, E) layer registry. AscendFusedMoE.__init__ registers every MoE
# layer when enable_moonep is on; the runtime pool is materialized from the
# registry on the first MoonEP forward (i.e. after the whole model graph has
# been constructed), because the collective ShmemRuntime init needs the
# complete descriptor set of all Buffers up front.
# ---------------------------------------------------------------------------
_moonep_layer_registry: list[dict] = []
_registry_lock = threading.Lock()
_SLOT_TLS = threading.local()  # per-layer balance state during fused_experts

# optional per-step slot statistics (MOONEP_SLOT_STATS=1, dumped at process exit)
_SLOT_STATS = {"calls": 0, "with_slots": 0, "slot_rows": 0, "prefetch_experts": 0}

# optional per-rank MoE load accounting (MOONEP_BALANCE_STATS=1)
_BALANCE_STATS_ON = os.environ.get("MOONEP_BALANCE_STATS", "0") == "1"
_BALANCE_STATS = {"calls": 0, "tokens": 0, "local_rows": 0, "all_rows": 0, "pad_rows": 0}

# MOONEP_PLAN_STATS=1: record, per layer, this rank's slot assignment
# (plan.experts_to_copy[rank], -1 = unused slot). This is the gate for any
# speculative-prefetch / slot-cache scheme: if the planner keeps putting the
# same expert in the same slot, a prefetch can be started a layer early (during
# attention) or served from a local cache; if the assignment churns, neither
# pays off. Measurement only: reads the plan back to the host every layer.
_PLAN_STATS_ON = os.environ.get("MOONEP_PLAN_STATS", "0") == "1"
_PLAN_STATS: list = []


# MOONEP_EARLY_PLAN=1: the speculation probe for "prefetch overlapped with
# attention".  The rationale (user's): the residual stream means this layer's
# MoE input (post-attention) is close to the previous layer's output
# (pre-attention), so running *this* layer's router on the pre-attention hidden
# -- i.e. during this layer's attention -- should predict the experts that will
# actually be hot, and therefore the experts the planner will mirror.
#
# The probe runs exactly that path in a decoder-layer forward *pre*-hook (where a
# real implementation would), and records the predicted per-expert histogram.
# token_dispatch records the real one.  Offline we compare: predicted vs real
# histogram, and both against the plan's actual experts_to_copy
# (MOONEP_PLAN_STATS), which is the hit rate that decides whether the scheme is
# worth building.  Measurement only; each layer pays one extra gate matmul.
_EARLY_PLAN_ON = os.environ.get("MOONEP_EARLY_PLAN", "0") == "1"
_EARLY: dict = {"pre": {}, "early": [], "real": [], "plan_pred": [],
                "pred_ids": None, "fail": 0, "done": False}


def _early_layers(model):
    for path in ("model.layers", "layers", "transformer.layers", "model.model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj):
            return list(obj)
    return []


def _early_route(blk, h_pre):
    """Route h_pre through this layer's own norm+gate -> (histogram, topk ids)."""
    import torch

    mlp = getattr(blk, "mlp", None)
    if mlp is None:
        return None
    gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
    norm = getattr(blk, "post_attention_layernorm", None)
    if gate is None or norm is None:
        return None
    k = getattr(mlp, "top_k", None)
    if k is None:
        k = getattr(getattr(mlp, "experts", None), "top_k", None)
    if k is None:
        return None
    with torch.no_grad():
        logits = gate(norm(h_pre))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        ids = logits.topk(int(k), dim=-1).indices
        return (torch.bincount(ids.reshape(-1).to(torch.int64)).cpu().tolist(), ids)


def maybe_install_early_router_probe(model) -> None:
    """Attach the MOONEP_EARLY_PLAN=1 pre-hooks (no-op when the env is off)."""
    if not _EARLY_PLAN_ON or _EARLY["done"]:
        return
    try:
        layers = _early_layers(model)
        if not layers:
            print("[early-plan] no decoder layers found; probe disabled", flush=True)
            return

        def _pre_hook(blk, args, kwargs):
            try:
                # Qwen3MoeDecoderLayer.forward(positions, hidden_states, residual):
                # the residual stream before attention is hidden_states + residual
                h = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
                res = kwargs.get("residual", args[2] if len(args) > 2 else None)
                if h is None:
                    return None
                if res is not None:
                    h = h + res
                got = _early_route(blk, h)
                if got is not None:
                    hist, ids = got
                    _EARLY["early"].append(hist)
                    # mailbox: the next MoE dispatch of this same layer consumes it
                    _EARLY["pred_ids"] = ids
            except Exception:            # noqa: BLE001 - probe must never break a step
                _EARLY["fail"] += 1
            return None

        for blk in layers:
            blk.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        _EARLY["done"] = True
        print(f"[early-plan] probe on {len(layers)} decoder layers", flush=True)
    except Exception as exc:             # noqa: BLE001
        print(f"[early-plan] install failed: {exc}", flush=True)


def _plan_on_predicted_routing(ctx, topk_pred, E, K):
    """Run the planning kernel on the *predicted* routing -> predicted plan."""
    from ascend_moonep.planning import allocate_planning_outputs

    plan_pred, cu_pred = allocate_planning_outputs(ctx)
    tpe_pred = torch.histc(topk_pred.to(torch.float32), bins=E, min=0, max=E - 1).to(torch.int32)
    ctx["kernels"]["planning"].compute(
        ctx, topk_pred.reshape(-1), tpe_pred, cu_pred, plan_pred, shmem_instance_id=0)
    return plan_pred


# One predicted-plan buffer per capacity rung, reused across layers: layers run
# sequentially and a layer's predicted plan is consumed by its own dispatch, so
# by the time the next layer predicts, the previous plan is dead. (Allocating a
# fresh plan per layer instead churned VMM memory badly enough to kill a run.)
_PRED_PLAN_BUF: dict = {}


def early_predicted_plan_ctx(blk, h_pre, top_k, E, H, B):
    """Predict this layer's routing from the pre-attention hidden and plan on it.

    Returns ``(buffer, predicted_plan, pad_expert)`` or None. Mirrors exactly what
    the real dispatch will do (same capacity rung, same pad expert) so the only
    difference is the routing the plan is computed from.
    """
    got = _early_route(blk, h_pre)
    if got is None:
        return None
    _, ids = got
    T = int(ids.shape[0])
    buffer, capacity = _select_buffer_for_step(E, int(top_k), H, T)
    ctx = buffer._require_ctx()
    epn = E // get_ep_group().world_size
    pad_expert = get_ep_group().rank_in_group * epn + (epn - 1)
    plan_pred, cu_pred = _PRED_PLAN_BUF.get(int(capacity), (None, None))
    if plan_pred is None:
        from ascend_moonep.planning import allocate_planning_outputs

        plan_pred, cu_pred = allocate_planning_outputs(ctx)
        _PRED_PLAN_BUF[int(capacity)] = (plan_pred, cu_pred)
    topk_c = torch.zeros(capacity, int(top_k), dtype=torch.int32, device=ids.device)
    topk_c[:T].copy_(ids.to(torch.int32))
    if T < capacity:
        topk_c[T:].fill_(pad_expert)
    tpe = torch.histc(topk_c.to(torch.float32), bins=E, min=0, max=E - 1).to(torch.int32)
    ctx["kernels"]["planning"].compute(
        ctx, topk_c.reshape(-1), tpe, cu_pred, plan_pred, shmem_instance_id=0)
    return buffer, plan_pred, pad_expert


def early_plan_fallback(state, top_k, T, H):
    """Run the planning kernel on a fixed, rank-symmetric dummy routing.

    The planning kernel is a cross-rank collective, so every rank must call it
    the same number of times: when the prediction is unavailable we still have to
    enter it, just with routing nobody will use. Uniform over this rank's own
    experts keeps the histogram well-formed (no division by zero).
    """
    E = state.E
    buffer, capacity = _select_buffer_for_step(E, int(top_k), H, T)
    ctx = buffer._require_ctx()
    epn = E // get_ep_group().world_size
    rank = get_ep_group().rank_in_group
    plan_pred, cu_pred = _PRED_PLAN_BUF.get(int(capacity), (None, None))
    if plan_pred is None:
        from ascend_moonep.planning import allocate_planning_outputs

        plan_pred, cu_pred = allocate_planning_outputs(ctx)
        _PRED_PLAN_BUF[int(capacity)] = (plan_pred, cu_pred)
    per = max(1, (capacity * int(top_k)) // epn)
    tpe = torch.zeros(E, dtype=torch.int32, device=plan_pred.dst.device)
    tpe[rank * epn:(rank + 1) * epn] = per
    topk_dummy = torch.zeros(capacity, int(top_k), dtype=torch.int32, device=tpe.device)
    ar = torch.arange(epn, dtype=torch.int32, device=tpe.device)
    topk_dummy[:] = (rank * epn + (ar[:int(top_k)] % epn))
    ctx["kernels"]["planning"].compute(
        ctx, topk_dummy.reshape(-1), tpe, cu_pred, plan_pred, shmem_instance_id=0)
    return buffer, plan_pred, (rank * epn + epn - 1)


def _dump_early_plan():
    if not _EARLY_PLAN_ON or not _EARLY["early"]:
        return
    import json as _json

    try:
        rank = int(torch.npu.current_device()) - int(os.environ.get("MOONEP_DEV_BASE", "0"))
    except Exception:  # noqa: BLE001
        rank = -1
    path = f"/tmp/moonep_earlyplan_{os.environ.get('TAG', 'early')}_rank{rank}.json"
    with open(path, "w") as fh:
        _json.dump({"early": _EARLY["early"], "real": _EARLY["real"],
                    "plan_pred": _EARLY["plan_pred"], "fail": _EARLY["fail"]}, fh)
    print(f"[early-plan r{rank}] early={len(_EARLY['early'])} real={len(_EARLY['real'])} "
          f"pred_plans={len(_EARLY['plan_pred'])} fail={_EARLY['fail']} -> {path}", flush=True)


atexit.register(_dump_early_plan)


def _dump_plan_stats():
    if not _PLAN_STATS_ON or not _PLAN_STATS:
        return
    import json as _json

    try:
        rank = int(torch.npu.current_device()) - int(os.environ.get("MOONEP_DEV_BASE", "0"))
    except Exception:  # noqa: BLE001
        rank = -1
    path = f"/tmp/moonep_planstats_{os.environ.get('TAG', 'plan')}_rank{rank}.json"
    with open(path, "w") as fh:
        _json.dump(_PLAN_STATS, fh)
    print(f"[moonep-planstats r{rank}] layers={len(_PLAN_STATS)} -> {path}", flush=True)


atexit.register(_dump_plan_stats)


# MOONEP_PAD_CUT=1: try to leave the pad rows out of the grouped GEMM by
# trimming the last local expert's group (pads are zero-hidden/zero-weight, and
# the matmul walks exactly sum(group_list) shard rows). **Off by default: it is
# WRONG.** The trim assumes the pad rows are the *trailing* rows of that group,
# but the planning kernel orders rows within a group by its own assignment (k
# slices), so the tail contains real rows too. Measured on the 4-rank smoke
# harness (E=8, K=2, S=512, T=384): PAD_CUT=0 -> rel err 0.6-0.8% PASS;
# PAD_CUT=1 -> rel err 74% FAIL. Keep the knob for experiments only; reducing
# the pad *volume* with the dense capacity ladder (below) is the safe lever.
_PAD_CUT = os.environ.get("MOONEP_PAD_CUT", "0") == "1"

# MOONEP_PAD_ROTATE=1: rotate the pad expert per layer (see pad_expert)
_PAD_ROT = {"i": 0}


def _dump_balance_stats():
    if not _BALANCE_STATS_ON:
        return
    s = _BALANCE_STATS
    rank = s.get("rank", "?")
    calls = max(s["calls"], 1)
    print(
        f"[moonep-balance r{rank}] layers={s['calls']} tokens={s['tokens']} "
        f"rows_local={s['local_rows']} ({s['local_rows'] / calls:.0f}/layer) "
        f"rows_all={s['all_rows']} ({s['all_rows'] / calls:.0f}/layer) "
        f"pad_rows={s['pad_rows']}",
        flush=True,
    )


atexit.register(_dump_balance_stats)

# MOONEP_TIME_STATS=1: per-phase wall totals [total_s, calls]
_TSTATS_ENABLED = os.environ.get("MOONEP_TIME_STATS", "0") == "1"
_MOONEP_TSTATS = {"dispatch": [0.0, 0], "disp_pre": [0.0, 0], "disp_pad": [0.0, 0],
                 "disp_tpe": [0.0, 0], "disp_buf": [0.0, 0],
                 "disp_lib": [0.0, 0], "disp_post": [0.0, 0],
                 "combine": [0.0, 0], "comb_pre": [0.0, 0], "comb_buf": [0.0, 0]}


def _dump_tstats():
    if not _TSTATS_ENABLED:
        return
    try:
        import torch
        dev = f"npu:{torch.npu.current_device()}" if torch.npu.is_available() else "?"
    except Exception:  # noqa: BLE001
        dev = "?"
    parts = [f"[moonep-tstats dev={dev}]"]
    for name, (total, calls) in _MOONEP_TSTATS.items():
        parts.append(
            f"{name}: calls={calls} total={total:.3f}s avg={total / max(calls, 1) * 1e3:.2f}ms")
    print(" ".join(parts), flush=True)


atexit.register(_dump_tstats)


def _dump_slot_stats():
    import os
    if os.environ.get("MOONEP_SLOT_STATS", "0") == "1":
        s = _SLOT_STATS
        print(f"[moonep-stats] dispatch_calls={s['calls']} calls_with_slot_rows={s['with_slots']} "
              f"total_slot_rows={s['slot_rows']} total_prefetched_expert_slots={s['prefetch_experts']}",
              flush=True)


# Step-adaptive dispatch capacity (upstream MoonEPBufferPool equivalent): a
# ``Buffer`` fixes its token capacity S at construction, so with a single
# full-capacity buffer every step pads to max_num_batched_tokens and the
# planning/dispatch/combine kernels plus the grouped GEMM run over the full
# ``S*K`` row space. A pow2 pool lets decode-sized steps use a fraction of it.
# Smallest rung of the capacity pool. The vLLM-side dispatch never needs
# more than the step token count, and decode steps are often tiny (32 tokens
# for 128 seqs on 4 ranks), so the floor is low by default; every padded row
# inside the MLP's group_list is real (wasted) GEMM work. Raise it (or set
# MOONEP_POOL_CAPS) if a rank's receive share can exceed S*K.
MOONEP_MIN_DISPATCH_TOKENS = int(os.environ.get("MOONEP_MIN_CAP", "16"))


def _moonep_pool_capacities(s_max: int, cost_per_token: int = 32_768) -> list[int]:
    """Capacity ladder for the buffer pool (MOONEP_POOL_CAPS overrides).

    A rung fixes the dispatched row space to S*K, so a step of T tokens pads
    (S-T)*K zero rows -- pure waste: the grouped GEMM multiplies zeros and the
    pile lands on the pad expert, which skews that rank's expert groups. The
    pool is idle memory, so trading pool bytes for smaller pads is a good deal;
    MOONEP_POOL_BUDGET_MB caps how much we spend on it (default 2 GiB, a small
    part of what releasing the vllm weight copies frees).

    Layout: dense where decode lives, geometric above it.

        step 8  : MIN .. 128        (decode shapes, cheap)
        step 32 : 160 .. 512        (big decode / small prefill)
        ratio r : 512 .. s_max      with r from the budget; a geometric ladder
                                    of ratio r costs ~s_max*cost*r/(r-1) bytes
                                    and leaves at most (r-1) relative pad

    ``cost_per_token`` is the hidden chunk's bytes per token of capacity
    (K*H*2), passed by the caller because the pool is built per (E, K, H).
    """
    override = os.environ.get("MOONEP_POOL_CAPS")
    if override:
        caps = sorted({int(c) for c in override.split(",") if c.strip()})
        if not caps or caps[-1] != int(s_max):
            caps = sorted(set(caps) | {int(s_max)})
        return [c for c in caps if c <= int(s_max)]
    s_max = int(s_max)
    caps: list[int] = []

    def _add(c: int) -> None:
        c = int(c)
        if 0 < c <= s_max and c not in caps:
            caps.append(c)

    # Shape: powers of two up to 512 (decode shapes, where the tail lives),
    # then a uniform step up to s_max. The waste per step is (S-T)*K rows, so a
    # uniform step bounds it by step*K rows whatever T is -- a geometric ladder
    # does not (its absolute waste grows with S, and where the rungs land is
    # luck: measured at T=4676, ratio 1.22 gave 6416 pad rows vs 688 for 1.35).
    #
    # Two caps bind, and the count one dominates: every Buffer also allocates
    # its op workspaces as separate VMM physical chunks (~10 per buffer, 16 MiB
    # scale each), and the driver starts failing with
    # "aclrtMallocPhysical(...) failed with 207001" past a few dozen buffers
    # (measured: 51 rungs -> pool init dies; 31 rungs -> fine).
    max_rungs = int(os.environ.get("MOONEP_POOL_MAX_RUNGS", "32"))
    budget = int(os.environ.get("MOONEP_POOL_BUDGET_MB", "2048")) * 2**20
    base = [c for c in (16, 32, 64, 128, 256, 512) if c <= s_max]
    for c in base:
        _add(c)
    top = base[-1] if base else s_max
    if s_max > top:
        # smallest uniform step that fits both caps
        step = 128
        while step < s_max:
            count = (s_max - top) // step
            span = (top + s_max) / 2.0 * count * cost_per_token
            if count + len(base) <= max_rungs and (budget <= 0 or span <= budget):
                break
            step *= 2
        c = top + step
        while c <= s_max:
            _add(c)
            c += step
    _add(s_max)
    return sorted(caps)


def _step_token_count(tokens: int):
    """Rank-symmetric token count for this step, or None when unknown.

    ``set_ascend_forward_context`` derives ``max_tokens_across_dp`` from the DP
    metadata all-reduce (or from the local count without DP), so every rank in
    the MoE collective reads the same value. A wrong capacity pick would
    desynchronize the collective rather than merely waste work, hence the
    conservative fallbacks: unknown context, or DP without metadata -> None
    (caller uses the full capacity, which every rank agrees on).
    """
    try:
        forward_context = get_forward_context()
    except Exception:  # noqa: BLE001
        return None
    if get_dp_group().world_size > 1 and forward_context.dp_metadata is None:
        return None
    max_across_dp = getattr(forward_context, "max_tokens_across_dp", None)
    if max_across_dp is None:
        return None
    return max(int(max_across_dp), tokens)


def _moonep_slot_count(E: int, epn: int) -> int:
    """Effective prefetch-slot count B: env MOONEP_B, default epn = E/R."""
    b = int(os.environ.get("MOONEP_B", epn))
    if b < 1 or b > epn:
        raise ValueError(f"MOONEP_B must be in [1, {epn}], got {b}")
    return b


def register_moonep_layer(*, E: int, K: int, H: int, nle: int, layer=None) -> None:
    """Record one MoE layer for MoonEP. Cheap; only called when enabled."""
    with _registry_lock:
        _moonep_layer_registry.append(
            {"E": int(E), "K": int(K), "H": int(H), "nle": int(nle), "layer": layer})


def _registry_keys() -> set[tuple]:
    with _registry_lock:
        return {(e["E"], e["K"], e["H"], e["nle"]) for e in _moonep_layer_registry}


# ---------------------------------------------------------------------------
# Runtime pool: one ascend_moonep.Buffer per unique (E, K, H) layer shape.
# Construction registers CANN VMM descriptors only; ShmemRuntime is initialized
# once, after every Buffer has been created (two-phase, collective).
# ---------------------------------------------------------------------------
class _MoonEPRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._fail: Exception | None = None
        self._buffers: dict[tuple, object] = {}
        self._pads: dict[tuple, dict] = {}

    def _slot_states(self) -> list:
        return getattr(self, "_states", [])

    def _ensure_initialized(self):
        with self._lock:
            if self._ready:
                return
            if self._fail is not None:
                raise RuntimeError(f"MoonEP runtime failed to initialize: {self._fail}") from self._fail
            try:
                self._initialize_locked()
            except Exception as exc:  # noqa: BLE001 - keep the first failure
                self._fail = exc
                raise
            self._ready = True

    def _initialize_locked(self):
        # Lazy imports: the ascend-moonep package must not be required on
        # configurations where MoonEP is never selected.
        from ascend_moonep import Buffer, ShmemRuntime

        if ShmemRuntime.is_initialized():
            raise RuntimeError(
                "MoonEP: ascend_moonep ShmemRuntime already initialized outside the MoonEP runtime."
            )
        ep_group = get_ep_group()
        group = ep_group.device_group
        R = ep_group.world_size
        if R <= 1:
            raise RuntimeError("MoonEP requires expert parallelism (EP world size > 1).")
        capacity = get_moonep_tokens_capacity()
        if capacity is None or capacity <= 0:
            raise RuntimeError("MoonEP: token capacity S was not configured (set_moonep_tokens_capacity).")
        use_udma = os.environ.get("MOONEP_DISABLE_UDMA", "0") != "1"

        keys = sorted(_registry_keys())
        if not keys:
            raise RuntimeError(
                "MoonEP: no MoE layers registered (register_moonep_layer). "
                "MoE layers must be created before the first forward."
            )
        print(f"[moonep-init r{get_ep_group().rank_in_group}] stage=creating-buffers nkeys={len(keys)}", flush=True)
        # Validate + create every Buffer *before* the single collective init.
        for E, K, H, nle in keys:
            if E % R != 0:
                raise ValueError(
                    f"MoonEP: num_experts E={E} must be divisible by EP size R={R}."
                )
            if nle != E // R:
                raise NotImplementedError(
                    f"MoonEP: local experts {nle} != E/R = {E // R}; only plain contiguous "
                    "expert sharding is supported (no dynamic EPLB / expert maps)."
                )
            if K > 16:
                raise ValueError(f"MoonEP: combine supports K <= 16, got K={K}.")
            if H % 16 != 0:
                raise ValueError(f"MoonEP: combine requires H % 16 == 0, got H={H}.")
            if K * H * capacity >= 2**31:
                raise ValueError("MoonEP: S*K exceeds the planning int32 range.")
            # B (prefetch slots) must equal epn for the planning kernel.
            capacities = _moonep_pool_capacities(int(capacity), cost_per_token=K * H * 2)
            self._buffers[(E, K, H)] = {
                cap: Buffer(
                    int(cap), H, K, E, R,
                    token_padding=1, B=_moonep_slot_count(E, E // R), group=group,
                    explicitly_destroy=True,
                    # MOONEP_NUM_SMS: block count for the MoonEP kernels
                    # (library default 32). Measured on the slot path:
                    # 32 -> 56 cuts planning_kernel 278 -> 242 us/layer (-13%),
                    # dispatch -20%, combine -13%, total device time -7%, but
                    # the step time is unchanged (71.3 vs 71.9 ms, noise) --
                    # planning is latency-bound on the cross-rank histogram
                    # exchange, so the saving is absorbed by the barriers.
                    # Kept as an env knob (unset = library default); the legal
                    # range is [8, 56] -- 64+ fails at init with
                    # "planning core_num must be in [8, 56]".
                    **({"num_sms": int(os.environ["MOONEP_NUM_SMS"])}
                       if os.environ.get("MOONEP_NUM_SMS") else {}),
                    **({"use_udma": True} if use_udma else {}),
                )
                for cap in capacities
            }
            print(
                f"[moonep-init r{get_ep_group().rank_in_group}] stage=pool-ready "
                f"E={E} K={K} H={H} capacities={capacities}",
                flush=True,
            )
            # Static [S_max, *] pad workspace shared across layers of one shape
            # (the step's capacity window is a prefix slice of it). ``tail``
            # records the (T, capacity, pad expert) the pad tail was last
            # written for: copies always overwrite [0:T] in full, so unchanged
            # steps need no zero/fill at all (3 launches per layer saved).
            dev = f"npu:{torch.npu.current_device()}"
            self._pads[(E, K, H)] = {
                "hidden": torch.zeros(capacity, H, dtype=torch.bfloat16, device=dev),
                "route": torch.zeros(capacity, K, dtype=torch.float32, device=dev),
                "topk": torch.zeros(capacity, K, dtype=torch.int32, device=dev),
                "tail": {"T": -1, "capacity": None, "pad_expert": None},
            }
        if os.environ.get("MOONEP_SLOT_MODE", "0") == "1":
            from vllm_ascend.ops.fused_moe.moonep_balance import _LayerWeightState

            dev = int(torch.npu.current_device())
            ep_grp = get_ep_group()
            rank = ep_grp.rank_in_group
            states = []
            seen = set()
            with _registry_lock:
                entries = list(_moonep_layer_registry)
            for e in entries:
                layer = e.get("layer")
                if layer is None or id(layer) in seen:
                    continue
                seen.add(id(layer))
                w13 = layer.w13_weight
                F = int(w13.shape[2]) // 2
                st = _LayerWeightState(layer, int(e["E"]), int(e["H"]), F, rank, dev)
                st.layer_index = len(states)      # registration order == layer order
                states.append(st)
            self._states = states
            self._state_by_w13 = {id(st._w13_src): st for st in states}

        if not ShmemRuntime.is_initialized():
            # Engine 环境默认走 ipport/config-store bootstrap：uid 路径要在
            # 每个 worker 的 HCCL 组上做 UID 广播（dist.broadcast over HCCL），
            # 在引擎里与刚创建的 UDMA-flagged VMM 上下文及模型加载后的设备态
            # 冲突（实测 UB_INLINE_WRITE/HD Put 任务异常 → 全体自旋）。ipport
            # 只在本机 rank0 起 config-store，无 HCCL 广播，引擎内实测通过
            # （2026-09-09：uid 卡 init_with_buffer:182，ipport 可到首层 MoE
            # dispatch）。standalone torchrun 无此问题，可用 MOONEP_SHMEM_BOOTSTRAP=uid 覆盖。
            bootstrap = os.environ.get("MOONEP_SHMEM_BOOTSTRAP", "ipport")
            print(
                f"[moonep-init r{get_ep_group().rank_in_group}] stage=shmem-init-enter "
                f"bootstrap={bootstrap}",
                flush=True,
            )
            if bootstrap == "ipport":
                # Config-store bootstrap (the path ascend-moonep's own multi-rank
                # tests validated) instead of the UID-over-HCCL broadcast.
                from ascend_moonep.buffer_c import symmetric_descriptors

                port = int(os.environ.get("MOONEP_SHMEM_IP_PORT", "28765"))
                ShmemRuntime.init_with_ip_port(
                    group=group,
                    ip_port=f"tcp://127.0.0.1:{port}",
                    descriptors=symmetric_descriptors(),
                    enable_udma=use_udma,
                    qp_count=0,
                )
            else:
                ShmemRuntime.init_with_buffer(group=group)
            print(f"[moonep-init r{get_ep_group().rank_in_group}] stage=shmem-init-done", flush=True)
        if os.environ.get("MOONEP_SLOT_MODE", "0") == "1":
            for st in self._states:
                st.publish()
            if os.environ.get("MOONEP_RELEASE_WEIGHTS", "1") == "1":
                # The freed vllm weight blocks sit in torch's cache otherwise, and
                # the engine sizes its KV cache from driver-level free memory
                # (torch.npu.mem_get_info), so hand them back before that runs.
                try:
                    dev = torch.npu.current_device()
                    torch.npu.empty_cache()
                    free_b, total_b = torch.npu.mem_get_info(dev)
                    print(
                        f"[moonep-release r{get_ep_group().rank_in_group}] home chunks now back "
                        f"the layers; free={free_b / 2**30:.1f}GB / {total_b / 2**30:.1f}GB",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[moonep-release] empty_cache/report failed: {exc}", flush=True)


_MOONEP_RUNTIME = _MoonEPRuntime()


def ensure_runtime_initialized() -> None:
    """MindSpeed-style wiring-time init for the MoonEP runtime.

    Mirrors MindSpeed-MM (init_moonep_layer_states): as soon as the model graph
    exists (end of worker load_model — every AscendFusedMoE layer is then
    registered), construct all symmetric Buffers and run the single collective
    ShmemRuntime.init_with_buffer, *before any forward*. Doing it lazily inside
    the first forward worked for the MTE path but the UDMA path stalls in the
    engine; early collective init on an otherwise-quiet device state is the
    MindSpeed pattern that keeps the UDMA/URMA kernels healthy.

    Idempotent; no-op when MoonEP is not enabled or EP world size <= 1.
    """
    from vllm_ascend.ascend_config import get_ascend_config

    if not get_ascend_config().enable_moonep:
        return
    if get_ep_group().world_size <= 1:
        return
    _MOONEP_RUNTIME._ensure_initialized()


# Host-path constants and caches. Everything here used to be re-evaluated per
# MoE layer: os.environ lookups, a sorted() of the capacity pool, and vLLM group
# lookups. At 48 layers/step each microsecond-class cost shows up in the step
# time (the MoE kernels themselves are only ~0.12 ms/layer), so they are lifted
# out of the hot path.
_ENV_REAL_T = os.environ.get("MOONEP_REAL_T", "0") == "1"
_ENV_REUSE_OUT = os.environ.get("MOONEP_REUSE_OUT", "1") == "1"
_ENV_SLOT_MODE = os.environ.get("MOONEP_SLOT_MODE", "0") == "1"
_ENV_NO_SYNC = os.environ.get("MOONEP_NO_SYNC", "1") != "0"
_ENV_PACE_SYNC = os.environ.get("MOONEP_PACE_SYNC", "0") == "1"
_ENV_CAP_TRACE = os.environ.get("MOONEP_CAP_TRACE", "0") == "1"
_ENV_POOL = os.environ.get("MOONEP_POOL", "1") == "1"
_ENV_PAD_EXPERT0 = os.environ.get("MOONEP_PAD_EXPERT0", "0") == "1"
_GEO_CACHE: dict = {}


def _geometry(E: int, K: int, H: int) -> dict:
    """Rank/epn/sorted-capacity for an (E, K, H) pool, computed once."""
    geo = _GEO_CACHE.get((E, K, H))
    if geo is None:
        ep_group = get_ep_group()
        pool = _MOONEP_RUNTIME._buffers[(E, K, H)]
        geo = {"rank": ep_group.rank_in_group,
               "world": ep_group.world_size,
               "epn": E // ep_group.world_size,
               "caps": sorted(pool),
               "s_max": max(pool),
               "B": int(pool[max(pool)]._require_ctx()["B"])}
        _GEO_CACHE[(E, K, H)] = geo
    return geo


_CAP_TRACE = {"capacity": None}


# Pre-allocated scratch for the per-layer hot path (space-for-time). The hot
# path used to allocate ~8 fresh tensors per layer (cu cast, the cat/sub for the
# counts, the contiguous copies, the weight snapshot's clone+cast+contiguous).
# Those allocations are what `disp_post` was really spending 0.34 ms on: once
# the device is saturated the caching allocator blocks while reclaiming blocks,
# which serialises the host. Reusing out= buffers removes both the kernels and
# the blocking. Sized for the largest capacity of the pool, so one scratch set
# per (E, K, H) covers every step.
_SCRATCH: dict = {}


def _scratch_for(E: int, K: int, H: int, s_max: int, extra: int, device) -> dict:
    """``extra`` = B, the per-rank prefetch-slot count: the library's cu_seqlens
    has E + B entries, which can exceed E + K."""
    key = (int(E), int(K), int(H))
    sc = _SCRATCH.get(key)
    if sc is not None:
        return sc
    nvs = int(s_max) * int(K)
    nrows = int(E) + int(extra)
    # cu_pad[0] is zero for the whole run (nothing ever writes index 0), so the
    # per-group row counts come out of a single subtract -- counts[i] =
    # cu_pad[i+1] - cu_pad[i] == cu[i] - cu[i-1] with the implicit cu[-1] = 0.
    # That replaces the old `counts[0] = cu64[0]` scalar splice, which cost a
    # kernel launch of its own every layer.
    cu_pad = torch.zeros(nrows + 1, dtype=torch.int64, device=device)
    sc = {
        "cu_pad": cu_pad,
        "counts": torch.empty(nrows, dtype=torch.int64, device=device),
        "own": torch.empty(int(E), dtype=torch.int64, device=device),
        # Compatibility-only tokens_per_expert; see the note in token_dispatch.
        # Zeroed once instead of rebuilt with histc every layer.
        "tpe": torch.zeros(int(E), dtype=torch.int32, device=device),
        "w": torch.empty(nvs, dtype=torch.float32, device=device),
        "nrows": nrows,
        "nvs": nvs,
    }
    _SCRATCH[key] = sc
    return sc


def _select_buffer_for_step(E: int, K: int, H: int, tokens: int):
    """Smallest pooled capacity that fits this step -> ``(buffer, capacity)``."""
    _MOONEP_RUNTIME._ensure_initialized()
    pool = _MOONEP_RUNTIME._buffers[(E, K, H)]
    geo = _geometry(E, K, H)
    s_max = geo["s_max"]
    step_tokens = _step_token_count(tokens) if _ENV_POOL else None
    if step_tokens is None or step_tokens > s_max:
        capacity = s_max
    else:
        # smallest rung that still fits the step: with the dense ladder this is
        # the "S as close to T as the pool allows" choice
        need = max(int(step_tokens), MOONEP_MIN_DISPATCH_TOKENS)
        capacity = next((c for c in geo["caps"] if c >= need), s_max)
    if capacity < tokens:
        raise RuntimeError(
            f"MoonEP: step has {tokens} tokens but the selected capacity is "
            f"{capacity} (s_max={s_max})."
        )
    if _ENV_CAP_TRACE and _CAP_TRACE["capacity"] != capacity:
        _CAP_TRACE["capacity"] = capacity
        print(
            f"[moonep-cap] tokens={tokens} step_tokens={step_tokens} "
            f"capacity={capacity} (s_max={s_max})",
            flush=True,
        )
    return pool[capacity], capacity


_COMBINE_ROW_IDS: dict = {}
_ZERO_BF16_SCALAR = None


def _zero_bf16_scalar(device) -> torch.Tensor:
    global _ZERO_BF16_SCALAR
    if _ZERO_BF16_SCALAR is None or _ZERO_BF16_SCALAR.device != torch.device(device):
        _ZERO_BF16_SCALAR = torch.zeros((), dtype=torch.bfloat16, device=device)
    return _ZERO_BF16_SCALAR


def _combine_row_ids(nvs: int, device) -> torch.Tensor:
    """Cached ``arange(nvs)`` for masking the combine input's unused tail."""
    key = (int(nvs), str(device))
    row_ids = _COMBINE_ROW_IDS.get(key)
    if row_ids is None:
        row_ids = torch.arange(int(nvs), dtype=torch.int64, device=device)
        _COMBINE_ROW_IDS[key] = row_ids
    return row_ids


def _pads_for(E: int, K: int, H: int):
    _MOONEP_RUNTIME._ensure_initialized()
    return _MOONEP_RUNTIME._pads[(E, K, H)]


# ---------------------------------------------------------------------------
# Token dispatcher
# ---------------------------------------------------------------------------
class TokenDispatcherWithMoonEP(MoETokenDispatcher[TMoECombineMetadata]):
    """EP token dispatch over MoonEP SHMEM symmetric memory (owner semantics).

    Rows are scattered to the ranks that own their experts; every rank then
    computes its owned experts' rows locally with its own weight tables, and
    ``token_combine`` gathers + K-sums each token's rows (route weights are
    applied per row before the unweighted combine, mirroring MindSpeed-MM).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.top_k = int(kwargs.get("top_k", 0))
        self.num_experts = int(kwargs.get("num_experts", 0))
        self.num_local_experts = int(kwargs.get("num_local_experts", 0))

    def _tstats(self, name: str, t0: float, t1: float | None = None) -> None:
        """MOONEP_TIME_STATS=1: per-phase wall accumulation (host, incl. syncs)."""
        if not _TSTATS_ENABLED:
            return
        dt = (time.perf_counter() if t1 is None else t1) - t0
        acc = _MOONEP_TSTATS[name]
        acc[0] += dt
        acc[1] += 1

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        _maybe_arm_udma_dump()
        if _EXTRA_CTX.capturing:
            raise RuntimeError(
                "MoonEP dispatch cannot run inside captured CUDA graphs; "
                "run with --enforce-eager (or disable cudagraph) when enable_moonep is on."
            )
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        if hidden_states.dim() != 2:
            raise ValueError(f"MoonEP dispatch expects 2D hidden_states, got shape {tuple(hidden_states.shape)}")
        T, H = hidden_states.shape
        E, K = self.num_experts, self.top_k
        if K <= 0 or E <= 0:
            raise ValueError(f"MoonEP: invalid geometry K={K}, E={E}")

        s_max = get_moonep_tokens_capacity()
        if s_max is None or T > s_max:
            raise RuntimeError(
                f"MoonEP: step has {T} tokens > static capacity S={s_max}. "
                "Raise --max-num-batched-tokens or disable enable_moonep."
            )
        # Pad the step up to this step's pooled capacity (<= S_max); every EP
        # rank picks the same one (see _step_token_count). Pad rows carry zero
        # hidden / zero weight, so they contribute exactly zero to the output.
        buffer, capacity = _select_buffer_for_step(E, K, H, T)
        _t0 = time.perf_counter()
        real_t = _ENV_REAL_T
        if real_t:
            # c-light: tell the library this step is exactly T tokens instead of
            # padding up to the pool rung. The shard stays allocated at the rung
            # (symmetric memory is registered once) but the launcher/planning/
            # dispatch/combine only touch rows [0, T*K), so there are no pad rows
            # to compute. Env-gated experimental path; needs bit-exact checks.
            ctx = buffer._require_ctx()
            if ctx.get("S") != T:
                ctx["S"] = T
                ctx["NvS"] = T * K
                ctx["N"] = T * K
        pads = _pads_for(E, K, H)
        # real-T: hand the library exactly T rows (its dispatch asserts
        # topk.numel() == N == S*K), so the window is T rather than the rung.
        window = T if real_t else capacity
        hidden_c = pads["hidden"][:window]
        route_c = pads["route"][:window]
        topk_c = pads["topk"][:window]
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        # Pad rows route to this rank's *own* first expert (rank*epn): routing
        # every rank's pads to expert 0 piles S-T rows onto one rank, which the
        # MoonEP planning/combine kernels misplace for high ranks (R >= 3).
        # Pads are zero-hidden / zero-weight, so their expert is irrelevant to
        # the result. (MOONEP_PAD_EXPERT0=1 restores the expert-0 behavior.)
        # The tail is written only when (T, capacity, pad expert) changes:
        # copy_ always overwrites [0:T] in full and nothing else touches the
        # tail, so unchanged steps skip three launches per layer.
        pad_expert = None
        pad_is_tail = False
        if T < capacity:
            if os.environ.get("MOONEP_PAD_ROTATE", "0") == "1":
                # Spread the pad pile over this rank's experts instead of always
                # dumping it on one: pads are the biggest artificial imbalance the
                # planner sees, and it pays for that imbalance with real
                # cross-rank mirrors (prefetch traffic on the critical path).
                ep_group = get_ep_group()
                epn = E // ep_group.world_size
                rot = _PAD_ROT["i"] % epn
                _PAD_ROT["i"] += 1
                pad_expert = ep_group.rank_in_group * epn + ((epn - 1 - rot) % epn)
                pad_is_tail = True
            elif os.environ.get("MOONEP_PAD_EXPERT0", "0") == "1":
                pad_expert = 0
            else:
                # own LAST expert: pads are then the trailing rows of this
                # rank's row block, which is what lets the grouped GEMM skip
                # them wholesale (see the pad cut below). Any expert owned by
                # this rank works for correctness -- routing every rank's pads
                # to expert 0 piles them onto one rank, which the planning/
                # combine kernels misplace for R >= 3.
                ep_group = get_ep_group()
                epn = E // ep_group.world_size
                pad_expert = ep_group.rank_in_group * epn + (epn - 1)
                pad_is_tail = True
        tail = pads["tail"]
        if not real_t and (tail["T"] != T or tail["capacity"] != capacity
                           or tail["pad_expert"] != pad_expert):
            hidden_c[T:].zero_()
            route_c[T:].zero_()
            if pad_expert is not None:
                topk_c[T:].fill_(pad_expert)
            tail["T"] = T
            tail["capacity"] = capacity
            tail["pad_expert"] = pad_expert
        # copy_ casts on the device: no per-call .to() temporaries.
        hidden_c[:T].copy_(hidden_states)
        route_c[:T].copy_(topk_weights)
        topk_c[:T].copy_(topk_ids)
        _tp0 = time.perf_counter()
        # tokens_per_expert is a compatibility argument, not an input: the
        # library documents it as "retained for MoonEP API compatibility ... the
        # kernel computes local expert counts from topk_flat and all-gathers
        # them before balancing, so the compatibility tensor is validated but
        # not consumed", and its _collect_inputs only checks shape/dtype. It
        # used to cost a histc plus two casts every layer (the histc alone was
        # ~40us of device time); now it is one zeroed buffer built once.
        _pool = _MOONEP_RUNTIME._buffers[(E, K, H)]
        sc = _scratch_for(E, K, H, max(_pool), buffer._require_ctx()["B"], topk_c.device)
        tpe = sc["tpe"]

        self._tstats("disp_pad", _t0, _tp0)
        self._tstats("disp_tpe", _tp0)
        self._tstats("disp_pre", _t0)
        _t1 = time.perf_counter()
        h_nvs, w_nvs, cu, plan = buffer.dispatch(
            hidden_c, route_c, topk_c, tpe,
            zero_copy=False, inter_rank_sync=True,
            reuse_outputs=(not real_t) and _ENV_REUSE_OUT,
        )
        _t2 = time.perf_counter()

        # Per-local-expert row counts = cu diff over this rank's owned global
        # expert range [rank*epn, (rank+1)*epn). Shard rows are prefix-packed
        # in ascending owned-expert order, which is exactly the row/group order
        # the grouped-matmul MLP expects (same contract as the all-to-all
        # dispatcher). Slot groups (cu beyond E) are unused (owner semantics).
        ep_group = get_ep_group()
        epn = E // ep_group.world_size
        rank = ep_group.rank_in_group
        # counts[i] = cu[i] - cu[i-1] with an implicit cu[-1] = 0 (the leading
        # zero term is why this is shifted: slicing per rank would be off by one
        # for rank 0). All three buffers are pre-allocated scratch: the old
        # cast+cat+sub+contiguous chain allocated four tensors per layer.
        counts = sc["counts"]
        cu_pad = sc["cu_pad"]
        n_cu = int(cu.numel())
        cu_pad[1:n_cu + 1].copy_(cu)                 # int32 -> int64 cast
        torch.sub(cu_pad[1:n_cu + 1], cu_pad[:n_cu], out=counts[:n_cu])
        own_counts = sc["own"]
        own_counts[:epn].copy_(counts[rank * epn:(rank + 1) * epn])
        own_counts = own_counts[:epn]
        slot_counts = None
        if os.environ.get("MOONEP_SLOT_MODE", "0") == "1":
            # load-balance path: rows for prefetched (slot) experts stay local
            slot_counts = counts[E:].contiguous()
        if (
            _PAD_CUT
            and pad_is_tail
            and not real_t
            and T < capacity
        ):
            # Drop the pad rows from the grouped GEMM. The grouped matmul walks
            # exactly sum(group_list) rows of the shard, and the (capacity-T)*K
            # zero rows are the *tail* of pad_expert's group (pad_expert is this
            # rank's last expert and pads are the last input rows), so trimming
            # that group removes them without shifting anything else.
            # Skipped when slot rows are present: borrowed groups follow the
            # owned ones in the shard, so shortening the tail would shift them.
            # Device-side (clamp + where), so no host sync per layer.
            cut = torch.clamp(own_counts[-1], max=(capacity - T) * K)
            if slot_counts is not None:
                cut = torch.where(slot_counts.sum() > 0, torch.zeros_like(cut), cut)
            own_counts = own_counts.clone()
            own_counts[-1] = own_counts[-1] - cut
            if _BALANCE_STATS_ON:
                _BALANCE_STATS["pad_cut_calls"] = _BALANCE_STATS.get("pad_cut_calls", 0) + 1
        local_counts = (
            torch.cat((own_counts, slot_counts)) if slot_counts is not None else own_counts
        )
        if os.environ.get("MOONEP_EARLY_PREFETCH", "0") == "1":
            # remember this layer's real slot->expert row: the next step's
            # speculative prefetch uses it as its (collective-free) prediction
            from vllm_ascend.ops.fused_moe.moonep_balance import stash_plan_row

            stash_plan_row(plan.experts_to_copy[rank])
        if _PLAN_STATS_ON:
            _PLAN_STATS.append([int(v) for v in plan.experts_to_copy[rank].tolist()])
        if _EARLY_PLAN_ON:
            # real routing histogram (pre-pad), for the early-plan comparison
            _EARLY["real"].append(
                torch.bincount(topk_ids.reshape(-1).to(torch.int64)).cpu().tolist())
            # second planning pass on the *predicted* routing: this yields the
            # plan the planner would have produced from the early router, which
            # is what the spec-prefetch scheme can act on a layer in advance.
            pid = _EARLY.get("pred_ids")
            _EARLY["pred_ids"] = None
            if pid is not None and int(pid.shape[0]) == T:
                try:
                    pred = torch.zeros(capacity, K, dtype=torch.int32, device=hidden_states.device)
                    pred[:T].copy_(pid.to(torch.int32))
                    if T < capacity and pad_expert is not None:
                        pred[T:].fill_(pad_expert)
                    _ctx = buffer._require_ctx()
                    plan_pred = _plan_on_predicted_routing(_ctx, pred, E, K)
                    _EARLY["plan_pred"].append(
                        [int(v) for v in plan_pred.experts_to_copy[rank].tolist()])
                except Exception:  # noqa: BLE001 - probe must never break a step
                    _EARLY["fail"] += 1
        if _BALANCE_STATS_ON:
            # per-rank (token, expert) entries: what this rank computes vs the
            # full step. Opt-in: reading the counts back costs a host sync.
            stat = _BALANCE_STATS
            stat["rank"] = get_ep_group().rank_in_group
            stat["calls"] += 1
            stat["tokens"] += T
            stat["local_rows"] += int(local_counts.sum().item())
            stat["all_rows"] += int(counts[:E].sum().item())
            stat["pad_rows"] += (capacity - T) * K
        if os.environ.get("MOONEP_SLOT_STATS", "0") == "1" and slot_counts is not None:
            # debug only: this host read re-introduces the per-layer sync
            s = _SLOT_STATS
            s["calls"] += 1
            sr = int(slot_counts.sum().item())
            s["slot_rows"] += sr
            if sr > 0:
                s["with_slots"] += 1
            s["prefetch_experts"] += int((plan.experts_to_copy[ep_group.rank_in_group] >= 0).sum().item())
        if _ENV_NO_SYNC:
            # Fast path: keep the row count on the device. The MLP takes the
            # full [NvS, H] shard plus the count-mode `group_list` and the
            # grouped matmul walks exactly sum(group_list) rows (rows past the
            # sum are not computed), so no `.item()` is needed -- the combine
            # masks the untouched tail on the device instead (see combine).
            hidden_trim = h_nvs
            # pre-allocated fp32 snapshot (one cast-copy instead of
            # clone + to + contiguous, i.e. three allocating ops)
            w_buf = sc["w"]
            n = min(int(w_buf.numel()), int(w_nvs.numel()))
            w_buf[:n].copy_(w_nvs[:n])
            w_trim = w_buf[:n]
            tail_masked = True
        else:
            # Validated default: read the dispatched row count back to the
            # host and slice, so the MLP and the combine only ever see real
            # rows.
            rows_total = int(local_counts.sum().item())
            if rows_total < 0 or rows_total > int(K) * capacity:
                raise RuntimeError(f"MoonEP: invalid dispatched row count {rows_total}.")
            hidden_trim = h_nvs[:rows_total]
            # clone: the symmetric weights_buf is shared by all layers
            w_trim = w_nvs[:rows_total].clone().to(torch.float32).contiguous()
            tail_masked = False
        if _ENV_PACE_SYNC:
            torch.npu.synchronize()
        _t3 = time.perf_counter()

        self._tstats("disp_lib", _t1, _t2)   # buffer.dispatch host side (planning + launches)
        self._tstats("disp_post", _t2, _t3)  # counts + the rows_total host sync + w clone
        self._tstats("disp_buf", _t1)
        self._tstats("dispatch", _t0)
        state = (
            getattr(_SLOT_TLS, "state", None)
            if os.environ.get("MOONEP_SLOT_MODE", "0") == "1"
            else None
        )
        return MoETokenDispatchOutput(
            hidden_states=hidden_trim,
            group_list=local_counts,
            group_list_type=1,  # count mode
            combine_metadata=MoonEPCombineMetadata(
                plan=plan,
                buffer=buffer,
                w_trim=w_trim,
                num_tokens=T,
                hidden_shape=hidden_states.shape,
                state=state,
                group_list=local_counts,
                tail_masked=tail_masked,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in TokenDispatcherWithMoonEP."
        _t0c = time.perf_counter()
        assert isinstance(combine_metadata, MoonEPCombineMetadata)
        T = combine_metadata.num_tokens
        # Apply the per-row route weights before the unweighted K-sum combine
        # (mirrors MindSpeed shmem_ep_forward; the Ascend combine kernel does
        # not apply weights itself). The MLP output rows map 1:1 onto the
        # dispatched shard rows (grouped matmul preserves row order).
        w = combine_metadata.w_trim
        assert hidden_states.shape[0] == w.shape[0], (
            f"MoonEP combine: MLP produced {hidden_states.shape[0]} rows but dispatch "
            f"captured {w.shape[0]} row weights."
        )
        buffer = combine_metadata.buffer
        nvs = int(buffer._require_ctx()["NvS"])
        S = int(buffer._require_ctx()["S"])
        H = int(buffer._require_ctx()["H"])
        # w_trim is fp32, so bf16 * fp32 promotes to fp32 on its own.
        if combine_metadata.tail_masked:
            # No-sync dispatch: the MLP ran over the full [NvS, H] shard with a
            # count-mode group_list, so only rows [0, sum(group_list)) hold
            # results -- the grouped matmul never writes the rest, and that tail
            # is *uninitialized* memory (the combine kernel reads the whole
            # tensor it is handed). Zero it on the device rather than reading
            # the count back to the host; `where` (not a multiply) is required
            # so a NaN in the tail cannot leak through 0 * NaN.
            group_list = combine_metadata.group_list
            assert group_list is not None, "MoonEP combine requires the dispatch group_list."
            rows_total = group_list.sum().to(torch.int64)
            row_mask = (_combine_row_ids(
                nvs, hidden_states.device) < rows_total).unsqueeze(-1)
            combine_input = torch.where(
                row_mask,
                (hidden_states * w.unsqueeze(-1)).to(torch.bfloat16),
                _zero_bf16_scalar(hidden_states.device),
            )
        else:
            weighted = (hidden_states * w.unsqueeze(-1)).to(torch.bfloat16)
            # Zero fill is load-bearing: reusing a scratch and only overwriting
            # the [0, rows_total) prefix leaves stale rows that the combine
            # kernel reads (measured: first tokens diverge from MC2).
            combine_input = torch.zeros(nvs, H, dtype=torch.bfloat16,
                                        device=weighted.device)
            combine_input[:weighted.shape[0]].copy_(weighted)
        self._tstats("comb_pre", _t0c)
        _t1c = time.perf_counter()
        out, _, _ = buffer.combine(
            plan=combine_metadata.plan,
            hidden_nvsh=combine_input,
            route_weights_nvs=None,
            inter_rank_sync=True,
        )
        assert out.shape == (S, H), f"MoonEP combine returned {tuple(out.shape)}"
        self._tstats("comb_buf", _t1c)
        self._tstats("combine", _t0c)
        return out[:T].view(combine_metadata.hidden_shape)


class MoonEPCommImpl(MoECommMethod):
    """Expert-parallel MoE over MoonEP SHMEM dispatch, reusing the all-to-all
    prepare/finalize (TP slicing/unpadding semantics) and the standard MLP."""

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMoonEP(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def fused_experts(self, fused_experts_input):
        if os.environ.get("MOONEP_SLOT_MODE", "0") != "1":
            return super().fused_experts(fused_experts_input)
        from vllm_ascend.ops.fused_moe.moonep_balance import run_slot_forward
        from vllm_ascend.ops.fused_moe.moe_runtime_args import build_token_dispatch_input

        w1 = fused_experts_input.weights.w1
        if isinstance(w1, (list, tuple)):
            raise NotImplementedError(
                "MoonEP slot mode requires single (non-fusion) weight tensors."
            )
        rt = _MOONEP_RUNTIME
        state = getattr(rt, "_state_by_w13", {}).get(id(w1))
        if state is None:
            raise RuntimeError(
                "MoonEP slot mode: no balance state for this layer's w13 "
                "(ensure_runtime_initialized must run after weight loading)."
            )
        _SLOT_TLS.state = state
        try:
            disp = self.token_dispatcher.token_dispatch(
                build_token_dispatch_input(fused_experts_input=fused_experts_input)
            )
            meta = disp.combine_metadata
            mlp_rows = run_slot_forward(
                state=meta.state,
                buffer=meta.buffer,
                plan=meta.plan,
                hidden_trim=disp.hidden_states,
                group_list=meta.group_list,
                fused_experts_input=fused_experts_input,
            )
            routed_out = self.token_dispatcher.token_combine(mlp_rows, meta)
        finally:
            _SLOT_TLS.state = None
        return FusedExpertsResult(
            routed_out=routed_out,
            group_list_type=disp.group_list_type,
            expert_tokens=disp.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

atexit.register(_dump_slot_stats)


# ---------------------------------------------------------------------------
# UDMA stall watchdog: MOONEP_UDMA_DUMP=1 arms a background torch_npu profiler
# at the first dispatch and dumps the device trace after a stall window, to
# capture which kernel never completes when UDMA deadlocks inside the engine.
# ---------------------------------------------------------------------------
_udma_dump_armed = False


def _maybe_arm_udma_dump() -> None:
    global _udma_dump_armed
    if _udma_dump_armed:
        return
    if os.environ.get("MOONEP_UDMA_DUMP", "0") != "1":
        return
    if os.environ.get("MOONEP_DISABLE_UDMA", "0") == "1":
        return
    _udma_dump_armed = True
    wait = int(os.environ.get("MOONEP_UDMA_DUMP_WAIT", "45"))
    import time

    def _dump():
        try:
            import torch_npu
            from torch_npu import profiler

            d = f"/tmp/udma_dump_r{os.environ.get('LOCAL_RANK', '?')}"
            os.makedirs(d, exist_ok=True)
            pr = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=profiler.tensorboard_trace_handler(d),
            )
            pr.start()
            time.sleep(wait)
            print("[udma-dump] stopping profiler (stall window elapsed)", flush=True)
            pr.stop()
            print("[udma-dump] DUMP DONE -> " + d, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[udma-dump] failed: {exc}", flush=True)

    threading.Thread(target=_dump, daemon=True).start()
