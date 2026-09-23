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
import time
import threading

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput

_slot_mode = os.environ.get("MOONEP_SLOT_MODE", "0") == "1"

# Persistent per-layer slot cache (FreeBalance-style "migrate once, reuse"):
# each layer keeps the weights of its N most recently mirrored remote experts in
# its own VMM chunk. A later step that mirrors the same expert into the same
# layer serves it with a *local* copy instead of a remote GET. Measured plan
# repeats make ~2/3 of the slot uses cacheable (docs/dev_results_2026-09-16.md).
# Off by default: measured 2026-09-17, the cache reaches a 68% hit rate and
# skips 80% of the prefetch launches, yet the step gets ~9 ms SLOWER -- the
# per-layer host read of the plan row costs more than the prefetch it saves
# (the prefetch is not on the critical path; the earlier 0.49 ms figure was
# a sync-based measurement that included queue drain).
MOONEP_SLOT_CACHE = int(os.environ.get("MOONEP_SLOT_CACHE", "0"))

# MOONEP_PF_TIMING=1: device-synchronised timing of prefetch_projections (and of
# the grouped GEMM that follows), so the speculative-prefetch work has a number
# to beat. Measurement only: it serialises the step.
_PF_TIMING: dict = {}


def _pf_time(name: str, dt: float) -> None:
    acc = _PF_TIMING.setdefault(name, [0.0, 0])
    acc[0] += dt
    acc[1] += 1


def _pf_stats_init(dev):
    if _PF.get("cnt") is None:
        _PF["cnt"] = {k: torch.zeros((), dtype=torch.int64, device=dev)
                      for k in ("layers", "spec_layers", "slots", "matched")}


def _dump_early_pf_stats() -> None:
    cnt = _PF.get("cnt")
    if cnt is None:
        return
    try:
        v = {k: int(t.item()) for k, t in cnt.items()}
    except Exception:  # noqa: BLE001
        return
    slots = max(v["slots"], 1)
    print(f"[early-pf-stats] layers={v['layers']} spec_layers={v['spec_layers']} "
          f"slots={v['slots']} matched={v['matched']} "
          f"match_rate={v['matched'] / slots:.3f}", flush=True)


import atexit as _atexit_epf

_atexit_epf.register(_dump_early_pf_stats)


_SLOT_CACHE_STATS: dict = {}
_PF_SKIPPED = [0]


def _dump_slot_cache_stats() -> None:
    """Sum the per-layer hit/miss counters once, at exit."""
    if not _SLOT_CACHE_STATS:
        return
    hits = sum(v[0] for v in _SLOT_CACHE_STATS.values())
    misses = sum(v[1] for v in _SLOT_CACHE_STATS.values())
    tot = max(hits + misses, 1)
    print(f"[slot-cache] layers={len(_SLOT_CACHE_STATS)} hits={hits} misses={misses} "
          f"hit_rate={hits / tot:.3f} prefetch_skipped={_PF_SKIPPED[0]}", flush=True)


import atexit as _atexit_sc

_atexit_sc.register(_dump_slot_cache_stats)


def _dump_pf_timing() -> None:
    if not _PF_TIMING:
        return
    parts = ["[moonep-pf-timing]"]
    for name, (tot, calls) in _PF_TIMING.items():
        parts.append(f"{name}: calls={calls} total={tot:.3f}s avg={tot / max(calls, 1) * 1e3:.3f}ms")
    print(" ".join(parts), flush=True)


import atexit as _atexit_pf

_atexit_pf.register(_dump_pf_timing)

# Release the vllm weight tensors once the home mirror holds them (slot mode
# computes against the [epn+B] alias, never against the vllm tensor). ~14.5 GB
# for Qwen3-30B-A3B at EP=4. See _LayerWeightState.publish.
_RELEASE_WEIGHTS = os.environ.get("MOONEP_RELEASE_WEIGHTS", "1") == "1"


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
                # per-layer weight cache: one [L, N] chunk per projection, so the
                # whole cache costs two VMM allocations rather than 2 per layer
                n_layers = max(1, int(os.environ.get("MOONEP_SLOT_CACHE_LAYERS", "64")))
                cache = {}
                if MOONEP_SLOT_CACHE > 0:
                    for name, (rows, cols) in (("w13", (H, 2 * F)), ("w2", (F, H))):
                        c, _, _ = create_vmm_physical(
                            [n_layers, MOONEP_SLOT_CACHE, rows, cols],
                            torch.bfloat16, dev)
                        c.zero_()
                        cache[name] = c
                # full [E+B] projections need home chunks; built per layer, so
                # keep slot chunks here and let _LayerState build the fulls.
                pool = {"slots": slots, "cache": cache, "B": B}
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
        self._layer = layer
        # persistent per-layer expert cache (FreeBalance-style reuse)
        self.layer_index = 0
        self.cache_w13 = shared["cache"].get("w13") if shared.get("cache") else None
        self.cache_w2 = shared["cache"].get("w2") if shared.get("cache") else None
        self.cache_experts = [-1] * max(MOONEP_SLOT_CACHE, 1)
        self.cache_next = 0
        self.cache_stats = [0, 0]      # [hits, misses] over used slots
        self._vmm_tensors = [self.home_w13, self.home_w2]

    def publish(self) -> None:
        """One-time mirror of the local vllm weight rows into the home chunks.

        Must run after ShmemRuntime init (copies only; no new descriptors).

        With MOONEP_RELEASE_WEIGHTS=1 the originals are then dropped: the home
        chunks become the layer's weights (in place -- ``.data`` of the same
        Parameter objects, because the runtime looks layer states up by
        ``id(w13_weight)``), so the rank keeps *one* copy of its expert weights
        instead of two. That is ~14.5 GB for Qwen3-30B-A3B at EP=4 (302 MB per
        layer of w13+w2 x 48 layers).

        Only safe because slot mode computes against the [epn+B] alias, never
        against the vllm tensor. Do not enable it where the weights are written
        again after init (weight reload / sleep mode / RL rollouts).
        """
        if self._published:
            return
        with torch.no_grad():
            self.home_w13.copy_(self._w13_src)
            self.home_w2.copy_(self._w2_src)
        self._published = True
        if _RELEASE_WEIGHTS:
            self._release_sources()

    def _release_sources(self) -> None:
        """Point the layer's weight Parameters at the home chunks (drop the copies)."""
        released = 0
        for name, src_attr, home in (("w13_weight", "_w13_src", self.home_w13),
                                     ("w2_weight", "_w2_src", self.home_w2)):
            src = getattr(self, src_attr)
            layer_param = getattr(self._layer, name, None)
            if layer_param is None:
                continue
            if layer_param is not src:
                # the layer was re-parameterised after the state was built: the
                # id() lookup would already be broken, so leave it alone
                continue
            if tuple(layer_param.shape) != tuple(home.shape) or layer_param.dtype != home.dtype:
                raise RuntimeError(
                    f"MoonEP release: {name} shape/dtype changed "
                    f"({tuple(layer_param.shape)} {layer_param.dtype} vs "
                    f"{tuple(home.shape)} {home.dtype}); refusing to swap")
            released += home.numel() * home.element_size()
            # in-place storage swap: the Parameter object (and therefore the
            # runtime's id()-keyed state lookup) is preserved
            layer_param.data = home
        if released:
            self._released = True
            print(
                f"[moonep-release] {type(self._layer).__name__}: vllm weight copies "
                f"released, home chunks now back the layer ({released / 2**20:.0f} MB)",
                flush=True)

    def destroy(self):
        if getattr(self, "_released", False):
            # after a release the home chunks *are* the layer's weights: freeing
            # them would leave the layer pointing at unmapped memory
            raise RuntimeError(
                "MoonEP: refusing to destroy weight state whose home chunks back "
                "the layer (MOONEP_RELEASE_WEIGHTS=1).")
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
# Early (speculative) prefetch: predict layer i's plan before its attention
# ---------------------------------------------------------------------------
# Measured (2026-09-16, GSM8K 256x64, EP4): `prefetch_projections` costs
# 0.49 ms per layer of *device* time -- more than the grouped GEMM that follows
# it (0.22 ms) -- because the MTE GET blocks its stream. Since a stream executes
# in order, moving the prefetch earlier on the same stream saves nothing; the
# transfer only overlaps the attention if it is issued on a *side* stream.
#
# The plan, however, is only known after this layer's dispatch... unless we
# predict it: running this layer's own router on the *pre-attention* residual
# (the previous layer's output) reproduces the planner's input histogram with
# cosine 0.986, and feeding that predicted routing through the planning kernel
# yields a plan whose slot->expert assignment matches the real one 80.5% of the
# time (vs 69% for simply reusing the previous step's plan). See
# docs/early_plan_prediction.html.
#
# So: in the decoder-layer pre-hook (i.e. while the attention is about to run)
# we predict the routing, run the planning kernel on it, and issue the prefetch
# on a side stream. When the real plan arrives after the dispatch, we re-prefetch
# only the slots that differ (usually none) and the grouped GEMM waits on the
# side stream's event.
#
# MOONEP_EARLY_PREFETCH=1 enables it; MOONEP_EARLY_PF_DEV=0 keeps the
# speculative issue on the main stream (no overlap, for A/B).
_EARLY_PF = os.environ.get("MOONEP_EARLY_PREFETCH", "0") == "1"
_EARLY_PF_DEV = os.environ.get("MOONEP_EARLY_PF_DEV", "1") == "1"

# MOONEP_PF_STREAM=1: 把 prefetch 发到 side stream，与同层的 dispatch 并行。
# 与 EARLY_PREFETCH 的区别：那个用「上一层的 plan 行」当预测、在 attention 前
# 发；这个用本层的**真实 plan**，在 planning 之后、dispatch 之前发 —— 不需要
# 预测，也不改变搬的东西。
#
# 必须配套 ASCEND_MOONEP_PREFETCH_RELAXED=1：prefetch 的两个跨卡 barrier 要
# 摘掉。不摘的话它们会跟着跑到 side stream 上，各 rank 的栅栏代际错开 → 死锁
# （prefetch.cpp:101 的注释写明了 RELAXED 就是为 side-stream 发射准备的）。
# MOONEP_PF_STREAM:
#   0 = 关（默认）
#   1 = 只走事件管道：钩子里 record_event + wait_event + 进出 stream 上下文都做，
#       但 prefetch 仍留在主 stream —— 输出正确，用来单独量「事件管道的开销」
#   2 = 全量：prefetch 真正发到 side stream
_PF_STREAM = int(os.environ.get("MOONEP_PF_STREAM", "0"))
_PF_STREAMED = [0]

# 与 moonep_comm._ALLOW_CAPTURE 同义（各自定义以避免模块级循环 import）。
# MOONEP_ALLOW_CAPTURE=1 放开 run_slot_forward 的捕获守卫，供可行性实验用。
_ALLOW_CAPTURE = os.environ.get("MOONEP_ALLOW_CAPTURE", "0") == "1"
_PF: dict = {"pred": {}, "stream": None, "ev_spec": None, "ev_main": None, "main": None,
             "done": False, "fail": 0, "spec": 0, "install_logged": False,
             # predictor state: the *previous step's* real plan row per layer.
             # It needs no collective and no router re-run -- measured to match
             # the current plan 69% of the time (vs 80.5% for the router-based
             # prediction, which required an extra planning-kernel call and that
             # call desynchronised the library's cross-rank pacing => hang).
             "cur_key": None, "last_rows": {}, "shell": {}, "pending": {},
             "shell_idx": 0, "spec_plan": None,
             # stats (device-side accumulation, read once at exit)
             "cnt": None}


def note_current_layer(key) -> None:
    """Tell the runtime which layer is executing (set by the pre-hook)."""
    if _EARLY_PF:
        _PF["cur_key"] = key


def stash_plan_row(row) -> None:
    """Remember this layer's real slot->expert row (called from the dispatcher)."""
    if not _EARLY_PF:
        return
    key = _PF.get("cur_key")
    if key is not None:
        _PF["last_rows"][key] = row.detach().clone()


def _pf_shell_plan(buffer):
    """A plan-shaped shell whose only meaningful field is experts_to_copy."""
    cap = int(buffer._require_ctx()["S"])
    shell = _PF["shell"].get(cap)
    if shell is not None:
        return shell
    from ascend_moonep.planning import allocate_planning_outputs

    plan, _cu = allocate_planning_outputs(buffer._require_ctx())
    plans = [plan]
    plan2, _cu2 = allocate_planning_outputs(buffer._require_ctx())
    plans.append(plan2)
    _PF["shell"][cap] = plans
    return plans


def take_predicted_plan(key):
    """Pop the speculative plan stashed for this layer (or None)."""
    if not _EARLY_PF:
        return None
    return _PF["pred"].pop(key, None)


def finish_speculative_prefetch():
    """Make the current stream wait for an in-flight speculative prefetch.

    Must run before anything that touches the slot chunks (the repair prefetch
    and the grouped GEMM).
    """
    if _PF.get("ev_spec") is None:
        return False
    import torch_npu

    torch_npu.npu.current_stream().wait_event(_PF["ev_spec"])
    _PF["ev_spec"] = None
    return True


def _pf_prefetch(buffer, state, plan):
    buffer.prefetch_projections(plan=plan, projections=[state.full_w13, state.full_w2])


def _pf_state_for(key):
    from vllm_ascend.ops.fused_moe.moonep_comm import _MOONEP_RUNTIME

    return getattr(_MOONEP_RUNTIME, "_state_by_w13", {}).get(key)


def maybe_install_early_prefetch(model) -> None:
    """MOONEP_EARLY_PREFETCH=1: predict the plan in the pre-attention hook and
    issue the expert-weight prefetch on a side stream, so the 0.49 ms MTE
    transfer overlaps this layer's attention instead of stalling the stream."""
    if not _EARLY_PF or _PF["done"]:
        return
    try:
        import torch_npu

        from vllm_ascend.ops.fused_moe import moonep_comm as mc

        layers = mc._early_layers(model)
        if not layers:
            print("[early-pf] no decoder layers found; disabled", flush=True)
            return

        def _hook(blk, args, kwargs):
            try:
                h = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
                res = kwargs.get("residual", args[2] if len(args) > 2 else None)
                if h is None:
                    return None
                if res is not None:
                    h = h + res                      # pre-attention residual stream
                experts = getattr(getattr(blk, "mlp", None), "experts", None)
                if experts is None:
                    return None
                key = id(experts.w13_weight)
                note_current_layer(key)              # the dispatcher stashes the real row
                state = _pf_state_for(key)
                if state is None:
                    return None
                row = _PF["last_rows"].get(key)
                if row is None:                      # first step of this layer
                    return None
                from vllm.distributed.parallel_state import get_ep_group

                T = int(h.shape[0])
                buffer, _cap = mc._select_buffer_for_step(
                    state.E, int(experts.top_k), state.H, T)
                shells = _pf_shell_plan(buffer)
                shell = shells[_PF["shell_idx"]]
                _PF["shell_idx"] ^= 1                # never rewrite an in-flight plan
                etc = shell.experts_to_copy
                etc.fill_(-1)
                etc[get_ep_group().rank_in_group] = row
                _PF["spec_plan"] = shell
                if _PF["main"] is None:
                    _PF["main"] = torch_npu.npu.current_stream()
                _PF["ev_main"] = _PF["main"].record_event()   # slots are safe after this
                if _EARLY_PF_DEV:
                    if _PF["stream"] is None:
                        _PF["stream"] = torch_npu.npu.Stream()
                    stream = _PF["stream"]
                    stream.wait_event(_PF["ev_main"])
                    with torch_npu.npu.stream(stream):
                        _pf_prefetch(buffer, state, shell)
                    _PF["ev_spec"] = stream.record_event()
                else:
                    _pf_prefetch(buffer, state, shell)
                    _PF["ev_spec"] = None
                _PF["spec"] += 1
                _PF["pending"][key] = row
            except Exception as exc:  # noqa: BLE001 - never break a step
                _PF["fail"] += 1
                if _PF["fail"] in (1, 50):
                    print(f"[early-pf] speculative prefetch failed (#{_PF['fail']}): {exc}", flush=True)
            return None

        for blk in layers:
            blk.register_forward_pre_hook(_hook, with_kwargs=True)
        _PF["done"] = True
        print(f"[early-pf] enabled on {len(layers)} decoder layers "
              f"(side stream={int(_EARLY_PF_DEV)})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[early-pf] install failed: {exc}", flush=True)


def maybe_install_pf_stream(model) -> None:
    """MOONEP_PF_STREAM=1: 让 after_planning 钩子知道"当前是哪一层"。

    钩子在库的 `_run_dispatch_on_current_stream` 里触发（planning 之后、
    dispatch 之前），需要拿到本层的 `_LayerWeightState` 才能发 prefetch。
    层的识别方式与 early-prefetch 一致：用 `experts.w13_weight` 的 id 当 key，
    由 decoder 层的 forward pre-hook 在每层开始时记下。
    """
    if not _PF_STREAM or _PF["done"]:
        return
    try:
        from vllm_ascend.ops.fused_moe import moonep_comm as mc

        layers = mc._early_layers(model)
        if not layers:
            print("[pf-stream] no decoder layers found; disabled", flush=True)
            return

        def _pre(blk, args, kwargs):
            try:
                experts = getattr(getattr(blk, "mlp", None), "experts", None)
                if experts is not None:
                    _PF["cur_key"] = id(experts.w13_weight)
            except Exception:  # noqa: BLE001 - 探测逻辑绝不能打断推理
                pass
            return None

        for blk in layers:
            blk.register_forward_pre_hook(_pre, with_kwargs=True)
        _PF["done"] = True
        print(f"[pf-stream] pre-hook installed on {len(layers)} decoder layers",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[pf-stream] install failed: {exc}", flush=True)


def _pf_stream_hook(buffer):
    """装在库的 `after_planning_hook` 上的回调：在 side stream 上发 prefetch。

    为什么必须挂在这里：prefetch 要读 planning 产出的 `experts_to_copy`，又希望
    和随后的 dispatch 并行。如果在 `buffer.dispatch()` 返回之后（`run_slot_forward`
    里）才 record_event，那事件已经把 dispatch 也圈进去了，side stream 会一直等
    dispatch 完成 —— 没有重叠。只有在本钩子这个时间点，主 stream 上排在最前面的
    才是 planning。
    """
    def _after(plan):
        try:
            key = _PF.get("cur_key")
            state = _pf_state_for(key) if key is not None else None
            if state is None:      # 本层第一次跑，还没建权重状态
                return
            import torch_npu

            main = torch_npu.npu.current_stream()
            if _PF_STREAM < 2:
                # 诊断档：只走一遍事件管道，prefetch 留给 run_slot_forward 在主
                # stream 上发。输出与档 0 完全一致，差值就是管道本身的开销。
                main.record_event()
                _PF_STREAMED[0] += 1
                return
            if _PF["stream"] is None:
                _PF["stream"] = torch_npu.npu.Stream()
            side = _PF["stream"]
            side.wait_event(main.record_event())
            with torch_npu.npu.stream(side):
                _pf_prefetch(buffer, state, plan)
            # 单槽握手：层是严格串行的，所以一个 slot 就够，run_slot_forward
            # 取走之后下一层才会再写。
            _PF["pf_ev"] = side.record_event()
            _PF_STREAMED[0] += 1
        except Exception as exc:  # noqa: BLE001 - 绝不能让探测逻辑打断推理
            _PF["fail"] += 1
            if _PF["fail"] in (1, 50):
                print(f"[pf-stream] issue failed: {exc}", flush=True)

    return _after


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
    if capturing and not _ALLOW_CAPTURE:
        raise RuntimeError(
            "MoonEP slot mode cannot run inside captured graphs. "
            "(MOONEP_ALLOW_CAPTURE=1 lifts this guard for the capture feasibility study; "
            "it is a deliberate override, not a claim that capture works.)"
        )
    plan_use = plan
    if _EARLY_PF:
        try:
            from vllm.distributed.parallel_state import get_ep_group

            rank = get_ep_group().rank_in_group
            # the row the speculative prefetch fetched (previous step's plan)
            spec_row = _PF["pending"].pop(id(state._w13_src), None)
            _pf_stats_init(plan.experts_to_copy.device)
            cnt = _PF["cnt"]
            cnt["layers"] += 1
            if spec_row is not None:
                cnt["spec_layers"] += 1
                real_row = plan.experts_to_copy[rank]
                same = spec_row == real_row
                cnt["slots"] += real_row.numel()
                cnt["matched"] += same.sum()
                repair_row = torch.where(same, torch.full_like(real_row, -1), real_row)
                etc = torch.full_like(plan.experts_to_copy, -1)
                etc[rank] = repair_row
                import dataclasses as _dc

                plan_use = _dc.replace(plan, experts_to_copy=etc)
        except Exception as exc:  # noqa: BLE001
            _PF["fail"] += 1
            if _PF["fail"] in (1, 50):
                print(f"[early-pf] repair-plan build failed: {exc}", flush=True)
            plan_use = plan
        # anything that touches the slot chunks must wait for the side-stream GET
        finish_speculative_prefetch()
    _sync = os.environ.get("MOONEP_PF_TIMING", "0") == "1"
    if _sync:
        # measurement-only: the device time of the prefetch itself (the host-side
        # timer under MOONEP_TIME_STATS only sees the launch)
        torch.npu.synchronize()
        _t_dev = time.perf_counter()
    # --- persistent per-layer cache: serve repeats with a local copy ---
    cache_plan = None
    skip_prefetch = False
    if MOONEP_SLOT_CACHE > 0 and state.cache_w13 is not None:
        try:
            from vllm.distributed.parallel_state import get_ep_group

            rank = get_ep_group().rank_in_group
            row = [int(v) for v in plan_use.experts_to_copy[rank].tolist()]
            hits, misses = [], []
            for i, e in enumerate(row):
                if e < 0:
                    continue
                if e in state.cache_experts:
                    hits.append((state.cache_experts.index(e), i))
                else:
                    misses.append((e, i))
            state.cache_stats[0] += len(hits)
            state.cache_stats[1] += len(misses)
            _SLOT_CACHE_STATS[id(state)] = state.cache_stats
            if hits:
                masked = list(row)
                for _, i in hits:
                    masked[i] = -1          # the library skips -1 slots
                etc = plan_use.experts_to_copy.clone()
                etc[rank] = torch.tensor(masked, dtype=etc.dtype, device=etc.device)
                import dataclasses as _dc

                plan_use = _dc.replace(plan_use, experts_to_copy=etc)
            cache_plan = (hits, misses)
            # The prefetch's cost is dominated by its fixed part (kernel launch
            # + two cross-rank barriers: ~0.38 ms of the measured 0.49 ms), not
            # by the bytes, so masking hits out of the plan saves almost nothing
            # -- the launch itself must be skipped. When this layer's cache
            # covers every used slot, skip the whole call.
            skip_prefetch = not misses
        except Exception as exc:  # noqa: BLE001 - never break a step
            _PF["fail"] += 1
            if _PF["fail"] in (1, 50):
                print(f"[slot-cache] skipped: {exc}", flush=True)
            cache_plan = None
    _t_pf = time.perf_counter() if os.environ.get("MOONEP_TIME_STATS", "0") == "1" else None
    pf_ev = None
    if _PF_STREAM:
        # 幂等安装：挂到本 buffer 的 ctx 上。第一次走到的 buffer 要从下一层
        # 才生效（钩子必须在 dispatch 之前就位），首层退回原路径，无副作用。
        _pf_ctx = buffer._require_ctx()
        if _pf_ctx.get('after_planning_hook') is None:
            _pf_ctx['after_planning_hook'] = _pf_stream_hook(buffer)
        pf_ev = _PF.pop("pf_ev", None)
    if skip_prefetch:
        _PF_SKIPPED[0] += 1
    elif pf_ev is not None:
        # 本层的 prefetch 已由 after_planning 钩子发到 side stream 上，与
        # dispatch 并行了；这里不重复发，只把 event 留到 GEMM 前 join。
        pass
    else:
        buffer.prefetch_projections(
            plan=plan_use,
            projections=[state.full_w13, state.full_w2],
        )
    if _sync:
        torch.npu.synchronize()
        _pf_time(name="prefetch", dt=time.perf_counter() - _t_dev)
    if _t_pf is not None:
        from vllm_ascend.ops.fused_moe.moonep_comm import _MOONEP_TSTATS

        acc = _MOONEP_TSTATS.setdefault("prefetch", [0.0, 0])
        acc[0] += time.perf_counter() - _t_pf
        acc[1] += 1
    if pf_ev is not None:
        # MOONEP_PF_STREAM：prefetch 在 side stream 上跑，主 stream 一路走到
        # dispatch 结束都不用等它。但下面 cache 回填和 GEMM 都要读 slot，所以在
        # 这里 join。此时 dispatch 的 host 时间和设备时间都已经过去，prefetch
        # 的 279us 被盖掉了。
        import torch_npu as _tn_pf

        _tn_pf.npu.current_stream().wait_event(pf_ev)
    if cache_plan is not None:
        # stream-ordered after the prefetch: hits come from this layer's cache
        # (no remote GET), the fetched experts refill the cache for later steps
        hits, misses = cache_plan
        with torch.no_grad():
            for j, i in hits:
                state.slot_w13[i].copy_(state.cache_w13[state.layer_index, j])
                state.slot_w2[i].copy_(state.cache_w2[state.layer_index, j])
            for e, i in misses:
                j = state.cache_next
                state.cache_next = (j + 1) % max(MOONEP_SLOT_CACHE, 1)
                state.cache_w13[state.layer_index, j].copy_(state.slot_w13[i])
                state.cache_w2[state.layer_index, j].copy_(state.slot_w2[i])
                state.cache_experts[j] = e
    if _sync:
        torch.npu.synchronize()
        _t_gemm = time.perf_counter()
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
    if _sync:
        torch.npu.synchronize()
        _pf_time("gemm_after_prefetch", time.perf_counter() - _t_gemm)
    return rows


__all__ = ["slot_mode_enabled", "_LayerWeightState", "_SharedSlots", "run_slot_forward"]
