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
  * Capacity envelope (empirical, this ascend-moonep build): the dispatch
    kernel silently drops a row once its destination offset exceeds that
    rank's S*K symmetric chunk (measured at R=4 with heavy per-expert skew and
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
from vllm_ascend.ops.fused_moe.moe_par_stats import note_route_counts as _note_route_counts
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
MOONEP_MIN_DISPATCH_TOKENS = 128


def _moonep_pool_capacities(s_max: int) -> list[int]:
    """Capacity ladder for the buffer pool (MOONEP_POOL_CAPS overrides)."""
    override = os.environ.get("MOONEP_POOL_CAPS")
    if override:
        caps = sorted({int(c) for c in override.split(",") if c.strip()})
        if not caps or caps[-1] != int(s_max):
            caps = sorted(set(caps) | {int(s_max)})
        return [c for c in caps if c <= int(s_max)]
    capacities = []
    cap = MOONEP_MIN_DISPATCH_TOKENS
    while cap < s_max:
        capacities.append(cap)
        cap *= 2
    capacities.append(int(s_max))
    return capacities


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
            capacities = _moonep_pool_capacities(int(capacity))
            self._buffers[(E, K, H)] = {
                cap: Buffer(
                    int(cap), H, K, E, R,
                    token_padding=1, B=_moonep_slot_count(E, E // R), group=group,
                    explicitly_destroy=True,
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
                # F（每分区的 ffn 维）改由 _LayerWeightState 按权重布局自己推并做校验，
                # 布局知识只留一处：量化后 w13 是 [epn, H/2, 2F]、w2 是 [epn, F/2, H]
                # （process_weights_after_loading 的 npu_format_cast + transpose(1,2)），
                # 两边的 scale 又各自不同，散落推导迟早会错。
                states.append(
                    _LayerWeightState(layer, int(e["E"]), int(e["H"]), rank, dev))
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


_CAP_TRACE = {"capacity": None}


def _select_buffer_for_step(E: int, K: int, H: int, tokens: int):
    """Smallest pooled capacity that fits this step -> ``(buffer, capacity)``."""
    _MOONEP_RUNTIME._ensure_initialized()
    pool = _MOONEP_RUNTIME._buffers[(E, K, H)]
    s_max = max(pool)
    step_tokens = None if os.environ.get("MOONEP_POOL", "1") != "1" else _step_token_count(tokens)
    if step_tokens is None or step_tokens > s_max:
        capacity = s_max
    else:
        capacity = min(
            s_max,
            max(MOONEP_MIN_DISPATCH_TOKENS,
                1 << max(step_tokens - 1, 0).bit_length()),
        )
        while capacity not in pool:
            capacity = min(s_max, capacity * 2)
    if capacity < tokens:
        raise RuntimeError(
            f"MoonEP: step has {tokens} tokens but the selected capacity is "
            f"{capacity} (s_max={s_max})."
        )
    if os.environ.get("MOONEP_CAP_TRACE", "0") == "1" and _CAP_TRACE["capacity"] != capacity:
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
        pads = _pads_for(E, K, H)
        hidden_c = pads["hidden"][:capacity]
        route_c = pads["route"][:capacity]
        topk_c = pads["topk"][:capacity]
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
        if T < capacity:
            if os.environ.get("MOONEP_PAD_EXPERT0", "0") == "1":
                pad_expert = 0
            else:
                ep_group = get_ep_group()
                pad_expert = ep_group.rank_in_group * (E // ep_group.world_size)
        tail = pads["tail"]
        if (tail["T"] != T or tail["capacity"] != capacity
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
        # histc instead of bincount: fixed-size output (no dynamic sizing round
        # trip) -> ~40us vs ~100us of latency per layer. Counts here are
        # integers <= S*K <= 8192, exact in float32.
        tpe = torch.histc(
            topk_c.to(torch.float32), bins=E, min=0, max=E - 1).to(torch.int32)

        self._tstats("disp_pad", _t0, _tp0)
        self._tstats("disp_tpe", _tp0)
        self._tstats("disp_pre", _t0)
        _t1 = time.perf_counter()
        h_nvs, w_nvs, cu, plan = buffer.dispatch(
            hidden_c, route_c, topk_c, tpe,
            zero_copy=False, inter_rank_sync=True,
            reuse_outputs=os.environ.get("MOONEP_REUSE_OUT", "1") == "1",
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
        cu64 = cu.to(torch.int64)
        # counts[i] = cu[i] - cu[i-1] with an implicit cu[-1] = 0 (the leading
        # zero term is why this keeps the concat: slicing per rank would be off
        # by one for rank 0).
        counts = cu64 - torch.cat((cu64.new_zeros(1), cu64[:-1]))
        local_counts = counts[rank * epn:(rank + 1) * epn].contiguous()
        # Hand the load recorder the *sending* side of this dispatch: ``tpe`` counts
        # the (token, expert) pairs by global (logical) expert id, pads included,
        # exactly as the router chose them. The plan rebuilds its own histogram from
        # ``topk_c`` and never reads this one (see the ascend-moonep planning kernel),
        # so passing it on changes nothing about the dispatch, the MLP's group_list
        # or the load EPLB sees.
        #
        # Why the sending side and not the counts above: the pad block is
        # (capacity - T) * K pairs aimed at this rank's own pad expert, and the
        # receive side mixes them with real rows and does not report rows the
        # dispatch dropped (see the module docstring's capacity envelope). Subtracting
        # them from the histogram instead is exact, and the histogram is per *logical*
        # expert id, so summing it over the ranks offline recovers the run's real
        # routing and stays correct when an expert is replicated (MoonEP's prefetch
        # slots / EPLB redundancy).
        #
        # Called on every dispatch -- pad_pairs=0 clears the note -- so a layer
        # without padding cannot inherit the previous layer's note.
        _note_route_counts(
            tpe,
            pad_pairs=(capacity - T) * K if pad_expert is not None else 0,
            pad_index=pad_expert if pad_expert is not None else -1,
        )
        slot_counts = None
        if os.environ.get("MOONEP_SLOT_MODE", "0") == "1":
            # load-balance path: rows for prefetched (slot) experts stay local
            slot_counts = counts[E:].contiguous()
            local_counts = torch.cat((local_counts, slot_counts))
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
        if os.environ.get("MOONEP_NO_SYNC", "1") != "0":
            # Fast path: keep the row count on the device. The MLP takes the
            # full [NvS, H] shard plus the count-mode `group_list` and the
            # grouped matmul walks exactly sum(group_list) rows (rows past the
            # sum are not computed), so no `.item()` is needed -- the combine
            # masks the untouched tail on the device instead (see combine).
            #
            # Fresh view object, not `h_nvs` itself: with reuse_outputs (the
            # default) the library caches the dispatch output *tensor object*
            # and hands that same object back on every later dispatch
            # (contract: "valid until the next dispatch"). The MLP consumes it
            # via dispose_tensor() -> Tensor.set_, which would re-point the
            # cached object to an empty 1-D tensor, so every later dispatch
            # returns [0] and the W4A8MXFP grouped matmul dies with
            # "GroupedMatmul x dim num should be in the range [2, 6], but
            # actual is 1". The view shares the storage but is disposable
            # without touching the library's cached tensor.
            hidden_trim = h_nvs[:]
            w_trim = w_nvs.clone().to(torch.float32).contiguous()
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
        if os.environ.get("MOONEP_PACE_SYNC", "0") == "1":
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

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

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
                token_dispatch_output=disp,
                fused_experts_input=fused_experts_input,
                use_fusion_ops=self.use_fusion_ops,
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
