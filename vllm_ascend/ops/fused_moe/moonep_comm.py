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
  * per-step token count must fit the static capacity S
    (``max_num_batched_tokens``), otherwise the step falls back to the
    SoC-default comm method.
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
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_moonep_tokens_capacity
from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod
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
            buf = Buffer(
                int(capacity), H, K, E, R,
                token_padding=1, B=E // R, group=group,
                explicitly_destroy=True,
                **({"use_udma": True} if use_udma else {}),
            )
            self._buffers[(E, K, H)] = buf
            # Static [S, *] pad workspace shared across layers of one shape.
            dev = f"npu:{torch.npu.current_device()}"
            self._pads[(E, K, H)] = {
                "hidden": torch.zeros(capacity, H, dtype=torch.bfloat16, device=dev),
                "route": torch.zeros(capacity, K, dtype=torch.float32, device=dev),
                "topk": torch.zeros(capacity, K, dtype=torch.int32, device=dev),
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
                states.append(
                    _LayerWeightState(layer, int(e["E"]), int(e["H"]), F, rank, dev))
            self._states = states
            self._state_by_w13 = {id(st._w13_src): st for st in states}

        if not ShmemRuntime.is_initialized():
            bootstrap = os.environ.get("MOONEP_SHMEM_BOOTSTRAP", "uid")
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


def _buffer_for(E: int, K: int, H: int):
    _MOONEP_RUNTIME._ensure_initialized()
    return _MOONEP_RUNTIME._buffers[(E, K, H)]


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

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
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

        capacity = get_moonep_tokens_capacity()
        if capacity is None or T > capacity:
            raise RuntimeError(
                f"MoonEP: step has {T} tokens > static capacity S={capacity}. "
                "Raise --max-num-batched-tokens or disable enable_moonep."
            )
        # Pad the step up to the static S. Pad rows route to expert 0 with
        # weight 0, so they contribute exactly zero to the final output.
        pads = _pads_for(E, K, H)
        hidden_p, route_p, topk_p = pads["hidden"], pads["route"], pads["topk"]
        hidden_p.zero_()
        route_p.zero_()
        topk_p.zero_()
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        hidden_p[:T].copy_(hidden_states)
        route_p[:T].copy_(topk_weights.to(torch.float32))
        topk_p[:T].copy_(topk_ids.to(torch.int32))
        # Pad rows route to this rank's *own* first expert (rank*epn): routing
        # every rank's pads to expert 0 piles S-T rows onto one rank, which the
        # MoonEP planning/combine kernels misplace for high ranks (R >= 3).
        # Pads are zero-hidden / zero-weight, so their expert is irrelevant to
        # the result. (MOONEP_PAD_EXPERT0=1 restores the expert-0 behavior.)
        if T < capacity and os.environ.get("MOONEP_PAD_EXPERT0", "0") != "1":
            epn_pad = self.num_experts // get_ep_group().world_size
            topk_p[T:].fill_(get_ep_group().rank_in_group * epn_pad)
        tpe = torch.bincount(topk_p.reshape(-1), minlength=E).to(torch.int32).contiguous()

        buffer = _buffer_for(E, K, H)
        h_nvs, w_nvs, cu, plan = buffer.dispatch(
            hidden_p, route_p, topk_p, tpe,
            zero_copy=False, inter_rank_sync=True,
        )

        # Per-local-expert row counts = cu diff over this rank's owned global
        # expert range [rank*epn, (rank+1)*epn). Shard rows are prefix-packed
        # in ascending owned-expert order, which is exactly the row/group order
        # the grouped-matmul MLP expects (same contract as the all-to-all
        # dispatcher). Slot groups (cu beyond E) are unused (owner semantics).
        ep_group = get_ep_group()
        epn = E // ep_group.world_size
        rank = ep_group.rank_in_group
        cu = cu.to(torch.int64)
        counts = cu - torch.cat((cu.new_zeros(1), cu[:-1]))
        local_counts = counts[rank * epn:(rank + 1) * epn].contiguous()
        slot_counts = None
        if os.environ.get("MOONEP_SLOT_MODE", "0") == "1":
            # load-balance path: rows for prefetched (slot) experts stay local
            slot_counts = counts[E:].contiguous()
            local_counts = torch.cat((local_counts, slot_counts))
        rows_total = int(local_counts.sum().item())
        if rows_total < 0 or rows_total > int(K) * capacity:
            raise RuntimeError(f"MoonEP: invalid dispatched row count {rows_total}.")
        hidden_trim = h_nvs[:rows_total]
        # Row-aligned per-row route weights (see module docstring / probe):
        # clone because the symmetric weights_buf is shared by all layers.
        if os.environ.get("MOONEP_PACE_SYNC", "0") == "1":
            torch.npu.synchronize()
        w_trim = w_nvs[:rows_total].clone().to(torch.float32).contiguous()

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
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in TokenDispatcherWithMoonEP."
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
        rows = hidden_states.to(torch.float32) if hidden_states.dtype != torch.float32 else hidden_states
        weighted = (rows * w.unsqueeze(-1)).to(torch.bfloat16)
        # The combine kernel expects the full [S*K, H] shard tensor: its prologue
        # copies the input into the symmetric chunk so remote ranks can RMA-GET
        # this rank's computed rows. Only the prefix (rows_total) is real; the
        # tail is never referenced by any plan, so pad with zeros.
        buffer = combine_metadata.buffer
        nvs = int(buffer._require_ctx()["NvS"])
        S = int(buffer._require_ctx()["S"])
        H = int(buffer._require_ctx()["H"])
        full = torch.zeros(nvs, H, dtype=torch.bfloat16, device=weighted.device)
        full[:weighted.shape[0]].copy_(weighted)
        out, _, _ = buffer.combine(
            plan=combine_metadata.plan,
            hidden_nvsh=full,
            route_weights_nvs=None,
            inter_rank_sync=True,
        )
        assert out.shape == (S, H), f"MoonEP combine returned {tuple(out.shape)}"
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
