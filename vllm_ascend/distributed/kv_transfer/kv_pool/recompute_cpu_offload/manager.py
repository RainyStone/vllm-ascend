# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for recompute CPU offloading."""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# `spec_manager_map` in vllm.v1.core.single_type_kv_cache_manager is a
# module-level dict keyed by the *exact* class object bound at import time.
# For MLA/DSA models vllm_ascend patches
# `vllm.v1.kv_cache_interface.MLAAttentionSpec` -> `AscendMLAAttentionSpec`,
# so the kv_cache_spec instances handed to get_kv_cache_coordinator() below are
# `AscendMLAAttentionSpec`. If `single_type_kv_cache_manager` was imported
# *before* that patch ran, the map only knows the original `MLAAttentionSpec`
# and get_manager_for_kv_cache_spec() raises
# `KeyError: AscendMLAAttentionSpec` while building our CPU coordinators.
#
# vllm_ascend/core/recompute_scheduler.py registers the subclass from
# RecomputeScheduler/AsyncRecomputeScheduler.__init__, but under
# CrossDPScheduler (DYCP) that path is never taken. Register eagerly at this
# module's import so the key is present no matter which outer scheduler drives
# the recompute CPU offload connector and regardless of import order. The
# `from ... import FullAttentionManager` line forces the target module to load
# if it isn't already, so this works whether or not it was pre-imported.
# Idempotent and additive; mirrors register_ascend_mla_spec_in_manager().
def _register_ascend_mla_spec_in_manager() -> None:
    import sys as _sys

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec

    _stm = _sys.modules.get("vllm.v1.core.single_type_kv_cache_manager")
    if _stm is not None and AscendMLAAttentionSpec not in _stm.spec_manager_map:
        _stm.spec_manager_map[AscendMLAAttentionSpec] = FullAttentionManager


_register_ascend_mla_spec_in_manager()


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class PreemptedRequestState:
    req_id: str
    cpu_block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    store_transfer_meta: TransferMeta
    store_event: int | None = None
    load_event: int | None = None
    load_transfer_meta: TransferMeta | None = None
    load_start_tokens: int = 0
    ready: bool = False
    finished: bool = False


class RecomputeCPUOffloadScheduler:
    """Preserve preempted requests' KV blocks in CPU memory.

    When offload prefix caching is enabled, full hashed blocks share CPU
    blocks. Otherwise every offloaded block is private to its request.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        enable_offload_prefix_caching: bool = True,
    ):
        assert kv_cache_config is not None
        self.vllm_config = vllm_config
        self.enable_offload_prefix_caching = enable_offload_prefix_caching
        self.cpu_kv_cache_config = self._derive_cpu_config(kv_cache_config, cpu_capacity_bytes)
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        self._group_is_sliding_window = self._get_group_is_sliding_window(kv_cache_config)
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None and vllm_config.kv_events_config.enable_kv_cache_events
        )

        logger.info(
            "RecomputeCPUOffloadScheduler: allocating %d CPU blocks (%.2f GB) for recompute offload, prefix caching=%s",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            self.enable_offload_prefix_caching,
        )

        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # [offload-adapt M1] Under DYCP (CrossDPScheduler) cp_world_size comes
        # from dp_per_domain, NOT dcp/pcp (which default to 1). The old
        # `assert dcp==1 and pcp==1` was a FALSE guard: it silently passed while
        # the single-pool model corrupted KV across the 16 CP ranks. Detect
        # cp_world_size from dp_per_domain and build per-rank pools when >1.
        self.cp_world_size = max(1, vllm_config.parallel_config.dp_per_domain or 1)
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if self.cp_world_size == 1:
            # CP==1 path (baseline / RecomputeScheduler): identical to before.
            self._cpu_coordinators: list[KVCacheCoordinator] = [
                get_kv_cache_coordinator(
                    kv_cache_config=self.cpu_kv_cache_config,
                    max_model_len=vllm_config.model_config.max_model_len,
                    use_eagle=False,
                    enable_caching=self.enable_offload_prefix_caching,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                    dcp_world_size=dcp_world_size,
                    pcp_world_size=pcp_world_size,
                    hash_block_size=vllm_config.cache_config.block_size,
                )
            ]
        else:
            # Per-rank CPU pools. cpu_capacity_bytes here is the FULL budget
            # (connector-capacity-owner.patch guarantees connector passes full
            # bytes under DYCP); partition equally across cp ranks.
            per_rank_bytes = max(1, cpu_capacity_bytes // self.cp_world_size)
            per_rank_config = self._derive_cpu_config(kv_cache_config, per_rank_bytes)
            self._cpu_coordinators = [
                get_kv_cache_coordinator(
                    kv_cache_config=per_rank_config,
                    max_model_len=vllm_config.model_config.max_model_len,
                    use_eagle=False,
                    enable_caching=self.enable_offload_prefix_caching,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                    dcp_world_size=dcp_world_size,
                    pcp_world_size=pcp_world_size,
                    hash_block_size=vllm_config.cache_config.block_size,
                )
                for _ in range(self.cp_world_size)
            ]
            logger.info(
                "[offload-adapt M1] DYCP per-rank CPU pools: cp_world_size=%d per_rank=%.2f GB",
                self.cp_world_size, (cpu_capacity_bytes // self.cp_world_size) / (1024**3),
            )
        # per-rank CPU block pools. cpu_block_pool (singular) kept as alias to
        # cpu_block_pools[0] for CP==1 backward compat (and any legacy caller).
        self.cpu_block_pools: list[BlockPool] = [c.block_pool for c in self._cpu_coordinators]
        self.cpu_block_pool: BlockPool = self.cpu_block_pools[0]
        # per-rank GPU block pools (M4 binds the list via bind_gpu_block_pools).
        self._gpu_block_pools: list[BlockPool] | None = None

        # Per-rank preempt states: dict[req_id][cp_rank] -> state.
        # CP==1: dict[req_id][0].
        self._preempted_req_states: dict[str, dict[int, PreemptedRequestState]] = {}
        # store/load event books are global-id-spaced (counter below) but each
        # event carries the cp_rank it belongs to; ack counts are rank-scoped.
        self._preempt_store_event_to_reqs: dict[int, list[str]] = {}
        self._preempt_store_event_to_blocks: dict[int, TransferMeta] = {}
        self._preempt_store_event_to_rank: dict[int, int] = {}
        self._preempt_load_event_to_reqs: dict[int, list[str]] = {}

        # Hash blocks created before build_connector_meta() are shared by all
        # requests preempted in the same scheduling step (per rank).
        self._pending_hash_blocks: dict[BlockHashWithGroupId, KVCacheBlock] = {}

        self._load_event_counter = 0
        self._store_event_counter = 0
        # [offload-adapt M2] Per-rank ack count. Default backend (world_size=1,
        # one worker per DP rank) => 1 ack per rank. external_launcher/TP>1 =>
        # tp_size. See M0 fact-sheet; expected is per-rank.
        self._expected_ack_per_rank = self._derive_expected_ack_per_rank(vllm_config)
        # (cp_rank, event_idx) -> count seen so far
        self._store_event_pending_counts: dict[tuple[int, int], int] = {}

    @staticmethod
    def _get_group_is_sliding_window(kv_cache_config: "KVCacheConfig") -> list[bool]:
        group_is_sliding_window: list[bool] = []
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                group_is_sliding_window.append(
                    any(isinstance(spec, SlidingWindowSpec) for spec in group.kv_cache_spec.kv_cache_specs.values())
                )
            else:
                group_is_sliding_window.append(isinstance(group.kv_cache_spec, SlidingWindowSpec))
        return group_is_sliding_window

    @staticmethod
    def _derive_cpu_config(gpu_config: "KVCacheConfig", cpu_capacity_bytes: int) -> "KVCacheConfig":
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert gpu_config.kv_cache_tensors
        gpu_kv_cache_tensors = []
        for t in gpu_config.kv_cache_tensors:
            if t.shared_by:
                gpu_kv_cache_tensors.append(t)
        gpu_total_bytes = sum(t.size for t in gpu_kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_kv_cache_tensors
        ]
        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    def _align_group_block_ids(
        self,
        group_idx: int,
        group_block_ids: list[int],
        logical_num_blocks: int,
    ) -> list[int]:
        if logical_num_blocks <= 0:
            return []
        aligned_group_block_ids = list(group_block_ids)
        if self._group_is_sliding_window[group_idx] and len(aligned_group_block_ids) < logical_num_blocks:
            aligned_group_block_ids = [0] * (
                logical_num_blocks - len(aligned_group_block_ids)
            ) + aligned_group_block_ids
        return aligned_group_block_ids[:logical_num_blocks]

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        # CP==1 bind (backward compat). Under DYCP must use bind_gpu_block_pools.
        self._gpu_block_pools = [gpu_block_pool] * self.cp_world_size

    def bind_gpu_block_pools(self, gpu_block_pools: list[BlockPool]) -> None:
        # [offload-adapt M4/M1] CrossDP binds the per-rank GPU pools list.
        if len(gpu_block_pools) < self.cp_world_size:
            raise RuntimeError(
                f"[offload-adapt] bind_gpu_block_pools got {len(gpu_block_pools)} pools "
                f"but cp_world_size={self.cp_world_size}"
            )
        self._gpu_block_pools = list(gpu_block_pools)

    @staticmethod
    def _derive_expected_ack_per_rank(vllm_config: VllmConfig) -> int:
        # [offload-adapt M2] Default backend => world_size=PP*TP*PCP, one worker
        # per DP rank => 1 ack per rank. external_launcher multiplies world_size
        # by data_parallel_size => per-rank ack = tp_size. Conservative default 1.
        ws = vllm_config.parallel_config.world_size
        tp = max(1, vllm_config.parallel_config.tensor_parallel_size or 1)
        if ws <= 1:
            return 1
        return tp

    def _resolve_cp_rank(self, cp_rank: int | None) -> int:
        return 0 if (cp_rank is None or self.cp_world_size == 1) else cp_rank

    def _get_state(self, req_id: str, cp_rank: int | None) -> PreemptedRequestState | None:
        ranks_map = self._preempted_req_states.get(req_id)
        if ranks_map is None:
            return None
        return ranks_map.get(self._resolve_cp_rank(cp_rank))

    def _ensure_rank_map(self, req_id: str) -> dict[int, PreemptedRequestState]:
        return self._preempted_req_states.setdefault(req_id, {})

    def has_preempted_request(self, req_id: str) -> bool:
        ranks_map = self._preempted_req_states.get(req_id)
        return bool(ranks_map)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        # [offload-adapt M1] Multi-rank: any ready per-rank state for this req
        # => restore. (For CP requests all per-rank states share num_computed;
        # restore hit_length is logical-token-level, H2D maps to per-rank shard.)
        ranks_map = self._preempted_req_states.get(request.request_id)
        if not ranks_map:
            return 0, False
        # pick a representative state (rank 0 preferred); all ranks capture the
        # same num_computed_tokens at preempt time.
        state = next(iter(ranks_map.values()))
        if not state.ready:
            return None, False

        restorable_tokens = min(state.num_computed_tokens, request.num_tokens)
        hit_length = max(0, restorable_tokens - num_computed_tokens)
        if hit_length <= 0:
            self._cleanup_preempt_cache_request(request.request_id)
            return 0, False

        for st in ranks_map.values():
            st.load_start_tokens = num_computed_tokens
        logger.debug(
            "[offload-adapt] offload cache hit req=%s ranks=%s load_start=%d load_tokens=%d stored=%d",
            request.request_id,
            list(ranks_map.keys()),
            num_computed_tokens,
            hit_length,
            state.num_computed_tokens,
        )
        return hit_length, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks | list[KVCacheBlocks]",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        # [offload-adapt M1/M7] CrossDP passes a rank-major list[KVCacheBlocks];
        # baseline passes a single KVCacheBlocks. Normalize and prepare H2D per
        # rank. The per-rank shard↔logical-token mapping itself is M7's
        # strided-mapping concern; here we hand each rank its block_ids so H2D
        # loads to the correct GPU pool.
        block_ids_by_group: tuple[list[int], ...] | list[tuple[list[int], ...]]
        if isinstance(blocks, list):
            block_ids_by_group = [b.get_block_ids() for b in blocks]
        else:
            block_ids_by_group = blocks.get_block_ids()
        prepared = self._prepare_preempt_load_after_alloc(
            request,
            block_ids_by_group,
            num_external_tokens,
        )
        if not prepared:
            raise RuntimeError(
                "Failed to prepare recompute H2D load after KV block "
                f"allocation: req_id={request.request_id}, "
                f"num_external_tokens={num_external_tokens}"
            )

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
        cp_rank: int | None = None,
    ) -> bool:
        rank = self._resolve_cp_rank(cp_rank)
        ranks_map = self._ensure_rank_map(request.request_id)
        if rank in ranks_map:
            # [offload-adapt M6] Re-preempt of an already-restored request: the
            # prior per-rank state has STALE block_ids/GPU blocks (re-allocated
            # on resume). Invalidate and re-capture instead of returning True.
            self._cleanup_one_rank(request.request_id, rank)
        return self._create_preempt_state(
            request.request_id,
            rank,
            block_ids,
            num_computed_tokens,
        )

    def discard_preempt_state(
        self,
        request: "Request",
        cp_rank: int | None = None,
    ) -> None:
        """[offload-adapt M5] Discard a per-rank CPU offload state for this
        request. Called by the scheduler all-or-nothing fail path when one rank
        of a CP request fails to offload, so the already-committed ranks CPU
        states do not leak (which would emit stale store specs for GPU blocks
        that _preempt_request is about to free/reallocate, and short-circuit a
        later re-preempt to a False-True). No-op if no state exists for the rank.
        """
        rank = self._resolve_cp_rank(cp_rank)
        ranks_map = self._preempted_req_states.get(request.request_id)
        if ranks_map is not None and rank in ranks_map:
            self._cleanup_one_rank(request.request_id, rank)
            logger.debug(
                "[offload-adapt M5] manager discarded state req=%s cp_rank=%s",
                request.request_id, rank)

    def _create_preempt_state(
        self,
        req_id: str,
        cp_rank: int,
        block_ids_by_group: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        gpu_pool = self._gpu_block_pools[cp_rank] if (self._gpu_block_pools is not None and cp_rank < len(self._gpu_block_pools)) else None
        cpu_pool = self.cpu_block_pools[cp_rank]
        if num_computed_tokens <= 0 or gpu_pool is None:
            return False

        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        group_gpu_blocks: list[list[KVCacheBlock | None]] = []
        group_gpu_hashes: list[list[BlockHashWithGroupId | None]] = []
        missing_hashes: set[BlockHashWithGroupId] = set()
        num_unhashed = 0

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            group_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            logical_num_blocks = cdiv(num_computed_tokens, group_block_size)
            aligned_group_gpu_ids = self._align_group_block_ids(g, group_gpu_ids, logical_num_blocks)
            eviction_group_gpu_ids = self._align_group_block_ids(
                g,
                group_gpu_ids,
                max(logical_num_blocks, len(group_gpu_ids)),
            )
            gpu_blocks: list[KVCacheBlock | None] = []
            effective_hashes: list[BlockHashWithGroupId | None] = []

            for block_idx, block_id in enumerate(eviction_group_gpu_ids):
                if block_id <= 0:
                    continue
                gpu_block = gpu_pool.blocks[block_id]
                block_is_computed = (block_idx + 1) * group_block_size <= num_computed_tokens
                if not block_is_computed and gpu_block.block_hash is not None:
                    # allocate_slots() may assign a hash using tokens planned
                    # for this scheduling step. If the request is then
                    # preempted before forward, that block does not contain the
                    # hashed KV and must not remain in the GPU prefix cache.
                    gpu_pool._maybe_evict_cached_block(gpu_block)

            for block_idx, block_id in enumerate(aligned_group_gpu_ids):
                if block_id <= 0:
                    gpu_blocks.append(None)
                    effective_hashes.append(None)
                    continue

                gpu_block = gpu_pool.blocks[block_id]
                block_is_computed = (block_idx + 1) * group_block_size <= num_computed_tokens
                block_hash = gpu_block.block_hash if block_is_computed and self.enable_offload_prefix_caching else None
                gpu_blocks.append(gpu_block)
                effective_hashes.append(block_hash)
                if block_hash is None:
                    num_unhashed += 1
                elif (
                    cpu_pool.cached_block_hash_to_block.get_one_block(block_hash) is None
                    and block_hash not in self._pending_hash_blocks
                ):
                    missing_hashes.add(block_hash)
            group_gpu_blocks.append(gpu_blocks)
            group_gpu_hashes.append(effective_hashes)

        num_needed = num_unhashed + len(missing_hashes)
        if not any(any(gpu_block is not None for gpu_block in group) for group in group_gpu_blocks):
            return False
        if num_needed > cpu_pool.get_num_free_blocks():
            logger.warning(
                "[offload-adapt] Skip recompute offload req=%s rank=%d: CPU free=%d need=%d",
                req_id, cp_rank, cpu_pool.get_num_free_blocks(), num_needed,
            )
            return False

        cpu_block_iter = iter(cpu_pool.get_new_blocks(num_needed))
        cpu_block_ids_by_group: list[list[int]] = []
        store_gpu_block_ids: list[int] = []
        store_cpu_block_ids: list[int] = []
        waiting_for_store = False

        for gpu_blocks, effective_hashes in zip(group_gpu_blocks, group_gpu_hashes):
            group_cpu_ids: list[int] = []
            for gpu_block, block_hash in zip(gpu_blocks, effective_hashes):
                if gpu_block is None:
                    group_cpu_ids.append(0)
                    continue

                cpu_block = None

                if block_hash is not None:
                    cpu_block = cpu_pool.cached_block_hash_to_block.get_one_block(block_hash)
                    if cpu_block is not None:
                        cpu_pool.touch([cpu_block])
                    else:
                        cpu_block = self._pending_hash_blocks.get(block_hash)
                        if cpu_block is not None:
                            cpu_pool.touch([cpu_block])
                            waiting_for_store = True
                        else:
                            cpu_block = next(cpu_block_iter)
                            cpu_block._block_hash = block_hash
                            self._pending_hash_blocks[block_hash] = cpu_block
                            store_gpu_block_ids.append(gpu_block.block_id)
                            store_cpu_block_ids.append(cpu_block.block_id)
                            waiting_for_store = True
                else:
                    cpu_block = next(cpu_block_iter)
                    store_gpu_block_ids.append(gpu_block.block_id)
                    store_cpu_block_ids.append(cpu_block.block_id)
                    waiting_for_store = True

                group_cpu_ids.append(cpu_block.block_id)
            cpu_block_ids_by_group.append(group_cpu_ids)

        store_transfer = TransferMeta(store_gpu_block_ids, store_cpu_block_ids)
        ranks_map = self._ensure_rank_map(req_id)
        ranks_map[cp_rank] = PreemptedRequestState(
            req_id=req_id,
            cpu_block_ids=tuple(cpu_block_ids_by_group),
            num_computed_tokens=num_computed_tokens,
            store_transfer_meta=store_transfer,
            ready=not waiting_for_store,
        )
        logger.info(
            "[offload-adapt M1] created state req=%s rank=%d computed=%d cpu_blocks=%d store_blocks=%d ready=%s",
            req_id, cp_rank, num_computed_tokens,
            sum(len(ids) for ids in cpu_block_ids_by_group),
            len(store_cpu_block_ids), not waiting_for_store,
        )

        return True

    def _prepare_preempt_store_specs(
        self,
        cp_rank: int,
    ) -> tuple[list[int], list[int], list[str]]:
        # [offload-adapt M1/M3] Filter to only this rank's states.
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        for req_id, ranks_map in self._preempted_req_states.items():
            state = ranks_map.get(cp_rank)
            if state is None or state.store_event is not None or state.ready:
                continue
            gpu_block_ids.extend(state.store_transfer_meta.gpu_block_ids)
            cpu_block_ids.extend(state.store_transfer_meta.cpu_block_ids)
            req_ids.append(req_id)
        return gpu_block_ids, cpu_block_ids, req_ids

    def _prepare_preempt_load_after_alloc(
        self,
        request: "Request",
        block_ids_by_group: tuple[list[int], ...] | list[tuple[list[int], ...]],
        num_external_tokens: int,
    ) -> bool:
        ranks_map = self._preempted_req_states.get(request.request_id)
        if not ranks_map:
            return False
        # [offload-adapt M7] block_ids_by_group may be:
        #   - single-rank form: tuple[list,...] (group-major, one rank) -> CP==1
        #   - rank-major form: list[tuple[list,...]] (one tuple per cp rank)
        rank_major = isinstance(block_ids_by_group, list) and bool(self.cp_world_size > 1)
        any_prepared = False
        cur_cp_ranks = list(getattr(request, "cp_ranks", []) or [])
        for cp_rank, state in ranks_map.items():
            if not state.ready:
                continue
            if rank_major:
                # [offload-adapt M7] cp_rank is the GLOBAL rank id stored at
                # preempt time (it also indexes _gpu_block_pools / cpu_block_pools
                # inside _prepare_load_one_rank). block_ids_by_group is POSITIONAL
                # over the *current* request.cp_ranks: block_ids_by_group[i] is the
                # blocks of rank cp_ranks[i]. For long reqs cp_ranks=[0..N-1] so
                # positional==global (no-op); for short reqs cp_ranks=[R] the global
                # id R must map to positional 0, else block_ids_by_group[R] overflows
                # (the IndexError). M6 routes offload-hit restores back to
                # prev_cp_ranks so cp_rank is normally present; if DyCP re-routed
                # it off the preempt ranks, M7 cross-rank reshard is not done ->
                # skip H2D for this rank (recompute fallback) instead of crashing.
                if cp_rank in cur_cp_ranks:
                    per_rank_groups = block_ids_by_group[cur_cp_ranks.index(cp_rank)]
                else:
                    logger.warning(
                        "[offload-adapt M7] restore rank mismatch req=%s "
                        "preempt_rank=%s cur_cp_ranks=%s -> skip H2D, recompute",
                        request.request_id, cp_rank, cur_cp_ranks)
                    continue
            else:
                per_rank_groups = block_ids_by_group
            prepared = self._prepare_load_one_rank(
                request, cp_rank, state, per_rank_groups, num_external_tokens)
            any_prepared = any_prepared or prepared
        return any_prepared

    def _prepare_load_one_rank(
        self,
        request: "Request",
        cp_rank: int,
        state: PreemptedRequestState,
        block_ids_by_group: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> bool:
        load_start_tokens = state.load_start_tokens
        load_end_tokens = min(
            load_start_tokens + num_external_tokens,
            state.num_computed_tokens,
        )
        if load_end_tokens <= load_start_tokens:
            return False

        if len(block_ids_by_group) != len(state.cpu_block_ids):
            raise RuntimeError(
                "Recompute H2D KV group count mismatch: "
                f"req_id={request.request_id}, rank={cp_rank}, "
                f"gpu_groups={len(block_ids_by_group)}, "
                f"cpu_groups={len(state.cpu_block_ids)}"
            )

        gpu_pool = self._gpu_block_pools[cp_rank] if (self._gpu_block_pools is not None and cp_rank < len(self._gpu_block_pools)) else None
        assert gpu_pool is not None, "GPU block pools not bound (M4)"
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        for g, group_cpu_ids in enumerate(state.cpu_block_ids):
            group_block_size = self.cpu_kv_cache_config.kv_cache_groups[g].kv_cache_spec.block_size
            start_block = load_start_tokens // group_block_size
            end_block = min(
                len(group_cpu_ids),
                len(
                    self._align_group_block_ids(
                        g,
                        block_ids_by_group[g],
                        max(
                            cdiv(load_end_tokens, group_block_size),
                            len(block_ids_by_group[g]),
                        ),
                    )
                ),
                cdiv(load_end_tokens, group_block_size),
            )
            if end_block == start_block:
                continue
            if end_block < start_block:
                raise RuntimeError(
                    "Recompute H2D produced an empty block range: "
                    f"req_id={request.request_id}, rank={cp_rank}, group={g}, "
                    f"start_block={start_block}, end_block={end_block}, "
                    f"gpu_blocks={len(block_ids_by_group[g])}, "
                    f"cpu_blocks={len(group_cpu_ids)}"
                )

            aligned_group_gpu_ids = self._align_group_block_ids(
                g,
                block_ids_by_group[g],
                end_block,
            )
            for block_idx in range(start_block, end_block):
                cpu_block_id = group_cpu_ids[block_idx]
                gpu_block_id = aligned_group_gpu_ids[block_idx]
                if cpu_block_id <= 0 or gpu_block_id <= 0:
                    continue
                cpu_block_ids.append(cpu_block_id)
                gpu_block_ids.append(gpu_block_id)

        if not gpu_block_ids:
            return False
        if len(cpu_block_ids) != len(gpu_block_ids):
            raise RuntimeError(
                "Recompute H2D block mapping is incomplete: "
                f"req_id={request.request_id}, rank={cp_rank}, "
                f"gpu_blocks={len(gpu_block_ids)}, cpu_blocks={len(cpu_block_ids)}"
            )

        gpu_pool.touch([gpu_pool.blocks[block_id] for block_id in gpu_block_ids])
        state.load_transfer_meta = TransferMeta(gpu_block_ids, cpu_block_ids)
        logger.info(
            "[offload-adapt M7] prepared H2D load req=%s rank=%d tokens=[%d,%d) blocks=%d",
            request.request_id, cp_rank, load_start_tokens, load_end_tokens, len(gpu_block_ids),
        )
        return True

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        cp_rank: int | None = None,
    ) -> RecomputeCPUOffloadMetadata:
        rank = self._resolve_cp_rank(cp_rank)
        # [offload-adapt M3] Each per-rank SchedulerOutput -> build only THIS
        # rank's store/load. CrossDP calls this 16x/step; without rank filtering
        # the first call would grab all 16 ranks' blocks and the rest be empty
        # -> workers copy the wrong physical KV.
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self._prepare_preempt_store_specs(rank)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._preempt_store_event_to_blocks[store_event] = TransferMeta(store_gpu, store_cpu)
            self._preempt_store_event_to_reqs[store_event] = store_req_ids
            self._preempt_store_event_to_rank[store_event] = rank
            for req_id in store_req_ids:
                self._preempted_req_states[req_id][rank].store_event = store_event
            if self.cp_world_size == 1:
                self._pending_hash_blocks.clear()

        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        _dbg_states = []
        for req_id, ranks_map in self._preempted_req_states.items():
            state = ranks_map.get(rank)
            if state is None:
                continue
            _dbg_states.append((req_id, state.load_transfer_meta is not None, state.load_event, state.ready))
            if state.load_transfer_meta is None or state.load_event is not None:
                continue
            load_gpu.extend(state.load_transfer_meta.gpu_block_ids)
            load_cpu.extend(state.load_transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if _dbg_states:
            logger.info(
                "[diag-load-build] cp_rank=%d states=%s load_req_ids=%s load_event_will=%d",
                rank, _dbg_states[:6], load_req_ids, self._load_event_counter if load_req_ids else -1)

        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._preempted_req_states[req_id][rank].load_event = load_event
            self._preempt_load_event_to_reqs[load_event] = load_req_ids

        return RecomputeCPUOffloadMetadata(
            cp_rank=rank,
            need_flush=bool(scheduler_output.preempted_req_ids),
            preempt_store_event=store_event,
            preempt_store_gpu_blocks=store_gpu,
            preempt_store_cpu_blocks=store_cpu,
            preempt_load_event=load_event,
            preempt_load_gpu_blocks=load_gpu,
            preempt_load_cpu_blocks=load_cpu,
            preempt_load_event_to_reqs=self._preempt_load_event_to_reqs,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        for req_id in list(connector_output.finished_recving or []):
            if req_id in self._preempted_req_states:
                self._cleanup_preempt_load_request(req_id)

        meta = connector_output.kv_connector_worker_meta
        _cse = getattr(meta, 'completed_store_events', None) if meta is not None else None
        if _cse:
            logger.info(
                "[diag-storeack] meta_type=%s completed_store_events=%s expected_ack=%s pending_counts=%s",
                type(meta).__name__, dict(_cse), self._expected_ack_per_rank,
                dict(self._store_event_pending_counts))
        if not isinstance(meta, RecomputeCPUOffloadWorkerMetadata):
            if _cse:
                logger.info(
                    "[diag-storeack] meta NOT offload type -> completed_store_events DROPPED (ready never set)")
            return
        for key, count in meta.completed_store_events.items():
            # [offload-adapt M2/M3] key may be int (CP==1, legacy) or
            # (cp_rank, event_idx) tuple (DYCP).
            if isinstance(key, tuple):
                cp_rank, event_idx = key
            else:
                cp_rank, event_idx = 0, key
            rk = (cp_rank, event_idx)
            total = self._store_event_pending_counts.get(rk, 0) + count
            if total >= self._expected_ack_per_rank:
                self._store_event_pending_counts.pop(rk, None)
                self._process_preempt_store_event(event_idx, cp_rank)
            else:
                self._store_event_pending_counts[rk] = total

    def _process_preempt_store_event(self, event_idx: int, cp_rank: int) -> None:
        transfer = self._preempt_store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            logger.info(
                "[diag-ready] event_idx=%d cp_rank=%d pop=None blocks_map_keys=%s to_reqs_keys=%s -> ready NOT set",
                event_idx, cp_rank, list(self._preempt_store_event_to_blocks.keys())[:8],
                list(self._preempt_store_event_to_reqs.keys())[:8])
            return
        req_ids = self._preempt_store_event_to_reqs.pop(event_idx, [])
        self._preempt_store_event_to_rank.pop(event_idx, None)
        cpu_pool = self.cpu_block_pools[cp_rank]

        for cpu_block_id in transfer.cpu_block_ids:
            cpu_block = cpu_pool.blocks[cpu_block_id]
            block_hash = cpu_block.block_hash
            if block_hash is None:
                continue
            cached_block = cpu_pool.cached_block_hash_to_block.get_one_block(block_hash)
            if cached_block is None:
                cpu_pool.cached_block_hash_to_block.insert(block_hash, cpu_block)
            elif cached_block.block_id != cpu_block.block_id:
                cpu_block.reset_hash()

        for req_id in req_ids:
            ranks_map = self._preempted_req_states.get(req_id)
            if ranks_map is None:
                logger.info(
                    "[diag-ready] event_idx=%d cp_rank=%d req=%s ranks_map=None -> ready NOT set",
                    event_idx, cp_rank, req_id)
                continue
            state = ranks_map.get(cp_rank)
            if state is not None:
                state.ready = True
                logger.info(
                    "[diag-ready] event_idx=%d cp_rank=%d req=%s -> ready=True",
                    event_idx, cp_rank, req_id)
                if state.finished:
                    self._cleanup_one_rank(req_id, cp_rank)
            else:
                logger.info(
                    "[diag-ready] event_idx=%d cp_rank=%d req=%s state=None (ranks_map_keys=%s) -> ready NOT set",
                    event_idx, cp_rank, req_id, list(ranks_map.keys()))

    def has_pending_transfers(self) -> bool:
        return bool(
            self._store_event_pending_counts
            or self._preempt_store_event_to_blocks
            or any(
                not state.ready or state.load_transfer_meta is not None
                for ranks_map in self._preempted_req_states.values()
                for state in ranks_map.values()
            )
        )

    def reset_cache(self) -> bool:
        if self.has_pending_transfers():
            logger.warning(
                "Failed to reset recompute offload cache because transfers or request states are still pending."
            )
            return False
        for req_id in list(self._preempted_req_states):
            self._cleanup_preempt_cache_request(req_id)
        self._preempt_store_event_to_reqs.clear()
        self._preempt_store_event_to_blocks.clear()
        self._preempt_store_event_to_rank.clear()
        self._preempt_load_event_to_reqs.clear()
        self._pending_hash_blocks.clear()
        ok = True
        for pool in self.cpu_block_pools:
            ok = self._safe_reset_pool(pool) and ok
        return ok

    def _safe_reset_pool(self, pool: BlockPool) -> bool:
        try:
            return pool.reset_prefix_cache()
        except NotImplementedError:
            # CrossDP pools may have prefix caching disabled (no-op).
            return True

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        ranks_map = self._preempted_req_states.get(request.request_id)
        if not ranks_map:
            return False, None
        for cp_rank in list(ranks_map.keys()):
            state = ranks_map[cp_rank]
            if state.load_event is None:
                if state.ready:
                    self._cleanup_one_rank(request.request_id, cp_rank)
                else:
                    state.finished = True
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def _cleanup_preempt_load_request(self, req_id: str) -> None:
        ranks_map = self._preempted_req_states.get(req_id)
        if not ranks_map:
            return
        for cp_rank in list(ranks_map.keys()):
            state = ranks_map[cp_rank]
            if state.load_event is not None:
                reqs = self._preempt_load_event_to_reqs.get(state.load_event)
                if reqs is not None:
                    with contextlib.suppress(ValueError):
                        reqs.remove(req_id)
                    if not reqs:
                        self._preempt_load_event_to_reqs.pop(state.load_event, None)
            if state.load_transfer_meta is not None and self._gpu_block_pools is not None and cp_rank < len(self._gpu_block_pools):
                gpu_pool = self._gpu_block_pools[cp_rank]
                gpu_pool.free_blocks(
                    gpu_pool.blocks[block_id] for block_id in state.load_transfer_meta.gpu_block_ids
                )
        self._cleanup_preempt_cache_request(req_id)

    def _cleanup_one_rank(self, req_id: str, cp_rank: int) -> None:
        ranks_map = self._preempted_req_states.get(req_id)
        if ranks_map is None:
            return
        state = ranks_map.pop(cp_rank, None)
        if state is None:
            return
        cpu_pool = self.cpu_block_pools[cp_rank] if cp_rank < len(self.cpu_block_pools) else self.cpu_block_pool
        cpu_pool.free_blocks(
            cpu_pool.blocks[block_id]
            for group_cpu_ids in state.cpu_block_ids
            for block_id in group_cpu_ids
            if block_id > 0
        )
        if not ranks_map:
            self._preempted_req_states.pop(req_id, None)

    def _cleanup_preempt_cache_request(self, req_id: str) -> None:
        ranks_map = self._preempted_req_states.pop(req_id, None)
        if not ranks_map:
            return
        for cp_rank, state in list(ranks_map.items()):
            cpu_pool = self.cpu_block_pools[cp_rank] if cp_rank < len(self.cpu_block_pools) else self.cpu_block_pool
            cpu_pool.free_blocks(
                cpu_pool.blocks[block_id]
                for group_cpu_ids in state.cpu_block_ids
                for block_id in group_cpu_ids
                if block_id > 0
            )

    def take_events(self) -> Iterable[KVCacheEvent]:
        # [offload-adapt M1] aggregate events across all per-rank CPU pools.
        for pool in self.cpu_block_pools:
            yield from pool.take_events()
