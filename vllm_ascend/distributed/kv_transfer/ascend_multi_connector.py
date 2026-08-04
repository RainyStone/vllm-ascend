import logging
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


logger = logging.getLogger(__name__)


class AscendMultiConnector(MultiConnector):
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks | list[KVCacheBlocks]", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        # [offload-adapt] CrossDP passes a per-cp_rank list[KVCacheBlocks];
        # baseline passes a single KVCacheBlocks. Build empty blocks matching
        # the shape so non-chosen connectors no-op correctly in both cases.
        if isinstance(blocks, list):
            empty_blocks = [b.new_empty() for b in blocks]
        else:
            empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
        cp_rank: int | None = None,
    ) -> bool:
        # [offload-adapt M5] Forward cp_rank so per-rank segregation survives the
        # multi-connector fan-out; only the offload sub-connector has the hook
        # (Mooncake has none, so OR-reduces to the offload result). The
        # all-or-nothing loop over the victim cp_ranks lives in CrossDP, so the
        # OR across sub-connectors here cannot break that invariant.
        offloaded = False
        for connector in self._connectors:
            hook = getattr(connector, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens, cp_rank=cp_rank)) or offloaded
        return offloaded

    def discard_preempt_state(
        self,
        request: "Request",
        cp_rank: int | None = None,
    ) -> None:
        # [offload-adapt M5] Fan out the all-or-nothing fail-path discard to the
        # offload sub-connector (others have no such hook -> no-op).
        for connector in self._connectors:
            hook = getattr(connector, "discard_preempt_state", None)
            if hook is not None:
                hook(request, cp_rank=cp_rank)
                logger.debug(
                    "[offload-adapt M5] multi discard req=%s cp_rank=%s via %s",
                    getattr(request, "request_id", "?"), cp_rank, type(connector).__name__)

    def clear_reqs_need_recv(self) -> None:
        # [offload-adapt] CrossDPScheduler calls this after build_connector_meta
        # (per cp_rank) to flush mooncake's _reqs_need_recv/_reqs_need_send/
        # _reqs_need_send_cp_ranks/_reqs_in_batch, which mooncake only clears
        # itself on the dp_per_domain==1 branch. Under DyCP (dp_per_domain>1)
        # those states survive across cp_rank meta builds, so the scheduler
        # forces the clear; forward it to the mooncake sub-connector that
        # defines it (offload has none -> no-op).
        for connector in self._connectors:
            hook = getattr(connector, "clear_reqs_need_recv", None)
            if hook is not None:
                hook()

    def bind_gpu_block_pools(self, gpu_block_pools: list) -> None:
        # [offload-adapt M4] CrossDPScheduler binds the per-rank GPU pool list
        # (kv_cache_manager.block_pools). Only the offload sub-connector
        # defines the plural bind (forwarded to the per-rank manager);
        # mooncake has none -> no-op. Without this passthrough the scheduler
        # fell back to single-pool bind and offload covered at most 1 rank.
        for connector in self._connectors:
            hook = getattr(connector, "bind_gpu_block_pools", None)
            if hook is not None:
                hook(gpu_block_pools)

    def has_preempted_request(self, req_id: str) -> bool:
        # [offload-adapt M6] Top-level passthrough so the scheduler can detect
        # an offload-captured preempted request (defined on the offload
        # sub-connector; mooncake has none -> no-op).
        for connector in self._connectors:
            hook = getattr(connector, "has_preempted_request", None)
            if callable(hook) and hook(req_id):
                return True
        return False
