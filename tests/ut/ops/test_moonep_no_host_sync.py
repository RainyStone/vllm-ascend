# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for device-resident dispatch counts and reusable outputs."""
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_moonep_real_tokens import Metadata, load_method


class NoHostCountSyncTests(unittest.TestCase):
    def make_dispatch(self, transport, slot):
        tokens, capacity, hidden, experts, topk, rows = 2, 4, 16, 4, 2, 8
        hidden_dtype = torch.float8_e4m3fn if transport == "mxfp8" else torch.bfloat16
        shard = torch.zeros(rows, hidden, dtype=hidden_dtype)
        weights = torch.arange(rows, dtype=torch.float32)
        scales = torch.zeros(rows, 1, dtype=torch.uint8)
        cu = torch.tensor([2, 3, 3, 3, 4, 4] if slot else [2, 3, 3, 3, 3, 3],
                          dtype=torch.int32)
        calls = []

        class Buffer:
            def _require_ctx(self):
                return {"B": 2}

            def dispatch(self, *args, **kwargs):
                calls.append(kwargs)
                return shard, weights, cu, SimpleNamespace(experts_to_copy=torch.tensor([[2, -1]]))

            def dispatch_mxfp8(self, *args, **kwargs):
                calls.append(kwargs)
                return shard, scales, weights, cu, SimpleNamespace(experts_to_copy=torch.tensor([[2, -1]]))

        buffer = Buffer()
        scratch = {
            "tpe_compat": torch.zeros(experts, dtype=torch.int32),
            "cu_pad": torch.zeros(7, dtype=torch.int64),
            "counts": torch.zeros(6, dtype=torch.int64),
            "own": torch.empty(2, dtype=torch.int64),
            "w": torch.empty(rows, dtype=torch.float32),
        }
        pads = {
            "hidden": torch.empty(capacity, hidden, dtype=torch.bfloat16),
            "route": torch.empty(capacity, topk),
            "topk": torch.empty(capacity, topk, dtype=torch.int32),
            "combine_route": torch.empty(capacity, 8),
        }
        namespace = {
            "torch": torch, "time": time, "os": os,
            "_EXTRA_CTX": SimpleNamespace(capturing=False),
            "_ENV_SLOT_MODE": slot, "_ENV_REUSE_OUT": True,
            "_ENV_DISPATCH_IN_SHARD": True, "_ENV_PACE_SYNC": False,
            "_DUMP_PLAN_DIR": None, "_PLAN_STATS_ON": False,
            "_EARLY_PLAN_ON": False, "_BALANCE_STATS_ON": False,
            "_SLOT_TLS": SimpleNamespace(),
            "get_ep_group": lambda: SimpleNamespace(world_size=2, rank_in_group=0),
            "get_moonep_tokens_capacity": lambda: capacity,
            "_select_buffer_for_step": lambda *args: (buffer, capacity),
            "_pads_for": lambda *args: pads,
            "_scratch_for": lambda *args: scratch,
            "_route_stats_enabled": lambda: False,
            "_route_histogram": lambda routes, e, t, compat, collect: compat,
            "_MOONEP_RUNTIME": SimpleNamespace(_buffers={(experts, topk, hidden): {capacity: buffer}}),
            "_note_route_counts": lambda *args, **kwargs: None,
            "_note_compute_load": lambda *args, **kwargs: None,
            "MoETokenDispatchOutput": SimpleNamespace,
            "MoonEPCombineMetadata": Metadata,
        }
        dispatch = load_method("token_dispatch", namespace)
        dispatcher = SimpleNamespace(num_experts=experts, top_k=topk,
                                     activation_transport=transport, _tstats=lambda *args: None)
        quant_type = SimpleNamespace(W4A8MXFP=object())
        inputs = SimpleNamespace(
            hidden_states=torch.ones(tokens, hidden, dtype=torch.bfloat16),
            topk_weights=torch.full((tokens, topk), 0.5),
            topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            quant=SimpleNamespace(quant_type=quant_type.W4A8MXFP),
        )
        modules = {"vllm_ascend.quantization.quant_type": SimpleNamespace(QuantType=quant_type)}
        return dispatch, dispatcher, inputs, shard, weights, scales, calls, modules

    def invoke(self, case, old_switch):
        dispatch, dispatcher, inputs, _, _, _, _, modules = case
        environment = {"MOONEP_NO_SYNC": old_switch, "MOONEP_EARLY_PREFETCH": "0",
                       "MOONEP_SLOT_STATS": "0"}
        with patch.dict(os.environ, environment), patch.dict(sys.modules, modules), patch.object(
            torch.Tensor, "item", side_effect=AssertionError("Dispatch must not read device counts to host")
        ):
            return dispatch(dispatcher, inputs)

    def test_all_dispatch_paths_keep_full_shard_and_device_counts(self):
        for transport, slot in (("bf16", False), ("bf16", True), ("mxfp8", True)):
            for old_switch in ("0", "1"):
                with self.subTest(transport=transport, slot=slot, old_switch=old_switch):
                    case = self.make_dispatch(transport, slot)
                    result = self.invoke(case, old_switch)
                    _, _, inputs, shard, weights, scales, calls, _ = case
                    self.assertEqual(tuple(result.hidden_states.shape), tuple(shard.shape))
                    self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
                    self.assertIsNot(result.hidden_states, shard)
                    self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0] if slot else [2, 1])
                    self.assertEqual(result.group_list_type, 1)
                    metadata = result.combine_metadata
                    self.assertTrue(metadata.tail_masked)
                    self.assertIs(metadata.group_list, result.group_list)
                    self.assertEqual(metadata.num_tokens, inputs.hidden_states.shape[0])
                    self.assertEqual(calls[0]["num_tokens"], metadata.num_tokens)
                    if transport == "bf16":
                        self.assertTrue(calls[0]["inter_rank_sync"])
                        self.assertTrue(calls[0]["zero_copy"])
                        self.assertTrue(calls[0]["reuse_outputs"])
                        self.assertIsNone(result.dynamic_scale)
                    else:
                        self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())
                    self.assertTrue(torch.equal(metadata.w_trim, weights))
                    expected_weights = weights.clone()
                    weights.fill_(99)
                    self.assertTrue(torch.equal(metadata.w_trim, expected_weights))

    def test_disposing_view_does_not_damage_reused_sdk_output(self):
        case = self.make_dispatch("bf16", True)
        first = self.invoke(case, "0")
        shard = case[3]
        first.hidden_states.set_(torch.empty(0, dtype=shard.dtype))
        self.assertEqual(tuple(shard.shape), (8, 16))
        second = self.invoke(case, "0")
        self.assertEqual(tuple(second.hidden_states.shape), (8, 16))
        self.assertEqual(second.hidden_states.data_ptr(), shard.data_ptr())


if __name__ == "__main__":
    unittest.main()
