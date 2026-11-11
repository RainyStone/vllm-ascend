# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for adaptive capacity selection and collective-safe fallbacks."""
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_comm.py"


class AdaptivePoolTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(dp_metadata=object(), max_tokens_across_dp=1)
        self.dp_world_size = 8
        self.namespace = {
            "os": os,
            "get_forward_context": self.get_context,
            "get_dp_group": lambda: SimpleNamespace(world_size=self.dp_world_size),
            "get_ep_group": lambda: SimpleNamespace(world_size=8, rank_in_group=0),
            "MOONEP_MIN_DISPATCH_TOKENS": 128,
            "_ENV_CAP_TRACE": False, "_CAP_TRACE": {"capacity": None},
            "_GEO_CACHE": {},
        }
        names = {"_min_dispatch_tokens", "_moonep_pool_capacities", "_step_token_count",
                 "_geometry", "_select_buffer_for_step"}
        tree = ast.parse(SOURCE.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), self.namespace)
        with patch.dict(os.environ, {}, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.install_pool(caps)

    def get_context(self):
        if self.context is None:
            raise RuntimeError("No forward context")
        return self.context

    def install_pool(self, caps):
        self.pool = {cap: SimpleNamespace(_require_ctx=lambda: {"B": 32}) for cap in caps}
        self.namespace["_MOONEP_RUNTIME"] = SimpleNamespace(
            _ensure_initialized=Mock(), _buffers={(256, 6, 4096): self.pool})
        self.namespace["_GEO_CACHE"].clear()

    def select(self, tokens):
        buffer, capacity = self.namespace["_select_buffer_for_step"](256, 6, 4096, tokens)
        self.assertIs(buffer, self.pool[capacity])
        return capacity

    def test_old_pool_environment_cannot_disable_adaptive_selection(self):
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        self.context.max_tokens_across_dp = 33
        for value in ("0", "1"):
            with self.subTest(old_switch=value), patch.dict(os.environ, {"MOONEP_POOL": value}):
                self.assertEqual(self.select(8), 64)

    def test_default_minimum_stays_128(self):
        with patch.dict(os.environ, {}, clear=True):
            minimum = self.namespace["_min_dispatch_tokens"]()
        self.assertEqual(minimum, 128)
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = minimum
        self.assertEqual(self.select(1), 128)

    def test_explicit_experiment_minimum_16_still_applies(self):
        with patch.dict(os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": "16"}, clear=True):
            self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = self.namespace["_min_dispatch_tokens"]()
        self.assertEqual(self.select(1), 16)

    def test_legacy_minimum_alias_and_primary_precedence_are_preserved(self):
        with patch.dict(os.environ, {"MOONEP_MIN_CAP": "16"}, clear=True):
            self.assertEqual(self.namespace["_min_dispatch_tokens"](), 16)
            with patch.dict(os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": "32"}):
                self.assertEqual(self.namespace["_min_dispatch_tokens"](), 32)

    def test_invalid_minimum_is_still_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    self.namespace["_min_dispatch_tokens"]()

    def test_global_dp_maximum_gives_all_ranks_the_same_capacity(self):
        self.context.max_tokens_across_dp = 513
        for local_tokens in (0, 1, 128, 512, 513):
            with self.subTest(local_tokens=local_tokens):
                self.assertEqual(self.select(local_tokens), 768)

    def test_single_dp_can_select_without_dp_metadata(self):
        self.dp_world_size = 1
        self.context = SimpleNamespace(dp_metadata=None, max_tokens_across_dp=33)
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        self.assertEqual(self.select(33), 64)

    def test_multi_dp_missing_metadata_uses_full_capacity(self):
        self.context.dp_metadata = None
        self.assertEqual(self.select(1), 4096)

    def test_missing_forward_context_uses_full_capacity(self):
        self.context = None
        self.assertEqual(self.select(1), 4096)

    def test_missing_global_token_count_uses_full_capacity(self):
        self.context = SimpleNamespace(dp_metadata=object())
        self.assertEqual(self.select(1), 4096)

    def test_oversized_global_count_keeps_existing_full_capacity_fallback(self):
        self.context.max_tokens_across_dp = 8192
        self.assertEqual(self.select(1), 4096)

    def test_local_tokens_beyond_full_capacity_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "selected capacity"):
            self.select(4097)

    def test_custom_capacity_list_still_controls_selection(self):
        with patch.dict(os.environ, {"MOONEP_POOL_CAPS": "1024,256,256"}, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096)
        self.assertEqual(caps, [256, 1024, 4096])
        self.install_pool(caps)
        self.context.max_tokens_across_dp = 300
        self.assertEqual(self.select(64), 1024)

    def test_historical_4k_capacity_ladder_is_preserved(self):
        expected = [16, 32, 64, 128, 256, 512, *range(768, 4097, 256)]
        self.assertEqual(sorted(self.pool), expected)
        with patch.dict(os.environ, {"MOONEP_POOL_MAX_RUNGS": "32",
                                     "MOONEP_POOL_BUDGET_MB": "2048"}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](
                4096, cost_per_token=6 * 4096 * 2), expected)

    def test_rung_limit_and_budget_still_change_ladder_density(self):
        with patch.dict(os.environ, {"MOONEP_POOL_MAX_RUNGS": "8"}, clear=True):
            sparse = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertLessEqual(len(sparse), 8)
        self.assertEqual(sparse[-1], 4096)
        with patch.dict(os.environ, {"MOONEP_POOL_BUDGET_MB": "1"}, clear=True):
            low_budget = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertLess(len(low_budget), len(self.pool))
        self.assertEqual(low_budget[-1], 4096)


if __name__ == "__main__":
    unittest.main()
