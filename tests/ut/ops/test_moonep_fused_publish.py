# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for publication selection and combine ordering."""
import ast
import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(path, class_name, method_name, namespace):
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    tree = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[method_name]


class Metadata(SimpleNamespace):
    pass


class DispatcherPublishTest(unittest.TestCase):
    def invoke(self, enabled, masked=True, counts=None, in_shard=False):
        source = torch.randn(6 if masked else 3, 4, dtype=torch.bfloat16)
        if masked:
            source[3:].fill_(float("nan"))
        calls = []
        shard = torch.full((6, 4), float("nan"), dtype=torch.bfloat16)
        class FakeBuffer:
            def _require_ctx(self):
                return {"NvS": 6, "hidden_buf_local": shard}
            def combine_weighted(self, **kwargs):
                calls.append(kwargs)
                return torch.ones(2, 4, dtype=torch.bfloat16), None, None
        namespace = {
            "torch": torch, "time": time, "MoonEPCombineMetadata": Metadata,
            "_combine_row_ids": lambda n, device: torch.arange(n, device=device),
            "_zero_bf16_scalar": lambda device: torch.zeros((), dtype=torch.bfloat16, device=device),
            "_ENV_COMBINE_IN_SHARD": in_shard,
        }
        fn = load_method(ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py",
                         "TokenDispatcherWithMoonEP", "token_combine", namespace)
        metadata = Metadata(num_tokens=2, buffer=FakeBuffer(), tail_masked=masked,
                            group_list=counts, source_topk_weights_sk=torch.ones(2, 8),
                            plan=object(), hidden_shape=(2, 4))
        obj = SimpleNamespace(fused_publish=enabled, _tstats=lambda *args: None)
        result = fn(obj, source, metadata)
        self.assertEqual(tuple(result.shape), (2, 4))
        return source, calls[0], shard

    def test_fused_passes_raw_tensor_and_device_counts_for_each_backend(self):
        class Counts:
            def sum(self):
                raise AssertionError("Fused path must not launch a standalone sum")
        for backend in ("mte", "udma"):
            with self.subTest(backend=backend):
                counts = Counts()
                with patch.object(torch, "where", side_effect=AssertionError("No temporary clean matrix")):
                    source, call, _ = self.invoke(True, counts=counts)
                self.assertIs(call["hidden_nvsh"], source)
                self.assertIs(call["row_counts"], counts)
                self.assertTrue(call["fused_publish"])
                self.assertEqual(source.dtype, torch.bfloat16)

    def test_default_path_still_cleans_nan_tail(self):
        source, call, _ = self.invoke(False, counts=torch.tensor([1, 2], dtype=torch.int64),
                                     in_shard=False)
        self.assertFalse(call["fused_publish"])
        self.assertIsNone(call["row_counts"])
        self.assertFalse(call["zero_copy"])
        self.assertTrue(torch.equal(call["hidden_nvsh"][:3], source[:3]))
        self.assertTrue(torch.equal(call["hidden_nvsh"][3:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_combine_stages_into_the_symmetric_shard(self):
        source, call, shard = self.invoke(False, counts=torch.tensor([1, 2], dtype=torch.int64),
                                          in_shard=True)
        self.assertIs(call["hidden_nvsh"], shard)
        self.assertTrue(call["zero_copy"])
        self.assertFalse(call["fused_publish"])
        self.assertTrue(torch.equal(shard[:3], source[:3]))
        self.assertTrue(torch.equal(shard[3:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_trimmed_in_shard_input_keeps_stale_rows_zero(self):
        with patch.object(torch, "where", side_effect=AssertionError("No temporary clean matrix")):
            source, call, shard = self.invoke(False, masked=False, in_shard=True)
        self.assertIs(call["hidden_nvsh"], shard)
        self.assertTrue(call["zero_copy"])
        self.assertEqual(source.shape[0], 3)
        self.assertTrue(torch.equal(shard[:3], source))
        self.assertTrue(torch.equal(shard[3:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_trimmed_fused_input_uses_its_shape(self):
        source, call, _ = self.invoke(True, masked=False)
        self.assertIs(call["hidden_nvsh"], source)
        self.assertEqual(source.shape[0], 3)
        self.assertIsNone(call["row_counts"])

    def test_missing_masked_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires counts"):
            self.invoke(True, counts=None)

    def test_default_disabled_and_strict_environment(self):
        spec = importlib.util.spec_from_file_location("publish_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        key = "VLLM_ASCEND_MOONEP_FUSED_PUBLISH"
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(getattr(envs, key))
        with patch.dict(os.environ, {key: "1"}):
            self.assertTrue(getattr(envs, key))
        with patch.dict(os.environ, {key: "2"}):
            with self.assertRaises(ValueError):
                getattr(envs, key)


class BufferPublishTest(unittest.TestCase):
    def setUp(self):
        sdk = ROOT.parent / "ascend-moonep"
        spec = importlib.util.spec_from_file_location(
            "publish_geometry", sdk / "ascend_moonep/combine_publish.py"
        )
        self.geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.geometry)
        self.order = []
        self.sym = torch.empty(8, 4, dtype=torch.bfloat16)
        owner = self
        class Publisher:
            def publish(self, source, destination, counts):
                owner.geometry.publish_geometry(source, destination, counts)
                owner.order.append("publish")
                rows = source.shape[0] if counts is None else int(counts.sum())
                rows = max(0, min(rows, source.shape[0]))
                destination.zero_()
                destination[:rows].copy_(source[:rows])
        class Plan:
            N, R, K = 8, 1, 2
        self.plan = Plan()
        self.weights = torch.ones(4, 8)
        self.ctx = {
            "S": 4, "H": 4, "K": 2, "R": 1, "hidden_buf_local": self.sym,
            "kernels": {"combine": SimpleNamespace(compute_weighted=self.compute)},
        }
        self.obj = SimpleNamespace(
            _require_ctx=lambda: self.ctx, _ensure_shmem_views=lambda: None,
            _combine_publisher=Publisher(),
        )
        namespace = {"torch": torch, "MoonEPCommPlan": Plan,
                     "cross_rank_barrier": lambda *args: self.order.append("barrier")}
        self.fn = load_method(sdk / "ascend_moonep/api.py", "Buffer", "combine_weighted", namespace)

    def compute(self, ctx, output, plan, weights):
        self.order.append("weighted")
        self.assertIs(plan, self.plan)
        self.assertIs(weights, self.weights)
        output.zero_()

    def test_publish_precedes_existing_barrier_and_weighted_kernel(self):
        source = torch.ones(8, 4, dtype=torch.bfloat16)
        source[5:].fill_(float("nan"))
        self.fn(self.obj, self.plan, source, self.weights, num_tokens=4,
                fused_publish=True, row_counts=torch.tensor([2, 3], dtype=torch.int64))
        self.assertEqual(self.order, ["publish", "barrier", "weighted"])
        self.assertTrue(torch.equal(self.sym[5:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_trimmed_prefix_zero_extends(self):
        source = torch.ones(3, 4, dtype=torch.bfloat16)
        self.fn(self.obj, self.plan, source, self.weights, num_tokens=4, fused_publish=True)
        self.assertTrue(torch.equal(self.sym[:3], source))
        self.assertTrue(torch.equal(self.sym[3:], torch.zeros(5, 4, dtype=torch.bfloat16)))

    def test_zero_copy_contract_is_not_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    zero_copy=True, fused_publish=True)
        self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4, zero_copy=True)
        self.assertEqual(self.order, ["barrier", "weighted"])

    def test_legacy_shape_and_fused_count_contracts(self):
        with self.assertRaisesRegex(ValueError, "Legacy weighted"):
            self.fn(self.obj, self.plan, self.sym[:3], self.weights, num_tokens=4)
        with self.assertRaisesRegex(ValueError, "row_counts requires"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    row_counts=torch.ones(3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "int64"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    fused_publish=True, row_counts=torch.ones(3, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
