"""CPU regressions for permanently disabled vLLM execution diagnostics.

Execute real dispatcher, prefetch and MLP functions with CPU tensors / fake
device submissions. No NPU initialization or device operators are required.
"""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Generic, TypeVar
import unittest
from unittest.mock import Mock, patch

import torch

import test_moonep_no_host_sync as counts_cases
import test_moonep_pool as pool_cases
import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "vllm_ascend/ops/fused_moe"
REMOVED = (
    "MOONEP_TIME_STATS", "MOONEP_TIME_STATS_EVERY", "MOONEP_MLP_SYNC",
    "MOONEP_PF_TIMING", "MOONEP_PACE_SYNC", "MOONEP_CAP_TRACE",
)


def environment(value):
    return {**dict.fromkeys(REMOVED, value), "MOONEP_SLOT_STATS": "0"}


def load_functions(relative, names, namespace):
    source = OPS / relative
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)


class NoExecutionDiagnosticsTests(unittest.TestCase):
    def test_removed_keys_have_no_runtime_string_references(self):
        found = []
        for source in (ROOT / "vllm_ascend").rglob("*.py"):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value in REMOVED:
                    found.append((str(source), node.lineno, node.value))
        self.assertEqual(found, [])

    def test_bf16_and_mxfp8_dispatch_ignore_old_values_without_clock_or_device_sync(self):
        for value in ("0", "1", "not-a-number"):
            for transport in ("bf16", "mxfp8"):
                with self.subTest(value=value, transport=transport):
                    case = counts_cases.NoHostCountSyncTests().make_dispatch(transport)
                    fn, dispatcher, inputs, shard, weights, scales, calls, modules = case
                    fn.__globals__.pop("_ENV_PACE_SYNC", None)
                    del dispatcher._tstats
                    forbidden = Mock(side_effect=AssertionError("Diagnostic device sync is forbidden"))
                    with patch.dict(os.environ, environment(value)), patch.dict(sys.modules, modules), patch.object(
                        torch, "npu", SimpleNamespace(synchronize=forbidden), create=True
                    ), patch("time.perf_counter", side_effect=AssertionError("Diagnostic clock is forbidden")), patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("No count D2H")
                    ):
                        result = fn(dispatcher, inputs)
                    forbidden.assert_not_called()
                    self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
                    self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0])
                    self.assertEqual(result.group_list_type, 1)
                    self.assertIs(result.combine_metadata.group_list, result.group_list)
                    self.assertIsNotNone(result.combine_metadata.state)
                    self.assertTrue(result.combine_metadata.tail_masked)
                    self.assertTrue(torch.equal(result.combine_metadata.w_trim, weights))
                    self.assertEqual(calls[0]["num_tokens"], 2)
                    if transport == "bf16":
                        self.assertTrue(calls[0]["zero_copy"])
                        self.assertTrue(calls[0]["reuse_outputs"])
                        self.assertTrue(calls[0]["inter_rank_sync"])
                    else:
                        self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())

    def test_capacity_selection_still_tracks_dp_metadata_without_trace_output(self):
        case = pool_cases.AdaptivePoolTests()
        case.setUp()
        case.namespace.pop("_ENV_CAP_TRACE", None)
        case.namespace.pop("_CAP_TRACE", None)
        case.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        case.install_pool([16, 64, 128, 4096])
        for value in ("0", "1", "invalid"):
            output = io.StringIO()
            with patch.dict(os.environ, environment(value)), redirect_stdout(output):
                case.context.max_tokens_across_dp = 33
                self.assertEqual(case.select(8), 64)
                case.context.max_tokens_across_dp = 129
                self.assertEqual(case.select(8), 4096)
            self.assertEqual(output.getvalue(), "")
        self.assertIn((256, 6, 4096), case.namespace["_GEO_CACHE"])

    def test_native_import_does_not_replace_mc2_ops_even_with_old_flag_one(self):
        from abc import ABC, abstractmethod
        source = OPS / "token_dispatcher.py"
        tree = ast.parse(source.read_text())
        body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body
        ], type_ignores=[])
        def operator(*args, **kwargs):
            raise AssertionError("An operator must not execute during module setup")
        names = ("npu_moe_distribute_dispatch_v2", "npu_moe_distribute_dispatch",
                 "npu_moe_distribute_combine_v2", "npu_moe_distribute_combine")
        native = SimpleNamespace(**dict.fromkeys(names, operator))
        namespace = dict(ABC=ABC, abstractmethod=abstractmethod, Generic=Generic,
                         TMoECombineMetadata=TypeVar("TMoECombineMetadata"),
                         torch=torch, torch_npu=native)
        for name in ("MoEMC2CombineMetadata", "MoEAllGatherCombineMetadata", "MoEAllToAllCombineMetadata"):
            namespace[name] = type(name, (), {})
        with patch.dict(os.environ, environment("1")):
            exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        for name in names:
            self.assertIs(getattr(native, name), operator)
        for name in ("TokenDispatcherWithAllGather", "TokenDispatcherWithAll2AllV"):
            self.assertFalse(getattr(namespace[name].token_dispatch, "_ts_wrapped", False))
            self.assertFalse(getattr(namespace[name].token_combine, "_ts_wrapped", False))



class NoTimingPrefetchTests(unittest.TestCase):
    def fixture(self):
        case = cache_cases.CacheStreamTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def forbid(self):
        return patch.object(
            torch, "npu",
            SimpleNamespace(synchronize=Mock(side_effect=AssertionError("No full device sync"))),
            create=True,
        )

    def test_serial_prefetch_and_cache_fill_ignore_old_timing_flags(self):
        for value in ("0", "1", "invalid"):
            case = self.fixture()
            case.ns["_PREFETCH_OVERLAP"] = "none"
            plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
            with patch.dict(os.environ, environment(value)), self.forbid(), patch(
                "time.perf_counter", side_effect=AssertionError("No diagnostic clocks")
            ):
                output = case.forward(plan)
            self.assertEqual(output.flatten().tolist(), [11.0, 22.0])
            self.assertEqual(case.state.cache_stats, [1, 1])
            self.assertEqual(case.events, ["prefetch", "gemm"])

    def test_dispatch_overlap_keeps_event_join_before_gemm(self):
        case = self.fixture()
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            case.hook(plan)
            case.forward(plan)
        self.assertEqual(len(case.calls), 1)
        self.assertLess(max(i for i, event in enumerate(case.events) if event == "wait"),
                        case.events.index("gemm"))
        self.assertNotIn("slot_prefetch_ready", case.ctx)

    def test_home_gemm_overlap_receives_original_prefetch_completion_event(self):
        case = self.fixture()
        case.ns.update(_PREFETCH_OVERLAP="dispatch|home_gemm1", _CACHED_EXPERT_COUNT=0)
        case.state.epn, case.state.F, case.state.B = 2, 1, 2
        runner = SimpleNamespace(activation_logged=True)
        case.state.home_gmm_runner = lambda: runner
        received = []
        def home(**kwargs):
            received.append(kwargs)
            return case.state.slots["w13"].clone(), None
        case.ns["home_gemm_apply_mlp"] = home
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            case.hook(plan)
            event = case.ctx["slot_prefetch_ready"][-1]
            case.forward(plan)
        self.assertIs(received[0]["prefetch_ready"], event)
        self.assertIs(received[0]["range_kernel"], runner)
        self.assertEqual(received[0]["home_experts"], 2)
        self.assertNotIn("gemm", case.events)
        self.assertNotIn("slot_prefetch_ready", case.ctx)

    def test_udma_independent_qp_guard_remains_before_submission(self):
        case = self.fixture()
        case.ctx["_prefetch_use_udma"] = True
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            with self.assertRaisesRegex(RuntimeError, "dedicated QP"):
                case.hook(plan)
        self.assertEqual(case.calls, [])
        self.assertEqual(case.events, [])


class NoTimingMlpTests(unittest.TestCase):
    def test_real_unquantized_mlp_math_ignores_old_diagnostic_switches(self):
        counts = torch.tensor([1, 3], dtype=torch.int64)
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2).div(10)
        w1 = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4).div(20)
        w2 = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).div(10)
        def gmm(*, x, weight, group_list, **kwargs):
            result, offset = [], 0
            for expert, count in enumerate(group_list.tolist()):
                result.append(x[0][offset:offset+count] @ weight[0][expert])
                offset += count
            return [torch.cat(result)]
        def activation(tensor):
            return torch.nn.functional.silu(tensor[:, :2]) * tensor[:, 2:]
        fake = SimpleNamespace(npu_grouped_matmul=gmm, npu_swiglu=activation)
        ns = dict(torch=torch, torch_npu=fake,
                  MoEActivation=SimpleNamespace(SWIGLUOAI="oai", SWIGLUSTEP="step",
                                                GELU="gelu", GELU_TANH="gelu_tanh"))
        load_functions("moe_mlp.py", {"unquant_apply_mlp"}, ns)
        expected = gmm(x=[activation(gmm(x=[x], weight=[w1], group_list=counts)[0])],
                       weight=[w2], group_list=counts)[0]
        for value in ("0", "1", "invalid"):
            forbidden = Mock(side_effect=AssertionError("No MLP timing device sync"))
            with patch.dict(os.environ, environment(value)), patch.object(
                torch, "npu", SimpleNamespace(synchronize=forbidden), create=True
            ), patch("time.perf_counter", side_effect=AssertionError("No MLP diagnostic clocks")):
                output, event = ns["unquant_apply_mlp"](
                    x, w1, w2, counts, activation="silu", need_trans=False)
            self.assertTrue(torch.equal(output, expected))
            self.assertIsNone(event)
            forbidden.assert_not_called()

    def test_unified_quant_entry_preserves_mxfp_scale_and_event_arguments(self):
        ns = dict(torch=torch,
                  QuantType=SimpleNamespace(W4A8MXFP="mx", W4A16MXFP4="mx4"))
        recorder = Mock(return_value=(object(), None))
        ns["quant_apply_mlp"] = recorder
        load_functions("moe_mlp.py", {"unified_apply_mlp"}, ns)
        weights = SimpleNamespace(**dict.fromkeys(
            ("w1", "w2", "w1_bias", "w2_bias", "w1_scale", "w2_scale",
             "w1_scale_bias", "w2_scale_bias", "w1_offset", "w2_offset"), None))
        weights.w1_scale, weights.w2_scale = object(), object()
        scales = object()
        mxfp = SimpleNamespace(act_quant_type=torch.float8_e4m3fn,
                               weight_quant_type=object(), scale_dtype=object(),
                               per_token_scale_dtype=object(), use_bf16=True)
        quant = SimpleNamespace(is_quant=True, is_int_quant=False, is_mxfp=True,
                                quant_type="mx", mxfp=mxfp,
                                use_w4a8_per_channel_gmm_swiglu=False)
        payload = SimpleNamespace(hidden_states=torch.ones(2, 4, dtype=torch.bfloat16),
                                  group_list=torch.tensor([2]), group_list_type=1,
                                  dynamic_scale=scales, topk_scales=None, weights=weights,
                                  activation="silu", need_trans=False, dynamic_eplb=False,
                                  fusion=True, swiglu_limit=10.0, quant=quant)
        with patch.dict(os.environ, environment("1")):
            result = ns["unified_apply_mlp"](mlp_compute_input=payload, record_before_gmm2=False)
        self.assertIs(result, recorder.return_value)
        self.assertIs(recorder.call_args.kwargs["dynamic_scale"], scales)
        self.assertFalse(recorder.call_args.kwargs["record_before_gmm2"])
        self.assertIs(recorder.call_args.kwargs["per_token_scale_type"], mxfp.per_token_scale_dtype)
        self.assertTrue(recorder.call_args.kwargs["use_mxfp_quant"])


if __name__ == "__main__":
    unittest.main()
