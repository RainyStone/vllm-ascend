"""Pure logic checks: no torch imports, NPU probes, service or benchmark processes."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_experiment as runner
import summarize

MODEL_CONFIG = {"model_type": "deepseek_v4", "quantization_config": {"quant_method": "fp8"},
                "n_routed_experts": 256, "num_experts_per_tok": 6}
NPU_TEXT = """
| NPU ID | Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| 0 | Ascend950DT | OK | 400 50 0 / 0 |
|   | NA | 0 0 / 0 87721 / 98304 |
| 4 | Ascend950DT | OK | 400 50 0 / 0 |
|   | NA | 0 0 / 0 4096 / 98304 |
| 5 | Ascend950DT | OK | 400 50 0 / 0 |
|   | NA | 0 0 / 0 4096 / 98304 |
| 6 | Ascend950DT | OK | 400 50 0 / 0 |
|   | NA | 0 0 / 0 4096 / 98304 |
| 7 | Ascend950DT | OK | 400 50 0 / 0 |
|   | NA | 0 0 / 0 4096 / 98304 |
| NPU ID | Process id | Process name | Process memory(MB) | Process id in container |
| 0 | 123456 | VLLM::Worker_TP | 82853 | 123 |
"""


def synthetic_result(prompts=32, input_tokens=262144, latency=100, duration=20):
    return {"completed": prompts, "failed": 0, "duration": duration,
            "total_input_tokens": input_tokens, "total_output_tokens": prompts,
            "mean_ttft_ms": latency, "mean_e2el_ms": latency + 1,
            "total_token_throughput": (input_tokens + prompts) / duration}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.load_config(runner.HERE / "config.json")

    def plan(self):
        # Model config is synthetic; shared route files are only read as text.
        with mock.patch.object(runner, "read_json", return_value=MODEL_CONFIG):
            return runner.make_plan(self.cfg)

    def test_complete_matrix_and_route_ratios(self):
        plan = self.plan()
        self.assertEqual(plan["service_starts"], 32)
        self.assertEqual(plan["benchmark_calls"], 128)
        ratios = [item["home_rank_peak_to_mean"] for item in plan["routes"]]
        for actual, expected in zip(ratios, [1.0, 1.6614, 2.4822, 2.9841], strict=True):
            self.assertAlmostEqual(actual, expected, places=4)
        for index in range(0, 32, 2):
            left, right = plan["cases"][index:index + 2]
            for key in ("route", "input_length", "batched_tokens"):
                self.assertEqual(left[key], right[key])
            self.assertEqual({left["mode"], right["mode"]}, {"off", "slot"})
        self.assertEqual(plan["cases"][0]["mode"], "off")
        self.assertEqual(plan["cases"][2]["mode"], "slot")

    def test_commands_and_environment_do_not_touch_front_cards(self):
        plan = self.plan()
        route = plan["routes"][0]
        off, slot = plan["cases"][:2]
        stale = {"MOONEP": "1", "MOONEP_SLOT_STATS": "1",
                 "ASCEND_MOONEP_NO_RECONCILE": "1", "VLLM_ASCEND_MOE_PAR_STATS": "1",
                 "VLLM_TOKENIZER_BATCH_WAIT_MS": "20"}
        with mock.patch.dict(runner.os.environ, stale):
            off_env = runner.service_env(self.cfg, off, route)
            slot_env = runner.service_env(self.cfg, slot, route)
        self.assertEqual(off_env["ASCEND_RT_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertNotIn("MOONEP", off_env)
        self.assertNotIn("ASCEND_MOONEP_NO_RECONCILE", off_env)
        self.assertNotIn("VLLM_ASCEND_MOE_PAR_STATS", slot_env)
        self.assertNotIn("VLLM_TOKENIZER_BATCH_WAIT_MS", slot_env)
        self.assertIn("ascend_model", slot_env["VLLM_PLUGINS"].split(","))
        self.assertIn("ascend_model", off_env["VLLM_PLUGINS"].split(","))
        self.assertIn("ascend_model_loader", off_env["VLLM_PLUGINS"].split(","))
        self.assertEqual(slot_env["MOONEP_SLOT_MODE"], "1")
        self.assertEqual(slot_env["ASCEND_MOONEP_NO_RECONCILE"], "1")
        self.assertEqual(off_env["VLLM_ASCEND_MOE_SIM_ROUTE"],
                         slot_env["VLLM_ASCEND_MOE_SIM_ROUTE"])
        off_cmd, slot_cmd = [runner.service_command(self.cfg, c) for c in (off, slot)]
        for command, expected in ((off_cmd, False), (slot_cmd, True)):
            item = json.loads(command[command.index("--additional-config") + 1])
            self.assertEqual(item["enable_moonep"], expected)
        left, right = [runner.bench_command(self.cfg, c, Path("/tmp/synthetic"), 2)
                       for c in (off, slot)]
        self.assertEqual(left[left.index("--seed") + 1], right[right.index("--seed") + 1])
        self.assertEqual(left[left.index("--random-output-len") + 1], "1")
        with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError):
            self.assertIn("serve", runner.with_cann(self.cfg, off_cmd))

    def test_npu_96g_parser_and_selected_cards_only(self):
        devices, processes = runner.parse_npu(NPU_TEXT)
        self.assertEqual(devices[4]["total_mb"], 98304)
        self.assertEqual(devices[4]["used_mb"], 4096)
        self.assertEqual(processes, {0: {123456}})
        with mock.patch.object(runner, "capture_npu", return_value=(devices, processes)):
            runner.wait_cards_free(self.cfg, None)  # Busy front card does not block us.
        cfg = {**self.cfg, "release_timeout_seconds": 0}
        with mock.patch.object(runner, "capture_npu", return_value=(devices, {4: {987}})):
            with self.assertRaises(RuntimeError):
                runner.wait_cards_free(cfg, None)
        with mock.patch.object(runner, "capture_npu", return_value=({}, {})):
            with self.assertRaises(RuntimeError):
                runner.wait_cards_free(cfg, None)

    def test_fail_fast_for_terminal_startup_error(self):
        with tempfile.TemporaryDirectory(prefix="dsv4_harness_") as tmp:
            path = Path(tmp) / "service.log"
            path.write_text("torch.OutOfMemoryError: synthetic startup failure")
            process = mock.Mock()
            process.poll.return_value = None
            with mock.patch.object(runner, "read_models", side_effect=AssertionError("No HTTP after fatal error")):
                with self.assertRaises(RuntimeError):
                    runner.wait_ready(self.cfg, process, path)
        cfg = {**self.cfg, "first_mode": "slot"}
        with mock.patch.object(runner, "read_json", return_value=MODEL_CONFIG):
            plan = runner.make_plan(cfg)
        self.assertEqual(plan["cases"][0]["mode"], "slot")

    def test_failed_requests_and_nonfinite_data_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dsv4_harness_") as tmp:
            path = Path(tmp) / "synthetic.json"
            for patch in ({"failed": 1}, {"completed": 31}, {"duration": 0},
                          {"mean_ttft_ms": float("nan")}, {"total_output_tokens": 31}):
                runner.write_json(path, {**synthetic_result(), **patch})
                with self.assertRaises(ValueError):
                    runner.validate_result(path, 32)

    def test_summary_paired_metrics_and_incomplete_exclusion(self):
        plan = self.plan()
        plan["cases"] = plan["cases"][:2]
        with tempfile.TemporaryDirectory(prefix="dsv4_harness_") as tmp:
            root = Path(tmp)
            runner.write_json(root / "plan.json", plan)
            for case in plan["cases"]:
                directory = root / case["id"]
                directory.mkdir()
                runner.write_json(directory / "metadata.json",
                                  {"case": case, "status": "complete", "route": plan["routes"][0]})
                for index in range(4):
                    # The huge discarded latency must not enter the means.
                    latency = 99999 if index == 1 else (100 if case["mode"] == "off" else 80)
                    prompts = 2 if index == 0 else 32
                    data = synthetic_result(prompts, 2048 if index == 0 else 262144,
                                            latency, 20 if case["mode"] == "off" else 16)
                    runner.write_json(directory / f"round_{index}.json", data)
            result = summarize.summarize(root)
            pair = result["comparisons"][0]
            self.assertTrue(result["all_complete"])
            self.assertAlmostEqual(pair["ttft_reduction_pct"], 20)
            self.assertAlmostEqual(pair["input_throughput_gain_pct"], 25)
            slot_path = root / plan["cases"][1]["id"] / "round_2.json"
            data = runner.read_json(slot_path)
            data["total_input_tokens"] -= 1
            runner.write_json(slot_path, data)
            self.assertEqual(summarize.summarize(root)["comparisons"][0]["status"], "incomplete")
            slot_path.unlink()
            self.assertFalse(summarize.summarize(root)["all_complete"])


if __name__ == "__main__":
    unittest.main()
