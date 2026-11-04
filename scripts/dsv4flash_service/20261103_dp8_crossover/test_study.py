"""Logic tests with synthetic data and Matplotlib plots; no GPU libraries or workloads."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import analyze
import plan_sweep
import run_sweep
import runtime
import study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.config = study.settings()

    def test_route_family_and_caps(self):
        for value in self.config["coarse_ratios"]:
            route = study.route(value)
            self.assertAlmostEqual(route["home_rank_peak_to_mean"], value, places=10)
            self.assertAlmostEqual(sum(route["rank_route_percent"]), 100, places=10)
            self.assertEqual(route["hot_experts"], [0, 16])
            self.assertLessEqual(route["hot_expert_share_pct"], 100 / 6)
            self.assertEqual(study.digest(route["text"].encode()), route["sha256"])
        self.assertAlmostEqual(study.route(1)["hot_expert_share_pct"], 100 / 256)
        self.assertAlmostEqual(study.route(2.4822164743083004)["hot_expert_share_pct"],
                               10.895172557876906)
        for value in (0.99, 4.0, float("nan")):
            with self.assertRaises(ValueError):
                study.route(value)

    def test_balanced_orders_and_paired_seeds(self):
        plan = study.make_plan(self.config, "coarse", self.config["coarse_ratios"], 2)
        self.assertEqual(plan["service_starts"], 32)
        self.assertEqual(plan["benchmark_calls"], 128)
        self.assertEqual(study.make_plan(self.config, "coarse", self.config["coarse_ratios"], 2), plan)
        study.validate_plan(plan)
        for ratio in plan["ratios"]:
            pairs = [pair for pair in plan["pairs"] if pair["ratio"] == ratio]
            self.assertEqual({tuple(pair["mode_order"]) for pair in pairs},
                             {("off", "slot"), ("slot", "off")})
            for pair in pairs:
                left, right = pair["cases"]
                for index in range(4):
                    a = runtime.bench_command(self.config["runtime"], left, Path("/tmp/preview"), index)
                    b = runtime.bench_command(self.config["runtime"], right, Path("/tmp/preview"), index)
                    self.assertEqual(a, b)
        a = plan["pairs"][0]["cases"][0]
        same_ratio = next(pair["cases"][0] for pair in plan["pairs"]
                          if pair["ratio"] == a["target_ratio"] and pair["replicate"] != a["replicate"])
        ca = runtime.bench_command(self.config["runtime"], a, Path("/tmp/preview"), 2)
        cb = runtime.bench_command(self.config["runtime"], same_ratio, Path("/tmp/preview"), 2)
        self.assertNotEqual(ca[ca.index("--seed")+1], cb[cb.index("--seed")+1])
        broken = copy.deepcopy(plan)
        broken["pairs"][0]["mode_order"].reverse()
        with self.assertRaises(ValueError):
            study.validate_plan(broken)

    def test_archived_entry_refuses_preview_and_execution(self):
        plan = study.make_plan(self.config, "coarse", self.config["coarse_ratios"], 2)
        with tempfile.TemporaryDirectory(prefix="crossover_test_") as temp:
            path = Path(temp) / "plan.json"
            study.write_new(path, plan)
            for extra in ([], ["--execute"]):
                with self.subTest(execute=bool(extra)):
                    error = io.StringIO()
                    argv = ["run_sweep.py", "--plan", str(path)] + extra
                    with mock.patch.object(sys, "argv", argv), \
                         mock.patch.object(run_sweep, "execute", side_effect=AssertionError("No execute")), \
                         mock.patch.object(runtime.subprocess, "Popen", side_effect=AssertionError("No process")), \
                         mock.patch.object(runtime, "capture_npu", side_effect=AssertionError("No NPU")), \
                         contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(error), \
                         self.assertRaises(SystemExit) as raised:
                        run_sweep.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("Superseded", error.getvalue())

    def fixture(self, root):
        plan = study.make_plan(self.config, "coarse", [1.0, 1.25, 1.5], 2)
        study.write_new(root / "plan.json", plan)
        routes = {route["name"]: route for route in plan["routes"]}
        factors = {1.0: 1.1, 1.25: 1.0, 1.5: 0.9}
        for pair in plan["pairs"]:
            for case in pair["cases"]:
                directory = root / case["id"]
                directory.mkdir()
                commands = []
                factor = factors[pair["ratio"]] if case["mode"] == "slot" else 1.0
                for index in range(4):
                    prompts = 2 if index == 0 else 32
                    length = 1024 if index == 0 else 8192
                    duration = 2 * factor
                    value = {"completed": prompts, "failed": 0, "duration": duration,
                             "mean_ttft_ms": (99999 if index == 1 else 100 * factor),
                             "mean_e2el_ms": 100 * factor,
                             "total_input_tokens": prompts * length, "total_output_tokens": prompts,
                             "total_token_throughput": (prompts * length + prompts) / duration}
                    study.write_new(directory / f"round_{index}.json", value)
                    commands.append(runtime.bench_command(self.config["runtime"], case, directory, index))
                study.write_new(directory / "metadata.json", {
                    "case": case, "status": "complete", "route": routes[case["route"]],
                    "bench_commands": commands})
        return plan

    def test_analysis_fine_plan_and_exploratory_only(self):
        with tempfile.TemporaryDirectory(prefix="crossover_test_") as temp:
            root = Path(temp)
            self.fixture(root)
            summary = analyze.summarize(root)
            self.assertTrue(summary["all_complete"])
            self.assertEqual(summary["boundaries"]["candidate_margin_brackets"], [[1.25, 1.5]])
            self.assertEqual(summary["boundaries"]["smallest_exploratory_margin_ratio"], 1.5)
            last = summary["ratios"][-1]
            self.assertAlmostEqual(last["ttft_gain_pct"], 10)
            self.assertAlmostEqual(last["throughput_gain_pct"], 100 / 9)
            self.assertEqual(last["complete_pairs"], 2)
            self.assertIsNone(last["ttft_ci95_pct"])
            self.assertFalse(last["confirmed_gain"])
            fine = plan_sweep.make_fine(self.config, summary)
            self.assertEqual(fine["ratios"], [1.25, 1.3, 1.35, 1.4, 1.45, 1.5])
            self.assertEqual(fine["service_starts"], 24)
            self.assertTrue((root / "curve.svg").is_file())
            self.assertTrue((root / "pairs.csv").is_file())

    def test_incomplete_pairs_and_whole_pair_resume(self):
        with tempfile.TemporaryDirectory(prefix="crossover_test_") as temp:
            root = Path(temp)
            plan = self.fixture(root)
            pair = plan["pairs"][0]
            (root / pair["cases"][0]["id"] / "round_2.json").unlink()
            summary = analyze.summarize(root)
            self.assertFalse(summary["all_complete"])
            self.assertEqual(analyze.collect_pair(root, pair, plan)["status"], "incomplete")
            with self.assertRaises(ValueError):
                plan_sweep.make_fine(self.config, summary)
            run_sweep.archive_partial_pair(root, pair)
            self.assertTrue(all(not (root / case["id"]).exists() for case in pair["cases"]))
            self.assertEqual(len(list((root / "attempts").glob("*/*/metadata.json"))), 2)
            unsafe = {"cases": [{"id": "../escape"}]}
            with self.assertRaises(ValueError):
                run_sweep.archive_partial_pair(root, unsafe)

    def test_bad_repeat_and_token_mismatch_are_not_averaged(self):
        with tempfile.TemporaryDirectory(prefix="crossover_test_") as temp:
            root = Path(temp)
            plan = self.fixture(root)
            pair = plan["pairs"][0]
            path = root / pair["cases"][0]["id"] / "round_2.json"
            original = study.read(path)
            for change in ({"failed": 1}, {"total_input_tokens": 262143}):
                path.write_text(json.dumps({**original, **change}))
                summary = analyze.summarize(root)
                row = next(item for item in summary["ratios"] if item["ratio"] == pair["ratio"])
                self.assertEqual(row["status"], "incomplete")
                self.assertNotIn("ttft_gain_pct", row)
                self.assertFalse(summary["all_complete"])
                path.write_text(json.dumps(original))

    def test_nonmonotonic_and_missing_points_not_bridged(self):
        rows = [{"ratio": value, "point_meets_margin": meets, "confirmed_gain": False,
                 "ttft_gain_pct": (5 if meets else -5)}
                for value, meets in [(1, False), (1.25, True), (1.5, False), (1.75, True)]]
        info = study.boundaries(rows, self.config)
        self.assertTrue(info["nonmonotonic_observed"])
        summary = {"settings": self.config, "all_complete": True,
                   "boundaries": info, "ratios": rows}
        with self.assertRaises(ValueError):
            plan_sweep.make_fine(self.config, summary)
        rows[1] = {"ratio": 1.25, "status": "incomplete"}
        self.assertNotIn([1, 1.5], study.boundaries(rows, self.config)["candidate_margin_brackets"])

    def test_small_samples_do_not_get_confidence_intervals(self):
        self.assertIsNone(study.bootstrap([0.1, 0.2]))
        interval = study.bootstrap([0.1] * 5)
        self.assertAlmostEqual(interval[0], 0.1)
        self.assertAlmostEqual(interval[1], 0.1)


if __name__ == "__main__":
    unittest.main()
