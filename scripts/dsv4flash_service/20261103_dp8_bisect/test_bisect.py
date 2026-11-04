"""Synthetic-only bisection tests; no model, NPU query, inference or benchmark."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import bisect_logic as logic
import measurements
import resources
import run_bisect
import runtime
import study


def pair(ttft, throughput, status="complete"):
    return {"status": status, "ttft_gain_pct": ttft, "throughput_gain_pct": throughput,
            "ttft_log_ratio": -math.log1p(-ttft / 100),
            "throughput_log_ratio": math.log1p(throughput / 100)}


class BisectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = logic.load_settings()

    def point(self, ratio, yes):
        values = [pair(5, 5), pair(5, 5)] if yes else [pair(1, 1), pair(1, 1)]
        return logic.classify_point(ratio, values, self.cfg)

    def test_ideal_search_paths_are_bounded_and_accurate(self):
        for true_threshold in (1.01, 1.2, 1.4, 1.6, 1.74):
            history = []
            for _ in range(10):
                action = logic.next_action(history, self.cfg)
                if action["action"] == "stop":
                    break
                self.assertTrue(1 <= action["ratio"] <= 1.75)
                history.append(self.point(action["ratio"], action["ratio"] >= true_threshold))
            self.assertEqual(action["status"], "bracket_found")
            self.assertLessEqual(len(history), 6)
            self.assertLessEqual(action["width"], 0.05)
            self.assertLess(action["bracket"][0], true_threshold)
            self.assertGreaterEqual(action["bracket"][1], true_threshold)
        self.assertEqual(logic.strategy(self.cfg)["max_service_starts"], 24)
        self.assertEqual(logic.strategy(self.cfg)["max_benchmark_calls"], 96)

    def test_endpoints_can_stop_early_without_false_global_claim(self):
        action = logic.next_action([], self.cfg)
        self.assertEqual(action["ratio"], 1.75)
        fail = logic.next_action([self.point(1.75, False)], self.cfg)
        self.assertEqual(fail["status"], "upper_not_gain")
        self.assertIsNone(fail["bracket"])
        next_action = logic.next_action([self.point(1.75, True)], self.cfg)
        self.assertEqual(next_action["ratio"], 1)
        done = logic.next_action([self.point(1.75, True), self.point(1, True)], self.cfg)
        self.assertEqual(done["status"], "lower_already_gain")

    def test_disagreement_and_invalid_data_never_shrink_bracket(self):
        history = [self.point(1.75, True), self.point(1, False)]
        middle = logic.classify_point(1.375, [pair(5, 5), pair(2, 5)], self.cfg)
        self.assertEqual(middle["status"], "uncertain")
        action = logic.next_action(history + [middle], self.cfg)
        self.assertEqual(action["status"], "uncertain")
        self.assertEqual(action["bracket"], [1, 1.75])
        invalid = logic.classify_point(1.375, [pair(5, 5), {"status": "incomplete"}], self.cfg)
        self.assertEqual(logic.next_action(history + [invalid], self.cfg)["status"], "invalid_data")

    def test_votes_require_both_metrics_and_do_not_pool_disagreements(self):
        self.assertEqual(logic.classify_point(1.3, [pair(3, 3), pair(3, 3)], self.cfg)["status"], "gain")
        result = logic.classify_point(1.3, [pair(4, 2), pair(2, 4)], self.cfg)
        self.assertEqual(result["status"], "no_gain")
        self.assertEqual(result["votes"], [False, False])
        self.assertEqual(logic.classify_point(1.3, [pair(4, 4), pair(2, 2)], self.cfg)["status"], "uncertain")

    def test_nonmonotonic_and_unsafe_history_stop(self):
        action = logic.next_action([self.point(1, True), self.point(1.75, False)], self.cfg)
        self.assertEqual(action["status"], "nonmonotonic")
        for history in ([self.point(2, True)], [self.point(1.75, True)] * 2):
            with self.assertRaises(ValueError):
                logic.next_action(history, self.cfg)
        with self.assertRaises(ValueError):
            run_bisect.point_plan(self.cfg, 1, 2.0)

    def test_route_and_seed_controls(self):
        for ratio in (1, 1.1875, 1.375, 1.421875, 1.75):
            route = study.route(ratio)
            self.assertAlmostEqual(route["home_rank_peak_to_mean"], ratio, places=10)
            self.assertEqual(route["hot_experts"], [0, 16])
            plan = run_bisect.point_plan(self.cfg, 1, ratio)
            self.assertEqual(plan["service_starts"], 4)
            self.assertEqual([p["mode_order"] for p in plan["pairs"]], [["off", "slot"], ["slot", "off"]])
            for p in plan["pairs"]:
                a, b = p["cases"]
                for i in range(4):
                    self.assertEqual(runtime.bench_command(self.cfg["runtime"], a, Path("/tmp/preview"), i),
                                     runtime.bench_command(self.cfg["runtime"], b, Path("/tmp/preview"), i))

    def test_default_preview_cannot_execute(self):
        with mock.patch.object(sys, "argv", ["run_bisect.py"]), \
             mock.patch.object(run_bisect, "execute", side_effect=AssertionError("No execute")), \
             mock.patch.object(runtime.subprocess, "Popen", side_effect=AssertionError("No process")), \
             mock.patch.object(runtime, "capture_npu", side_effect=AssertionError("No NPU")), \
             contextlib.redirect_stdout(io.StringIO()):
            run_bisect.main()

    def write_point(self, root, index, ratio, factors):
        step = root / "points" / f"{index:02d}_s{study.ratio_key(ratio)}"
        step.mkdir(parents=True)
        plan = run_bisect.point_plan(self.cfg, index, ratio)
        study.write_new(step / "plan.json", plan)
        for p, factor in zip(plan["pairs"], factors, strict=True):
            for case in p["cases"]:
                directory = step / case["id"]
                directory.mkdir()
                f = factor if case["mode"] == "slot" else 1.0
                commands = []
                for i in range(4):
                    count, length = (2, 1024) if i == 0 else (32, 8192)
                    data = {"completed": count, "failed": 0, "duration": 2 * f,
                            "total_input_tokens": count * length, "total_output_tokens": count,
                            "mean_ttft_ms": 99999 if i == 1 else 100 * f,
                            "mean_e2el_ms": 100 * f,
                            "total_token_throughput": count * (length + 1) / (2 * f)}
                    study.write_new(directory / f"round_{i}.json", data)
                    commands.append(runtime.bench_command(self.cfg["runtime"], case, directory, i))
                study.write_new(directory / "metadata.json", {
                    "case": case, "status": "complete", "route": plan["routes"][0], "bench_commands": commands})
        return step, plan

    def test_disk_history_report_and_partial_resume(self):
        with tempfile.TemporaryDirectory(prefix="bisect_test_") as temp:
            root = Path(temp)
            self.write_point(root, 1, 1.75, [.95, .95])
            first = run_bisect.report(root, self.cfg)
            self.assertFalse(first["search_stopped"])
            self.assertEqual(first["decision"]["ratio"], 1)
            step, plan = self.write_point(root, 2, 1, [.99, .99])
            history, pending = run_bisect.history_from_disk(root, self.cfg)
            self.assertIsNone(pending)
            self.assertEqual(logic.next_action(history, self.cfg)["ratio"], 1.375)
            bad = step / plan["pairs"][0]["cases"][0]["id"] / "round_2.json"
            bad.unlink()
            history, pending = run_bisect.history_from_disk(root, self.cfg)
            self.assertEqual(len(history), 1)
            self.assertIsNotNone(pending)
            report = run_bisect.report(root, self.cfg)
            self.assertEqual(report["decision"]["status"], "invalid_data")
            self.assertFalse(report["candidate_bracket_ready"])
            resources.archive_partial_pair(step, plan["pairs"][0])
            self.assertEqual(len(list((step / "attempts").glob("*/*/metadata.json"))), 2)

    def test_uncertain_point_stops_without_third_pair(self):
        with tempfile.TemporaryDirectory(prefix="bisect_test_") as temp:
            root = Path(temp)
            self.write_point(root, 1, 1.75, [.95, .95])
            self.write_point(root, 2, 1, [.99, .99])
            self.write_point(root, 3, 1.375, [.95, 1.02])
            result = run_bisect.report(root, self.cfg)
            self.assertEqual(result["decision"]["status"], "uncertain")
            self.assertEqual(result["decision"]["bracket"], [1, 1.75])
            self.assertEqual(len(result["points"][2]["pairs"]), 2)


if __name__ == "__main__":
    unittest.main()
