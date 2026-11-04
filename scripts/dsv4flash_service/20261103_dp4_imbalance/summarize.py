#!/usr/bin/env python3
"""Summarize a recorded experiment directory without contacting an NPU or service."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from run_experiment import read_json, validate_result, write_json

AGGREGATE_KEYS = (
    "duration", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "std_ttft_ms", "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
    "request_throughput", "input_throughput", "output_throughput",
    "total_token_throughput", "total_input_tokens", "total_output_tokens",
)


def case_summary(root, case, cfg):
    directory = root / case["id"]
    result = {"case": case, "status": "missing", "reason": "", "rounds": []}
    try:
        metadata = read_json(directory / "metadata.json")
        if metadata["case"] != case or metadata["status"] != "complete":
            raise ValueError("Case metadata is incomplete or does not match the plan: "
                             + metadata.get("error", metadata["status"]))
        # Validate warmup and discarded rounds too; a partial run is not a sample.
        for index in range(cfg["rounds"] + 1):
            value = validate_result(directory / f"round_{index}.json",
                                    2 if index == 0 else cfg["num_prompts"])
            if index > cfg["discard_rounds"]:
                value["input_throughput"] = value["total_input_tokens"] / value["duration"]
                result["rounds"].append({"round": index, "metrics": value})
        expected = cfg["rounds"] - cfg["discard_rounds"]
        if len(result["rounds"]) != expected:
            raise ValueError("Measured round count does not match plan.")
        means, stdevs = {}, {}
        for key in AGGREGATE_KEYS:
            values = [r["metrics"].get(key) for r in result["rounds"]]
            if any(value is None for value in values):
                continue
            means[key] = statistics.mean(values)
            stdevs[key] = statistics.stdev(values) if len(values) > 1 else None
        result.update(status="complete", route=metadata["route"],
                      sample_count=expected, means=means, across_round_stdev=stdevs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="incomplete", reason=f"{type(exc).__name__}: {exc}")
    return result


def compare_pair(off, on):
    comparison = {"route": off["case"]["route"],
                  "input_length": off["case"]["input_length"],
                  "batched_tokens": off["case"]["batched_tokens"],
                  "moonep_mode": on["case"]["mode"], "status": "incomplete"}
    if off["status"] != "complete" or on["status"] != "complete":
        comparison["reason"] = "Both complete modes are required; missing/failed cases are not averaged."
        return comparison
    if off["route"]["sha256"] != on["route"]["sha256"]:
        comparison["reason"] = "Route profile hashes differ."
        return comparison
    # Equal seeds/request shapes should also yield equal token totals per paired round.
    for left, right in zip(off["rounds"], on["rounds"], strict=True):
        if left["round"] != right["round"]:
            comparison["reason"] = "Measured round numbers differ."
            return comparison
        for key in ("completed", "total_input_tokens", "total_output_tokens"):
            if left["metrics"][key] != right["metrics"][key]:
                comparison["reason"] = f"Round {left['round']} differs in {key}; no speedup claimed."
                return comparison
    a, b = off["means"], on["means"]
    comparison.update(
        status="complete",
        home_rank_peak_to_mean=off["route"]["home_rank_peak_to_mean"],
        off_ttft_ms=a["mean_ttft_ms"], on_ttft_ms=b["mean_ttft_ms"],
        ttft_reduction_pct=100 * (1 - b["mean_ttft_ms"] / a["mean_ttft_ms"]),
        off_input_tps=a["input_throughput"], on_input_tps=b["input_throughput"],
        input_throughput_gain_pct=100 * (b["input_throughput"] / a["input_throughput"] - 1),
        off_total_tps=a["total_token_throughput"], on_total_tps=b["total_token_throughput"],
        total_throughput_gain_pct=100 * (
            b["total_token_throughput"] / a["total_token_throughput"] - 1),
        off_duration_s=a["duration"], on_duration_s=b["duration"],
        duration_reduction_pct=100 * (1 - b["duration"] / a["duration"]),
        samples_per_mode=off["sample_count"],
    )
    return comparison


def save_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def summarize(root):
    plan = read_json(root / "plan.json")
    cfg = plan["config"]
    cases = [case_summary(root, case, cfg) for case in plan["cases"]]
    groups = {}
    for item in cases:
        case = item["case"]
        key = (case["route"], case["input_length"], case["batched_tokens"])
        groups.setdefault(key, {})[case["mode"]] = item
    pairs = [compare_pair(group["off"], group[cfg["modes"][1]])
             for group in groups.values()]
    result = {"source": str(root), "config": cfg, "cases": cases, "comparisons": pairs,
              "all_complete": all(pair["status"] == "complete" for pair in pairs),
              "statistic": "Arithmetic mean across retained rounds; no confidence claim with n=2.",
              "routing_scope": "Synthetic home-rank route peak/mean, not measured post-slot compute skew."}
    write_json(root / "summary.json", result)
    rows = []
    for item in cases:
        row = {**item["case"], "status": item["status"], "reason": item["reason"]}
        if item["status"] == "complete":
            row.update(item["means"])
            row["sample_count"] = item["sample_count"]
            row["home_rank_peak_to_mean"] = item["route"]["home_rank_peak_to_mean"]
            for key, value in item["across_round_stdev"].items():
                row["across_round_std_" + key] = value
        rows.append(row)
    fields = ["id", "route", "mode", "input_length", "batched_tokens", "status",
              "reason", "sample_count", "home_rank_peak_to_mean", *AGGREGATE_KEYS,
              *("across_round_std_" + key for key in AGGREGATE_KEYS)]
    save_csv(root / "metrics.csv", rows, fields)
    pair_fields = ["route", "home_rank_peak_to_mean", "input_length", "batched_tokens",
                   "moonep_mode", "status", "reason", "samples_per_mode",
                   "off_ttft_ms", "on_ttft_ms", "ttft_reduction_pct",
                   "off_input_tps", "on_input_tps", "input_throughput_gain_pct",
                   "off_total_tps", "on_total_tps", "total_throughput_gain_pct",
                   "off_duration_s", "on_duration_s", "duration_reduction_pct"]
    save_csv(root / "comparison.csv", pairs, pair_fields)
    lines = [
        "# MoonEP off / slot performance comparison",
        "",
        "Home-rank peak/mean describes the injected pre-balancing route distribution.",
        "It does not measure post-MoonEP compute imbalance or numerical correctness.",
        f"Retained rounds: {cfg['discard_rounds'] + 1}..{cfg['rounds']}; "
        "all values are arithmetic means. Positive percentages mean MoonEP improves.",
        "Per-request percentiles remain in raw JSON; table reports mean TTFT.",
        "",
        "| Route | Peak/mean | Input / budget | Status | TTFT off/on (ms) | "
        "TTFT reduction | Input tok/s off/on | Input gain | Duration off/on (s) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        shape = f"{pair['input_length']} / {pair['batched_tokens']}"
        if pair["status"] != "complete":
            lines.append(f"| {pair['route']} | — | {shape} | incomplete | — | — | — | — | — |")
            continue
        lines.append(
            f"| {pair['route']} | {pair['home_rank_peak_to_mean']:.4f} | {shape} | complete | "
            f"{pair['off_ttft_ms']:.2f}/{pair['on_ttft_ms']:.2f} | "
            f"{pair['ttft_reduction_pct']:+.2f}% | "
            f"{pair['off_input_tps']:.2f}/{pair['on_input_tps']:.2f} | "
            f"{pair['input_throughput_gain_pct']:+.2f}% | "
            f"{pair['off_duration_s']:.3f}/{pair['on_duration_s']:.3f} |")
    incomplete = [item for item in cases if item["status"] != "complete"]
    if incomplete:
        lines.extend(["", "## Incomplete cases", ""])
        lines.extend(f"- {item['case']['id']}: {item['reason']}" for item in incomplete)
    lines.extend(["", "No aggregate speedup is claimed for unmatched/failed pairs.",
                  "These scripts do not automatically certify external resource isolation or output equivalence.",
                  "Raw service logs, NPU snapshots, commands, seeds and Git versions accompany the results.", ""])
    (root / "summary.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="A data/<timestamp> directory containing plan.json")
    args = parser.parse_args()
    root = args.session.resolve()
    result = summarize(root)
    complete = sum(pair["status"] == "complete" for pair in result["comparisons"])
    print(f"Summarized {complete}/{len(result['comparisons'])} complete pairs in {root}")


if __name__ == "__main__":
    main()
