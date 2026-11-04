#!/usr/bin/env python3
"""Analyze recorded pairs without contacting a model service or NPU."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import runtime
import study


def case_data(root, case, cfg, expected_route):
    directory = root / case["id"]
    metadata = study.read(directory / "metadata.json")
    if metadata["status"] != "complete" or metadata["case"] != case:
        raise ValueError(f"Case incomplete or mismatched: {case['id']}")
    if metadata["route"]["sha256"] != expected_route["sha256"]:
        raise ValueError("Route hash differs from frozen plan.")
    rounds = []
    for index in range(cfg["rounds"] + 1):
        value = runtime.validate_result(directory / f"round_{index}.json",
                                        2 if index == 0 else cfg["num_prompts"])
        command = metadata["bench_commands"][index]
        expected_seed = cfg["bench_seed_base"] + case["replicate"] * 100000 + (
            0 if index == 0 else case["input_length"] + index)
        if int(command[command.index("--seed") + 1]) != expected_seed:
            raise ValueError("Benchmark seed differs from paired plan.")
        if index > cfg["discard_rounds"]:
            if value["total_input_tokens"] != cfg["num_prompts"] * case["input_length"]:
                raise ValueError("Input token count differs from requested fixed-length workload.")
            rounds.append({"round": index, "completed": value["completed"],
                           "input_tokens": value["total_input_tokens"],
                           "output_tokens": value["total_output_tokens"],
                           "ttft_ms": value["mean_ttft_ms"],
                           "input_tps": value["total_input_tokens"] / value["duration"],
                           "duration_s": value["duration"],
                           "total_tps": value["total_token_throughput"]})
    return {"rounds": rounds, "ttft_ms": statistics.mean(x["ttft_ms"] for x in rounds),
            "input_tps": statistics.mean(x["input_tps"] for x in rounds),
            "duration_s": statistics.mean(x["duration_s"] for x in rounds)}


def collect_pair(root, pair, plan):
    result = {"id": pair["id"], "ratio": pair["ratio"], "replicate": pair["replicate"],
              "mode_order": pair["mode_order"], "status": "incomplete"}
    routes = {route["name"]: route for route in plan["routes"]}
    try:
        data = {case["mode"]: case_data(root, case, plan["settings"]["runtime"],
                                        routes[case["route"]]) for case in pair["cases"]}
        off, slot = data["off"], data["slot"]
        for left, right in zip(off["rounds"], slot["rounds"], strict=True):
            for key in ("round", "completed", "input_tokens", "output_tokens"):
                if left[key] != right[key]:
                    raise ValueError(f"Paired rounds differ in {key}.")
        result.update(status="complete", off=off, slot=slot,
                      off_ttft_ms=off["ttft_ms"], slot_ttft_ms=slot["ttft_ms"],
                      off_input_tps=off["input_tps"], slot_input_tps=slot["input_tps"],
                      ttft_log_ratio=math.log(off["ttft_ms"] / slot["ttft_ms"]),
                      throughput_log_ratio=math.log(slot["input_tps"] / off["input_tps"]))
        result["ttft_gain_pct"] = study.gain(result["ttft_log_ratio"], "ttft")
        result["throughput_gain_pct"] = study.gain(result["throughput_log_ratio"], "throughput")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def csv_file(path, records, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value)
            writer.writerow(row)


def render_svg(rows, path, margin):
    # A standalone scientific plot; no GPU/model dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    valid = [row for row in rows if "ttft_gain_pct" in row]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    xs = [row["ratio"] for row in valid]
    ax.plot(xs, [row["ttft_gain_pct"] for row in valid], "o-",
            color="#2563eb", label="TTFT reduction")
    ax.plot(xs, [row["throughput_gain_pct"] for row in valid], "s-",
            color="#ea580c", label="Input throughput gain")
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
    ax.axhline(margin, color="#16a34a", linewidth=1, linestyle=":",
               label=f"Target {margin:g}%")
    ax.set(xlabel="Target home-rank peak / mean (s)", ylabel="Paired geometric-mean gain (%)",
           title="DP8 exploratory crossover sweep")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(path, format="svg")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)



def summarize(root):
    root = Path(root)
    plan = study.read(root / "plan.json")
    study.validate_plan(plan)
    pairs = [collect_pair(root, pair, plan) for pair in plan["pairs"]]
    config = plan["settings"]
    rows = study.aggregate(pairs, config)
    result = {"schema": "dp8-crossover-summary-v1", "phase": plan["phase"],
              "settings": config, "plan_sha256": plan["plan_sha256"],
              "all_complete": all(pair["status"] == "complete" for pair in pairs),
              "pairs": pairs, "ratios": rows, "boundaries": study.boundaries(rows, config),
              "inference": "Exploratory only: two independent service pairs per ratio; no confidence claim.",
              "aggregation": "Geometric mean of paired speed ratios; rounds within one service are not independent repeats."}
    runtime.write_json(root / "summary.json", result)
    csv_file(root / "pairs.csv", pairs, ("id", "ratio", "replicate", "mode_order", "status", "reason",
                                        "off_ttft_ms", "slot_ttft_ms", "off_input_tps", "slot_input_tps",
                                        "ttft_gain_pct", "throughput_gain_pct"))
    csv_file(root / "curve.csv", rows, ("ratio", "planned_pairs", "complete_pairs", "status",
                                       "off_ttft_ms", "slot_ttft_ms", "off_input_tps", "slot_input_tps",
                                       "ttft_gain_pct", "throughput_gain_pct", "joint_win_fraction",
                                       "point_meets_margin", "confirmed_gain", "ttft_ci95_pct",
                                       "throughput_ci95_pct", "reason"))
    render_svg(rows, root / "curve.svg", config["practical_gain_pct"])
    lines = ["# DP8 MoonEP slot收益交叉区间：探索性结果", "",
             "每点为两对独立重启的off/slot服务；每台服务丢弃第1轮，统计第2/3轮。",
             "表中收益为配对速度比的几何平均，原始均值与逐对结果见CSV；不把同一服务内的轮次视作独立样本。",
             "收益判据：TTFT和输入吞吐均改善至少3%。样本不足以确认稳定阈值，不报告置信区间或统计显著性。",
             "", "| 目标s | 有效对数 | TTFT改善 | 输入吞吐改善 | 双指标同向获益比例 | ≥3%探索点 |",
             "|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if "ttft_gain_pct" not in row:
            lines.append(f"| {row['ratio']:.4f} | {row['complete_pairs']}/{row['planned_pairs']} | — | — | — | 数据不完整 |")
        else:
            lines.append(f"| {row['ratio']:.4f} | {row['complete_pairs']}/{row['planned_pairs']} | "
                         f"{row['ttft_gain_pct']:+.2f}% | {row['throughput_gain_pct']:+.2f}% | "
                         f"{row['joint_win_fraction']:.0%} | {'是' if row['point_meets_margin'] else '否'} |")
    lines += ["", "候选交叉区间（未达标→达标）：" + json.dumps(result["boundaries"]["candidate_margin_brackets"]),
              "观察到非单调/回落：" + str(result["boundaries"]["nonmonotonic_observed"]),
              "![收益曲线](curve.svg)", "",
              "阈值仅针对本机、固定两热点路由族、8K负载和当前KV/并发参数；不代表数值正确性认证或其它拓扑的普遍阈值。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    summary = summarize(args.session)
    print(f"Complete pairs: {sum(x['status']=='complete' for x in summary['pairs'])}/{len(summary['pairs'])}")
    print(json.dumps(summary["boundaries"], ensure_ascii=False))


if __name__ == "__main__":
    main()
