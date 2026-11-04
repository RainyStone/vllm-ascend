"""Collect archived measurements and render a standalone plot; no model calls."""
from __future__ import annotations
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
           title="DP8 adaptive bisection samples")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(path, format="svg")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)
