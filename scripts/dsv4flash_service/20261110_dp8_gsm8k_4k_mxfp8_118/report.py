"""Summarize formal GSM8K performance and separate diagnostic load replays."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rank_log_reader import describe, peak, read_rank

ROOT = Path(__file__).resolve().parent
MODES = ("off", "slot")


def round_result(phase, mode):
    root = ROOT / phase / "data" / phase / "aisbench"
    candidates = list(root.glob(f"*-{mode}-*/round_1/metrics.json"))
    if not candidates:
        return None
    path, = candidates
    metrics = json.loads(path.read_text())
    assert metrics["valid"] and metrics["requests"] == 1319 and metrics["input_tokens"] == 77756
    return path.parent, metrics


def formal_results():
    cases = {}
    for mode in MODES:
        result = round_result("formal", mode)
        if not result:
            continue
        directory, metrics = result
        raw = Path(metrics["raw_details"])
        summary = json.loads((raw.parent / "gsm8k-test.json").read_text())
        with (raw.parent / "gsm8k-test.csv").open() as stream:
            ttft, = [row for row in csv.DictReader(stream)
                     if row["Performance Parameters"] == "TTFT" and row["Stage"] == "total"]
        key = next(key for key in summary if key.lower() in ("request throughput", "request goodput"))
        metrics["p50_ttft_ms"] = float(ttft["Median"].split()[0].replace(",", ""))
        metrics["request_rps"] = float(str(summary[key]["total"]).split()[0].replace(",", ""))
        metrics["debug_log"] = str(directory / "aisbench_debug.log")
        cases[mode] = metrics
    keys = ("mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "input_tps", "request_rps")
    change = {key: 100 * (cases["slot"][key] / cases["off"][key] - 1) for key in keys} if len(cases) == 2 else {}
    return cases, change


def analyze_load(mode):
    result = round_result("diagnostic", mode)
    if not result:
        return None
    directory, _ = result
    before, after = (json.loads((directory / f"rank_boundary_{name}.json").read_text())
                     for name in ("before", "after"))
    assert before["directory"] == after["directory"]
    assert before["flush_every_steps"] == after["flush_every_steps"] == 1
    assert set(after["ranks"]) == set(map(str, range(8)))
    rank_data = []
    raw = []
    for rank in range(8):
        key = str(rank)
        begin = before["ranks"].get(key, {}).get("tail_max_step")
        end = after["ranks"][key]["tail_max_step"]
        # The last preflight step has not yet been flushed at the before snapshot.
        start = 0 if begin is None else int(begin) + 2
        path = Path(after["ranks"][key]["path"])
        assert path.parent == Path(after["directory"])
        rank_data.append(read_rank(path, rank, start, int(end), mode))
        raw.append(str(path))
    layers = set.intersection(*(data["layers"] for data in rank_data))
    assert layers and all(len(layers) == int(data["meta"]["moe_layers"]) for data in rank_data)
    views = ("compute",) if mode == "off" else ("owner", "compute")
    counters = {view: dict(ratios=[], weights=[], totals=np.zeros(8, dtype=np.int64),
                           busy=np.zeros(8, dtype=np.int64)) for view in views}
    methods = Counter()
    discarded = 0
    common = set.intersection(*(set(data["records"]) for data in rank_data))
    out = ROOT / f"{mode}_rank_windows.csv"
    with out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "layer", "method", "view", *[f"ep{r}_pairs" for r in range(8)],
                         "peak_mean_ratio"])
        for step in sorted(common):
            if any(not data["start"] <= step <= data["end"] for data in rank_data):
                continue
            rows = [data["records"][step] for data in rank_data]
            method = rows[0][0]
            if any(other != method for other, _ in rows):
                discarded += len(layers)
                continue
            contents = [content for _, content in rows]
            required = ("local",) if mode == "off" else ("global", "compute")
            for layer in sorted(layers):
                if any(content.get((layer, scope)) is None for scope in required for content in contents):
                    discarded += 1
                    continue
                if mode == "off":
                    load = {"compute": np.asarray([content[layer, "local"] for content in contents])}
                else:
                    load = {"owner": np.sum([content[layer, "global"] for content in contents], axis=0),
                            "compute": np.asarray([content[layer, "compute"] for content in contents])}
                if any(not vector.sum() for vector in load.values()):
                    discarded += 1
                    continue
                methods[method] += 1
                for view, vector in load.items():
                    ratio = peak(vector)
                    counter = counters[view]
                    counter["ratios"].append(ratio)
                    counter["weights"].append(int(vector.sum()))
                    counter["totals"] += vector
                    counter["busy"][vector == vector.max()] += 1
                    writer.writerow([step, layer, method, view, *vector, ratio])
    observed_keys = {"owner": "owner_observed", "compute": "owner_observed" if mode == "off" else "compute_observed"}
    analyzed = {}
    for view, data in counters.items():
        observed = np.sum([rank[observed_keys[view]] for rank in rank_data], axis=0)
        stats = describe(**data) if data["ratios"] else {}
        stats.update(observed_pairs_by_rank=list(map(int, observed)),
                     observed_cumulative_ratio=peak(observed),
                     observed_pairs=int(observed.sum()))
        analyzed[view] = stats
    return dict(views=analyzed, methods=dict(methods), discarded_windows=discarded,
                moe_layers=len(layers), rank_bounds=[[data["start"], data["end"]] for data in rank_data],
                reference_prefill_pairs=77756 * int(rank_data[0]["meta"]["topk"]) * len(layers),
                pad_pairs_subtracted=sum(data["pad_pairs"] for data in rank_data),
                raw_rank_logs=raw, window_csv=str(out))


def main():
    cases, change = formal_results()
    result = dict(host="141.61.45.118", formal=cases, slot_vs_off_pct=change)
    if "--formal-only" not in sys.argv:
        result["rank_diagnostic"] = {mode: analyze_load(mode) for mode in MODES}
    name = "FORMAL_RESULTS.json" if "--formal-only" in sys.argv else "RESULTS.json"
    (ROOT / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
