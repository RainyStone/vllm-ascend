"""Pure noisy-bisection controller. Does not assume an empirical curve is proven monotone."""
from __future__ import annotations

import math

import study

EPSILON = 1e-9


def load_settings(path=None):
    cfg = study.read(path or study.HERE / "settings.json")
    if cfg["bounds"] != [1.0, 1.75] or cfg["tolerance"] != 0.05:
        raise ValueError("Confirmed design is restricted to [1.00,1.75], tolerance 0.05.")
    if cfg["pairs_per_point"] != 2 or not cfg["stop_on_disagreement"]:
        raise ValueError("Exactly two pairs; disagreements must stop the search.")
    if cfg["objective"] != "joint" or cfg["practical_gain_pct"] != 3.0:
        raise ValueError("Both TTFT and input throughput must improve at least 3%.")
    r = cfg["runtime"]
    if (r["cards"] != list(range(8)) or r["input_lengths"] != [8192]
            or r["batched_tokens"] != [8192] or r["kv_cache_bytes"] != 4294967296
            or r["max_model_len"] != 8193 or r["max_concurrency"] != 16
            or r["num_prompts"] != 32 or r["output_length"] != 1
            or r["rounds"] != 3 or r["discard_rounds"] != 1
            or r["modes"] != ["off", "slot"] or not r["slot_optimizations"]):
        raise ValueError("Fixed validated DP8/8K workload settings changed.")
    return cfg


def strategy(cfg):
    lower, upper = cfg["bounds"]
    midpoints = math.ceil(math.log2((upper - lower) / cfg["tolerance"]))
    return {"bounds": cfg["bounds"], "tolerance": cfg["tolerance"],
            "first_point": upper, "second_point_if_upper_gains": lower,
            "first_midpoint_if_bracketed": (lower + upper) / 2,
            "max_midpoints": midpoints, "max_distinct_points": midpoints + 2,
            "max_service_starts": (midpoints + 2) * cfg["pairs_per_point"] * 2,
            "max_benchmark_calls": (midpoints + 2) * cfg["pairs_per_point"] * 2
                                   * (cfg["runtime"]["rounds"] + 1),
            "ideal_final_width": (upper - lower) / 2 ** midpoints,
            "point_policy": "both pairs gain -> gain; both fail -> no_gain; mixed -> stop",
            "assumption": "Approximately monotone gain predicate, not proved by sparse sampling.",
            "execution": "Deferred; no workloads until explicit --execute following user notification."}


def classify_point(ratio, pairs, cfg):
    result = {"ratio": ratio, "pairs": pairs, "status": "invalid",
              "reason": "Incomplete or invalid pair data."}
    if len(pairs) != cfg["pairs_per_point"] or any(p["status"] != "complete" for p in pairs):
        return result
    votes = [all(p[key] + EPSILON >= cfg["practical_gain_pct"]
                 for key in ("ttft_gain_pct", "throughput_gain_pct")) for p in pairs]
    result["votes"] = votes
    result["status"] = "gain" if all(votes) else "no_gain" if not any(votes) else "uncertain"
    result["reason"] = ("Both independent pairs meet the joint 3% criterion." if all(votes) else
                        "Neither independent pair meets the joint 3% criterion." if not any(votes) else
                        "Pairs disagree; bracket must not shrink.")
    for metric in ("ttft", "throughput"):
        logs = [p[metric + "_log_ratio"] for p in pairs]
        result[metric + "_gain_pct"] = study.gain(sum(logs) / len(logs), metric)
    return result


def next_action(history, cfg):
    lower, upper = cfg["bounds"]
    seen = {}
    for point in history:
        ratio = point["ratio"]
        if not math.isfinite(ratio) or not lower <= ratio <= upper or ratio in seen:
            raise ValueError("Out-of-range or duplicate measurement.")
        if point["status"] not in ("gain", "no_gain", "uncertain", "invalid"):
            raise ValueError("Unknown point classification.")
        seen[ratio] = point
    gains = [p["ratio"] for p in history if p["status"] == "gain"]
    misses = [p["ratio"] for p in history if p["status"] == "no_gain"]
    bracket = [max(misses), min(gains)] if gains and misses and max(misses) < min(gains) else None
    if any(p["status"] == "invalid" for p in history):
        return {"action": "stop", "status": "invalid_data", "bracket": bracket,
                "reason": "Invalid/incomplete data; no further branch selection."}
    if any(p["status"] == "uncertain" for p in history):
        return {"action": "stop", "status": "uncertain", "bracket": bracket,
                "pending_ratio": next(p["ratio"] for p in history if p["status"] == "uncertain"),
                "reason": "Pair decisions disagree. No extra repeat was authorized."}
    if any(g < m for g in gains for m in misses):
        return {"action": "stop", "status": "nonmonotonic", "bracket": None,
                "reason": "A smaller tested ratio gains but a larger tested ratio does not."}
    if upper not in seen:
        return {"action": "measure", "ratio": upper, "reason": "Test upper endpoint first."}
    if seen[upper]["status"] != "gain":
        return {"action": "stop", "status": "upper_not_gain", "bracket": None,
                "reason": "Upper endpoint does not gain; no bisection bracket established. "
                          "This does not prove every interior ratio has no gain."}
    if lower not in seen:
        return {"action": "measure", "ratio": lower, "reason": "Upper gains; test lower endpoint."}
    if seen[lower]["status"] == "gain":
        return {"action": "stop", "status": "lower_already_gain", "bracket": None,
                "reason": "The smallest allowed ratio already gains; no fail-to-gain bracket here."}
    if bracket is None:
        return {"action": "stop", "status": "no_bracket", "bracket": None}
    width = bracket[1] - bracket[0]
    if width <= cfg["tolerance"] + EPSILON:
        return {"action": "stop", "status": "bracket_found", "bracket": bracket, "width": width,
                "reason": "Exploratory candidate threshold in (low, high] conditional on approximate monotonicity."}
    if len(history) >= strategy(cfg)["max_distinct_points"]:
        return {"action": "stop", "status": "budget_reached", "bracket": bracket, "width": width}
    return {"action": "measure", "ratio": sum(bracket) / 2, "bracket": bracket, "width": width,
            "reason": "Measure midpoint; do not round the tested ratio."}
