"""Pure planning and statistics helpers. No GPU libraries or network calls."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
EXPERTS = 256
TOP_K = 6
RANKS = 8
HOT_EXPERTS = 2


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different content: {path}")
    else:
        with path.open("x") as stream:
            stream.write(text)


def settings(path=None):
    value = read(path or HERE / "settings.json")
    cfg = value["runtime"]
    if cfg["cards"] != list(range(RANKS)) or cfg["modes"] != ["off", "slot"]:
        raise ValueError("Requires eight cards and an off/slot comparison.")
    if cfg["rounds"] <= cfg["discard_rounds"] or cfg["discard_rounds"] < 0:
        raise ValueError("No retained rounds.")
    if cfg["input_lengths"] != [8192] or cfg["batched_tokens"] != [8192]:
        raise ValueError("This study holds both input and token budget at 8192.")
    if cfg["kv_cache_bytes"] != 4294967296 or cfg["max_model_len"] != 8193:
        raise ValueError("This study holds 4 GiB KV and max-model-len=8193 fixed.")
    if cfg["num_prompts"] != 32 or cfg["max_concurrency"] != 16 or cfg["output_length"] != 1:
        raise ValueError("This study fixes 32 prompts, concurrency 16, output 1.")
    if value.get("objective", "joint") not in ("joint", "ttft", "throughput"):
        raise ValueError("Unknown objective.")
    if not 0 <= value["hot_rank"] < RANKS:
        raise ValueError("Invalid hot rank.")
    if value["practical_gain_pct"] < 0:
        raise ValueError("Invalid gain margin.")
    if value.get("exploratory_only") and any(value[k] != 2 for k in ("coarse_pairs", "fine_pairs")):
        raise ValueError("Confirmed lightweight design requires exactly two pairs per point.")
    for key in ("coarse_pairs", "fine_pairs"):
        if not isinstance(value[key], int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    return value


def ratio_key(ratio):
    return format(float(ratio), ".10f").rstrip("0").rstrip(".").replace(".", "p")


def route(ratio, hot_rank=0):
    ratio = float(ratio)
    per_rank = EXPERTS // RANKS
    # Sparse two-hot profile; all remaining experts share one background value.
    target_share = ratio * 100 / RANKS
    other = EXPERTS - HOT_EXPERTS
    cold_in_hot_rank = per_rank - HOT_EXPERTS
    hot_share = (target_share - cold_in_hot_rank * 100 / other) / (
        HOT_EXPERTS - cold_in_hot_rank * HOT_EXPERTS / other)
    if not math.isfinite(ratio) or ratio < 1 or hot_share > 100 / TOP_K + 1e-10:
        raise ValueError("Ratio exceeds the fixed two-hot family's feasible range.")
    if not 0 <= hot_rank < RANKS:
        raise ValueError("Invalid hot rank.")
    hot_ids = [hot_rank * per_rank, hot_rank * per_rank + per_rank // 2]
    background = (100 - HOT_EXPERTS * hot_share) / other
    shares = [hot_share if index in hot_ids else background for index in range(EXPERTS)]
    rank_shares = [math.fsum(shares[i:i + per_rank]) for i in range(0, EXPERTS, per_rank)]
    text = (f"# E=256 K=6 EP=8; fixed two hot experts on rank {hot_rank}.\n"
            f"# Target home-rank peak/mean={ratio:.12f}; not post-slot compute imbalance.\n"
            + "".join(f"{index}:{hot_share:.15f}\n" for index in hot_ids))
    finite = {}
    for tokens in (16, 128, 1024, 8192, 65536):
        ideal = [share * tokens * TOP_K / 100 for share in shares]
        counts = [math.floor(x) for x in ideal]
        residual = tokens * TOP_K - sum(counts)
        order = sorted(range(EXPERTS), key=lambda i: (-(ideal[i] - counts[i]), i))
        for index in order[:residual]:
            counts[index] += 1
        if max(counts) > tokens:
            # The tested range ends at 3.0, below the two-hot cap.
            raise ValueError("Integer apportionment exceeds per-token expert cap.")
        rank_counts = [sum(counts[i:i + per_rank]) for i in range(0, EXPERTS, per_rank)]
        finite[str(tokens)] = max(rank_counts) / (tokens * TOP_K / RANKS)
    return {"name": "s" + ratio_key(ratio), "target_ratio": ratio,
            "home_rank_peak_to_mean": max(rank_shares) / (100 / RANKS),
            "rank_route_percent": rank_shares, "hot_experts": hot_ids,
            "hot_expert_share_pct": hot_share,
            "text": text, "filename": f"route_s{ratio_key(ratio)}.txt",
            "sha256": digest(text.encode()), "finite_token_ratio": finite}


def make_plan(config, phase, ratios, pairs, parent_sha256=None):
    values = sorted(set(float(x) for x in ratios))
    if len(values) != len(ratios) or not values or pairs < 1:
        raise ValueError("Unique ratios and positive pair count required.")
    routes = [route(x, config["hot_rank"]) for x in values]
    randomizer = random.Random(config["plan_seed"])
    planned_pairs = []
    for replicate in range(1, pairs + 1):
        indices = list(range(len(routes)))
        randomizer.shuffle(indices)
        for index in indices:
            item = routes[index]
            pair_id = f"{phase}_s{ratio_key(item['target_ratio'])}_p{replicate:02d}"
            modes = ["off", "slot"] if (replicate - 1 + index) % 2 == 0 else ["slot", "off"]
            cases = [{"id": pair_id + "_" + mode, "pair_id": pair_id,
                      "route": item["name"], "mode": mode,
                      "target_ratio": item["target_ratio"], "replicate": replicate,
                      "input_length": 8192, "batched_tokens": 8192}
                     for mode in modes]
            planned_pairs.append({"id": pair_id, "ratio": item["target_ratio"],
                                  "replicate": replicate, "route": item["name"],
                                  "mode_order": modes, "cases": cases})
    result = {"schema": "dp8-crossover-plan-v1", "phase": phase, "settings": config,
              "parent_summary_sha256": parent_sha256, "routes": routes,
              "pairs": planned_pairs, "ratios": values,
              "service_starts": len(planned_pairs) * 2,
              "benchmark_calls": len(planned_pairs) * 2 * (config["runtime"]["rounds"] + 1),
              "independent_pairs_per_ratio": pairs}
    result["plan_sha256"] = digest(result)
    return result


def validate_plan(plan):
    raw = dict(plan)
    expected = raw.pop("plan_sha256")
    if digest(raw) != expected:
        raise ValueError("Plan hash mismatch.")
    for item in plan["routes"]:
        if digest(item["text"].encode()) != item["sha256"]:
            raise ValueError("Route hash mismatch.")


def gain(log_ratio, metric):
    return 100 * (1 - math.exp(-log_ratio)) if metric == "ttft" else 100 * math.expm1(log_ratio)


def percentile(values, quantile):
    position = (len(values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] * (high - position) + values[high] * (position - low) if low != high else values[low]


def bootstrap(values, minimum_pairs=5, samples=10000, seed=20261103):
    if len(values) < minimum_pairs:
        return None
    n = len(values)
    if n ** n <= samples:
        means = [statistics.mean(values[index] for index in indices)
                 for indices in itertools.product(range(n), repeat=n)]
    else:
        rng = random.Random(seed)
        means = [statistics.mean(rng.choices(values, k=n)) for _ in range(samples)]
    means.sort()
    return [percentile(means, 0.025), percentile(means, 0.975)]


def point_meets(row, config):
    keys = (("ttft", "throughput") if config["objective"] == "joint"
            else (config["objective"],))
    return all(row[key + "_gain_pct"] > 0 and row[key + "_gain_pct"] >= config["practical_gain_pct"]
               for key in keys)


def aggregate(pairs, config):
    rows = []
    all_ratios = sorted(set(pair["ratio"] for pair in pairs))
    for ratio in all_ratios:
        group = [pair for pair in pairs if pair["ratio"] == ratio]
        complete = [pair for pair in group if pair["status"] == "complete"]
        row = {"ratio": ratio, "planned_pairs": len(group), "complete_pairs": len(complete),
               "status": "incomplete", "confirmed_gain": False}
        if len(complete) != len(group):
            row["reason"] = "Incomplete pair(s); no threshold classification."
            rows.append(row)
            continue
        ttft = [pair["ttft_log_ratio"] for pair in complete]
        throughput = [pair["throughput_log_ratio"] for pair in complete]
        for metric, values in (("ttft", ttft), ("throughput", throughput)):
            row[metric + "_gain_pct"] = gain(statistics.mean(values), metric)
            interval = bootstrap(values, config["minimum_confirm_pairs"],
                                 config["bootstrap_samples"], config["bootstrap_seed"])
            row[metric + "_ci95_pct"] = None if interval is None else [gain(x, metric) for x in interval]
        keys = ("ttft", "throughput") if config["objective"] == "joint" else (config["objective"],)
        row["joint_win_fraction"] = sum(
            all(pair[key + "_log_ratio"] > 0 for key in keys) for pair in complete) / len(complete)
        row["point_meets_margin"] = point_meets(row, config)
        row["confirmed_gain"] = (
            not config.get("exploratory_only", False)
            and len(complete) >= config["minimum_confirm_pairs"]
            and row["point_meets_margin"]
            and row["joint_win_fraction"] >= config["minimum_joint_win_fraction"]
            and all(row[key + "_ci95_pct"] is not None and row[key + "_ci95_pct"][0] > 0 for key in keys))
        row["status"] = "confirmed_gain" if row["confirmed_gain"] else (
            "exploratory_positive" if row["point_meets_margin"] else "not_confirmed_gain")
        for metric in ("off_ttft_ms", "slot_ttft_ms", "off_input_tps", "slot_input_tps"):
            row[metric] = statistics.mean(pair[metric] for pair in complete)
        rows.append(row)
    return rows


def boundaries(rows, config):
    complete = [row for row in rows if "point_meets_margin" in row]
    rising, falling, zero = [], [], []
    for left, right in zip(rows, rows[1:]):
        if "point_meets_margin" not in left or "point_meets_margin" not in right:
            continue
        if left["point_meets_margin"] != right["point_meets_margin"]:
            (rising if right["point_meets_margin"] else falling).append([left["ratio"], right["ratio"]])
        if left["ttft_gain_pct"] <= 0 < right["ttft_gain_pct"]:
            zero.append([left["ratio"], right["ratio"]])
    confirmed = [row["ratio"] for row in complete if row["confirmed_gain"]]
    exploratory = [row["ratio"] for row in complete if row["point_meets_margin"]]
    return {"smallest_exploratory_margin_ratio": min(exploratory) if exploratory else None,"candidate_margin_brackets": rising, "falling_margin_brackets": falling,
            "ttft_zero_crossing_brackets": zero,
            "nonmonotonic_observed": bool(falling or len(rising) > 1),
            "smallest_confirmed_tested_ratio": min(confirmed) if confirmed else None,
            "interpretation": "Discrete tested points only; no universal or monotonic threshold assumed."}
