"""Pure routing and frozen point-plan helpers; no GPU imports."""
from __future__ import annotations
import hashlib
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



def gain(log_ratio, metric):
    return 100 * (1 - math.exp(-log_ratio)) if metric == "ttft" else 100 * math.expm1(log_ratio)



def validate_plan(plan):
    raw = dict(plan)
    expected = raw.pop("plan_sha256")
    if digest(raw) != expected:
        raise ValueError("Plan hash mismatch.")
    for item in plan["routes"]:
        if digest(item["text"].encode()) != item["sha256"]:
            raise ValueError("Route hash mismatch.")
