"""Select three length-matched workloads from measured DP16 owner histograms.

Routing input: {"prepared_sha256": "...", "samples": [
    {"data_id": 0, "owner_pairs": [16 integers, summed over 43 MoE layers]}
]}
Only real prompt routes belong in owner_pairs. Run on the same model, expert
mapping and prepared prompts as the benchmark. Route measurements may cover
a subset of the prepared dataset; feasibility then applies to that subset.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


RANKS = 16
PAIRS_PER_TOKEN = 43 * 6
TARGETS = (1.2, 1.4, 1.6)


def select_workloads(lengths, owner_pairs, request_count, tolerance, time_limit):
    """Find integer multiplicities with an identical exact length histogram."""
    candidates = len(lengths)
    width = candidates + RANKS
    variable_count = len(TARGETS) * width
    # Scaling keeps the coefficients near one for the long prompts.
    normalized_pairs = owner_pairs / owner_pairs.sum(axis=1).max()
    totals = normalized_pairs.sum(axis=1)
    rows, lower, upper = [], [], []

    def constraint(terms, low, high):
        row = np.zeros(variable_count)
        for positions, values in terms:
            row[positions] = values
        rows.append(row)
        lower.append(low)
        upper.append(high)

    length_groups = [np.flatnonzero(lengths == length) for length in np.unique(lengths)]
    for group, target in enumerate(TARGETS):
        start = group * width
        sample_vars = slice(start, start + candidates)
        peak_vars = slice(start + candidates, start + width)
        constraint([(sample_vars, 1)], request_count, request_count)
        constraint([(peak_vars, 1)], 1, 1)
        if group:
            for indices in length_groups:
                constraint([(indices, 1), (indices + start, -1)], 0, 0)
        for rank in range(RANKS):
            scaled_rank_load = RANKS * normalized_pairs[:, rank]
            constraint(
                [(sample_vars, scaled_rank_load - (target + tolerance) * totals)],
                -np.inf, 0,
            )
            # A binary variable picks one rank meeting the lower edge. For
            # unpicked ranks the relaxed bound follows from totals <= 1.
            relax = (target - tolerance) * request_count
            constraint(
                [(sample_vars, scaled_rank_load - (target - tolerance) * totals),
                 (start + candidates + rank, -relax)],
                -relax, np.inf,
            )

    limits = np.tile(
        np.r_[np.full(candidates, request_count), np.ones(RANKS)], len(TARGETS)
    )
    solution = milp(
        c=np.zeros(variable_count),
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), limits),
        constraints=LinearConstraint(np.array(rows), lower, upper),
        options={"time_limit": time_limit},
    )
    if solution.x is None:
        raise ValueError(f"No usable selection for this measured pool: {solution.message}")
    counts = np.rint(solution.x).astype(np.int64).reshape(len(TARGETS), width)
    return counts[:, :candidates]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=503)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args()
    if args.requests <= 0 or not 0 < args.tolerance < 0.1 or args.time_limit <= 0:
        parser.error("Expected positive request count, 0 < tolerance < 0.1, and positive time limit")
    if args.output.exists():
        parser.error("The output directory already exists")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    source_bytes = args.prepared.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if manifest["prepared_sha256"] != source_sha or routing["prepared_sha256"] != source_sha:
        raise ValueError("Prepared dataset SHA differs from the manifest or measured routing")
    original = json.loads(source_bytes)
    samples = {sample["data_id"]: sample for sample in manifest["samples"]}
    measurements = sorted(routing["samples"], key=lambda row: row["data_id"])
    ids, lengths, pairs = [], [], []
    for row in measurements:
        sample_id = row["data_id"]
        counts = row["owner_pairs"]
        if type(sample_id) is not int or not 0 <= sample_id < len(original):
            raise ValueError(f"Invalid data_id: {sample_id}")
        length = samples[sample_id]["prompt_tokens"]
        if len(counts) != RANKS or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError(f"Invalid owner histogram for sample {sample_id}")
        if length <= 0 or sum(counts) != length * PAIRS_PER_TOKEN:
            raise ValueError(f"Sample {sample_id} has incomplete or extra routing pairs")
        ids.append(sample_id)
        lengths.append(length)
        pairs.append(counts)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected a non-empty pool with one routing row per data_id")

    lengths, pairs = np.array(lengths), np.array(pairs, dtype=np.int64)
    ceiling = float((RANKS * pairs.max(axis=1) / pairs.sum(axis=1)).max())
    print(f"Measured candidates: {len(ids)}. Natural-routing upper bound: {ceiling:.6f}")
    if ceiling < max(TARGETS) - args.tolerance:
        raise ValueError("The measured candidate pool cannot reach the top target by resampling")

    selections = select_workloads(lengths, pairs, args.requests, args.tolerance, args.time_limit)
    verified, reference_histogram = [], None
    for target, counts in zip(TARGETS, selections):
        if (counts < 0).any() or int(counts.sum()) != args.requests:
            raise ValueError("Selection failed integer request-count validation")
        indices = np.repeat(np.arange(len(ids)), counts)
        length_histogram = dict(sorted(Counter(map(int, lengths[indices])).items()))
        if reference_histogram is not None and length_histogram != reference_histogram:
            raise ValueError("Selection failed exact length matching")
        reference_histogram = length_histogram
        rank_totals = counts @ pairs
        ratio = RANKS * int(rank_totals.max()) / int(rank_totals.sum())
        if abs(ratio - target) > args.tolerance + 1e-9:
            raise ValueError(f"Predicted ratio {ratio:.6f} is outside the band for {target}")
        verified.append({
            "target": target,
            "predicted_ratio": ratio,
            "owner_pairs": rank_totals.tolist(),
            "source_data_ids": [ids[index] for index in indices],
        })

    for entry in verified:
        entry["dataset"] = f"longbenchv2_owner_{entry['target']:.1f}.json"
    report = {
        "prepared_sha256": source_sha,
        "measured_candidates": len(ids),
        "natural_routing_upper_bound": ceiling,
        "scope": "owner-rank loads summed across all prompt tokens and all 43 MoE layers",
        "verification": "offline prediction from recorded routes",
        "tolerance": args.tolerance,
        "requests_per_group": args.requests,
        "input_tokens_per_group": sum(length * n for length, n in reference_histogram.items()),
        "exact_length_histogram": reference_histogram,
        "groups": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lb_imbalance_", dir=args.output.parent) as scratch:
        staging = Path(scratch) / "workloads"
        staging.mkdir()
        for entry in verified:
            rows = [original[index] for index in entry["source_data_ids"]]
            with (staging / entry["dataset"]).open("x", encoding="utf-8") as output:
                json.dump(rows, output, ensure_ascii=False)
        (staging / "selection.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(args.output)


if __name__ == "__main__":
    main()
