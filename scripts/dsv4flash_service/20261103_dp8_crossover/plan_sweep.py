#!/usr/bin/env python3
"""Create coarse/fine plans only. Never starts vLLM or probes NPU devices."""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import study


def interval_points(low, high, step):
    low, high, step = map(lambda x: Decimal(str(x)), (low, high, step))
    if not (Decimal("1") <= low < high <= Decimal("2.5")) or step <= 0:
        raise ValueError("Require 1 <= low < high <= 2.5 and a positive step.")
    values = [low]
    current = low + step
    while current < high:
        values.append(current)
        current += step
    values.append(high)
    return [float(value) for value in values]


def make_fine(config, summary, explicit_range=None):
    if summary["settings"] != config:
        raise ValueError("Settings differ from source session; cannot silently mix studies.")
    if not summary["all_complete"]:
        raise ValueError("Source session is incomplete; finish/resolve it before planning fine scan.")
    if explicit_range is None:
        info = summary["boundaries"]
        if info["nonmonotonic_observed"]:
            raise ValueError("Non-monotonic/multiple crossings: review the curve and pass --range LOW HIGH.")
        brackets = info["candidate_margin_brackets"]
        if len(brackets) == 1:
            low, high = brackets[0]
        elif not brackets and summary["ratios"][0].get("point_meets_margin"):
            low, high = [row["ratio"] for row in summary["ratios"][:2]]
        else:
            raise ValueError("No bracket found within tested range; do not invent a threshold. Review data.")
    else:
        low, high = explicit_range
    ratios = interval_points(low, high, config["fine_step"])
    if len(ratios) > config["fine_max_points"]:
        raise ValueError("Fine range is too wide; choose a narrower explicit interval.")
    return study.make_plan(config, "fine", ratios, config["fine_pairs"], study.digest(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("coarse", "fine"))
    parser.add_argument("--settings", type=Path, default=study.HERE / "settings.json")
    parser.add_argument("--from-session", type=Path)
    parser.add_argument("--range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = study.settings(args.settings)
    if config.get("superseded_by"):
        parser.error("Superseded design; use ../20261103_dp8_bisect/run_bisect.py")
    if args.phase == "coarse":
        if args.from_session or args.range:
            parser.error("coarse does not take --from-session/--range")
        plan = study.make_plan(config, "coarse", config["coarse_ratios"], config["coarse_pairs"])
    else:
        if args.from_session is None:
            parser.error("fine requires --from-session containing summary.json")
        summary = study.read(args.from_session / "summary.json")
        plan = make_fine(config, summary, args.range)
    output = args.out or study.HERE / "plans" / f"{args.phase}.json"
    study.write_new(output, plan)
    print(f"Plan: {output.resolve()}")
    print(f"Ratios: {plan['ratios']}")
    print(f"Independent pairs/ratio: {plan['independent_pairs_per_ratio']}")
    print(f"Service starts: {plan['service_starts']}; benchmark calls: {plan['benchmark_calls']}")
    print("No service or benchmark executed.")


if __name__ == "__main__":
    main()
