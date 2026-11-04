#!/usr/bin/env python3
"""Compare one-token text consistency; this is not a LongBench task accuracy score."""
import csv
import json
import sys
from pathlib import Path

def collect(session):
    grouped = {"off": {}, "slot": {}}
    with (session / "request_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            bucket = grouped[row["mode"]].setdefault(int(row["id"]), {})
            bucket[f'{row["case"]}/round_{row["round"]}'] = row["first_token_text"]
    assert len(grouped["off"]) == len(grouped["slot"]) == 316
    assert all(len(v) == 6 for mode in grouped.values() for v in mode.values())
    stable = {mode: {idx: next(iter(v.values())) for idx, v in rows.items()
                     if len(set(v.values())) == 1}
              for mode, rows in grouped.items()}
    both = set(stable["off"]) & set(stable["slot"])
    equal = sorted(i for i in both if stable["off"][i] == stable["slot"][i])
    different = sorted(both - set(equal))
    paired = []
    for pair in ("pair01", "pair02"):
        for round_no in (1, 2, 3):
            matches = sum(grouped["off"][i][f"{pair}_off/round_{round_no}"] ==
                          grouped["slot"][i][f"{pair}_slot/round_{round_no}"]
                          for i in grouped["off"])
            paired.append({"pair": pair, "round": round_no, "matches": matches,
                           "samples": 316, "measurement_round": round_no in (2, 3)})
    return {"session": str(session), "samples": 316, "repeats_per_mode": 6,
            "stable_off": len(stable["off"]), "stable_slot": len(stable["slot"]),
            "stable_both": len(both), "stable_both_equal_ids": equal,
            "stable_both_different_ids": different, "paired_rounds": paired}

session = Path(sys.argv[1])
result = collect(session)
result["limitation"] = (
    "Only one-token text is compared. Equal text does not establish equal token IDs "
    "or task accuracy. Batch scheduling can cause within-mode variation.")
if len(sys.argv) > 2:
    previous = collect(Path(sys.argv[2]))
    result["previous_run"] = previous
    result["previous_stable_mismatch_now_stable_match"] = sorted(
        set(previous["stable_both_different_ids"]) & set(result["stable_both_equal_ids"]))
(session / "FIRST_TOKEN_COMPARISON.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in
                  ("previous_run", "stable_both_equal_ids", "stable_both_different_ids")}))
