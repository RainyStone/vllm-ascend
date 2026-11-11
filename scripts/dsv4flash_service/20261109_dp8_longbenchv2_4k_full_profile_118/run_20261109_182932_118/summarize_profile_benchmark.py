#!/usr/bin/env python3
"""Report end-to-end AISBench data for the two profiler-enabled rounds."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118"))
from metrics import read_round
from summarize_formal import percentile_95

def main():
    stats = {}
    for mode in ("off", "slot"):
        client, = (ROOT / "formal/data/formal/aisbench").glob(f"*-{mode}-*")
        case = read_round(client, 1)
        case.pop("predictions")
        p95, raw_mean = percentile_95(Path(case["raw_file"]))
        assert abs(raw_mean - case["mean_ttft_ms"]) < 0.11, (raw_mean, case)
        case["p95_ttft_ms"] = p95
        case["debug_log"] = str(client / "round_1/aisbench_debug.log")
        assert case["input_tokens"] == 18_518_962
        stats[mode] = case
    keys = ("mean_ttft_ms", "p95_ttft_ms", "input_tps", "duration_seconds")
    stats["slot_vs_off_pct"] = {
        key: 100 * (stats["slot"][key] / stats["off"][key] - 1) for key in keys
    }
    (ROOT / "PROFILED_AISBENCH_RESULTS.json").write_text(json.dumps(stats, indent=2) + "\n")
    for mode, case in stats.items():
        print(mode, {key: case[key] for key in keys})

if __name__ == "__main__":
    main()
