"""Recompute preserved operator results without source snapshots or NPU execution."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = {"host": "141.61.45.124", "scope": "historical operator results", "runs": {}}
for world in (2, 8):
    ranks = [json.loads(p.read_text()) for p in sorted(
        (root / f"results/operator_dp{world}").glob("rank*.json"))]
    assert len(ranks) == world and all(d["status"] == "passed" for d in ranks)
    checks = [x for d in ranks for x in d["records"] if "max_abs" in x and "elements" in x]
    assert all(x["finite"] and x["mismatches"] == 0 for x in checks)
    native = [x for x in checks if x["name"].endswith("vs_native")
              or x["name"] == "real_planning_dispatch_weighted_roundtrip"]
    timings = []
    for item in ranks[0]["timings"]:
        assert item["regression_percent"] <= 2.0
        timings.append({k: item[k] for k in (
            "S", "K", "T", "kind", "medians_ms", "regression_percent",
            "max_rank_samples_ms", "peak_allocated_bytes")})
    old = [x for d in ranks for x in d["records"]
           if x.get("name", "").endswith("old_rounding_evidence")]
    summary["runs"][str(world)] = {
        "ranks_passed": len(ranks), "checks": len(checks),
        "native_comparison_elements": sum(x["elements"] for x in native),
        "native_mismatches": sum(x["mismatches"] for x in native),
        "all_check_elements": sum(x["elements"] for x in checks),
        "all_check_mismatches": sum(x["mismatches"] for x in checks),
        "old_rounding_mismatches": sum(x["mismatches"] for x in old),
        "old_rounding_elements": sum(x["elements"] for x in old),
        "overhead_gate_passed": True, "timings": timings,
    }
for label in ("operator_skipdup_guard", "operator_installed_smoke"):
    ranks = [json.loads(p.read_text()) for p in (root / "results" / label).glob("rank*.json")]
    assert len(ranks) == 2 and all(d["status"] == "passed" for d in ranks)
    summary[label] = "passed"
historical = json.loads((root / "results/summary.json").read_text())
assert summary["runs"] == historical["runs"], "Metrics differ from historical summary"
target = root / "results/recomputed_operator_summary.json"
target.write_text(json.dumps(summary, indent=2))
print(f"Historical metrics reproduced exactly: {target}")
