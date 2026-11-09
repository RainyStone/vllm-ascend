#!/usr/bin/env python3
"""Validate tiny-test evidence; never certifies planning/dispatch/expert prefetch."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("directory",type=Path)
p.add_argument("--backend",choices=["mte","udma"],required=True)
p.add_argument("--world-size",type=int,choices=[2,16],required=True)
a=p.parse_args()
rows=[json.loads((a.directory/f"rank{i}.json").read_text()) for i in range(a.world_size)]
assert sorted(r["rank"] for r in rows)==list(range(a.world_size))
assert {r["node_ip"] for r in rows}=={"141.61.45.118","141.61.45.124"}
expected="CombineUrma.compute_weighted" if a.backend=="udma" else "Combine.compute_weighted"
for r in rows:
    assert r["status"]=="passed" and r["backend"]==a.backend
    assert r["world_size"]==a.world_size and not r["model_loaded"] and r["all_valid_routes_remote"]
    assert r["node_ip"]==("141.61.45.118" if r["rank"]<a.world_size//2 else "141.61.45.124")
    assert len(r["called_backend_methods"])>=22 and set(r["called_backend_methods"])=={expected}
    assert len(r["checks"])>=39
    assert all(c["mismatches"]==0 and c["finite"] and c["max_abs"]==0 for c in r["checks"])
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.directory.glob("rank*.json")}
result=dict(valid=True,backend=a.backend,world_size=a.world_size,valid_ranks=len(rows),
    cross_node_weighted_combine_validated=True,precision_passed=True,model_loaded=False,
    planning_dispatch_validated=False,expert_prefetch_validated=False,
    evidence_sha256=hashes,
    limit="Only weighted/unweighted combine transfer and arithmetic; not an end-to-end Slot certificate.")
(a.directory/"combine_validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
