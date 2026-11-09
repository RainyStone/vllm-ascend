#!/usr/bin/env python3
"""Require fresh certificates from both real nodes before an aisbench round."""
import json
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
mode,pair,output=sys.argv[1:4]
phase="post" if "--post" in sys.argv else "pre"
nodes=[]
for node in ("118","124"):
    p=root/"audits"/f"{pair}_{mode}_{node}{'_post' if phase=='post' else ''}.json"
    d=json.loads(p.read_text())
    assert d["phase"]==phase
    assert d["prefetch_stream_mode"]==0 and d["side_stream_disabled_verified"] and not d["side_stream_activation_verified"]
    assert d["valid"] and (d["mode"],d["pair"],d["node"])==(mode,pair,node)
    assert 0<=time.time()-d["captured_at"]<=90, "Run fresh audits on both nodes outside the timed round"
    nodes.append(d)
assert sorted(rank for d in nodes for rank in d["global_ranks"])==list(range(16))
assert nodes[0]["source_sha256"]==nodes[1]["source_sha256"]
p=Path(output);p.mkdir(parents=True,exist_ok=True)
(p/("cluster_mode_audit_post.json" if phase=="post" else "cluster_mode_audit.json")).write_text(json.dumps(dict(mode=mode,phase=phase,valid=True,nodes=nodes),indent=2))
print("CLUSTER_MODE_AUDIT_OK",mode,"16/16 real workers, both nodes")
