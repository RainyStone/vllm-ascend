#!/usr/bin/env python3
"""Read and validate one saved main-stream-only round; sends no requests."""
import csv
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ABBR="longbenchv2-all503-midcut128k"
CASES=["pair01_off", "pair01_slot_mte", "pair02_off", "pair02_slot_mte", "pair01_slot_udma", "pair02_slot_udma"]
def number(value):
    if isinstance(value,(int,float)):return float(value)
    found=re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",str(value).replace(",",""))
    if not found:raise ValueError(value)
    return float(found.group())
def read_round(client,i,expected_input,mode,pair):
    folder=client/f"round_{i}"
    check=json.loads((folder/"validation.json").read_text())
    assert check["valid"] and check["requests"]==503
    audit=json.loads((folder/"cluster_mode_audit.json").read_text())
    assert audit["mode"]==mode and audit["phase"]=="pre"
    assert all(n["pair"]==pair for n in audit["nodes"])
    post=json.loads((folder/"cluster_mode_audit_post.json").read_text())
    assert post["valid"] and post["phase"]=="post" and post["mode"]==mode
    assert all(n["pair"]==pair for n in post["nodes"])
    assert sorted(r for n in post["nodes"] for r in n["global_ranks"])==list(range(16))
    assert [(n["node"],n["service_pid"]) for n in audit["nodes"]]==[(n["node"],n["service_pid"]) for n in post["nodes"]]
    assert all(n["prefetch_stream_mode"]==0 and n["side_stream_disabled_verified"] and not n["side_stream_activation_verified"] for n in post["nodes"])
    assert audit["valid"] and len(audit["nodes"])==2
    assert sorted(r for d in audit["nodes"] for r in d["global_ranks"])==list(range(16))
    paths=list((folder/"aisbench").rglob(ABBR+".json"));assert len(paths)==1,paths
    path=paths[0];data=json.loads(path.read_text());total=lambda k:number(data[k]["total"])
    assert total("Total Requests")==total("Success Requests")==total("Total Generated Tokens")==503
    assert total("Failed Requests")==0 and total("Total Input Tokens")==expected_input
    assert total("Max Concurrency")==16
    with path.with_suffix(".csv").open() as f:csvrows=list(csv.DictReader(f))
    ttft=[r for r in csvrows if r["Performance Parameters"]=="TTFT" and r["Stage"]=="total"]
    assert len(ttft)==1 and "ms" in ttft[0]["Average"]
    return dict(round=i,requests=503,mean_ttft_ms=number(ttft[0]["Average"]),
        p90_ttft_ms=number(ttft[0]["P90"]),p99_ttft_ms=number(ttft[0]["P99"]),
        input_tps=total("Input Token Throughput"),input_tokens=expected_input,
        raw_jsonl=str(path.with_name(ABBR+"_details.jsonl")),summary=str(path))
