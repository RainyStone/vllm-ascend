#!/usr/bin/env python3
"""Summarize only validated cases from a specified execution directory."""
import csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=["pair01_off","pair01_slot","pair02_slot","pair02_off"]

def number(value):
    if isinstance(value,(int,float)):return float(value)
    match=re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",str(value).replace(",",""))
    if not match:raise ValueError("Not numeric: "+repr(value))
    return float(match.group())

def read_round(client,i):
    root=Path(client)/f"round_{i}"
    validation=json.loads((root/"validation.json").read_text())
    assert validation["valid"] and validation["requests"]==316
    candidates=list((root/"aisbench").rglob("longbenchv2-full-le128k.json"))
    assert len(candidates)==1,candidates
    summary_path=candidates[0]
    summary=json.loads(summary_path.read_text())
    total=lambda key:number(summary[key]["total"])
    duration_text = str(summary["Benchmark Duration"]["total"]).strip()
    assert duration_text.endswith((" ms", " s")), duration_text
    duration_seconds = number(duration_text) / (1000 if duration_text.endswith(" ms") else 1)
    assert total("Total Requests")==total("Success Requests")==316
    assert total("Failed Requests")==0 and total("Total Generated Tokens")==316
    with summary_path.with_suffix(".csv").open() as f:
        metrics=list(csv.DictReader(f))
    ttft=[r for r in metrics if r["Performance Parameters"]=="TTFT" and r["Stage"]=="total"]
    assert len(ttft)==1
    assert "ms" in ttft[0]["Average"],ttft
    raw=summary_path.parent/"longbenchv2-full-le128k_details.jsonl"
    predictions={}
    with raw.open() as stream:
        for line in stream:
            record=json.loads(line)
            predictions[int(record["id"])]=record.get("prediction")
    assert len(predictions)==316
    return dict(round=i,mean_ttft_ms=number(ttft[0]["Average"]),
                median_ttft_ms=number(ttft[0]["Median"]),p90_ttft_ms=number(ttft[0]["P90"]),
                p99_ttft_ms=number(ttft[0]["P99"]),input_tps=total("Input Token Throughput"),
                duration_seconds=duration_seconds,input_tokens=int(total("Total Input Tokens")),
                output_tokens=int(total("Total Generated Tokens")),requests=316,
                concurrency=total("Concurrency"),max_concurrency=total("Max Concurrency"),
                summary_file=str(summary_path),raw_file=str(raw),predictions=predictions)

