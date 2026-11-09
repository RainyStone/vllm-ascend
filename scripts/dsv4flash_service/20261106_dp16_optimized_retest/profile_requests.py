#!/usr/bin/env python3
"""Separate, bounded diagnostic profiling; never part of formal AIS measurements."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from tests.test_home_gmm_npu import require_time

ROOT=Path(__file__).resolve().parent
def request(path,payload=None,timeout=1800,headers=None):
    data=b"" if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request("http://141.61.45.118:8445"+path,data=data,
                              headers={"Content-Type":"application/json",**(headers or {})})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as f:
        raw=f.read()
        return json.loads(raw) if raw else {}

def main():
    require_time()
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["off","slot_mte","slot_udma"],required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    prepared=ROOT/"data/longbenchv2_all503_midcut128k.json"
    manifest=json.loads((ROOT/"data/full503_manifest.json").read_text())
    raw=prepared.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest["prepared_sha256"]
    rows=json.loads(raw)
    template_path=Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py")
    tree=ast.parse(template_path.read_text())
    templates=[node.value.value for node in ast.walk(tree) if isinstance(node,ast.keyword)
               and node.arg=="prompt" and isinstance(node.value,ast.Constant)]
    assert len(templates)==1
    ordered=sorted(manifest["samples"],key=lambda x:(x["prompt_tokens"],x["source_index"]))
    chosen=[ordered[round(i*(len(ordered)-1)/15)] for i in range(16)]
    chosen.sort(key=lambda x:x["source_index"])
    assert len({x["source_index"] for x in chosen})==16
    prompts=[]
    for item in chosen:
        prompt=templates[0].format(**rows[item["source_index"]])
        assert hashlib.sha256(prompt.encode()).hexdigest()==item["prompt_sha256"]
        prompts.append((item,prompt))
    pin_by_index={item["source_index"]: rank for rank,(item,prompt) in enumerate(prompts)}
    expected_model="dsv4flash_dp16_longbench_"+args.mode
    def run_one(pair):
        item,prompt=pair
        began=time.monotonic()
        value=request("/v1/completions",dict(model=expected_model,prompt=prompt,
            max_tokens=1,temperature=0,seed=1024,ignore_eos=True,stream=False),headers={"X-data-parallel-rank":str(pin_by_index[item["source_index"]])})
        usage=value["usage"]
        assert usage["prompt_tokens"]==item["prompt_tokens"],(item,usage)
        assert usage["completion_tokens"]==1,usage
        return dict(source_index=item["source_index"],data_id=item["data_id"],target_dp_rank=pin_by_index[item["source_index"]],
            prompt_tokens=item["prompt_tokens"],prompt_sha256=item["prompt_sha256"],
            response=value,seconds=time.monotonic()-began)
    def wave(label):
        with ThreadPoolExecutor(max_workers=16) as pool:
            values=list(pool.map(run_one,prompts))
        return dict(label=label,successes=len(values),requests=values)
    # Diagnostic-only pre-run avoids attributing one-time operator loading to the
    # steady-state trace. These16 requests never enter the cold formal comparison.
    warmup=wave("DIAGNOSTIC_ONLY_pre_profile16")
    print("DIAGNOSTIC_PRE_RUN_DONE",args.mode,flush=True)
    request("/start_profile",timeout=180)
    try:
        profiled=wave("DIAGNOSTIC_ONLY_profile16")
    finally:
        request("/stop_profile",timeout=600)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(mode=args.mode,formal_result=False,
        profiler_config=dict(delay_iterations=0,max_iterations=3,ignore_frontend=True),
        note="WorkerProfiler stops after its bounded counter; count actual trace steps. These timings include instrumentation and are not formal performance.",
        prepared_sha256=manifest["prepared_sha256"],warmup=warmup,profiled=profiled),indent=2))
    print("DIAGNOSTIC_PROFILE_REQUESTS_DONE",args.mode,output,flush=True)
if __name__=="__main__":main()
