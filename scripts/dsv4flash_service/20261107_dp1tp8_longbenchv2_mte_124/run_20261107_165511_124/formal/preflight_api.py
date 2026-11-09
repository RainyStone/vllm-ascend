#!/usr/bin/env python3
"""Validate API tokenization and one longest request, outside formal measurements."""
import ast
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = Path('/mnt/share/x00893896/vllm_ascend_moonep_integration')
BASE = "http://127.0.0.1:8444"

def request(path, payload=None, timeout=20):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)

def main():
    mode, session = sys.argv[1:]
    assert mode in ("off", "slot")
    output = Path(session) / ("preflight_" + mode + ".json")
    name = "dsv4flash_longbench_tp8_" + mode
    began = time.monotonic()
    while time.monotonic() - began < 1500:
        try:
            models = request("/v1/models", timeout=5)["data"]
            if name in [x["id"] for x in models]:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        waited = int(time.monotonic()-began)
        if waited % 60 < 3:
            print("WAIT_API", mode, waited, flush=True)
        time.sleep(3)
    else:
        raise RuntimeError("Service not ready after 1500 seconds")
    path = WS / "benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
    tree = ast.parse(path.read_text())
    templates = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.keyword)
                 and n.arg=="prompt" and isinstance(n.value,ast.Constant)]
    assert len(templates)==1
    records = json.loads((ROOT/"data/longbenchv2_full_le128k.json").read_text())
    manifest = json.loads((ROOT/"data/selection_manifest.json").read_text())
    samples = manifest["samples"]
    smallest = min(samples, key=lambda x:x["prompt_tokens"])
    longest = max(samples, key=lambda x:x["prompt_tokens"])
    audit = []
    for entry in [smallest,longest]:
        prompt = templates[0].format(**records[entry["data_id"]])
        response = request("/tokenize",dict(model=name,prompt=prompt,add_special_tokens=True),timeout=300)
        count = response["count"]
        audit.append(dict(id=entry["id"],expected=entry["prompt_tokens"],service_tokens=count))
        assert count==entry["prompt_tokens"], audit
    prompt=templates[0].format(**records[longest["data_id"]])
    print("LONGEST_REQUEST_BEGIN",mode,longest["prompt_tokens"],"output=1",flush=True)
    start=time.monotonic()
    response=request("/v1/completions",dict(model=name,prompt=prompt,max_tokens=1,
                        stream=False,temperature=0,ignore_eos=True,seed=1024),timeout=1800)
    usage=response["usage"]
    report=dict(mode=mode,formal_measurement=False,tokenize_audit=audit,
                longest_id=longest["id"],usage=usage,elapsed_seconds=time.monotonic()-start,
                response=response)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    assert usage["prompt_tokens"]==longest["prompt_tokens"],usage
    assert usage["completion_tokens"]==1,usage
    print("PREFLIGHT_OK",mode,"input",usage["prompt_tokens"],"output",usage["completion_tokens"],
          "seconds",round(report["elapsed_seconds"],3),flush=True)

if __name__=="__main__":
    try: main()
    except Exception as error:
        if len(sys.argv)>2:
            (Path(sys.argv[2])/("preflight_"+sys.argv[1]+"_error.txt")).write_text(repr(error)+"\n")
        raise
