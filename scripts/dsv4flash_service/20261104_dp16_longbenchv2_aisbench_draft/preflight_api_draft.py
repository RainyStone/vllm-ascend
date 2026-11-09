#!/usr/bin/env python3
"""Use only on a separate validation service, never on the six measured startups."""
import ast
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_REVIEWED=True
ROOT=Path(__file__).resolve().parent
BASE="http://141.61.45.118:8445"
TEMPLATE=Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py")
def request(path,payload=None,timeout=30):
    body=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(BASE+path,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as f:
        return json.load(f)
def main():
    if not SCRIPT_REVIEWED:raise SystemExit("Draft only; no API request sent.")
    if "--validation-service" not in sys.argv:raise SystemExit("Only a separate validation service may receive inference probes.")
    mode,pair=[x for x in sys.argv[1:] if x!="--validation-service"]
    assert mode in ("off","slot_mte","slot_udma") and pair in ("pair01","pair02")
    model="dsv4flash_dp16_longbench_"+mode
    assert model in [x["id"] for x in request("/v1/models")["data"]]
    manifest=json.loads((ROOT/"data/full503_manifest.json").read_text())
    raw=(ROOT/"data/longbenchv2_all503_midcut128k.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest["prepared_sha256"]
    rows=json.loads(raw);assert len(rows)==503
    tree=ast.parse(TEMPLATE.read_text())
    templates=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.keyword)
               and n.arg=="prompt" and isinstance(n.value,ast.Constant)]
    assert len(templates)==1
    selected=[min(manifest["samples"],key=lambda x:x["prompt_tokens"]),
              max(manifest["samples"],key=lambda x:x["prompt_tokens"])]
    checks=[]
    for item in selected:
        prompt=templates[0].format(**rows[item["source_index"]])
        assert hashlib.sha256(prompt.encode()).hexdigest()==item["prompt_sha256"]
        tokenized=request("/tokenize",dict(model=model,prompt=prompt),timeout=180)
        assert tokenized["count"]==item["prompt_tokens"]
        began=time.monotonic()
        result=request("/v1/completions",dict(model=model,prompt=prompt,max_tokens=1,
            temperature=0,seed=1024,ignore_eos=True,stream=False),timeout=1900)
        assert result["usage"]["prompt_tokens"]==item["prompt_tokens"]
        assert result["usage"]["completion_tokens"]==1
        checks.append(dict(data_id=item["data_id"],prompt_tokens=item["prompt_tokens"],
                           prompt_sha256=item["prompt_sha256"],seconds=time.monotonic()-began))
    out=ROOT/"data/preflight";out.mkdir(parents=True,exist_ok=True)
    (out/f"{pair}_{mode}_api.json").write_text(json.dumps(dict(valid=True,mode=mode,pair=pair,checks=checks),indent=2))
    print("PREFLIGHT_OK",mode,pair)
if __name__=="__main__":main()
