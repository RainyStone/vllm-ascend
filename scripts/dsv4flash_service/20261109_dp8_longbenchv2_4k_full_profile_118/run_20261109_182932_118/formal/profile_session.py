import ast
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
API = "http://127.0.0.1:8444"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, payload=None, timeout=1800):
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(API+path, data=data, method="POST",
                                 headers={"Content-Type":"application/json"})
    with OPENER.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def event(output, step, **fields):
    row = dict(at=datetime.datetime.now().astimezone().isoformat(), step=step, **fields)
    with (output/"profile_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)


def warmup(mode):
    data = json.loads((ROOT/"data/longbenchv2_full_le128k.json").read_text())
    manifest = json.loads((ROOT/"data/selection_manifest.json").read_text())
    items = sorted(manifest["samples"],key=lambda v:int(v["data_id"]))
    assert len(items)==316
    path = WORKSPACE/"benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
    templates = [n.value.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.keyword)
                 and n.arg=="prompt" and isinstance(n.value,ast.Constant)]
    template, = templates

    def one(item):
        prompt = template.format(**data[int(item["data_id"])])
        assert hashlib.sha256(prompt.encode()).hexdigest()==item["prompt_sha256"]
        result = post("/v1/completions",dict(model=f"dsv4flash_longbench_dp8_4k_{mode}",prompt=prompt,max_tokens=1,
                                             temperature=0,ignore_eos=True,seed=1024,stream=False))
        usage = result["usage"]
        assert usage["prompt_tokens"]==item["prompt_tokens"] and usage["completion_tokens"]==1
        return {"data_id":item["data_id"],"prompt_tokens":usage["prompt_tokens"],"completion_tokens":1}

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(one,items[:16]))


def main():
    mode,folder,separator,*command = sys.argv[1:]
    assert mode in ("off","slot") and separator=="--"
    output = Path(folder)
    event(output,"warmup_start",mode=mode,requests=16)
    warm = warmup(mode)
    (output/"warmup_validated.json").write_text(json.dumps(warm,indent=2)+"\n")
    event(output,"warmup_validated",mode=mode,requests=len(warm))
    post("/start_profile",timeout=180)
    event(output,"profile_started",mode=mode,delay_iterations=0,max_iterations=0)
    try:
        subprocess.run(command,check=True)
        event(output,"aisbench_complete",mode=mode)
    finally:
        event(output,"profile_stop_requested",mode=mode)
        post("/stop_profile",timeout=3600)
        event(output,"profile_stopped",mode=mode)


if __name__=="__main__":
    main()

