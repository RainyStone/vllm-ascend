#!/usr/bin/env python3
"""Flush the final full503 diagnostic step without mixing control requests into it.

Requires PAR_FLUSH_EVERY=1. A recorder flushes completed step j when step j+1
begins. At idle after503, latest dumped step is j-1 and j is buffered. A short
control request on every DP rank causes that j to be written. Seal only <=j;
all control steps remain excluded. Preserve raw snapshots and boundary evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from tests.test_home_gmm_npu import require_time

def metadata_and_tail(path):
    with path.open("rb") as f:
        first=f.read(16384).decode()
        f.seek(max(0,path.stat().st_size-16384))
        tail=f.read().decode()
    meta={}
    for line in (first+"\n"+tail).splitlines():
        if line.startswith("# meta "):meta.update(dict(re.findall(r"(\w+)=([^\s]+)",line)))
    steps=[int(v) for v in re.findall(r"(?:^|\n)step=(\d+)",tail)]
    assert steps,("No flushed rows",path)
    return meta,max(steps)

def main():
    require_time()
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["off","slot_mte","slot_udma"],required=True)
    p.add_argument("--routing",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    files=sorted(Path(args.routing).glob("moe_par_*_rank*.log"))
    assert len(files)==16
    before={}
    for path in files:
        meta,latest=metadata_and_tail(path)
        rank=int(meta["rank"])
        assert int(meta["flush_every"])==1
        before[rank]=dict(path=str(path),metadata=meta,latest_dumped=latest,sealed_max_step=latest+1)
    assert set(before)==set(range(16))
    time.sleep(1)
    for rank,d in before.items():
        _,latest=metadata_and_tail(Path(d["path"]))
        assert latest==d["latest_dumped"],("Traffic is not idle; do not choose a boundary",rank)
    target=Path(args.output)
    assert not target.exists(),"Do not overwrite a sealed diagnostic boundary"
    target.mkdir(parents=True)
    (target/"BOUNDARY_BEFORE_CONTROL.json").write_text(json.dumps(before,indent=2))
    def control(rank):
        payload=dict(model="dsv4flash_dp16_longbench_"+args.mode,prompt="hello",
                     max_tokens=1,temperature=0,ignore_eos=True,seed=1024,stream=False)
        req=urllib.request.Request("http://141.61.45.118:8445/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type":"application/json","X-data-parallel-rank":str(rank)})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=1800) as f:
            result=json.load(f)
        assert result["usage"]["prompt_tokens"]<8192
        assert result["usage"]["completion_tokens"]==1
        return dict(target_dp_rank=rank,response=result,diagnostic_control_only=True)
    with ThreadPoolExecutor(max_workers=16) as pool:controls=list(pool.map(control,range(16)))
    (target/"EXCLUDED_CONTROL_REQUESTS.json").write_text(json.dumps(controls,indent=2))
    sealed=target/"sealed";sealed.mkdir()
    raw=target/"raw_after_flush";raw.mkdir()
    summary=[]
    for rank,d in sorted(before.items()):
        path=Path(d["path"]);raw_data=path.read_bytes()
        (raw/path.name).write_bytes(raw_data)
        cap=d["sealed_max_step"]
        kept=[];last_layers={};header_layers=set()
        for line in raw_data.decode().splitlines(True):
            if line.startswith("# meta partial_steps="):continue  # running ledger may include control traffic
            if line.startswith("# layer "):
                found=re.search(r"layer_ord=(\d+)",line)
                if found:header_layers.add(int(found.group(1)))
            if line.startswith("step="):
                step=int(re.search(r"step=(\d+)",line).group(1))
                if step>cap:continue
                if step==cap:
                    layer=int(re.search(r"layer=(\d+)",line).group(1))
                    scope_match=re.search(r"scope=(\w+)",line)
                    scope=scope_match.group(1) if scope_match else d["metadata"]["counts_scope"].split("+")[0]
                    last_layers.setdefault(scope,set()).add(layer)
            kept.append(line)
        primary="local" if args.mode=="off" else "global"
        assert last_layers.get(primary)==header_layers,(rank,"Final dataset step was not fully flushed",last_layers,header_layers)
        if args.mode!="off":assert last_layers.get("compute")==header_layers
        # Carry the original pre-control warning counters without claiming its
        # running ledger totals describe this newly sealed range.
        after_meta,_=metadata_and_tail(path)
        warnings={k:str(max(int(d["metadata"].get(k,"0")),int(after_meta.get(k,"0")))) for k in
                  ("partial_steps","negative_cells_clamped","pad_notes_misplaced")}
        kept.append("# meta "+" ".join(f"{k}={v}" for k,v in warnings.items())+
                    f" sealed_max_step={cap} control_steps_excluded=1\n")
        data="".join(kept).encode();(sealed/path.name).write_bytes(data)
        summary.append(dict(rank=rank,sealed_max_step=cap,rows=sum(x.startswith("step=") for x in kept),
            sealed_sha256=hashlib.sha256(data).hexdigest(),
            raw_sha256=hashlib.sha256(raw_data).hexdigest(),last_complete_layers=len(header_layers)))
    (target/"SEAL.json").write_text(json.dumps(dict(mode=args.mode,valid=True,
        traffic="full503 diagnostic only",excluded_control_requests=16,
        rule="latest pre-control flushed step+1; all rank's final observed MoE layers checked complete",
        ranks=summary),indent=2))
    print("FULL503_RANKLOAD_SEALED",sealed,flush=True)
if __name__=="__main__":main()
