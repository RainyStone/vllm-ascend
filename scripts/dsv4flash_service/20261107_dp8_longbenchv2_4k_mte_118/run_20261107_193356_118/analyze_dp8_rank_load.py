import collections
import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ABBR = "longbenchv2-full-le128k"
RANKS = set(range(8))
ROW = re.compile(r"^step=(\d+) layer=(\d+) method=(\S+)(?: scope=(\S+))? counts=\[([^\]]*)\]")
OFF_ALLTOALL = "AlltoAllCommImpl"
SLOT_COMM = "MoonEPCommImpl"

def boundary(mode):
    candidates = list((ROOT/"diagnostic/data/diagnostic/aisbench").glob(f"*-{mode}-*/round_1"))
    round_dir, = [p for p in candidates if (p/"validation.json").is_file()]
    validation = json.loads((round_dir/"validation.json").read_text())
    assert validation["valid"] and validation["requests"]==316, validation
    before = json.loads((round_dir/"rank_boundary_before.json").read_text())
    after = json.loads((round_dir/"rank_boundary_after.json").read_text())
    assert before["directory"]==after["directory"]
    assert set(map(int,after["ranks"]))==RANKS
    return round_dir,before,after

def read_rank(path, ep, start, end, mode):
    meta = {}
    known_layers = set()
    records = {}
    owner_observed = np.zeros(8,dtype=np.int64)
    compute_observed = np.zeros(8,dtype=np.int64)
    pad_pairs = 0
    method_rows = collections.Counter()
    with path.open() as stream:
        for line in stream:
            if line.startswith("# meta "):
                meta.update(dict(re.findall(r"(\w+)=([^\s]+)",line)))
            elif line.startswith("# layer "):
                layer = int(re.search(r"layer_ord=(\d+)",line)[1])
                known_layers.add(layer)
                logical = re.search(r"log2phy=(\[[^\]]*\]|None)",line)
                if logical and logical[1]!="None":
                    values = json.loads(logical[1])
                    assert values == list(range(len(values))), (path,layer,"nonidentity logical placement")
                mapping = re.search(r"expert_map=(\[[^\]]*\]|None)",line)
                if mapping and mapping[1]!="None":
                    values = json.loads(mapping[1])
                    assert len(values)==256, (path,layer,len(values))
                    assert [i for i,v in enumerate(values) if v>=0]==list(range(ep*32,(ep+1)*32)), (path,layer,ep)
            elif line.startswith("step="):
                match = ROW.match(line)
                assert match,(path,line[:120])
                step,layer,method,scope,text = match.groups()
                step,layer = int(step),int(layer)
                if step > end and scope!="pad":
                    break
                scope = scope or ("local" if mode=="off" else "global")
                valid = start <= step <= end
                if scope=="pad":
                    if valid:
                        pad_pairs += int(np.fromstring(text,sep=",",dtype=np.int64).sum())
                    continue
                item = records.setdefault(step,[method,{}])
                assert item[0]==method, (path,step,item[0],method)
                key = (layer,scope)
                assert key not in item[1],(path,step,key)
                if not valid:
                    item[1][key]=None
                    continue
                counts = np.fromstring(text,sep=",",dtype=np.int64)
                assert counts.size==(32 if scope=="local" else 256) and counts.min()>=0, (path,step,key,counts.size)
                if scope=="global":
                    value = tuple(map(int,counts.reshape(8,32).sum(axis=1)))
                    owner_observed += value
                else:
                    value = int(counts.sum())
                    if scope=="compute":
                        compute_observed[ep] += value
                    else:
                        owner_observed[ep] += value
                item[1][key]=value
                method_rows[(method,scope)] += 1
    assert int(meta["ep_rank"])==ep and int(meta["ep_size"])==8, meta
    assert int(meta["global_experts"])==256 and int(meta["local_experts"])==32, meta
    assert int(meta["redundant_experts"])==0, meta
    return dict(meta=meta, records=records, layers=known_layers, start=start, end=end,
                pad_pairs=pad_pairs, owner_observed=owner_observed,compute_observed=compute_observed,
                methods={f"{m}:{s}":c for (m,s),c in method_rows.items()})

def describe(ratios, weights, totals, busy):
    a=np.asarray(ratios,dtype=float)
    return dict(windows=len(a), mean=float(a.mean()), weighted_mean=float(np.average(a,weights=weights)),
                p50=float(np.percentile(a,50)), p95=float(np.percentile(a,95)), max=float(a.max()),
                aligned_pairs_by_rank=list(map(int,totals)), aligned_cumulative_ratio=peak(totals),
                busiest_counts_ties_included=list(map(int,busy)))

def peak(values):
    return float(8*values.max()/values.sum()) if values.sum() else None

def analyze(mode):
    round_dir,before,after = boundary(mode)
    rank_data = []
    logs = []
    for rank in sorted(RANKS):
        begin = before["ranks"].get(str(rank),{}).get("tail_max_step")
        finish = after["ranks"][str(rank)]["tail_max_step"]
        assert finish is not None,(mode,rank,after)
        # A completed preflight step can still occupy the unflushed window.
        # Retain a conservative prefix gap when locating the diagnostic interval.
        start = (begin if begin is not None else -1) + int(before["flush_every_steps"]) + 1
        path=Path(after["ranks"][str(rank)]["path"])
        assert path.parent==Path(after["directory"]) and path.is_file(),path
        logs.append(str(path))
        rank_data.append(read_rank(path,rank,start,int(finish),mode))
    layers = set.intersection(*(p["layers"] for p in rank_data))
    assert layers and all(len(layers)==int(p["meta"]["moe_layers"]) for p in rank_data),(mode,layers)
    method = OFF_ALLTOALL if mode=="off" else SLOT_COMM
    # ALLTOALL and SHMEM are collectives over the eight ranks. Match the
    # ordinal of that same method starting at service startup, independently
    # of local step IDs or the flush boundary of each rank.
    sequences = [[(step,content) for step,(comm,content) in sorted(p["records"].items()) if comm==method] for p in rank_data]
    assert all(sequences),(mode,method,[len(s) for s in sequences])
    if mode=="slot":
        step_sequences=[[step for step,_ in seq] for seq in sequences]
        prefix = min(map(len,step_sequences))
        assert all(s[:prefix]==step_sequences[0][:prefix] for s in step_sequences), (mode,"SHMEM raw invocation IDs disagree")
    labels=("owner",) if mode=="off" else ("owner","compute")
    ratio_data={k:dict(ratios=[],weights=[],totals=np.zeros(8,dtype=np.int64),busy=np.zeros(8,dtype=np.int64)) for k in labels}
    path = ROOT/f"{mode}_rank_windows.csv"
    skipped=0
    aligned_steps=0
    with path.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["method","collective_index","layer","view",*[f"ep{r}_step" for r in range(8)],
                         *[f"ep{r}_pairs" for r in range(8)],"total_pairs","peak_mean_ratio","busiest_ep_ranks"])
        for ordinal,group in enumerate(zip(*sequences)):
            step_ids,contents=zip(*group)
            if any(not p["start"]<=step<=p["end"] for p,step in zip(rank_data,step_ids)):
                continue
            aligned_steps += 1
            for layer in sorted(layers):
                required=("local",) if mode=="off" else ("global","compute")
                if any(content.get((layer,scope)) is None for content in contents for scope in required):
                    skipped+=1
                    continue
                if mode=="off":
                    owner=np.asarray([content[layer,"local"] for content in contents],dtype=np.int64)
                    views={"owner":owner}
                else:
                    owner=np.sum([content[layer,"global"] for content in contents],axis=0,dtype=np.int64)
                    compute=np.asarray([content[layer,"compute"] for content in contents],dtype=np.int64)
                    views={"owner":owner,"compute":compute}
                if any(v.sum()==0 for v in views.values()):
                    skipped+=1
                    continue
                for label,v in views.items():
                    ratio=peak(v)
                    total=int(v.sum())
                    busy=np.flatnonzero(v==v.max())
                    writer.writerow([method,ordinal,layer,label,*step_ids,*v,total,ratio,";".join(map(str,busy))])
                    r=ratio_data[label]
                    r["ratios"].append(ratio)
                    r["weights"].append(total)
                    r["totals"]+=v
                    r["busy"][busy]+=1
    assert ratio_data["owner"]["ratios"], (mode,method,aligned_steps,skipped)
    observed={k:np.sum([p[f"{k}_observed"] for p in rank_data],axis=0,dtype=np.int64) for k in labels}
    topk = {int(p["meta"]["topk"]) for p in rank_data}
    assert len(topk)==1,topk
    manifest=json.loads((ROOT/"diagnostic/data/selection_manifest.json").read_text())
    expected_pairs=sum(s["prompt_tokens"] for s in manifest["samples"])*topk.pop()*len(layers)
    views={k:describe(**data) for k,data in ratio_data.items()}
    for k,v in observed.items():
        view=views[k]
        view.update(observed_pairs_by_rank=list(map(int,v)), observed_cumulative_ratio=peak(v),
                    aligned_fraction_of_observed=sum(view["aligned_pairs_by_rank"])/int(v.sum()) if v.sum() else None)
    return dict(phase="diagnostic",mode=mode,aligned_method=method,aligned_steps=aligned_steps,discarded_windows=skipped,
                num_moe_layers=len(layers),prefill_reference_pairs=expected_pairs,
                observed_route_pairs_fraction_of_reference=int(observed["owner"].sum())/expected_pairs,
                scope_pad_pairs=sum(p["pad_pairs"] for p in rank_data), views=views,
                rank_local_bounds_and_methods=[dict(start=p["start"],end=p["end"],methods=p["methods"]) for p in rank_data],
                raw_logs=logs, debug_log=str(round_dir/"aisbench_debug.log"), window_csv=str(path))

if __name__=="__main__":
    result={}
    for mode in ("off","slot"):
        result[mode]=analyze(mode)
        (ROOT/"RANK_LOAD_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result[mode].items() if k not in ("rank_local_bounds_and_methods","raw_logs")},ensure_ascii=False),flush=True)

