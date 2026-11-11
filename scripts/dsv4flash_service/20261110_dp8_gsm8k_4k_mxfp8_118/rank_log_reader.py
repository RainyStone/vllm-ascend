import collections
import csv
import json
import re
from pathlib import Path

import numpy as np


ROW = re.compile(r"^step=(\d+) layer=(\d+) method=(\S+)(?: scope=(\S+))? counts=\[([^\]]*)\]")
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

