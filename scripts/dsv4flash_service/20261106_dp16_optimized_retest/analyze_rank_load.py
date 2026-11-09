#!/usr/bin/env python3
"""Whole-run rank max/mean; no summing per-step maxima or step alignment assumed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np

def peak_mean(values):
    values=np.asarray(values,dtype=np.int64)
    total=int(values.sum())
    return None if total==0 else len(values)*int(values.max())/total

def analyze(root,world=16):
    routing=np.zeros(world,dtype=np.int64)
    compute=np.zeros(world,dtype=np.int64)
    bylayer_route=defaultdict(lambda:np.zeros(world,dtype=np.int64))
    bylayer_compute=defaultdict(lambda:np.zeros(world,dtype=np.int64))
    metadata={};seen=set();ranks=set();row_counts=defaultdict(int);files=[]
    expert_route=None;expert_compute=None;layer_expert_route={};layer_expert_compute={}
    for path in sorted(Path(root).rglob("moe_par_*_rank*.log")):
        meta={}
        rank=None
        with path.open() as stream:
            for line in stream:
                if line.startswith("# meta "):
                    meta.update(dict(re.findall(r"(\w+)=([^\s]+)",line)))
                    if "rank" in meta:
                        rank=int(meta["rank"])
                    continue
                if not line.startswith("step="):continue
                assert rank is not None,(path,"Missing rank header")
                fields=dict(re.findall(r"(\w+)=([^\s]+)",line.split("counts=",1)[0]))
                scope=fields.get("scope",meta.get("counts_scope","local").split("+")[0])
                if scope=="pad":continue
                assert scope in ("local","global","compute"),(path,scope)
                match=re.search(r"counts=\[([^\]]*)\]",line)
                assert match,(path,line[:100])
                counts=np.fromstring(match.group(1),dtype=np.int64,sep=",")
                assert counts.size and np.all(counts>=0)
                key=(rank,int(fields["step"]),int(fields["layer"]),scope)
                assert key not in seen,("Duplicate route rows / reused dump directory",key)
                seen.add(key)
                layer=key[2]
                width=counts.size*world if scope=="local" else counts.size
                if expert_route is None:
                    expert_route=np.zeros(width,dtype=np.int64);expert_compute=np.zeros(width,dtype=np.int64)
                assert expert_route.size==width
                if layer not in layer_expert_route:
                    layer_expert_route[layer]=np.zeros(width,dtype=np.int64);layer_expert_compute[layer]=np.zeros(width,dtype=np.int64)
                if scope=="global":
                    expert_route+=counts;layer_expert_route[layer]+=counts
                elif scope=="compute":
                    expert_compute+=counts;layer_expert_compute[layer]+=counts
                else:
                    sl=slice(rank*counts.size,(rank+1)*counts.size)
                    expert_route[sl]+=counts;expert_compute[sl]+=counts
                    layer_expert_route[layer][sl]+=counts;layer_expert_compute[layer][sl]+=counts
                row_counts[(rank,scope)]+=1
                if scope=="global":
                    assert counts.size%world==0
                    owner=counts.reshape(world,-1).sum(axis=1)
                    routing+=owner;bylayer_route[layer]+=owner
                elif scope=="compute":
                    compute[rank]+=counts.sum();bylayer_compute[layer][rank]+=counts.sum()
                else:
                    # Native off: the count vector is already local to its owner.
                    routing[rank]+=counts.sum();bylayer_route[layer][rank]+=counts.sum()
                    compute[rank]+=counts.sum();bylayer_compute[layer][rank]+=counts.sum()
        assert rank is not None
        assert rank not in ranks,("More than one dump per rank",rank)
        ranks.add(rank);metadata[rank]=meta;files.append(str(path))
    assert ranks==set(range(world)),("Missing ranks",ranks)
    warnings=[]
    for rank,meta in metadata.items():
        for key in ("partial_steps","negative_cells_clamped","pad_notes_misplaced"):
            value=int(meta.get(key,"0"))
            if value:warnings.append(dict(rank=rank,key=key,value=value))
    layers=sorted(set(bylayer_route)|set(bylayer_compute))
    per_layer=[dict(layer=layer,routing_owner_pairs=bylayer_route[layer].tolist(),
        compute_pairs=bylayer_compute[layer].tolist(),
        routing_owner_peak_mean=peak_mean(bylayer_route[layer]),
        compute_peak_mean=peak_mean(bylayer_compute[layer])) for layer in layers]
    return dict(world=world,scope="whole recorded diagnostic run; sum loads per rank first, then max/mean",
        routing_owner_pairs=routing.tolist(),compute_pairs=compute.tolist(),
        routing_owner_peak_mean=peak_mean(routing),compute_peak_mean=peak_mean(compute),
        total_routing_pairs=int(routing.sum()),total_compute_pairs=int(compute.sum()),
        layer_count=len(layers),per_layer=per_layer,metadata=metadata,warnings=warnings,
        expert_routing_pairs=expert_route.tolist(),expert_compute_pairs=expert_compute.tolist(),
        expert_pair_max_abs_delta=int(np.abs(expert_compute-expert_route).max()),
        per_layer_expert_pair_max_abs_delta={str(k):int(np.abs(layer_expert_compute[k]-layer_expert_route[k]).max()) for k in layers},
        row_counts={f"rank{rank}:{scope}":n for (rank,scope),n in row_counts.items()},
        files=files,step_aligned_ratio_computed=False,
        limitations="Diagnostic traffic only, excluded from formal timing. Check completeness and compare total pairs to dataset tokens x topk x observed MoE layers; do not silently remove service dummy work or dropped pairs.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    result=analyze(args.root)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("world","routing_owner_peak_mean","compute_peak_mean",
        "total_routing_pairs","total_compute_pairs","layer_count","warnings")},indent=2))
if __name__=="__main__":main()
