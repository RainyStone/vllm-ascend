#!/usr/bin/env python3
"""Read the captured Ascend device events without launching workers."""
import bisect
import collections
import csv
import json
import mmap
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).resolve().parent
MODES = ("off", "slot")
RANKS = set(range(8))


@dataclass(slots=True)
class Event:
    name: str
    tid: int
    start: float
    end: float

    @property
    def duration(self):
        return self.end-self.start


def load_trace(path):
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        with memoryview(mapped) as data:
            obj = orjson.loads(data)
    events = obj["traceEvents"] if isinstance(obj, dict) else obj
    names = {e["pid"]:e.get("args",{}).get("name") for e in events
             if e.get("ph")=="M" and e.get("name")=="process_name"}
    hardware = {k for k,v in names.items() if v=="Ascend Hardware"}
    assert hardware,(path,names)
    selected = [e for e in events if e.get("ph")=="X" and e.get("pid") in hardware and "dur" in e and "ts" in e]
    assert selected,path
    base = min(Decimal(str(e["ts"])) for e in selected)
    out = []
    for e in selected:
        t = float(Decimal(str(e["ts"]))-base)
        out.append(Event(e["name"],e["tid"],t,t+float(e["dur"])))
    steps = [e["name"] for e in events if e.get("ph")=="X" and e.get("name","").startswith("ProfilerStep#")]
    return sorted(out,key=lambda e:e.start),steps


def slice_between(events, starts, lo, hi):
    return events[bisect.bisect_left(starts,lo-0.02):bisect.bisect_left(starts,hi-0.02)]


def only(events, predicate):
    found=[e for e in events if predicate(e.name.lower())]
    return found[0] if len(found)==1 else None


def sum_where(events,predicate):
    return sum(e.duration for e in events if predicate(e.name.lower()))


def read_rank(mode, rank, path, emit):
    hw, steps = load_trace(path)
    grouped = [x for x in hw if "groupedmatmul" in x.name.lower()]
    assert grouped,path
    streams=collections.Counter(x.tid for x in grouped)
    main_tid=streams.most_common(1)[0][0]
    main=[x for x in hw if x.tid==main_tid]
    main_start=[x.start for x in main]
    gates=[x for x in main if x.name=="MoeGatingTopKHash"]
    assert gates,(path,streams)
    hw_start=[x.start for x in hw]
    routers=[x for x in main if x.name=="aclnnMatmul_MatMulV3Common_MatMulV3"]
    router_starts=[x.start for x in routers]
    prefetch=[x for x in hw if "prefetch_weight" in x.name.lower()]
    pf_start=[x.start for x in prefetch]
    attention=[x for x in hw if x.name=="KvQuantSparseAttnSharedkv"]
    attention_starts=[x.start for x in attention]
    types=collections.Counter()
    skipped=collections.Counter()
    observed=collections.Counter(x.name for x in hw)
    valid=0
    for i,gate in enumerate(gates):
        next_gate=gates[i+1].start if i+1<len(gates) else main[-1].end+0.1
        span=slice_between(main,main_start,gate.end,next_gate)
        g=[x for x in span if "groupedmatmul" in x.name.lower()]
        if len(g)!=2:
            skipped[f"expected_two_grouped_matmul_got_{len(g)}"]+=1
            continue
        g1,g2=g
        post=slice_between(main,main_start,g2.end,next_gate)
        pre=slice_between(main,main_start,gate.end,g1.start)
        activation=slice_between(main,main_start,g1.end,g2.start)
        if mode=="off":
            unperm=[x for x in post if "moefinalizeroutingv2" in x.name.lower()]
            if len(unperm)==2:
                typ="native_alltoall"
                endpoint=unperm[-1].end
                extra=dict(native_dispatch_permute_us=sum_where(pre,lambda n:"moeinitrouting" in n or "tokenpermute" in n),
                           native_combine_local_unpermute_us=unperm[0].duration,
                           native_combine_collective_span_us=unperm[1].start-unperm[0].end,
                           native_combine_final_unpermute_us=unperm[1].duration)
            else:
                candidate=[x for x in post if "moedistributecombine" in x.name.lower()]
                if len(candidate)==1:
                    typ="native_mc2"
                    endpoint=candidate[0].end
                    extra=dict(native_mc2_combine_kernel_us=candidate[0].duration)
                else:
                    skipped["native_combine_marker_not_unique"]+=1
                    continue
        else:
            p=only(pre,lambda n:n=="planning_kernel")
            d=only(pre,lambda n:"moonep_dispatch" in n)
            pk=only(pre,lambda n:n=="moonep_mxfp8_pack_kernel")
            up=only(pre,lambda n:n=="moonep_mxfp8_unpack_kernel")
            pub=only(post,lambda n:n=="moonep_publish_kernel")
            c=only(post,lambda n:"combine" in n and "weighted" in n)
            if not all((p,d,pk,up,pub,c)):
                skipped["slot_missing_stage_marker"]+=1
                continue
            if not (gate.end-0.02<=p.start<=pk.start<=d.start<=up.start<=g1.start and g2.end-0.02<=pub.start<=c.start):
                skipped["slot_stage_order"]+=1
                continue
            typ="slot_mte"
            endpoint=c.end
            extra=dict(planning_kernel_us=p.duration,pack_kernel_us=pk.duration,
                       dispatch_kernel_us=d.duration,unpack_kernel_us=up.duration,
                       fused_publish_kernel_us=pub.duration,weighted_combine_kernel_us=c.duration,
                       slot_dispatch_prepare_gap_us=p.start-gate.end,
                       slot_publish_to_combine_gap_us=c.start-pub.end)
            pf=slice_between(prefetch,pf_start,p.end-1,g1.start+1)
            if pf and len({x.tid for x in pf})==1 and pf[0].tid!=main_tid:
                pf_end=max(x.end for x in pf)
                extra.update(prefetch_calls=len(pf),prefetch_active_us=sum(x.duration for x in pf),
                    prefetch_span_us=pf_end-pf[0].start,
                    prefetch_overlaps_dispatch_kernel_us=sum(max(0,min(x.end,d.end)-max(x.start,d.start)) for x in pf),
                    prefetch_tail_after_dispatch_us=max(0,pf_end-d.end),
                    prefetch_tail_after_unpack_us=max(0,pf_end-up.end))
            else:
                skipped["slot_prefetch_marker_unmatched"]+=1

        prev=gates[i-1].end if i else main[0].start
        sparse=slice_between(attention,attention_starts,prev,gate.start)
        row=dict(mode=mode,rank=rank,gate_ordinal=i,path=typ,local_sparse_attention=bool(sparse),
                 sparse_attention_kernel_us=sum(x.duration for x in sparse),
                 topk_us=gate.duration,dispatch_span_us=g1.start-gate.end,
                 input_quant_kernel_us=sum_where(pre,lambda n:n=="dynamicmxquant"),
                 gemm1_us=g1.duration,activation_quant_span_us=g2.start-g1.end,
                 activation_quant_kernel_us=sum_where(activation,lambda n:"swiglugroupquant" in n),
                 gemm2_us=g2.duration,combine_span_us=endpoint-g2.end,
                 routed_wall_us=endpoint-gate.start,**extra)
        for tag,lo,hi in (("dispatch",gate.end,g1.start),("combine",g2.end,endpoint)):
            comm=slice_between(hw,hw_start,lo,hi)
            row[f"{tag}_hccl_alltoall_task_us"]=sum_where(comm,lambda n:"alltoall" in n)
        index=bisect.bisect_left(router_starts,gate.start)-1
        if index>=0 and routers[index].start>=prev:
            router=routers[index]
            row["router_gemm_us"]=router.duration
            tail=slice_between(main,main_start,endpoint,next_gate)
            add=next((x for x in tail if x.name=="aclnnAdd_AddAiCore_Add"),None)
            shared=[x for x in tail if add and x.start<add.start and
                    (x.name=="DynamicMxQuant" or "aclnnQuantMatmulV5_" in x.name)]
            if add and len(shared)>=3:
                row["shared_expert_span_us"]=shared[-1].end-shared[0].start
                row["routed_shared_merge_us"]=add.duration
                row["router_to_merge_us"]=add.end-router.start
        parts=("topk_us","dispatch_span_us","gemm1_us","activation_quant_span_us","gemm2_us","combine_span_us")
        assert abs(sum(row[k] for k in parts)-row["routed_wall_us"])<0.05, (mode,rank,i)
        emit(row)
        valid+=1
        types[typ]+=1
    return dict(rank=rank,trace=str(path),trace_bytes=path.stat().st_size,
                hardware_event_count=len(hw),profiler_step_markers=len(steps),gate_count=len(gates),
                parsed_calls=valid,paths=dict(types),skipped=dict(skipped),
                key_operators={k:v for k,v in observed.items() if re.search(r"moe|alltoall|prefetch_weight|groupedmatmul|mxfp8",k,re.I)})


def mode_traces(mode):
    base=ROOT/"exported"/mode
    traces={}
    for path in sorted(base.rglob("trace_view.json")):
        parts=path.relative_to(base).parts
        match=next((m for p in parts if (m:=re.match(r"dp(\d+)(?:_|$)",p))),None)
        assert match,(mode,path)
        rank=int(match[1])
        assert rank not in traces,(rank,traces.get(rank),path)
        traces[rank]=path
    assert set(traces)==RANKS,(mode,list(traces))
    return traces


def stats(values):
    a=np.asarray(values,dtype=np.float64)/1000
    return dict(n=len(a),mean_ms=float(a.mean()),p50_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()))


def analyze(mode):
    vectors=collections.defaultdict(lambda:collections.defaultdict(list))
    out=ROOT/f"per_call_{mode}.jsonl"
    with out.open("w") as stream:
        def emit(row):
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
            for cohort in ("all_recognized", "sparse_attention" if row["local_sparse_attention"] else "other"):
                for k,v in row.items():
                    if k.endswith("_us"):
                        vectors[(row["path"],cohort)][k].append(v)
        inventory=[]
        for rank,path in sorted(mode_traces(mode).items()):
            record=read_rank(mode,rank,path,emit)
            inventory.append(record)
            print(mode,rank,record["gate_count"],record["paths"],record["skipped"],flush=True)
    return dict(inventory=inventory,per_call_jsonl=str(out),
                stages={f"{k[0]}:{k[1]}":{n:stats(v) for n,v in m.items()} for k,m in vectors.items()})


if __name__=="__main__":
    target=ROOT/"STAGE_PROFILE_RESULTS.json"
    result=json.loads(target.read_text()) if target.is_file() else {}
    for mode in sys.argv[1:] or MODES:
        assert mode in MODES,mode
        result[mode]=analyze(mode)
        tmp=target.with_suffix(".tmp")
        tmp.write_text(json.dumps(result,indent=2)+"\n")
        tmp.replace(target)

