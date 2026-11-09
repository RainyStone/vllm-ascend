#!/usr/bin/env python3
"""CPU timeline partition for PF_STREAM=2 and unsplit GMM1. No NPU execution."""
import bisect,json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
base=root/"data/diagnostics/optimized_profiling_20261106"
modes=sys.argv[1:] or ["off","slot_mte","slot_udma"]
assert set(modes)<=set(["off","slot_mte","slot_udma"])
all_rows={};summaries={}
def stats(rows):
 out={}
 for key in rows[0]:
  if key in ("rank","index","layer","prefill_ordinal","long_prefill","timeline_us"):continue
  a=np.array([x[key] for x in rows],dtype=float)
  out[key]=dict(mean_us=float(a.mean()),p50_us=float(np.median(a)),p95_us=float(np.percentile(a,95)),max_us=float(a.max()))
 return out
for mode in modes:
 rows=[]
 for rank in range(16):
  inv=json.loads((base/mode/f"rank{rank:02d}_trace_inventory.json").read_text())
  trace=json.loads(Path(inv["path"]).read_text());trace=trace["traceEvents"] if isinstance(trace,dict) else trace
  pids={int(k) for k,v in inv["processes"].items() if v=="Ascend Hardware"}
  hw=[x for x in trace if x.get("ph")=="X" and x.get("pid") in pids]
  origin=min(Decimal(str(x["ts"])) for x in hw)
  for x in hw:x["start"]=float(Decimal(str(x["ts"]))-origin);x["end"]=x["start"]+float(x["dur"])
  hw.sort(key=lambda x:x["start"])
  g=[x for x in hw if "groupedmatmul" in x["name"].lower()]
  assert g and len({x["tid"] for x in g})==1
  main=[x for x in hw if x["tid"]==g[0]["tid"]];starts=[x["start"] for x in main]
  gates=[x for x in main if x["name"]=="MoeGatingTopKHash"]
  attn=[x for x in hw if x["name"]=="KvQuantSparseAttnSharedkv"]
  n=len(gates);assert len(g)==2*n,(mode,rank,n,len(g))
  assert not any("home_gmm_mxfp4" in x["name"] for x in hw),"Home split unexpectedly active"
  planning=[x for x in main if x["name"]=="planning_kernel"]
  dispatch=[x for x in main if "moonep_dispatch" in x["name"]]
  combines=[x for x in main if "combine" in x["name"].lower() and "weighted" in x["name"].lower()]
  pack=[x for x in main if x["name"]=="moonep_mxfp8_pack_kernel"]
  unpack=[x for x in main if x["name"]=="moonep_mxfp8_unpack_kernel"]
  publish=[x for x in main if x["name"]=="moonep_publish_kernel"]
  pf=[x for x in hw if "prefetch_weight" in x["name"]]
  if mode!="off":
   assert all(len(x)==n for x in (planning,dispatch,combines,pack,unpack,publish)),(mode,rank,n,[len(x) for x in (planning,dispatch,combines,pack,unpack,publish)])
   assert len(pf)==4*n and len({x["tid"] for x in pf})==1 and pf[0]["tid"]!=g[0]["tid"]
  selected=[]
  for i in range(n):
   previous=g[2*i-1]["end"] if i else -float("inf")
   if any(previous<=x["start"] and x["end"]<=g[2*i]["start"] for x in attn):selected.append(i)
  local_prefill=set(selected)
  for prefill_ordinal,i in enumerate(selected[:86]):
   gate=gates[i];g1,g2=g[2*i:2*i+2]
   nextgate=gates[i+1]["start"] if i+1<n else float("inf")
   after=main[bisect.bisect_left(starts,g2["end"]-1e-5):bisect.bisect_left(starts,nextgate)]
   if mode=="off":
    unperm=[x for x in after if "MoeFinalizeRoutingV2" in x["name"]]
    assert len(unperm)==2,(mode,rank,i,"expected native AlltoAll finalizers")
    end=unperm[-1]["end"]
   else:end=combines[i]["end"]
   prior=main[:bisect.bisect_left(starts,gate["start"])]
   routers=[x for x in prior if x["name"]=="aclnnMatmul_MatMulV3Common_MatMulV3"]
   assert routers,(mode,rank,i,"router projection missing")
   router=routers[-1]
   row=dict(rank=rank,index=i,layer=i%43,prefill_ordinal=prefill_ordinal,long_prefill=True,router_gemm_us=router["dur"],
    router_to_gate_gap_us=gate["start"]-router["end"],topk_us=gate["dur"],
    dispatch_inclusive_us=g1["start"]-gate["end"],gemm1_kernel_us=g1["dur"],
    activation_quant_span_us=g2["start"]-g1["end"],gemm2_kernel_us=g2["dur"],
    combine_inclusive_us=end-g2["end"],core_wall_us=end-gate["start"])
   row["activation_quant_kernel_us"]=sum(x["dur"] for x in main if g1["end"]-1e-5<=x["start"]<g2["start"] and x["name"]=="SwigluGroupQuant")
   assert row["activation_quant_kernel_us"]>0
   if mode=="off":
    between=[x for x in main if gate["end"]-1e-5<=x["start"]<g1["start"]]
    row.update(input_quant_kernel_us=sum(x["dur"] for x in between if x["name"]=="DynamicMxQuant"),
     dispatch_histogram_kernel_us=sum(x["dur"] for x in between if "Histogram" in x["name"]),
     dispatch_permute_kernel_us=sum(x["dur"] for x in between if "MoeInitRouting" in x["name"]),
     combine_local_unpermute_us=unperm[0]["dur"],
     combine_collective_span_us=unperm[1]["start"]-unperm[0]["end"],
     combine_final_unpermute_us=unperm[1]["dur"])
   else:
    p,d,pk,up,pub,c=planning[i],dispatch[i],pack[i],unpack[i],publish[i],combines[i]
    assert gate["end"]<=p["start"]<=p["end"]<=pk["start"]<=pk["end"]<=d["start"]<=d["end"]<=up["start"]<=up["end"]<=g1["start"]
    assert g2["end"]<=pub["start"]<=pub["end"]<=c["start"]<=c["end"]
    q=[x for x in main if p["end"]-1e-5<=x["start"]<pk["start"] and x["name"]=="DynamicMxQuant"]
    assert len(q)==1,(mode,rank,i,"expected one sender-side MX quantizer",len(q))
    pref=pf[4*i:4*i+4];ps,pe=pref[0]["start"],pref[-1]["end"]
    assert ps>=p["end"]-1 and pe<=g1["start"]+1
    row.update(dispatch_prepare_us=p["start"]-gate["end"],planning_kernel_us=p["dur"],
     planning_to_quant_gap_us=q[0]["start"]-p["end"],input_quant_kernel_us=q[0]["dur"],
     quant_to_pack_gap_us=pk["start"]-q[0]["end"],pack_kernel_us=pk["dur"],
     pack_to_dispatch_gap_us=d["start"]-pk["end"],dispatch_kernel_us=d["dur"],
     dispatch_to_unpack_gap_us=up["start"]-d["end"],unpack_kernel_us=up["dur"],
     unpack_to_gemm1_gap_us=g1["start"]-up["end"],fused_publish_kernel_us=pub["dur"],
     combine_before_publish_gap_us=pub["start"]-g2["end"],
     publish_to_combine_gap_us=c["start"]-pub["end"],combine_kernel_us=c["dur"],
     prefetch_active_us=sum(x["dur"] for x in pref),prefetch_span_us=pe-ps,
     prefetch_pending_after_dispatch_us=max(0.,pe-d["end"]),
     prefetch_ready_before_dispatch_end=int(pe<=d["end"]+1),
     prefetch_ready_before_unpack_end=int(pe<=up["end"]+1),
     timeline_us=dict(planning=[p["start"],p["end"]],pack=[pk["start"],pk["end"]],
      dispatch=[d["start"],d["end"]],unpack=[up["start"],up["end"]],gemm1=[g1["start"],g1["end"]],
      gemm2=[g2["start"],g2["end"]],publish=[pub["start"],pub["end"]],combine=[c["start"],c["end"]],
      prefetch=[[x["start"],x["end"]] for x in pref]))
    parts=("dispatch_prepare_us","planning_kernel_us","planning_to_quant_gap_us","input_quant_kernel_us","quant_to_pack_gap_us","pack_kernel_us","pack_to_dispatch_gap_us","dispatch_kernel_us","dispatch_to_unpack_gap_us","unpack_kernel_us","unpack_to_gemm1_gap_us")
    assert abs(sum(row[k] for k in parts)-row["dispatch_inclusive_us"])<.01
   pieces=sum(row[k] for k in ("topk_us","dispatch_inclusive_us","gemm1_kernel_us","activation_quant_span_us","gemm2_kernel_us","combine_inclusive_us"))
   assert abs(pieces-row["core_wall_us"])<.01
   # Shared expert computation follows routed combine on the same stream.
   tail=main[bisect.bisect_left(starts,end-1e-5):bisect.bisect_left(starts,nextgate)]
   adds=[x for x in tail if x["name"]=="aclnnAdd_AddAiCore_Add"];assert adds
   add=adds[0];part=[x for x in tail if x["start"]<add["start"]]
   mm=[x for x in part if "aclnnQuantMatmulV5_" in x["name"]]
   act=[x for x in part if x["name"]=="SwigluGroupQuant"]
   sq=[x for x in part if x["name"]=="DynamicMxQuant"]
   assert len(mm)==2 and len(act)==1 and len(sq)==1,(mode,rank,i,"shared path changed")
   row.update(shared_input_quant_us=sq[0]["dur"],shared_gemm1_us=mm[0]["dur"],
    shared_activation_quant_us=act[0]["dur"],shared_gemm2_us=mm[1]["dur"],
    shared_span_us=mm[1]["end"]-sq[0]["start"],merge_add_us=add["dur"],
    combine_end_to_merge_end_us=add["end"]-end,total_router_to_merge_wall_us=add["end"]-router["start"])
   rows.append(row)
  print("PF2_PHASES_OK",mode,rank,n,"local_prefill",len(local_prefill),flush=True)
 all_rows[mode]=rows;summaries[mode]=dict(calls=len(rows),ranks=16,stages=stats(rows))
all_captured=all_rows
common_counts={rank:min(sum(x["rank"]==rank for x in all_captured[m]) for m in modes) for rank in range(16)}
all_rows={m:[x for x in all_captured[m] if x["prefill_ordinal"]<common_counts[x["rank"]]] for m in modes}
summaries={m:dict(calls=len(v),ranks=len({x["rank"] for x in v}),stages=stats(v)) for m,v in all_rows.items()}
out=base/"PF2_STAGE_BREAKDOWN";out.mkdir(exist_ok=True)
(out/"per_call_all_captured.json").write_text(json.dumps(all_captured,indent=2))
(out/"per_call.json").write_text(json.dumps(all_rows,indent=2))
(out/"summary.json").write_text(json.dumps(dict(data=summaries,common_prefill_counts_per_rank=common_counts,
 cohort="Same16 selected prompts. Match each rank by ordinal among captured sparse-attention local-prefill calls, cap at two passes, and use the minimum available count across modes. Retain actual trace call indices. Dummy-only calls excluded. Captured ordinals are not proven global request-step IDs; this is diagnostic sampling, not full503 attribution.",
 boundaries="Main-stream disjoint gate -> GMM1 -> activation/quant -> GMM2 -> final routed output; shared expert continuation measured separately. Side-stream prefetch is parallel and must not be added.",
 limits="Communication durations include protocol and cross-rank waits. Gaps may include unrecorded device work or host enqueue time. Per-call rank means are not global iteration maxima or exact whole503 TTFT attribution."),indent=2))
print("PF2_STAGE_SUMMARY",str(out/"summary.json"),flush=True)
