#!/usr/bin/env python3
"""CPU-only MoE phase partition from existing local GPU timelines.
Serial main-stream wall intervals are disjoint. Parallel prefetch is reported separately.
"""
import bisect,collections,json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_profiling_v2_20261106'
native=json.loads((b/'NATIVE_GMM_SUBSET.json').read_text())['rows']
all_rows={};details={}
for mode in ('off','slot_mte','slot_udma'):
 result=[]
 for rank in range(16):
  inv=json.loads((b/mode/f'rank{rank:02d}_trace_inventory.json').read_text())
  ev=json.loads(Path(inv['path']).read_text());ev=ev['traceEvents'] if isinstance(ev,dict) else ev
  pids={int(p) for p,n in inv['processes'].items() if n=='Ascend Hardware'}
  hw=[x for x in ev if x.get('ph')=='X' and x.get('pid') in pids]
  origin=min(Decimal(str(x['ts'])) for x in hw)
  for x in hw:x['start']=float(Decimal(str(x['ts']))-origin);x['end']=x['start']+float(x['dur'])
  hw.sort(key=lambda x:x['start'])
  grouped=[x for x in hw if 'groupedmatmul' in x['name'].lower()]
  home=[x for x in hw if 'home_gmm_mxfp4_kernel' in x['name']]
  gates=[x for x in hw if x['name']=='MoeGatingTopKHash']
  n=len(gates);assert n>0
  if mode=='off':
   assert len(grouped)==2*n
   selected=[x['index'] for x in native if x['rank']==rank and x['long_prefill']][:86]
  else:
   assert len(home)==2*n and len(grouped)==n
   old=json.loads((b/mode/f'rank{rank:02d}_prefetch_overlap.json').read_text())['rows']
   assert len(old)==n
   selected=[x['index'] for x in old if x['long_prefill']][:86]
  assert len(selected)==86
  mainid=grouped[0]['tid'];main=[x for x in hw if x['tid']==mainid]
  starts=[x['start'] for x in main]
  planning=[x for x in main if x['name']=='planning_kernel']
  dispatch=[x for x in main if 'moonep_dispatch' in x['name']]
  combines=[x for x in main if 'weighted' in x['name'].lower() and 'combine' in x['name'].lower()]
  if mode!='off':assert len(planning)==len(dispatch)==len(combines)==n
  for i in selected:
   gate=gates[i]
   if mode=='off':g1,g2=grouped[2*i:2*i+2];lastg1=g1;g1kernel=g1['dur'];g1span=g1['dur']
   else:
    g1,lastg1=home[2*i:2*i+2];g2=grouped[i];g1kernel=g1['dur']+lastg1['dur'];g1span=lastg1['end']-g1['start']
   assert gate['end']<=g1['start']<=lastg1['end']<=g2['start']
   nextgate=gates[i+1]['start'] if i+1<n else float('inf')
   lo=bisect.bisect_left(starts,g2['end']-1e-5);hi=bisect.bisect_left(starts,nextgate)
   after=main[lo:hi]
   if mode=='off':
    unperm=[x for x in after if 'MoeFinalizeRoutingV2' in x['name']]
    assert len(unperm)==2,(mode,rank,i,'native path not AlltoAll',[(x['name'],x['dur']) for x in unperm])
    end=unperm[1]['end']
   else:end=combines[i]['end']
   # Router projection is the last dense MatMulV3 before gate, excluding quantized attention matmuls.
   prior=main[:bisect.bisect_left(starts,gate['start'])]
   routers=[x for x in prior if x['name']=='aclnnMatmul_MatMulV3Common_MatMulV3']
   assert routers,(mode,rank,i,'router projection missing')
   router=routers[-1];assert router['end']<=gate['start']
   row=dict(rank=rank,index=i,layer=i%43,
    router_gemm_us=router['dur'],router_to_gate_gap_us=gate['start']-router['end'],topk_us=gate['dur'],
    dispatch_inclusive_us=g1['start']-gate['end'],
    gemm1_kernel_us=g1kernel,gemm1_span_us=g1span,
    activation_quant_span_us=g2['start']-lastg1['end'],gemm2_us=g2['dur'],
    combine_inclusive_us=end-g2['end'],
    core_wall_us=end-gate['start'],router_through_combine_wall_us=end-router['start'])
   row['activation_quant_kernel_us']=sum(x['dur'] for x in main if lastg1['end']-1e-5<=x['start']<g2['start'] and x['name']=='SwigluGroupQuant')
   assert row['activation_quant_kernel_us']>0
   row['native_unpermute_kernel_us']=sum(x['dur'] for x in unperm) if mode=='off' else 0.
   if mode!='off':
    p,d,c=planning[i],dispatch[i],combines[i];o=old[i]
    assert gate['end']<=p['start']<=p['end']<=d['start']<=d['end']<=g1['start']
    row.update(dispatch_prepare_us=p['start']-gate['end'],planning_kernel_us=p['dur'],
     planning_to_dispatch_gap_us=d['start']-p['end'],dispatch_kernel_us=d['dur'],
     dispatch_post_quant_span_us=g1['start']-d['end'],
     input_quant_kernel_us=sum(x['dur'] for x in main if d['end']-1e-5<=x['start']<g1['start'] and x['name']=='DynamicMxQuant'),
     post_dispatch_tensor_move_us=sum(x['dur'] for x in main if d['end']-1e-5<=x['start']<g1['start'] and 'TensorMoveAiCore_TensorMove' in x['name']),
     home_gemm1_us=g1['dur'],slot_gemm1_us=lastg1['dur'],home_slot_gap_us=lastg1['start']-g1['end'],
     combine_prepare_us=c['start']-g2['end'],combine_kernel_us=c['dur'],
     prefetch_active_us=o['prefetch_active_us'],prefetch_span_us=o['prefetch_span_us'],
     prefetch_ready_before_dispatch_end=o['prefetch_ready_before_dispatch_end'])
    assert abs(sum(row[k] for k in ('dispatch_prepare_us','planning_kernel_us','planning_to_dispatch_gap_us','dispatch_kernel_us','dispatch_post_quant_span_us'))-row['dispatch_inclusive_us'])<.01
   else:
    # A2A has two expert permutations surrounding two payload collectives.
    between=[x for x in main if gate['end']-1e-5<=x['start']<g1['start']]
    row.update(dispatch_histogram_kernel_us=sum(x['dur'] for x in between if 'Histogram' in x['name']),
     input_quant_kernel_us=sum(x['dur'] for x in between if x['name']=='DynamicMxQuant'),
     dispatch_permute_kernel_us=sum(x['dur'] for x in between if 'MoeInitRouting' in x['name']),
     combine_local_unpermute_us=unperm[0]['dur'],combine_collective_span_us=unperm[1]['start']-unperm[0]['end'],
     combine_final_unpermute_us=unperm[1]['dur'])
   pieces=sum(row[k] for k in ('topk_us','dispatch_inclusive_us','gemm1_span_us','activation_quant_span_us','gemm2_us','combine_inclusive_us'))
   assert abs(pieces-row['core_wall_us'])<.01,(mode,rank,i,pieces,row['core_wall_us'])
   result.append(row)
  print('PHASES_OK',mode,rank,len(selected),flush=True)
 all_rows[mode]=result
 values={}
 for key in result[0]:
  if key in ('rank','index','layer','prefetch_ready_before_dispatch_end'):continue
  a=np.array([x[key] for x in result],float)
  values[key]=dict(mean_us=float(a.mean()),p50_us=float(np.median(a)),p95_us=float(np.percentile(a,95)),max_us=float(a.max()))
 details[mode]=dict(calls=len(result),ranks=16,stages=values)
out=b/'MOE_STAGE_BREAKDOWN';out.mkdir(exist_ok=True)
(out/'per_call.json').write_text(json.dumps(all_rows,indent=2))
(out/'summary.json').write_text(json.dumps(dict(data=details,
 cohort='Each rank first2 captured sparse-attention prefill chunks x43 layers =1376 rank-layer calls per mode. Same pinned inputs; no new NPU execution.',
 boundaries='Main-stream disjoint wall intervals: gate -> GMM1 -> activation/quant -> GMM2 -> final combine output. Router projection shown separately. Prefetch side stream overlaps dispatch, must not add to wall total.',
 limits='Independent diagnostic cohort, not full503. Communication durations include cross-rank synchronization and waits; wall gaps may include unrecorded device tasks or host delay. Mean over ranks/calls is not an iteration maximum or exact attribution of full503 TTFT differences.'),indent=2))
print('BREAKDOWN_COMPLETE',out,flush=True)
