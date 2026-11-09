#!/usr/bin/env python3
"""Shared-expert continuation after routed combine, verified by actual main-stream kernel sequence."""
import bisect,json
from decimal import Decimal
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_profiling_v2_20261106'
native=json.loads((b/'NATIVE_GMM_SUBSET.json').read_text())['rows'];out={}
for mode in ('off','slot_mte','slot_udma'):
 rows=[]
 for rank in range(16):
  inv=json.loads((b/mode/f'rank{rank:02d}_trace_inventory.json').read_text())
  ev=json.loads(Path(inv['path']).read_text());ev=ev['traceEvents'] if isinstance(ev,dict) else ev
  pids={int(k) for k,v in inv['processes'].items() if v=='Ascend Hardware'}
  hw=[x for x in ev if x.get('ph')=='X' and x.get('pid') in pids]
  origin=min(Decimal(str(x['ts'])) for x in hw)
  for x in hw:x['start']=float(Decimal(str(x['ts']))-origin);x['end']=x['start']+float(x['dur'])
  hw.sort(key=lambda x:x['start'])
  g=[x for x in hw if 'groupedmatmul' in x['name'].lower()];tid=g[0]['tid']
  main=[x for x in hw if x['tid']==tid];starts=[x['start'] for x in main]
  gates=[x for x in main if x['name']=='MoeGatingTopKHash']
  comb=[x for x in main if 'weighted' in x['name'].lower() and 'combine' in x['name'].lower()]
  if mode=='off':sel=[x['index'] for x in native if x['rank']==rank and x['long_prefill']][:86]
  else:sel=[x['index'] for x in json.loads((b/mode/f'rank{rank:02d}_prefetch_overlap.json').read_text())['rows'] if x['long_prefill']][:86]
  assert len(sel)==86
  for i in sel:
   if mode=='off':
    g2=g[2*i+1]
    z=[x for x in main[bisect.bisect_left(starts,g2['end']-1e-5):bisect.bisect_left(starts,gates[i+1]['start']) if i+1<len(gates) else len(main)] if 'MoeFinalizeRoutingV2' in x['name']]
    assert len(z)==2;end=z[-1]['end']
   else:end=comb[i]['end']
   lo=bisect.bisect_left(starts,end-1e-5);hi=bisect.bisect_left(starts,gates[i+1]['start']) if i+1<len(gates) else len(main)
   tail=main[lo:hi];adds=[x for x in tail if x['name']=='aclnnAdd_AddAiCore_Add'];assert adds,(mode,rank,i)
   add=adds[0];part=[x for x in tail if x['start']<add['start']]
   mm=[x for x in part if 'aclnnQuantMatmulV5_' in x['name']];act=[x for x in part if x['name']=='SwigluGroupQuant'];q=[x for x in part if x['name']=='DynamicMxQuant']
   assert len(mm)==2 and len(act)==1 and len(q)==1,(mode,rank,i,[x['name'] for x in part])
   assert q[0]['end']<=mm[0]['start']<=mm[0]['end']<=act[0]['start']<=act[0]['end']<=mm[1]['start']<=mm[1]['end']<=add['start']
   rows.append(dict(rank=rank,index=i,shared_input_quant_us=q[0]['dur'],shared_gemm1_us=mm[0]['dur'],shared_activation_quant_us=act[0]['dur'],shared_gemm2_us=mm[1]['dur'],shared_span_us=mm[1]['end']-q[0]['start'],merge_add_kernel_us=add['dur'],combine_end_to_shared_start_us=q[0]['start']-end,shared_to_merge_gap_us=add['start']-mm[1]['end'],combine_end_to_merge_end_us=add['end']-end))
  print('SHARED_PHASES_OK',mode,rank,len(sel),flush=True)
 stages={}
 for k in rows[0]:
  if k in ('rank','index'):continue
  a=np.array([x[k] for x in rows]);stages[k]=dict(mean_us=float(a.mean()),p95_us=float(np.percentile(a,95)))
 out[mode]=dict(calls=len(rows),stages=stages)
target=b/'MOE_STAGE_BREAKDOWN';target.mkdir(exist_ok=True)
(target/'shared_experts.json').write_text(json.dumps(dict(data=out,cohort='Same first2 prefill chunks per rank as routed phase table. All1376 per mode checked: quant -> dense matmul -> activation quant -> dense matmul -> Add, after routed combine, on same stream. No new NPU requests.'),indent=2))
print('SHARED_COMPLETE',flush=True)
