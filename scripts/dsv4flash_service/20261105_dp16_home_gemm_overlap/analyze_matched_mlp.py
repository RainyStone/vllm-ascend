#!/usr/bin/env python3
"""Match first two observed long-prefill forwards per rank (86 MoE layers).
No causal claim about Home on/off: the formal experiment has no same-version PF2 arm.
"""
import json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_profiling_v2_20261106'
native=json.loads((b/'NATIVE_GMM_SUBSET.json').read_text())['rows']
result={}
def stats(rows,keys):
 return dict(calls=len(rows),means_us={k:float(np.mean([x[k] for x in rows])) for k in keys})
off=[]
for rank in range(16):
 a=[x for x in native if x['rank']==rank and x['long_prefill']][:86];assert len(a)==86
 off+=a
result['off']=stats(off,['gmm1_us','gmm2_us'])
for mode in ('slot_mte','slot_udma'):
 rows=[]
 for rank in range(16):
  d=json.loads((b/mode/f'rank{rank:02d}_trace_inventory.json').read_text())
  o=json.loads((b/mode/f'rank{rank:02d}_prefetch_overlap.json').read_text())['rows']
  ev=json.loads(Path(d['path']).read_text());pids={int(p) for p,n in d['processes'].items() if n=='Ascend Hardware'}
  hw=[x for x in ev if x.get('ph')=='X' and x.get('pid') in pids]
  g=sorted([x for x in hw if 'groupedmatmul' in x['name'].lower()],key=lambda x:Decimal(str(x['ts'])))
  assert len(g)==len(o)
  a=[]
  for x in o:
   if x['long_prefill']:
    x=dict(x,gmm2_us=g[x['index']]['dur']);x['gemm1_total_us']=x['home_us']+x['slot_us'];a.append(x)
  assert len(a)>=86
  rows+=a[:86]
  print('MATCHED',mode,rank,flush=True)
 result[mode]=stats(rows,['planning_us','dispatch_us','home_us','slot_us','gemm1_total_us','gmm2_us','combine_us','prefetch_active_us'])
result['limits']='Same pinned input sample per rank; first2 captured long-prefill forwards per rank, not full503. Timings include kernel waits and can overlap. Changes in placement alter group sizes; this does not isolate Home feature cost or explain exact formal latency percentages.'
(b/'MATCHED_MLP_SUBSET.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
