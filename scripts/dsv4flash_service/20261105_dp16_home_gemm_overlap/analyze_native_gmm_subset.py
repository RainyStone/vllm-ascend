#!/usr/bin/env python3
"""Exact sparse-attention marker identifies long-prefill native expert GMM pairs."""
import json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
base=r/'data/diagnostics/home_profiling_v2_20261106'
mode='off';all_rows=[]
for rank in range(16):
 d=json.loads((base/mode/f'rank{rank:02d}_trace_inventory.json').read_text())
 ev=json.loads(Path(d['path']).read_text());pids={int(p) for p,n in d['processes'].items() if n=='Ascend Hardware'}
 hw=[x for x in ev if x.get('ph')=='X' and x.get('pid') in pids]
 origin=min(Decimal(str(x['ts'])) for x in hw)
 for x in hw:x['start']=float(Decimal(str(x['ts']))-origin);x['end']=x['start']+float(x['dur'])
 hw.sort(key=lambda x:x['start'])
 g=[x for x in hw if 'groupedmatmul' in x['name'].lower()]
 attn=[x for x in hw if x['name']=='KvQuantSparseAttnSharedkv']
 assert len(g)%2==0 and len({x['tid'] for x in g})==1
 rows=[]
 for i in range(len(g)//2):
  a,b=g[2*i:2*i+2];previous=g[2*i-1]['end'] if i else -float('inf')
  long=any(previous<=x['start'] and x['end']<=a['start'] for x in attn)
  rows.append(dict(rank=rank,index=i,long_prefill=long,gmm1_us=a['dur'],gmm2_us=b['dur']))
 assert sum(x['long_prefill'] for x in rows)==len(attn),(rank,len(attn))
 all_rows+=rows
 print('NATIVE_GMM',rank,len(rows),len(attn),flush=True)
s={}
for key,rows in [('all',all_rows),('long_prefill',[x for x in all_rows if x['long_prefill']])]:
 s[key]=dict(calls=len(rows),gmm1_mean_us=float(np.mean([x['gmm1_us'] for x in rows])),gmm2_mean_us=float(np.mean([x['gmm2_us'] for x in rows])))
(base/'NATIVE_GMM_SUBSET.json').write_text(json.dumps(dict(summary=s,rows=all_rows),indent=2))
print(json.dumps(s),flush=True)
