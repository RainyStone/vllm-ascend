#!/usr/bin/env python3
"""Describe rank0 dispatch->Home main-stream interval; do not call unrecorded gaps CPU overhead."""
import collections,json
from decimal import Decimal
from pathlib import Path
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_profiling_v2_20261106';out={}
for mode in ('slot_mte','slot_udma'):
 d=json.loads((b/mode/'rank00_trace_inventory.json').read_text());events=json.loads(Path(d['path']).read_text());pids={int(p) for p,n in d['processes'].items() if n=='Ascend Hardware'}
 hw=[x for x in events if x.get('ph')=='X' and x.get('pid') in pids]
 origin=min(Decimal(str(x['ts'])) for x in hw)
 for x in hw:x['start']=float(Decimal(str(x['ts']))-origin);x['end']=x['start']+float(x['dur'])
 main_ids={x['tid'] for x in hw if 'home_gmm_mxfp4_kernel' in x['name']};assert len(main_ids)==1
 main=[x for x in hw if x['tid'] in main_ids]
 rows=[x for x in json.loads((b/mode/'rank00_prefetch_overlap.json').read_text())['rows'] if x['long_prefill']]
 tasks=collections.defaultdict(lambda:dict(calls=0,sum_us=0.));gaps=[]
 for row in rows:
  a=row['timeline_us']['dispatch'][1];z=row['timeline_us']['home'][0];gaps.append(z-a)
  for x in main:
   overlap=max(0.,min(z,x['end'])-max(a,x['start']))
   if overlap:
    tasks[x['name']]['calls']+=1;tasks[x['name']]['sum_us']+=overlap
 out[mode]=dict(rank=0,long_prefill_calls=len(rows),mean_dispatch_to_home_gap_us=sum(gaps)/len(gaps),
  device_tasks=[dict(name=k,**v,mean_us_per_moe_call=v['sum_us']/len(rows)) for k,v in sorted(tasks.items(),key=lambda kv:kv[1]['sum_us'],reverse=True)],
  limit='One rank, local device timeline. Remaining unrecorded gaps are not proven CPU overhead.')
(b/'DISPATCH_TO_HOME.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
