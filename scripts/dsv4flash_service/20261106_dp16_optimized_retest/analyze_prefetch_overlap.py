#!/usr/bin/env python3
"""Local GPU-timeline overlap, preserving stream order and measured kernel intervals."""
import bisect,json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;mode=sys.argv[1];rank=int(sys.argv[2]) if len(sys.argv)>2 else 0
inventory=json.loads((r/'data/diagnostics/optimized_profiling_20261106'/mode/f'rank{rank:02d}_trace_inventory.json').read_text())
events=json.loads(Path(inventory['path']).read_text());hardware={int(pid) for pid,name in inventory['processes'].items() if name=='Ascend Hardware'}
hw=[e for e in events if e.get('ph')=='X' and e.get('pid') in hardware]
origin=min(Decimal(str(e['ts'])) for e in hw)
for e in hw:e['start']=float(Decimal(str(e['ts']))-origin);e['end']=e['start']+float(e['dur'])
hw.sort(key=lambda e:e['start'])
def select(part):return [e for e in hw if part.lower() in e['name'].lower()]
home=select('home_gmm_mxfp4_kernel');planning=select('planning_kernel');dispatch=select('moonep_dispatch');combine=[e for e in hw if 'combine' in e['name'].lower() and 'weighted' in e['name'].lower()];prefetch=select('prefetch_weight')
assert len({e['tid'] for e in home})==1
assert len({e['tid'] for e in prefetch})==1
n=len(planning);assert len(home)==2*n and len(dispatch)==n and len(combine)==n and len(prefetch)==4*n,(n,len(home),len(dispatch),len(combine),len(prefetch))
main_id=home[0]['tid'];main=[e for e in hw if e['tid']==main_id];starts=[e['start'] for e in main];attn=[e for e in hw if e['name']=='KvQuantSparseAttnSharedkv']
def overlap(a,b,c,d):return max(0.,min(b,d)-max(a,c))
def union_length(intervals):
 intervals=sorted((a,b) for a,b in intervals if b>a)
 if not intervals:return 0.
 total=0.;a,b=intervals[0]
 for c,d in intervals[1:]:
  if c<=b:b=max(b,d)
  else:total+=b-a;a,b=c,d
 return total+b-a
rows=[]
for i in range(n):
 p=planning[i];d=dispatch[i];h,s=home[2*i:2*i+2];c=combine[i];pf=prefetch[4*i:4*i+4]
 assert p['end']<=d['start']+1 and d['end']<=h['start']+1 and h['end']<=s['start']+1 and s['end']<=c['start']+1
 ps,pe=pf[0]['start'],pf[-1]['end'];assert ps>=p['start']-1 and pe<=s['start']+1
 previous=combine[i-1]['end'] if i else -float('inf')
 long_prefill=any(e['start']>=previous and e['end']<=p['start'] for e in attn)
 lo=max(0,bisect.bisect_left(starts,h['end'])-2);hi=bisect.bisect_right(starts,s['start'])
 between=main[lo:hi]
 intervals=[(max(e['start'],h['end']),min(e['end'],s['start'])) for e in between]
 waits=[(max(e['start'],h['end']),min(e['end'],s['start'])) for e in between if 'WAIT' in e['name']]
 rows.append(dict(index=i,layer=i%43,pass_index=i//43,long_prefill=long_prefill,planning_us=p['dur'],dispatch_us=d['dur'],home_us=h['dur'],slot_us=s['dur'],combine_us=c['dur'],prefetch_active_us=sum(e['dur'] for e in pf),prefetch_span_us=pe-ps,prefetch_ready_before_home=pe<=h['start'],prefetch_ready_before_dispatch_end=pe<=d['end'],prefetch_active_overlap_home_us=sum(overlap(e['start'],e['end'],h['start'],h['end']) for e in pf),prefetch_span_overlap_home_us=overlap(ps,pe,h['start'],h['end']),prefetch_remaining_after_home_us=max(0.,pe-h['end']),home_slot_gap_us=s['start']-h['end'],home_slot_wait_task_us=union_length(waits),home_slot_no_recorded_device_task_us=max(0.,s['start']-h['end']-union_length(intervals)),timeline_us=dict(planning=[p['start'],p['end']],dispatch=[d['start'],d['end']],home=[h['start'],h['end']],slot=[s['start'],s['end']],combine=[c['start'],c['end']],prefetch=[[e['start'],e['end']] for e in pf])))
def summary(a):
 result=dict(calls=len(a))
 for key in ('prefetch_ready_before_home','prefetch_ready_before_dispatch_end'):result[key+'_count']=sum(x[key] for x in a)
 for key in rows[0]:
  if key in ('index','layer','pass_index','long_prefill','timeline_us','prefetch_ready_before_home','prefetch_ready_before_dispatch_end'):continue
  v=np.array([x[key] for x in a],dtype=float);result[key]=dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),max=float(v.max()),sum=float(v.sum()))
 return result
result=dict(mode=mode,rank=rank,all=summary(rows),long_prefill=summary([x for x in rows if x['long_prefill']]),rows=rows,limits='Per-rank local stream timeline; planning/dispatch/combine kernel durations include cross-rank waits, not pure network transfer. Unrecorded device intervals are not proven CPU overhead.')
out=r/'data/diagnostics/optimized_profiling_20261106'/mode/f'rank{rank:02d}_prefetch_overlap.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'})[:19000])
