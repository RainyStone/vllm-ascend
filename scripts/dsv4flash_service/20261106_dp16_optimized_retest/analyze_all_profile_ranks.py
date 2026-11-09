#!/usr/bin/env python3
"""CPU-only all-rank measured inventories and overlap; preserves individual traces."""
import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'data/diagnostics/optimized_profiling_20261106'
results={}
for mode in ('off','slot_mte','slot_udma'):
 for rank in range(16):
  for tool in ['inventory_profile_trace.py']+([] if mode=='off' else ['analyze_prefetch_overlap.py']):
   p=subprocess.run([sys.executable,str(r/tool),mode,str(rank)],capture_output=True,text=True)
   (out/mode/f'rank{rank:02d}_{tool}.log').write_text(p.stdout+p.stderr)
   assert p.returncode==0,(mode,rank,tool,p.stderr)
  print('ANALYZED',mode,rank,flush=True)
 if mode=='off':continue
 rows=[json.loads((out/mode/f'rank{i:02d}_prefetch_overlap.json').read_text()) for i in range(16)]
 summary={}
 for section in ('all','long_prefill'):
  ss=[x[section] for x in rows]
  summary[section]=dict(calls=sum(x['calls'] for x in ss),
   ready_before_home=sum(x['prefetch_ready_before_home_count'] for x in ss),
   ready_before_dispatch_end=sum(x['prefetch_ready_before_dispatch_end_count'] for x in ss),
   prefetch_active_overlap_home_us=sum(x['prefetch_active_overlap_home_us']['sum'] for x in ss),
   prefetch_remaining_after_home_us=sum(x['prefetch_remaining_after_home_us']['sum'] for x in ss),
   calls_per_rank=[x['calls'] for x in ss],
   stages_sum_us={k:sum(x[k]['sum'] for x in ss) for k in ('planning_us','dispatch_us','home_us','slot_us','combine_us','prefetch_active_us','home_slot_gap_us')})
 results[mode]=summary
(out/'ALL_RANK_OVERLAP.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
