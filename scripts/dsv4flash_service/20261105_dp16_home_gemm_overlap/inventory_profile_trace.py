#!/usr/bin/env python3
"""CPU-only Chrome trace inventory; durations are GPU task totals, not critical-path percentages."""
import collections,json,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
mode=sys.argv[1];rank=int(sys.argv[2]) if len(sys.argv)>2 else 0
base=root/'diagnostics/profiling/data/profiling_v2'/mode
dirs=[p for p in base.glob('*_ascend_pt') if p.name.split('_')[0]=='dp'+str(rank)]
assert len(dirs)==1,dirs
paths=list(dirs[0].rglob('trace_view.json'));assert len(paths)==1,paths
events=json.loads(paths[0].read_text());events=events['traceEvents'] if isinstance(events,dict) else events
processes={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
hardware_ids={pid for pid,name in processes.items() if name=='Ascend Hardware'}
hw=[e for e in events if e.get('ph')=='X' and e.get('pid') in hardware_ids]
assert hw
origin=min(Decimal(str(e['ts'])) for e in hw)
groups=collections.defaultdict(list)
for e in hw:groups[e['name']].append(e)
top=[]
for name,a in groups.items():
 dur=np.array([float(e['dur']) for e in a])
 top.append(dict(name=name,count=len(a),sum_ms=float(dur.sum()/1000),mean_us=float(dur.mean()),max_us=float(dur.max()),task_types=sorted({str(e.get('args',{}).get('Task Type')) for e in a})))
top.sort(key=lambda x:x['sum_ms'],reverse=True)
steps=[dict(name=e['name'],duration_us=e.get('dur')) for e in events if e.get('ph')=='X' and e.get('name','').startswith('ProfilerStep#')]
out=dict(mode=mode,rank=rank,path=str(paths[0]),hardware_event_count=len(hw),processes=processes,steps=steps,hardware_top=top,warning='Task-duration sums can overlap across streams; do not interpret as wall-time percentages.')
dest=root/'data/diagnostics/home_profiling_v2_20261106'/mode;dest.mkdir(parents=True,exist_ok=True)
(dest/f'rank{rank:02d}_trace_inventory.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(mode=mode,rank=rank,hardware_event_count=len(hw),steps=steps,top=top[:14],sample_hardware_args=hw[:2]),default=str)[:10500])
