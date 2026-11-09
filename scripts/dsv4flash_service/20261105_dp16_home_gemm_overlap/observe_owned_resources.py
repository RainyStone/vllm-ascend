#!/usr/bin/env python3
"""Read-only resource ownership observation for one running experiment node."""
import json,re,subprocess,sys,time
from pathlib import Path
record,output=sys.argv[1:]
owner=json.loads(Path(record).read_text());known={x['pid'] for x in owner['owned']}
text=subprocess.check_output(['npu-smi','info'],text=True)
rows=[dict(card=int(c),host_pid=int(h),name=name.strip(),memory_mb=int(mem),container_pid=int(local))
 for c,h,name,mem,local in re.findall(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text)]
foreign=[x for x in rows if x['container_pid'] not in known]
result=dict(at=time.time(),valid=bool(rows) and not foreign,processes=rows,foreign=foreign,owned_root=owner['root_pid'])
p=Path(output)
with p.open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result))
if foreign:raise SystemExit('Foreign NPU process detected: stop only owned experiment tasks')
