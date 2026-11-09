#!/usr/bin/env python3
import json,subprocess,sys,re
from pathlib import Path
r=Path(__file__).resolve().parent
node,mode=sys.argv[1:]
d=json.loads((r/'STATE.json').read_text());run=r/'data/formal'/d['formal_run_id']/mode
owner=json.loads((run/f'node{node}_ownership.json').read_text())
z=subprocess.run(['python3',str(r/'observe_owned_resources.py'),str(run/f'node{node}_ownership.json'),str(run/f'node{node}_resources.jsonl')],text=True,capture_output=True)
out=dict(node=node,root_exit=owner.get('root_exit'),resources=json.loads(z.stdout) if z.stdout.strip() else dict(valid=False,error=z.stderr))
if node=='118':
 p=run/'bench_ownership.json';b=json.loads(p.read_text())
 out['bench_exit']=b.get('root_exit');out['bench_cleanup']=b.get('cleanup_completed')
 with (run/'bench.log').open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-5000));out['bench_tail']=f.read().decode(errors='replace')
 if 'Task [' in out['bench_tail']:(run/'bench_ownership.quiet').touch()
out['resources']={k:v for k,v in out['resources'].items() if k in ('valid','foreign','error')}
if 'bench_tail' in out:
 text=out['bench_tail']
 matches=re.findall(r'POST=(\d+).*?RECV=(\d+).*?FAIL=(\d+).*?FIN=(\d+)',text)
 if matches:out['progress']=dict(zip(('post','received','failed','finished'),map(int,matches[-1])))
 out['bench_tail']=text[-1600:] if out.get('bench_exit') is not None else ('' if matches else text[-300:])
print(json.dumps(out))
