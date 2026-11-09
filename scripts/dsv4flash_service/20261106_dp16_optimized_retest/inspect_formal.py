#!/usr/bin/env python3
import json,sys,urllib.request,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
node=sys.argv[1];mode=sys.argv[2]
d=json.loads((r/'STATE.json').read_text());run=r/'data/formal'/d['formal_run_id']/mode
p=run/f'node{node}_ownership.json'
own=json.loads(p.read_text());pid=own['root_pid']
matches=[p.parent for p in (r/'data/services').glob(f'*-{mode}-node{node}-*/service.pid') if p.read_text().strip()==str(pid)]
out=dict(node=node,mode=mode,pid=pid,root_exit=own.get('root_exit'),cleanup=own.get('cleanup_completed'),services=list(map(str,matches)))
for path in matches:
 log=(path/'service.log').read_text(errors='replace')
 out['log_tail']=log[-1800:];out['errors']=log.count('Traceback (most recent call last)')
if node=='118':
 try:
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://141.61.45.118:8445/v1/models',timeout=2) as resp:out['models']=json.load(resp)
 except Exception as error:out['api']=str(error)
if out['root_exit'] is None:
 z=subprocess.run(['python3',str(r/'observe_owned_resources.py'),str(p),str(run/f'node{node}_startup_resources.jsonl')],text=True,capture_output=True)
 observed=json.loads(z.stdout) if z.stdout.strip() else {'error':z.stderr}
 out['resources']={k:v for k,v in observed.items() if k in ('valid','foreign','error')}
print(json.dumps(out))
