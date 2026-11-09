#!/usr/bin/env python3
import json,subprocess,sys,urllib.request
from pathlib import Path
r=Path(__file__).resolve().parent
kind,node,mode=sys.argv[1:]
d=json.loads((r/'STATE.json').read_text());entry=d['diagnostics'][kind];run=r/'data/diagnostics'/entry['run_id']/mode
owner_path=run/f'node{node}_ownership.json';owner=json.loads(owner_path.read_text());pid=owner['root_pid']
q=r/'diagnostics'/kind
matches=[p.parent for p in (q/'data/services').glob(f'*-{mode}-node{node}-*/service.pid') if p.read_text().strip()==str(pid)]
out=dict(node=node,mode=mode,kind=kind,pid=pid,root_exit=owner.get('root_exit'),cleanup=owner.get('cleanup_completed'),service_dirs=list(map(str,matches)))
for directory in matches:
 text=(directory/'service.log').read_text(errors='replace');out['errors']=text.count('Traceback (most recent call last)');out['tail']=text[-900:]
if owner.get('root_exit') is None:
 z=subprocess.run(['python3',str(r/'observe_owned_resources.py'),str(owner_path),str(run/f'node{node}_resources.jsonl')],text=True,capture_output=True)
 observed=json.loads(z.stdout) if z.stdout.strip() else dict(error=z.stderr)
 out['resources']={k:v for k,v in observed.items() if k in ('valid','foreign','error')}
if node=='118':
 try:
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://141.61.45.118:8445/v1/models',timeout=2) as response:out['models']=json.load(response)
 except Exception as error:out['api']=str(error)
print(json.dumps(out))
