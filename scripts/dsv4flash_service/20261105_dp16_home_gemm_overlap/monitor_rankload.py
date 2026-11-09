#!/usr/bin/env python3
import json,re,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;node,mode=sys.argv[1:]
run=r/'data/diagnostics/home_rankload_20261106'/mode
owner=run/f'node{node}_ownership.json';d=json.loads(owner.read_text())
p=subprocess.run(['python',str(r/'observe_owned_resources.py'),str(owner),str(run/f'node{node}_resources.jsonl')],capture_output=True,text=True)
resources=json.loads(p.stdout) if p.stdout.strip() else dict(error=p.stderr)
out=dict(node=node,mode=mode,service_exit=d.get('root_exit'),cleanup=d.get('cleanup_completed'),resources={k:v for k,v in resources.items() if k in ('valid','foreign','error')})
if node=='118':
 b=run/'bench_ownership.json';d=json.loads(b.read_text()) if b.exists() else {}
 s=(run/'bench_client.log').read_text(errors='replace') if (run/'bench_client.log').exists() else ''
 out.update(bench_exit=d.get('root_exit'),bench_cleanup=d.get('cleanup_completed'),progress=re.findall(r'POST=\d+[^\r\n]*',s)[-1:],validation='FULL503' in s or 'ROUND_VALIDATION_OK' in s,tail=s[-240:] if d.get('root_exit') is not None else '')
print(json.dumps(out),flush=True)
if resources.get('foreign') or d.get('remaining_owned'):sys.exit(2)
