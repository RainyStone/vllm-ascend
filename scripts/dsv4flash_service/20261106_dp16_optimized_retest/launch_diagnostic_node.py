#!/usr/bin/env python3
"""Launch one separately recorded diagnostic node; never modify formal records."""
import datetime as dt,fcntl,json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent
kind,node,mode=sys.argv[1:]
assert kind in ('profiling','rankload') and node in ('118','124') and mode in ('off','slot_mte','slot_udma')
with (r/'.diagnostic_controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=r/'STATE.json';d=json.loads(state.read_text())
 assert all(d['formal'][x].get('benchmark_passed') and d['formal'][x].get('post_audit_passed') and d['formal'][x].get('cleanup_completed') for x in ('off','slot_mte','slot_udma'))
 diag=d.setdefault('diagnostics',{}).setdefault(kind,dict(run_id=f'optimized_{kind}_20261106'))
 run=r/'data/diagnostics'/diag['run_id']/mode;run.mkdir(parents=True,exist_ok=True)
 f=(run/f'node{node}_runner.log').open('xb')
 assert not (run/f'node{node}_ownership.json').exists()
 script=r/'diagnostics'/kind/f'start_{mode}_{node}.sh';assert script.is_file()
 entry=diag.setdefault('modes',{}).setdefault(mode,{})
 entry.setdefault('launch_intents',{})[node]=dict(at=dt.datetime.now(dt.timezone.utc).isoformat(),script=str(script),launcher_pid=os.getpid())
 d['phase']=f'diagnostic_{kind}_{mode}_starting'
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,state)
os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
py='/usr/local/python3.11.10/bin/python'
os.execv(py,[py,str(r/'owned_task_runner.py'),str(script),str(run/f'node{node}_ownership.json')])
