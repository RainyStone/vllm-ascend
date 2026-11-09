#!/usr/bin/env python3
"""Launch exactly one approved node script, preserving a durable launch intent."""
import datetime as dt
import fcntl,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
node,mode=sys.argv[1:]
assert node in ('118','124') and mode in ('off','slot_mte','slot_udma')
with (root/'.formal_controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=root/'STATE.json';d=json.loads(state.read_text())
 run=root/'data/formal'/d['formal_run_id']/mode
 run.mkdir(parents=True,exist_ok=True)
 log=(run/f'node{node}_runner.log').open('xb')
 assert not (run/f'node{node}_ownership.json').exists()
 script=root/f'start_{mode}_{node}.sh'
 assert script.is_file()
 intent=dict(node=node,mode=mode,at=dt.datetime.now(dt.timezone.utc).isoformat(),script=str(script),launcher_pid=os.getpid())
 (run/f'node{node}_launch.json').write_text(json.dumps(intent,indent=2))
 d['phase']=f'formal_{mode}_starting'
 d.setdefault('formal',{}).setdefault(mode,{}).setdefault('launch_intents',{})[node]=intent
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,state)
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
py='/usr/local/python3.11.10/bin/python'
os.execv(py,[py,str(root/'owned_task_runner.py'),str(script),str(run/f'node{node}_ownership.json')])
