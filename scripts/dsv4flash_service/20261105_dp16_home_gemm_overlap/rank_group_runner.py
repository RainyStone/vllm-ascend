#!/usr/bin/env python3
"""Launch explicit global ranks; stop only this launch's exact children normally."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from tests.test_home_gmm_npu import require_time
require_time()
p=argparse.ArgumentParser()
p.add_argument("--world",type=int,choices=[2,16],required=True)
p.add_argument("--node",type=int,choices=[0,1],required=True)
p.add_argument("--backend",choices=["mte","udma"],required=True)
p.add_argument("--output",required=True)
p.add_argument("--library",required=True)
p.add_argument("--master-port",type=int,required=True)
p.add_argument("--shmem-port",type=int,required=True)
args=p.parse_args()
root=Path(__file__).resolve().parent
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
record=out/f"node{args.node}_children.json"
assert not record.exists(),"Do not restart an existing run ID"
children=[];logs=[]
stop=False
def signal_stop(_s,_f):
    global stop
    stop=True
signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
for local in range(args.world//2):
    rank=args.node*(args.world//2)+local
    env=dict(os.environ,RANK=str(rank),WORLD_SIZE=str(args.world),LOCAL_RANK=str(local),
             MASTER_ADDR="141.61.45.118",MASTER_PORT=str(args.master_port))
    log=(out/f"rank{rank}.log").open("xb")
    child=subprocess.Popen([sys.executable,str(root/"tests/test_home_cross.py"),
        "--backend",args.backend,"--output",str(out),"--library",args.library,
        "--shmem-port",str(args.shmem_port)],env=env,stdout=log,stderr=subprocess.STDOUT)
    children.append(child);logs.append(log)
    print("RANK_STARTED",rank,"pid",child.pid,"log",out/f"rank{rank}.log",flush=True)
record.write_text(json.dumps(dict(node=args.node,pids=[c.pid for c in children],
    world=args.world,backend=args.backend),indent=2))
began=time.monotonic()
while not stop and any(c.poll() is None for c in children):
    if any(c.poll() not in (None,0) for c in children) or time.monotonic()-began>1200:stop=True;break
    time.sleep(1)
if stop:
    for child in children:
        if child.poll() is None:child.terminate()  # direct Popen children only, no PID search
    deadline=time.monotonic()+60
    while any(c.poll() is None for c in children) and time.monotonic()<deadline:time.sleep(1)
data=json.loads(record.read_text());data["exit_codes"]=[c.poll() for c in children]
data["cleanup_completed"]=all(c.poll() is not None for c in children)
record.write_text(json.dumps(data,indent=2))
for log in logs:log.close()
if not data["cleanup_completed"]:print("BLOCKED: own rank did not exit normally; no force kill",flush=True)
raise SystemExit(0 if all(c.poll()==0 for c in children) else 1)
