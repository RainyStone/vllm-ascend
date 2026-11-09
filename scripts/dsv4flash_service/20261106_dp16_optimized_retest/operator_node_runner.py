#!/usr/bin/env python3
"""Recorded children only; resource conflicts invalidate and stop this test."""
import argparse,datetime,fcntl,json,os,re,signal,socket,struct,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ("node","local-size","master-port","shmem-port"):p.add_argument("--"+name,type=int,required=True)
for name in ("output","backend","library"):p.add_argument("--"+name,required=True)
a=p.parse_args();root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
assert a.node in (0,1) and a.local_size in (1,8) and a.backend in ("mte","udma")
cards=[1] if a.local_size==1 else list(range(8))
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"]==",".join(map(str,cards))
ip=["141.61.45.118","141.61.45.124"][a.node]
lock=(root/f".operator_node{a.node}.lock").open("a");fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sk:
 address=fcntl.ioctl(sk.fileno(),0x8915,struct.pack("256s",b"enp34s0f1"))
assert socket.inet_ntoa(address[20:24])==ip
marker=out/f"started_node{a.node}.json"
assert not marker.exists(),"fresh RUN_ID required"
if a.node==0:
 for port in (a.master_port,a.shmem_port):
  with socket.socket() as sk:sk.bind((ip,port))
def sample():
 raw=subprocess.check_output(["npu-smi","info"],text=True)
 rows=re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|",raw)
 return raw,[row for row in rows if int(row[0]) in cards]
raw,rows=sample();assert not rows,("selected cards occupied",rows)
marker.write_text(json.dumps(dict(node=a.node,controller_pid=os.getpid(),cards=cards,at=time.time()),indent=2))
def stat(pid):
 text=Path(f"/proc/{pid}/stat").read_text();fields=text[text.rindex(")")+2:].split()
 return dict(pid=pid,ppid=int(fields[1]),start_ticks=int(fields[19]))
children=[];owned={};logs=[];stop=None;began=time.monotonic();abort=out/"ABORT.json"
for local in range(a.local_size):
 rank=a.node*a.local_size+local
 env=dict(os.environ,RANK=str(rank),WORLD_SIZE=str(2*a.local_size),LOCAL_RANK=str(local),MASTER_ADDR="141.61.45.118",MASTER_PORT=str(a.master_port))
 log=(out/f"rank{rank}.log").open("xb");logs.append(log)
 child=subprocess.Popen([sys.executable,str(root/("test_optimized_full_cross.py" if a.local_size==8 else "test_optimized_cross.py")),"--backend",a.backend,"--output",str(out),"--library",a.library,"--shmem-port",str(a.shmem_port),"--bootstrap-ip","141.61.45.118"],env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append((rank,child));owned[child.pid]=stat(child.pid)
record=out/f"children_node{a.node}.json"
def snapshot():return dict(node=a.node,cards=cards,children=[dict(rank=rr,pid=cc.pid,exit_code=cc.poll(),start_ticks=owned[cc.pid]["start_ticks"]) for rr,cc in children],owned=list(owned.values()))
record.write_text(json.dumps(snapshot(),indent=2))
def alive_owned():
 current={}
 for q in Path("/proc").iterdir():
  if not q.name.isdigit():continue
  try:current[int(q.name)]=stat(int(q.name))
  except (OSError,ValueError):pass
 live={pid for pid,rec in owned.items() if pid in current and current[pid]["start_ticks"]==rec["start_ticks"]}
 todo=list(live)
 while todo:
  pid=todo.pop()
  for sub,rec in current.items():
   if rec["ppid"]==pid and sub not in live:
    owned[sub]=rec;live.add(sub);todo.append(sub)
 return live
def stopped(signum,frame):
 global stop
 stop=f"received signal {signum}"
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
next_sample=began;known={}
while any(cc.poll() is None for _,cc in children) and stop is None:
 if abort.exists():stop="peer abort: "+abort.read_text();break
 failed=[(rr,cc.returncode) for rr,cc in children if cc.poll() not in (None,0)]
 if failed:stop=f"rank failure {failed}";break
 if time.monotonic()-began>1200:stop="1200 second deadline";break
 if time.monotonic()>=next_sample:
  live=alive_owned();raw,rows=sample()
  bindings=out/f"host_pid_audit_node{a.node}.json"
  if bindings.exists():
   for v in json.loads(bindings.read_text()):
    cid=v["container_pid"]
    if cid in owned and owned[cid]["start_ticks"]==v["start_ticks"]:known[v["host_pid"]]=v
  for row in rows:
   cid=int(row[4])
   if cid in live:known[int(row[1])]=dict(host_pid=int(row[1]),container_pid=cid,start_ticks=owned[cid]["start_ticks"])
  completed={cc.pid for rr,cc in children if cc.poll()==0 and (out/f"rank{rr}.json").exists() and json.loads((out/f"rank{rr}.json").read_text())["status"]=="passed"}
  foreign=[]
  for row in rows:
   if int(row[4]) in live:continue
   bound=known.get(int(row[1]))
   if bound and bound["container_pid"] in live|completed:continue
   foreign.append(row)
  audit=dict(at=time.time(),raw=raw,foreign=foreign,known_host_pids=list(known.values()),owned=list(owned.values()))
  with (out/f"resource_history_node{a.node}.jsonl").open("a") as ff:ff.write(json.dumps(audit)+"\n")
  if foreign:stop=f"foreign process on selected cards: {foreign}";break
  next_sample=time.monotonic()+5
 time.sleep(.5)
if any(cc.poll() not in (None,0) for _,cc in children) and stop is None:stop="rank failed"
if stop:
 try:
  with abort.open("x") as ff:json.dump(dict(node=a.node,reason=stop,at=time.time()),ff,indent=2)
 except FileExistsError:pass
 for rr,cc in children:
  if cc.poll() is None:cc.terminate()
 deadline=time.monotonic()+60
 while any(cc.poll() is None for _,cc in children) and time.monotonic()<deadline:time.sleep(.5)
data=snapshot();data.update(stop_reason=stop,cleanup_completed=all(cc.poll() is not None for _,cc in children),force_kill_permitted=False)
record.write_text(json.dumps(data,indent=2))
for log in logs:log.close()
print("OPERATOR_NODE_ENDED",json.dumps(data),flush=True)
raise SystemExit(0 if all(cc.poll()==0 for _,cc in children) and stop is None else 1)
