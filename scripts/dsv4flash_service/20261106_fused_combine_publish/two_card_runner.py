#!/usr/bin/env python3
"""Explicit local two-rank launcher; detects contention and stops only its Popen children."""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--backend",choices=["mte","udma"],required=True);p.add_argument("--output",required=True);p.add_argument("--library",required=True);p.add_argument("--master-port",type=int,required=True);p.add_argument("--shmem-port",type=int,required=True);p.add_argument("--master-addr",required=True);p.add_argument("--test-file",choices=["test_two_publish.py","test_udma_publish_diagnostic.py","test_udma_sync_probe.py","test_udma_extended_regression.py","test_udma_fixed_timing.py","test_udma_constant_timing.py"],default="test_two_publish.py");a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);record=out/"children.json";assert not record.exists()
cards=list(map(int,os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")));assert len(cards)==2
root=Path(__file__).resolve().parent;children=[];logs=[];stop=False;reason=None;began=time.monotonic()
def signaled(s,f):
 global stop,reason
 stop=True;reason=f"received signal {s}"
signal.signal(signal.SIGTERM,signaled);signal.signal(signal.SIGINT,signaled)
def stat(pid):
 t=Path(f"/proc/{pid}/stat").read_text();f=t[t.rindex(")")+2:].split()
 return dict(pid=pid,ppid=int(f[1]),start_ticks=int(f[19]))
owned={}
for local in range(2):
 env=dict(os.environ,RANK=str(local),WORLD_SIZE="2",LOCAL_RANK=str(local),MASTER_ADDR=a.master_addr,MASTER_PORT=str(a.master_port))
 log=(out/f"rank{local}.log").open("xb")
 ch=subprocess.Popen([sys.executable,str(root/a.test_file),"--backend",a.backend,"--output",str(out),"--library",a.library,"--shmem-port",str(a.shmem_port),"--bootstrap-ip",a.master_addr],env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append(ch);logs.append(log);owned[ch.pid]=stat(ch.pid)
 print("OWNED_RANK_STARTED",local,ch.pid,flush=True)
def sample_owned():
 current={}
 for path in Path("/proc").iterdir():
  if not path.name.isdigit():continue
  try:current[int(path.name)]=stat(int(path.name))
  except (OSError,ValueError):pass
 roots={pid for pid,rec in owned.items() if pid in current and current[pid]["start_ticks"]==rec["start_ticks"]}
 todo=list(roots)
 while todo:
  pid=todo.pop()
  for sub,rec in current.items():
   if rec["ppid"]==pid and sub not in roots:roots.add(sub);owned[sub]=rec;todo.append(sub)
 return roots
next_check=began
record.write_text(json.dumps(dict(pids=[c.pid for c in children],cards=cards,backend=a.backend),indent=2))
while not stop and any(c.poll() is None for c in children):
 if any(c.poll() not in (None,0) for c in children):stop=True;reason="rank failed";break
 if time.monotonic()-began>600:stop=True;reason="600 second deadline";break
 if time.monotonic()>=next_check:
  live=sample_owned()
  raw=subprocess.check_output(["npu-smi","info"],text=True)
  rows=re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|",raw)
  foreign=[row for row in rows if int(row[0]) in cards and int(row[4]) not in live]
  (out/"last_resources.json").write_text(json.dumps(dict(raw=raw,foreign=foreign,owned=list(owned.values())),indent=2))
  if foreign:stop=True;reason=f"foreign process appeared on selected cards: {foreign}";break
  next_check=time.monotonic()+10
 time.sleep(1)
if stop:
 print("STOP_OWN_TASK",reason,flush=True)
 for ch in children:
  if ch.poll() is None:ch.terminate()
 deadline=time.monotonic()+60
 while any(c.poll() is None for c in children) and time.monotonic()<deadline:time.sleep(1)
data=json.loads(record.read_text());data.update(exit_codes=[c.poll() for c in children],cleanup_completed=all(c.poll() is not None for c in children),stop_reason=reason,owned=list(owned.values()))
record.write_text(json.dumps(data,indent=2))
for log in logs:log.close()
print("RANK_GROUP_ENDED",json.dumps(data),flush=True)
raise SystemExit(0 if all(c.poll()==0 for c in children) else 1)
