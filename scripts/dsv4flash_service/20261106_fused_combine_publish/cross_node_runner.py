#!/usr/bin/env python3
"""One rank per physical host; only recorded task-owned children may be stopped."""
import argparse, datetime, fcntl, json, os, re, signal, socket, struct, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--node", type=int, choices=[0,1], required=True)
p.add_argument("--output", required=True)
p.add_argument("--master-port", type=int, required=True)
p.add_argument("--shmem-port", type=int, required=True)
p.add_argument("--library", required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
node_ip=["141.61.45.118","141.61.45.124"][a.node]
lock=(root/f".cross_weighted_node{a.node}.lock").open("a")
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
    v=fcntl.ioctl(s.fileno(),0x8915,struct.pack("256s",b"enp34s0f1"))
assert socket.inet_ntoa(v[20:24])==node_ip
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"]=="1"
started=out/f"started_node{a.node}.json"
assert not started.exists(),"Use a fresh RUN_ID"
if a.node==0:
    for port in [a.master_port,a.shmem_port]:
        with socket.socket() as s:s.bind((node_ip,port))
started.write_text(json.dumps(dict(node=a.node,node_ip=node_ip,controller_pid=os.getpid(),
    card=1,at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
env=dict(os.environ,RANK=str(a.node),WORLD_SIZE="2",LOCAL_RANK="0",
         MASTER_ADDR="141.61.45.118",MASTER_PORT=str(a.master_port))
log=(out/f"rank{a.node}.log").open("xb")
child=subprocess.Popen([sys.executable,str(root/"test_cross_node_udma_sync.py"),
    "--output",str(out),"--shmem-port",str(a.shmem_port),"--library",a.library],env=env,
    stdout=log,stderr=subprocess.STDOUT)
owned={};stop=None;began=time.monotonic()
def stat(pid):
    text=Path(f"/proc/{pid}/stat").read_text()
    f=text[text.rindex(")")+2:].split()
    return dict(pid=pid,ppid=int(f[1]),start_ticks=int(f[19]))
owned[child.pid]=stat(child.pid)
def live_owned():
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
                live.add(sub);owned[sub]=rec;todo.append(sub)
    return live
def signal_stop(signum,frame):
    global stop
    stop=f"received signal {signum}"
signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
record=out/f"children_node{a.node}.json"
record.write_text(json.dumps(dict(pid=child.pid,node=a.node,node_ip=node_ip,card=1,owned=list(owned.values())),indent=2))
next_check=began
known_host_pids={}
abort=out/"ABORT.json"
while child.poll() is None and stop is None:
    if abort.exists():stop="peer/local abort: "+abort.read_text();break
    if time.monotonic()-began>600:stop="600 second deadline";break
    if time.monotonic()>=next_check:
        live=live_owned()
        raw=subprocess.check_output(["npu-smi","info"],text=True)
        rows=re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|",raw)
        # Keep verified host/container PID bindings before namespace IDs disappear
        # during context teardown. Unknown host PIDs never become task-owned.
        host_audit=out/f"host_pid_audit_node{a.node}.json"
        if host_audit.exists():
            binding=json.loads(host_audit.read_text())
            cid=binding["container_pid"]
            if cid in owned and owned[cid]["start_ticks"]==binding["start_ticks"]:
                known_host_pids[binding["host_pid"]]=binding
        selected=[row for row in rows if int(row[0])==1]
        for row in selected:
            container_pid=int(row[4])
            if container_pid in live:
                known_host_pids[int(row[1])]=dict(host_pid=int(row[1]),
                    container_pid=container_pid,start_ticks=owned[container_pid]["start_ticks"])
        exited=child.poll()
        rank_path=out/f"rank{a.node}.json"
        completed_ok=exited==0 and rank_path.exists() and json.loads(rank_path.read_text())["status"]=="passed"
        foreign=[];teardown_rows=[]
        for row in selected:
            if int(row[4]) in live:continue
            binding=known_host_pids.get(int(row[1]))
            if binding and binding["container_pid"] in live and owned[binding["container_pid"]]["start_ticks"]==binding["start_ticks"]:
                continue
            if binding and completed_ok:
                teardown_rows.append(row)
                continue
            foreign.append(row)
        audit=dict(at=time.time(),raw=raw,foreign=foreign,teardown_rows=teardown_rows,
                   known_host_pids=list(known_host_pids.values()),child_exit_during_sample=exited,
                   owned=list(owned.values()))
        (out/f"resources_node{a.node}.json").write_text(json.dumps(audit,indent=2))
        with (out/f"resource_history_node{a.node}.jsonl").open("a") as f:
            f.write(json.dumps(audit)+"\n")
        if foreign:stop=f"foreign process on physical card1: {foreign}";break
        next_check=time.monotonic()+10
    time.sleep(0.5)
if child.poll() not in (None,0) and stop is None:stop=f"rank failed with {child.returncode}"
if stop:
    payload=dict(node=a.node,reason=stop,at=time.time())
    try:
        with abort.open("x") as f:json.dump(payload,f,indent=2)
    except FileExistsError:pass
    if child.poll() is None:child.terminate()
    deadline=time.monotonic()+60
    while child.poll() is None and time.monotonic()<deadline:time.sleep(0.5)
data=dict(node=a.node,node_ip=node_ip,card=1,pid=child.pid,exit_code=child.poll(),
          stop_reason=stop,cleanup_completed=child.poll() is not None,owned=list(owned.values()),
          force_kill_permitted=False)
record.write_text(json.dumps(data,indent=2));log.close()
print("CROSS_NODE_RANK_ENDED",json.dumps(data),flush=True)
raise SystemExit(0 if child.poll()==0 and stop is None else 1)
