#!/usr/bin/env python3
"""Record descendants, gracefully stop only exact task-owned PIDs. No force kill."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

script,record=sys.argv[1:]
target=Path(record)
assert not target.exists(),target
target.parent.mkdir(parents=True,exist_ok=True)
child=subprocess.Popen(["bash",script],start_new_session=True)
owned={}
stop=False
def stat(pid):
    p=Path("/proc")/str(pid)
    data=(p/"stat").read_text()
    fields=data[data.rindex(")")+2:].split()
    return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),
                start_ticks=int(fields[19]),
                cmdline=(p/"cmdline").read_bytes().decode(errors="replace").replace("\0"," ").strip())
def live_owned():
    result=[]
    for rec in owned.values():
        try:
            now=stat(rec["pid"])
            if now["start_ticks"]==rec["start_ticks"] and now["state"]!="Z":result.append(rec)
        except (OSError,ValueError):pass
    return result
def sample():
    current={}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:current[int(p.name)]=stat(int(p.name))
        except (OSError,ValueError):pass
    todo=[child.pid]
    # Previously recorded living descendants may spawn after parent reaping.
    todo.extend(r["pid"] for r in live_owned())
    seen=set()
    while todo:
        pid=todo.pop()
        if pid in seen:continue
        seen.add(pid)
        if pid in current:owned[str(pid)]=current[pid]
        todo.extend(x["pid"] for x in current.values() if x["ppid"]==pid and x["pid"]!=pid)
    data=dict(runner_pid=os.getpid(),root_pid=child.pid,script=script,owned=list(owned.values()),
              root_exit=child.poll(),updated_at=time.time(),force_kill_permitted=False)
    tmp=target.with_suffix(".tmp");tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,target)
def request_stop(_signum,_frame):
    global stop
    stop=True
# Block SIGTERM only after Popen: the child retains its normal signal mask.
# sigwaitinfo records the real sender so unexplained orchestration stops are auditable.
import threading
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
def receive_term():
    info=signal.sigwaitinfo({signal.SIGTERM})
    source=Path('/proc')/str(info.si_pid)/'comm'
    sender=source.read_text().strip() if source.exists() else '<exited>'
    print('OWNED_RUNNER_SIGTERM', json.dumps(dict(sender_pid=info.si_pid,sender_uid=info.si_uid,signal_code=info.si_code,sender_comm=sender,at=time.time())),flush=True)
    request_stop(info.si_signo,None)
threading.Thread(target=receive_term,daemon=True).start()
signal.signal(signal.SIGINT,request_stop)
sample()
try:
    while child.poll() is None and not stop:
        if not target.with_suffix(".quiet").exists():sample()
        time.sleep(1)
finally:
    sample()
    # Normal shutdown starts at root, giving the service time to reap its workers.
    records=sorted(live_owned(),key=lambda x:x["pid"]!=child.pid)
    for rec in records[:1]:
        try:os.kill(rec["pid"],signal.SIGTERM)
        except ProcessLookupError:pass
    until=time.monotonic()+90
    while live_owned() and time.monotonic()<until:time.sleep(1)
    # Remaining descendants get one normal SIGTERM, exact start-tick checked.
    for rec in live_owned():
        try:
            if stat(rec["pid"])["start_ticks"]==rec["start_ticks"]:
                os.kill(rec["pid"],signal.SIGTERM)
        except ProcessLookupError:pass
    until=time.monotonic()+30
    while live_owned() and time.monotonic()<until:time.sleep(1)
    sample()
    data=json.loads(target.read_text())
    data.update(cleanup_completed=not live_owned(),remaining_owned=live_owned())
    target.write_text(json.dumps(data,indent=2))
    print("OWNED_TASK_ENDED",child.poll(),"cleanup",data["cleanup_completed"],flush=True)
    if not data["cleanup_completed"]:
        print("BLOCKED: own processes did not exit normally; no force kill, no next job",flush=True)
        sys.exit(2)
sys.exit(child.returncode if child.returncode is not None else 1)
