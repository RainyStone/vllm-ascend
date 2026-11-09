#!/usr/bin/env python3
"""Single-card test launcher; stop only its exact child if another job appears."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
cards = set(map(int, os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")))
child = subprocess.Popen([sys.executable, str(root / "test_single_publish.py"),
                          "--library", args.library, "--output", args.output])
owned = {}
stop_reason = None


def stat(pid):
    path = Path("/proc") / str(pid)
    value = (path / "stat").read_text()
    fields = value[value.rindex(")") + 2:].split()
    return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]))


owned[child.pid] = stat(child.pid)


def live_descendants():
    current = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            current[int(path.name)] = stat(int(path.name))
        except (OSError, ValueError):
            pass
    live = {pid for pid, rec in owned.items()
            if pid in current and current[pid]["start_ticks"] == rec["start_ticks"]}
    todo = list(live)
    while todo:
        pid = todo.pop()
        for sub, rec in current.items():
            if rec["ppid"] == pid and sub not in live:
                live.add(sub)
                owned[sub] = rec
                todo.append(sub)
    return live


def stop(signum, frame):
    global stop_reason
    stop_reason = f"received signal {signum}"


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
began = time.monotonic()
next_check = began
while child.poll() is None and stop_reason is None:
    if time.monotonic() - began > 600:
        stop_reason = "600 second test deadline"
        break
    if time.monotonic() >= next_check:
        live = live_descendants()
        raw = subprocess.check_output(["npu-smi", "info"], text=True)
        rows = re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|", raw)
        foreign = [row for row in rows if int(row[0]) in cards and int(row[4]) not in live]
        (output.parent / "last_resources.json").write_text(
            json.dumps(dict(raw=raw, foreign=foreign, owned=list(owned.values())), indent=2)
        )
        if foreign:
            stop_reason = f"foreign processes appeared: {foreign}"
            break
        next_check = time.monotonic() + 5
    time.sleep(0.5)
if stop_reason is not None and child.poll() is None:
    child.terminate()
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
record = dict(child_pid=child.pid, exit_code=child.poll(), stop_reason=stop_reason,
              cleanup_completed=child.poll() is not None, owned=list(owned.values()))
(output.parent / "single_children.json").write_text(json.dumps(record, indent=2))
print("SINGLE_CHILD_ENDED", child.poll(), stop_reason, flush=True)
raise SystemExit(0 if child.poll() == 0 and stop_reason is None else 1)
