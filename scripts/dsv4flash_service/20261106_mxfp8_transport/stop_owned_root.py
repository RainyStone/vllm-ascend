#!/usr/bin/env python3
"""Request normal shutdown of one exact root launched by this experiment."""
import argparse
import json
import os
from pathlib import Path
import signal

p=argparse.ArgumentParser();p.add_argument("ownership");args=p.parse_args()
root=Path(__file__).resolve().parent
path=Path(args.ownership).resolve()
assert path.is_relative_to(root/"data"),path
owner=json.loads(path.read_text())
pid=owner["root_pid"]
record=next(x for x in owner["owned"] if x["pid"]==pid)
proc=Path("/proc")/str(pid)
if not proc.exists():
    print("OWNED_ROOT_ALREADY_EXITED",pid)
    raise SystemExit(0)
text=(proc/"stat").read_text();fields=text[text.rindex(")")+2:].split()
assert int(fields[19])==record["start_ticks"],"PID was reused: never signal it"
command=(proc/"cmdline").read_bytes().decode(errors="replace").replace("\0"," ")
assert str(root) in record["cmdline"] or "vllm.entrypoints.cli.main" in record["cmdline"],record
assert str(root) in command or "vllm.entrypoints.cli.main" in command,command
os.kill(pid,signal.SIGTERM)
print("NORMAL_SHUTDOWN_REQUESTED_EXACT_OWNED_ROOT",pid)
