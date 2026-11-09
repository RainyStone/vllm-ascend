#!/usr/bin/env python3
"""Host-side, bounded read-only SIGTERM tracing for exact current task PIDs on 124."""
import json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
d=json.loads((r/'STATE.json').read_text());mode=sys.argv[1] if len(sys.argv)>1 else 'off';assert mode in ('off','slot_mte','slot_udma');run=r/'data/formal'/d['formal_run_id']/mode
o=json.loads((run/'node124_ownership.json').read_text())
assert o['root_exit'] is None
expected={o['root_pid'],o['runner_pid']}
init=int(subprocess.check_output(['docker','inspect','-f','{{.State.Pid}}','xzx_moonep'],text=True))
namespace=os.readlink(f'/proc/{init}/ns/pid')
mapping={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if os.readlink(p/'ns/pid')!=namespace:continue
  status=(p/'status').read_text().splitlines()
  ids=next(x.split()[1:] for x in status if x.startswith('NSpid:'))
  local=int(ids[-1])
  if local not in expected:continue
  stat=(p/'stat').read_text();fields=stat[stat.rindex(')')+2:].split()
  rec=next(x for x in o['owned'] if x['pid']==local) if local==o['root_pid'] else None
  if rec is not None:assert int(fields[19])==rec['start_ticks']
  mapping[str(local)]=dict(host_pid=int(p.name),start_ticks=int(fields[19]))
 except (OSError,StopIteration):continue
assert set(map(int,mapping))==expected,mapping
(run/'signal_trace_mapping.json').write_text(json.dumps(dict(namespace=namespace,container='xzx_moonep',mapping=mapping),indent=2))
predicate=' || '.join('args->pid == '+str(x['host_pid']) for x in mapping.values())
program='tracepoint:signal:signal_generate /args->sig == 15 && ('+predicate+')/ { printf("OWNED_SIGTERM sender_host_pid=%d sender_comm=%s parent_host_pid=%d uid=%d target_host_pid=%d target_comm=%s code=%d result=%d elapsed_ns=%llu\\n", pid, comm, curtask->real_parent->tgid, uid, args->pid, args->comm, args->code, args->result, elapsed); }\ninterval:s:1200 { exit(); }\n'
bt=run/'signal_trace_124_v2.bt';bt.write_text(program)
script=run/'trace_sigterm_124.sh';script.write_text('#!/usr/bin/env bash\nset -eu\nexec /usr/bin/bpftrace '+str(bt)+'\n')
log=(run/'signal_trace_124_v2.log').open('xb')
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
py='/usr/bin/python3'
os.execv(py,[py,str(r/'owned_task_runner.py'),str(script),str(run/'signal_trace_124_v2_ownership.json')])
