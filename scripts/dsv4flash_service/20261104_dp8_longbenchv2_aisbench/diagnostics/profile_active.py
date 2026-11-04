#!/usr/bin/env python3
import json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
D=Path(__file__).resolve().parent
run=Path(json.loads((D/'CURRENT_DIAGNOSIS.json').read_text())['execution'])
state=json.loads((run/'state.json').read_text())
mode=sys.argv[1]
assert state['mode']==mode and state['event']=='census_running',state
case=run/mode
audit=json.loads(next(case.glob('*_mode_audit.json')).read_text())
pids=[int(x['pid']) for x in audit['workers']]
assert len(pids)==8 and all((Path('/proc')/str(p)).exists() for p in pids)
output=case/'dynamic_profile';assert not output.exists()
command=['/usr/local/Ascend/cann/tools/profiler/bin/msprof','--dynamic=on',
         '--pid='+','.join(map(str,pids)),'--duration=10','--output='+str(output),
         '--task-time=on','--runtime-api=on','--ai-core=on','--aicpu=on',
         '--type=text','--storage-limit=1000MB']
record=dict(mode=mode,command=command,worker_pids=pids,formal_performance_data=False,
            start=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
with (case/'dynamic_profile_tool.log').open('w') as f:
    result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=180)
record.update(exit_code=result.returncode,end=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
(case/'dynamic_profile_attempt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
