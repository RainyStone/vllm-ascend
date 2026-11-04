#!/usr/bin/env python3
import json,re
from pathlib import Path
D=Path(__file__).resolve().parent
root=D.parent
run=Path(json.loads((D/'CURRENT_DIAGNOSIS.json').read_text())['execution'])
state=json.loads((run/'state.json').read_text())
out=dict(event=state['event'],mode=state.get('mode'),completed_modes=len(list(run.glob('*/case_result.json'))))
if state['event']=='census_running':
    mode=state['mode']
    folders=list((root/'data/aisbench').glob(f"diagnostic-{mode}-*-{state['client_pid']}"))
    if folders:
        folder=folders[0]/'round_1';out['client_dir']=str(folders[0])
        if (folder/'validation.json').exists():out['validation']=json.loads((folder/'validation.json').read_text())
        log=folder/'aisbench_debug.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-10000));text=f.read().decode(errors='replace')
            values=re.findall(r'(\d+)/316',text)
            out['completed_requests']=int(values[-1]) if values else 0
            values=re.findall(r'FAIL=(\d+)',text)
            out['observed_failed_requests']=int(values[-1]) if values else 0
elif state['event']=='precision_running':
    p=Path(state['output'])/'responses.jsonl'
    if p.exists():
        with p.open() as f:out['completed_requests']=sum(1 for _ in f)
elif state['event']=='stopped_on_error':out['error']=state.get('error')
print(json.dumps(out,ensure_ascii=False))
