#!/usr/bin/env python3
"""Run the approved off/Slot load census and first-token diagnosis separately."""
import json,os,re,shutil,signal,socket,subprocess,time,urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
DIAG=ROOT/'diagnostics'
PYTHON='/usr/local/python3.11.10/bin/python'
EXPERIMENT=DIAG/'data/executions'/('diagnosis_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
EXPERIMENT.mkdir(parents=True,exist_ok=False)
def log(event,**kw):
    record=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),event=event,**kw)
    (EXPERIMENT/'state.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    with (EXPERIMENT/'events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
def free_devices(timeout=180):
    began=time.monotonic()
    while True:
        npu=subprocess.check_output(['npu-smi','info'],text=True)
        free=set(re.findall(r'No running processes found in NPU (\d+)',npu))
        available=False
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',8444))
            available=True
        except OSError:pass
        if free==set(map(str,range(8))) and available:return npu
        if time.monotonic()-began>timeout:raise RuntimeError('Eight NPUs / port8444 unavailable; will not touch unowned processes')
        time.sleep(5)
def stop(process):
    if process is None:return
    assert process.pid!=os.getpgrp()
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    time.sleep(5)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:pass
def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))
def models():
    with opener().open('http://127.0.0.1:8444/v1/models',timeout=5) as r:
        return [x['id'] for x in json.load(r)['data']]
def marker(mode,name,folder):
    body=dict(model='dsv4flash_longbench_'+mode,prompt='Hello',max_tokens=1,temperature=0,
              seed=1024,ignore_eos=True,stream=False)
    req=urllib.request.Request('http://127.0.0.1:8444/v1/completions',data=json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    with opener().open(req,timeout=300) as r:response=json.load(r)
    assert response['usage']['completion_tokens']==1
    (folder/(name+'.json')).write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
def census_files(stats):
    files=sorted(stats.glob('moe_par_*_rank*.log'));assert len(files)==8,files
    output={}
    for file in files:
        text=file.read_text();meta=re.search(r'^# meta .*?ep_rank=(\d+)',text,re.M)
        assert meta,file
        rank=int(meta[1]);steps=[int(x) for x in re.findall(r'^step=(\d+)',text,re.M)]
        assert steps,(file,'No complete logged step')
        output[rank]=dict(file=file,last_step=max(steps))
    assert set(output)==set(range(8));return output
def await_process(child,service,limit):
    began=time.monotonic()
    while child.poll() is None:
        if service.poll() is not None:raise RuntimeError('Service exited during diagnosis')
        if time.monotonic()-began>limit:raise TimeoutError('Diagnostic phase exceeded timeout')
        time.sleep(3)
    assert child.returncode==0,('Diagnostic command failed',child.returncode)
def run_mode(mode):
    case=EXPERIMENT/mode;case.mkdir()
    service=None;bench=None;probe=None;handles=[]
    try:
        (case/'npu_before.txt').write_text(free_devices())
        stats=DIAG/'data'/mode/'routing'
        assert not list(stats.glob('*.log')),'Existing routing files must be preserved before a rerun'
        handle=(case/'service_controller.log').open('w');handles.append(handle)
        service=subprocess.Popen(['bash',str(DIAG/f'start_{mode}_diagnostic.sh')],cwd=ROOT,
                                 stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        log('starting_service',mode=mode,service_pid=service.pid,execution=str(EXPERIMENT))
        began=time.monotonic()
        while True:
            if service.poll() is not None:raise RuntimeError('Service exited during startup')
            try:
                if 'dsv4flash_longbench_'+mode in models():break
            except Exception:pass
            if time.monotonic()-began>1500:raise TimeoutError('Service startup exceeded1500s')
            time.sleep(3)
        candidates=[p for p in (ROOT/'data/services').glob(f'diagnostic-{mode}-*')
                    if (p/'service.pid').exists() and int((p/'service.pid').read_text())==service.pid]
        assert len(candidates)==1;service_dir=candidates[0]
        subprocess.run([PYTHON,str(DIAG/'audit_diagnostic_mode.py'),mode,str(service_dir),str(case)],check=True)
        log('isolating_census_start',mode=mode,service_dir=str(service_dir))
        marker(mode,'before_census_marker1',case)
        marker(mode,'before_census_marker2',case)
        before=census_files(stats)
        baseline=case/'routing_before';baseline.mkdir()
        for row in before.values():shutil.copy2(row['file'],baseline/row['file'].name)
        handle=(case/'census_client.log').open('w');handles.append(handle)
        bench=subprocess.Popen(['bash',str(DIAG/f'bench_{mode}_diagnostic.sh')],cwd=ROOT,
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        log('census_running',mode=mode,service_pid=service.pid,client_pid=bench.pid,service_dir=str(service_dir))
        await_process(bench,service,10800)
        candidates=list((ROOT/'data/aisbench').glob(f'diagnostic-{mode}-*-{bench.pid}'));assert len(candidates)==1
        client_dir=candidates[0]
        validation=json.loads((client_dir/'round_1/validation.json').read_text())
        assert validation['valid'] and validation['requests']==316
        marker(mode,'after_census_marker',case)
        after=census_files(stats)
        captured=case/'routing_capture';captured.mkdir()
        bounds=dict(method='A pre-census one-forward marker is held in the recorder; the first census forward flushes it. Post marker flushes the final census step.',
                    ranks={})
        for rank in range(8):
            bounds['ranks'][rank]=dict(start_step=before[rank]['last_step']+2,end_step=after[rank]['last_step'])
            assert bounds['ranks'][rank]['end_step']>=bounds['ranks'][rank]['start_step']
            shutil.copy2(after[rank]['file'],captured/after[rank]['file'].name)
        (case/'phase_boundaries.json').write_text(json.dumps(bounds,indent=2)+'\n')
        subprocess.run([PYTHON,str(DIAG/'analyze_rank_load.py'),mode,str(captured),str(case/'load_analysis'),
                        str(case/'phase_boundaries.json')],check=True)
        log('census_complete',mode=mode,client_dir=str(client_dir),load_summary=str(case/'load_analysis/SUMMARY.json'))
        handle=(case/'precision_client.log').open('w');handles.append(handle)
        precision=case/'precision'
        probe=subprocess.Popen([PYTHON,str(DIAG/'probe_first_tokens.py'),mode,str(precision)],cwd=ROOT,
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        log('precision_running',mode=mode,probe_pid=probe.pid,output=str(precision))
        await_process(probe,service,5400)
        check=json.loads((precision/'VALIDATION.json').read_text());assert check['valid'] and check['requests']==48
        result=dict(mode=mode,valid=True,service_dir=str(service_dir),client_dir=str(client_dir),
                    census_requests=316,precision_requests=48,marker_requests=3,phase_boundaries=bounds,
                    formal_performance_data=False,case_dir=str(case))
        (case/'case_result.json').write_text(json.dumps(result,indent=2)+'\n')
        log('mode_complete',mode=mode,census_requests=316,precision_requests=48)
    finally:
        stop(probe);stop(bench);stop(service)
        for h in handles:h.close()
        (case/'npu_after.txt').write_text(free_devices())
def main():
    approval=json.loads((DIAG/'APPROVAL.json').read_text());assert approval['approved'] is True
    current=json.loads((ROOT/'CURRENT_EXECUTION.json').read_text());assert current['status']=='completed'
    (DIAG/'CURRENT_DIAGNOSIS.json').write_text(json.dumps(dict(execution=str(EXPERIMENT),status='running'),indent=2)+'\n')
    try:
        for mode in ['off','slot']:run_mode(mode)
        log('all_diagnostics_complete',modes=2)
    except Exception as error:
        log('stopped_on_error',error=repr(error));raise
if __name__=='__main__':main()
