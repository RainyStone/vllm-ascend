#!/usr/bin/env python3
"""Execute all four approved service scripts sequentially, with hard mode gates."""
import json,os,re,signal,socket,subprocess,sys,time,urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
PYTHON="/usr/local/python3.11.10/bin/python"
SESSION=Path(sys.argv[1])
STATE=SESSION/"remaining_state.json"

def log(event,**kw):
    record=dict(at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),event=event,**kw)
    print(json.dumps(record,ensure_ascii=False),flush=True)
    with (SESSION/"remaining_events.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    STATE.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")

def api_models():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8444/v1/models",timeout=5) as response:
        return [row["id"] for row in json.load(response)["data"]]

def assert_free(timeout=180):
    began=time.monotonic()
    while True:
        npu=subprocess.check_output(["npu-smi","info"],text=True)
        free=set(re.findall(r"No running processes found in NPU (\d+)",npu))
        port_free=False
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(("127.0.0.1",8444))
            port_free=True
        except OSError:pass
        if free==set(map(str,range(8))) and port_free:return npu
        if time.monotonic()-began>timeout:
            raise RuntimeError("Expected eight released NPUs and free port; will not touch unowned processes.")
        time.sleep(5)

def stop_owned(process):
    if process is None:return
    pgid=process.pid
    assert pgid!=os.getpgrp()
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError:return
    time.sleep(5)
    try:os.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:pass

def set_pair(pair):
    for name in ["start_slot.sh","bench_slot.sh"]:
        path=ROOT/name;source=path.read_text()
        result,count=re.subn(r"(?m)^PAIR_LABEL='[^']+'",f"PAIR_LABEL='{pair}'",source)
        assert count==1,(name,count)
        path.write_text(result)

def run_case(pair,mode):
    name=pair+"_"+mode
    folder=SESSION/"cases"/name;folder.mkdir(parents=True)
    service=None;bench=None
    handles=[]
    try:
        (folder/"npu_before.txt").write_text(assert_free())
        set_pair(pair)
        service_log=(folder/"controller_service.log").open("w");handles.append(service_log)
        service=subprocess.Popen(["bash",f"start_{mode}.sh"],cwd=ROOT,
                 stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
        log("starting_service",case=name,service_pid=service.pid)
        began=time.monotonic()
        while True:
            if service.poll() is not None:
                raise RuntimeError(f"{name}: service exited {service.returncode} during startup")
            try:
                if f"dsv4flash_longbench_{mode}" in api_models():break
            except Exception:pass
            if time.monotonic()-began>1500:raise TimeoutError(name+": startup exceeded 1500 seconds")
            time.sleep(3)
        directories=[p for p in (ROOT/"data/services").glob(f"{pair}-{mode}-*")
                     if (p/"service.pid").exists() and int((p/"service.pid").read_text())==service.pid]
        assert len(directories)==1,directories
        service_dir=directories[0]
        subprocess.run([PYTHON,str(ROOT/"audit_service_mode.py"),mode,str(service_dir),str(folder)],check=True)
        log("validating_longest_request",case=name,service_dir=str(service_dir))
        subprocess.run([PYTHON,str(ROOT/"preflight_api.py"),mode,str(folder)],check=True,timeout=1900)
        if mode == "slot":
            subprocess.run([PYTHON,str(ROOT/"audit_optimizations.py"),str(service_dir),str(folder)],check=True)
        (folder/"npu_ready.txt").write_text(subprocess.check_output(["npu-smi","info"],text=True))
        bench_log=(folder/"controller_client.log").open("w");handles.append(bench_log)
        bench=subprocess.Popen(["bash",f"bench_{mode}.sh"],cwd=ROOT,
                 stdout=bench_log,stderr=subprocess.STDOUT,start_new_session=True)
        log("benchmark_running",case=name,service_pid=service.pid,client_pid=bench.pid,service_dir=str(service_dir))
        began=time.monotonic()
        while bench.poll() is None:
            if service.poll() is not None:raise RuntimeError(name+": service exited during benchmark")
            if time.monotonic()-began>10800:raise TimeoutError(name+": benchmark exceeded 3 hours")
            time.sleep(5)
        if bench.returncode!=0:raise RuntimeError(f"{name}: benchmark exited {bench.returncode}")
        outputs=list((ROOT/"data/aisbench").glob(f"{pair}-{mode}-*-{bench.pid}"))
        assert len(outputs)==1,outputs
        client_dir=outputs[0]
        rounds=[]
        for i in [1]:
            check=json.loads((client_dir/f"round_{i}/validation.json").read_text())
            assert check["valid"] and check["requests"]==128,check
            rounds.append(check)
        report=dict(case=name,service_dir=str(service_dir),client_dir=str(client_dir),
                    service_pid=service.pid,client_pid=bench.pid,rounds=rounds,
                    valid=True,formal_requests=128)
        (folder/"case_result.json").write_text(json.dumps(report,indent=2)+"\n")
        log("case_complete",case=name,client_dir=str(client_dir),formal_requests=128)
    finally:
        stop_owned(bench)
        stop_owned(service)
        for handle in handles:handle.close()
        try:(folder/"npu_after.txt").write_text(assert_free())
        except Exception as error:
            log("cleanup_failure",case=name,error=repr(error));raise

if __name__ == "__main__":
    raise SystemExit("Use run_pair.py; only one Off/Slot pair is authorized.")
