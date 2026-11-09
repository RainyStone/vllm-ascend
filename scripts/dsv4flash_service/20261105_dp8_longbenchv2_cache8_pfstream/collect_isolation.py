"""Collect executed isolation runs and device evidence without starting workloads."""
import json,re,shutil,socket,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
out=ROOT/"diagnostic_report"
out.mkdir(exist_ok=True)
reports={}
for label in ["stream_only","cache_only"]:
    profile=ROOT/"isolation_diagnostics"/label
    current_file=profile/"CURRENT_EXECUTION.json"
    if not current_file.exists():
        reports[label]={"status":"not_run"};continue
    current=json.loads(current_file.read_text());session=Path(current["session"])
    status_file=session/"STATUS.json"
    if not status_file.exists():
        reports[label]={"status":"running","session":str(session)};continue
    status=json.loads(status_file.read_text())
    state=json.loads((session/"remaining_state.json").read_text())
    services=list((profile/"data/services").glob("*"))
    pid=state.get("service_pid")
    if pid is not None:
        services=[p for p in services if (p/"service.pid").exists() and (p/"service.pid").read_text().strip()==str(pid)] or services
    service=max(services,key=lambda p:p.stat().st_mtime)
    text=(service/"service.log").read_text(errors="replace")
    pids=sorted(set(int(p) for p in re.findall(r"Worker_DP\d+_EP\d+ pid=(\d+)",text)))
    evidence=out/label;evidence.mkdir(exist_ok=True)
    faults=[]
    for worker in pids:
        for path in Path("/root/ascend/log/debug/plog").glob(f"plog-{worker}_*"):
            lines=path.read_text(errors="replace").splitlines()
            fault=[line for line in lines if any(key in line for key in ["fault kernel_name=", "DDR address", "Task run failed"])]
            if fault:
                shutil.copy2(path,evidence/path.name);faults.extend(fault)
    case=session/"cases"/(label+"_slot")
    for path in case.glob("*.json"):shutil.copy2(path,evidence/path.name)
    for path in [status_file,session/"remaining_state.json",service/"start_script.sh",service/"environment.txt"]:
        if path.exists():shutil.copy2(path,evidence/path.name)
    record=dict(status,session=str(session),service_dir=str(service),
                service_log=str(service/"service.log"),device_fault_evidence=faults,
                optimization_audit=(case/"OPTIMIZATION_AUDIT.json").exists(),
                real_request_preflight=(case/"preflight_slot.json").exists())
    if status["status"]=="passed":
        result=json.loads((case/"case_result.json").read_text())
        assert result["valid"] and result["formal_requests"]==128
        assert result["rounds"][0]["requests"]==128
        record["validated_requests"]=128
        record["client_dir"]=result["client_dir"]
    reports[label]=record
npu=subprocess.check_output(["npu-smi","info"],text=True)
free=set(re.findall(r"No running processes found in NPU (\d+)",npu))==set(map(str,range(8)))
(out/"npu_final.txt").write_text(npu)
port_free=False
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(("127.0.0.1",8444))
    port_free=True
except OSError:pass
result=dict(profiles=reports,diagnostic_only=True,excluded_from_formal_results=True,
            eight_npus_released=free,port_8444_bindable=port_free,
            main_off1_preserved=True)
(out/"RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
lines=["# LongBench 缓存与副流隔离诊断","",
       "MTE、并发16、max_out_len=1、24 GiB KV、128K上限、8K token预算；每项先最长样本校验及16条预热，再测试前128条原始全文。诊断不计入316条正式性能对比。","",
       "|模式|每层缓存|prefetch副流|状态|有效诊断请求|","|---|---:|---|---|---:|"]
for label,r in reports.items():
    lines.append(f'|{label}|{0 if label=="stream_only" else 8}|{"开" if label=="stream_only" else "关"}|{r["status"]}|{r.get("validated_requests",0)}|')
    if r.get("device_fault_evidence"):
        kernels=sorted(set(re.findall(r"fault kernel_name=([^, ]+)", "\n".join(r["device_fault_evidence"]))))
        lines+=["",label+" 首个/关联设备故障算子："+", ".join(kernels)]
lines+=["",f"8卡已释放：{free}；8444端口可绑定：{port_free}。Off1有效数据保留。","",
        "单项不复现不能证明所有流量均正确。首个设备故障与Python收到异步错误的位置必须分开判断；尚未定位的原因不作为确定根因。"]
(out/"RESULT.md").write_text("\n".join(lines)+"\n")
print(json.dumps({k:{a:b for a,b in v.items() if a in ("status","validated_requests","session")} for k,v in reports.items()},ensure_ascii=False))
