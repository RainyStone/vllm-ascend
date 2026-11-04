#!/usr/bin/env python3
"""Record the user's requested pause after Slot, without starting further services."""
import json
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from summarize_results import read_round

ROOT = Path(__file__).resolve().parent
session = Path(sys.argv[1]).resolve()
request_path = session / "USER_STOP_AFTER_SLOT.json"
request = json.loads(request_path.read_text())
deadline = time.monotonic() + 3600
while True:
    state = json.loads((session / "remaining_state.json").read_text())
    case_file = session / "cases/pair02_slot/case_result.json"
    proc = Path("/proc") / str(request["controller_pid"]) / "cmdline"
    try:
        controller_live = b"run_full_matrix.py" in proc.read_bytes()
    except FileNotFoundError:
        controller_live = False
    if state.get("event") == "stopped_on_error" and "service exited 75" not in state.get("error", ""):
        raise RuntimeError(state)
    if case_file.exists() and not controller_live:
        break
    if time.monotonic() > deadline:
        raise TimeoutError("Requested pause did not finish within one hour")
    time.sleep(3)

cases = {}
for name in ("pair01_off", "pair01_slot", "pair02_slot"):
    c = json.loads((session / "cases" / name / "case_result.json").read_text())
    assert c["valid"] and c["formal_requests"] == 948
    rows = [read_round(c["client_dir"], i) for i in (1, 2, 3)]
    cases[name] = {
        "service_dir": c["service_dir"], "client_dir": c["client_dir"],
        "rounds": [{k: v for k, v in r.items() if k != "predictions"} for r in rows],
        "mean_last_two": {k: statistics.mean(r[k] for r in rows[1:])
                          for k in ("mean_ttft_ms", "input_tps")},
    }
assert not (session / "cases/pair02_off/case_result.json").exists()
off = cases["pair01_off"]["mean_last_two"]
slot = cases["pair01_slot"]["mean_last_two"]
result = {
    "status": "paused_by_user", "valid_formal_requests": 2844, "completed_rounds": 9,
    "completed_cases": list(cases), "deferred_cases": ["pair02_off"],
    "cases": cases,
    "first_pair_comparison": {
        "slot_ttft_increase_pct": (slot["mean_ttft_ms"] / off["mean_ttft_ms"] - 1) * 100,
        "slot_input_tps_change_pct": (slot["input_tps"] / off["input_tps"] - 1) * 100,
    },
    "limitation": "One Off service and two Slot services completed; the full two-pair matrix is incomplete.",
}
(session / "PARTIAL_RESULTS.json").write_text(json.dumps(result, indent=2))
npu = subprocess.check_output(["npu-smi", "info"], text=True)
(session / "npu_paused.txt").write_text(npu)
free = all(f"No running processes found in NPU {i}" in npu for i in range(8))
port_free = False
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 8444))
    port_free = True
except OSError:
    pass
cleanup = {"eight_npus_released": free, "port_8444_bindable": port_free,
           "no_final_off_service_started": True}
(session / "CLEANUP_CHECK.json").write_text(json.dumps(cleanup, indent=2))
now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
state = {"at": now, "event": "paused_after_pair02_slot", "completed_cases": 3,
         "completed_rounds": 9, "valid_formal_requests": 2844, "deferred_case": "pair02_off",
         "reason": "User requested completion of current Slot round 3 only; final Off deferred."}
(session / "remaining_state.json").write_text(json.dumps(state, indent=2))
with (session / "remaining_events.jsonl").open("a") as stream:
    stream.write(json.dumps(state) + "\n")
current = json.loads((ROOT / "CURRENT_EXECUTION.json").read_text())
current.update(status="paused_by_user", result=str(session / "PARTIAL_RESULTS.json"),
               cleanup=str(session / "CLEANUP_CHECK.json"), deferred_cases=["pair02_off"])
(ROOT / "CURRENT_EXECUTION.json").write_text(json.dumps(current, indent=2))
report = ["已按用户要求在第二对Slot第3轮后暂停。",
          "完成3台服务、9轮、2844条正式请求，全部校验通过；最后Off的3轮未执行。"]
for name, case in cases.items():
    v = case["mean_last_two"]
    report.append(f"{name}: TTFT={v['mean_ttft_ms']:.3f} ms，input TPS={v['input_tps']:.3f}")
report += [f"8卡无运行进程={free}；8444端口可绑定={port_free}。",
           "完整两对矩阵尚未完成，不能将当前结果标记为完整实验最终结果。"]
(session / "PARTIAL_RESULT.txt").write_text("\n".join(report) + "\n")
complete = {"paused": True, "at": now, "cleanup": cleanup,
            "partial_result": str(session / "PARTIAL_RESULTS.json")}
(session / "STOP_AFTER_SLOT_COMPLETE.json").write_text(json.dumps(complete, indent=2))
print("\n".join(report), flush=True)
