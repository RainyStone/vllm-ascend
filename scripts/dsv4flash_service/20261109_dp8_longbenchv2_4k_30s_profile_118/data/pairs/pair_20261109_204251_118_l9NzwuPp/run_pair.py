import datetime
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parents[2]
PYTHON = "/usr/local/python3.11.10/bin/python"
NPU = "/usr/local/Ascend/driver/tools/npu-smi"
NPU = NPU if Path(NPU).is_file() else "npu-smi"
NAMES = {mode: "dsv4flash_longbench_dp8_4k_" + mode for mode in ("off", "slot")}
STATE = ROOT / "STATE.json"

def log(stage, **extra):
    row = dict(at=datetime.datetime.now().astimezone().isoformat(), stage=stage, **extra)
    print(json.dumps(row), flush=True)
    with (ROOT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2) + "\n")
    tmp.replace(STATE)

def resources_free(timeout=0):
    end = time.monotonic() + timeout
    while True:
        cards = subprocess.check_output([NPU, "info"], text=True)
        free = set(re.findall(r"No running processes found in NPU (\d+)", cards))
        try:
            for port in (8444, 28765):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", port))
            ports_free = True
        except OSError:
            ports_free = False
        if free == set(map(str, range(8))) and ports_free:
            return cards
        if time.monotonic() >= end:
            raise RuntimeError(f"Eight NPUs and both ports must be free. Idle={sorted(free)} ports_free={ports_free}")
        time.sleep(3)

def wait_ready(service, mode):
    end = time.monotonic() + 1500
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < end:
        if service.poll() is not None:
            raise RuntimeError(f"{mode} service exited {service.returncode} before readiness")
        try:
            with opener.open("http://127.0.0.1:8444/v1/models", timeout=5) as response:
                names = {model["id"] for model in json.load(response)["data"]}
            if NAMES[mode] in names:
                return
        except (OSError, ValueError):
            pass
        time.sleep(3)
    raise TimeoutError(f"{mode} readiness exceeded 1500 seconds")

def stop_owned(process, grace):
    if process is None:
        return
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=20)

def run_case(phase, mode):
    case = ROOT / phase / mode
    case.mkdir(parents=True, exist_ok=False)
    (case / "npu_before.txt").write_text(resources_free())
    service = bench = None
    try:
        with (case / "service_console.log").open("w") as srv_out, (case / "bench_console.log").open("w") as cli_out:
            service = subprocess.Popen(["bash", f"start_{mode}.sh"], cwd=EXPERIMENT,
                                       stdout=srv_out, stderr=subprocess.STDOUT, start_new_session=True)
            log("service_starting", phase=phase, mode=mode, pid=service.pid)
            wait_ready(service, mode)
            log("service_ready", phase=phase, mode=mode, pid=service.pid)
            bench = subprocess.Popen(["bash", "bench_profile.sh", mode], cwd=EXPERIMENT,
                                     stdout=cli_out, stderr=subprocess.STDOUT, start_new_session=True)
            log("preflight_and_bench", phase=phase, mode=mode, pid=bench.pid)
            end = time.monotonic() + 7200
            while bench.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError(f"{phase} {mode}: service exited {service.returncode} during client")
                if time.monotonic() > end:
                    raise TimeoutError(f"{phase} {mode}: client exceeded two hours")
                time.sleep(2)
            if bench.returncode:
                raise RuntimeError(f"{phase} {mode}: client exited {bench.returncode}, see {case/'bench_console.log'}")
            log("case_validated", phase=phase, mode=mode)
    finally:
        if bench is not None and bench.poll() is None:
            stop_owned(bench, 5)
        if service is not None:
            stop_owned(service, 25)
        (case / "npu_after.txt").write_text(resources_free(120))

def on_signal(signum, _frame):
    raise RuntimeError(f"Controller interrupted by signal {signum}")

if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    with (EXPERIMENT / ".dp8_4k_118_controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for phase in ("formal",):
                for mode in ("off", "slot"):
                    run_case(phase, mode)
                log("phase_complete", phase=phase)
            log("runs_complete", role="diagnostic_30s_profile")
        except BaseException as exc:
            log("stopped", error=repr(exc))
            traceback.print_exc()
            sys.exit(1)

