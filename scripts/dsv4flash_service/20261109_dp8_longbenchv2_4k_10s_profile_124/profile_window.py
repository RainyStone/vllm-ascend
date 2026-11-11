"""Capture a bounded profiling window during an existing AISBench replay."""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
API = "http://127.0.0.1:8444"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, payload=None, timeout=1800):
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(API+path, data=data, method="POST",
                                 headers={"Content-Type":"application/json"})
    with OPENER.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def event(output, step, **fields):
    row = dict(at=datetime.datetime.now().astimezone().isoformat(), step=step, **fields)
    with (output/"profile_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)


def warmup(mode):
    data = json.loads((ROOT/"data/longbenchv2_full_le128k.json").read_text())
    manifest = json.loads((ROOT/"data/selection_manifest.json").read_text())
    items = sorted(manifest["samples"],key=lambda v:int(v["data_id"]))
    assert len(items)==316
    path = WORKSPACE/"benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
    templates = [n.value.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.keyword)
                 and n.arg=="prompt" and isinstance(n.value,ast.Constant)]
    template, = templates

    def one(item):
        prompt = template.format(**data[int(item["data_id"])])
        assert hashlib.sha256(prompt.encode()).hexdigest()==item["prompt_sha256"]
        result = post("/v1/completions",dict(model=f"dsv4flash_longbench_dp8_4k_{mode}",prompt=prompt,max_tokens=1,
                                             temperature=0,ignore_eos=True,seed=1024,stream=False))
        usage = result["usage"]
        assert usage["prompt_tokens"]==item["prompt_tokens"] and usage["completion_tokens"]==1
        return {"data_id":item["data_id"],"prompt_tokens":usage["prompt_tokens"],"completion_tokens":1}

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(one,items[:16]))


def read_metrics(text, model):
    fields = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:prompt_tokens_total": "prompt_tokens",
    }
    ranks = {i: {} for i in range(8)}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            field = fields.get(sample.name)
            if not field or sample.labels.get("model_name") != model:
                continue
            rank = int(sample.labels["engine"])
            value = sample.value
            if rank not in ranks or field in ranks[rank] or not math.isfinite(value):
                raise ValueError(f"Unexpected metric {sample}")
            if value < 0 or value != int(value):
                raise ValueError(f"Expected a nonnegative integer metric: {sample}")
            ranks[rank][field] = int(value)
    if any(set(row) != set(fields.values()) for row in ranks.values()):
        raise ValueError(f"Expected running, waiting, and prompt tokens on all 8 engines: {ranks}")
    return ranks


def sample_metrics(output, model, phase, previous):
    record = dict(at=datetime.datetime.now().astimezone().isoformat(), phase=phase,
                  monotonic=time.monotonic())
    current = None
    try:
        with OPENER.open(API + "/metrics", timeout=2) as response:
            current = read_metrics(response.read().decode("utf-8"), model)
        record["ranks"] = current
        record["running"] = sum(row["running"] for row in current.values())
        record["waiting"] = sum(row["waiting"] for row in current.values())
        record["in_service"] = record["running"] + record["waiting"]
        if previous is not None:
            increments = {rank: row["prompt_tokens"] - previous[rank]["prompt_tokens"]
                          for rank, row in current.items()}
            if min(increments.values()) < 0:
                raise ValueError("Input token counter reset during the round")
            record["prompt_tokens_delta_by_engine"] = increments
            record["prompt_tokens_delta"] = sum(increments.values())
    except (OSError, ValueError, KeyError) as exc:
        record["error"] = str(exc)
        current = None
    with (output / "metrics_samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record, current


def wait_idle(output, model):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row, _ = sample_metrics(output, model, "before_replay", None)
        if row.get("in_service") == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"Service did not drain after warmup: {row}")


def gate_ready(samples, target, required):
    return (
        len(samples) == required
        and all("error" not in row and row["in_service"] == target for row in samples)
        and all(row.get("prompt_tokens_delta", 0) > 0 for row in list(samples)[1:])
    )


def wait_for_load(bench, output, model, target, required, interval, timeout):
    deadline = time.monotonic() + timeout
    samples = deque(maxlen=required)
    previous = None
    while bench.poll() is None and time.monotonic() < deadline:
        began = time.monotonic()
        row, previous = sample_metrics(output, model, "gate", previous)
        samples.append(row)
        if gate_ready(samples, target, required) and time.monotonic() < deadline:
            return row, previous
        time.sleep(max(0, min(began + interval, deadline) - time.monotonic()))
    return None, previous


def check_service_profile(output, mode):
    audit, = output.glob("*_mode_audit.json")
    report = json.loads(audit.read_text())
    if not report["valid"] or report["mode"] != mode:
        raise ValueError("Service mode audit failed")
    command = report["actual_command"]
    config = json.loads(command[command.index("--profiler-config") + 1])
    profile_dir = Path(report["service_dir"]) / "profiling"
    required = dict(profiler="torch", torch_profiler_dir=str(profile_dir),
                    delay_iterations=0, max_iterations=0, ignore_frontend=True,
                    torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                    torch_profiler_with_memory=False)
    if config != required or any(profile_dir.iterdir()):
        raise ValueError("Expected an empty, per-service profiling directory and the reviewed config")
    workers = {row["rank"]: row["pid"] for row in report["workers"]}
    return profile_dir, workers


def inspect_traces(profile_dir, workers):
    traces = {}
    for path in sorted(profile_dir.glob("dp*_ascend_pt")):
        match = re.fullmatch(r"dp(\d+)_pp\d+_tp\d+_dcp\d+_ep(\d+)_rank\d+_(\d+)_\d+_ascend_pt",
                             path.name)
        if not match:
            raise ValueError(f"Unexpected trace directory: {path}")
        rank, ep, pid = map(int, match.groups())
        if rank != ep or workers.get(rank) != pid or rank in traces:
            raise ValueError(f"Trace does not match the audited worker: {path}")
        info, = path.glob("profiler_info_*.json")
        meta = json.loads(info.read_text())
        device, = path.glob("PROF_*/device_*")
        if not any((device / "data").iterdir()) or not any(device.glob("end_info.*.done")):
            raise ValueError(f"Device capture missing data or end marker: {path}")
        if (path / "FRAMEWORK/torch.op_range").stat().st_size == 0:
            raise ValueError(f"CPU capture empty: {path}")
        seconds = (meta["end_info"]["MonotonicTimeEnd"] -
                   meta["start_info"]["start_monotonic"]) / 1e9
        if seconds <= 0 or meta["rank_id"] != rank:
            raise ValueError(f"Invalid rank or timing metadata: {info}")
        traces[rank] = dict(rank=rank, worker_pid=pid, seconds=seconds,
                            trace_dir=str(path), profiler_info=str(info))
    if set(traces) != set(workers) or len(traces) != 8:
        raise ValueError(f"Expected 8 original traces, found {sorted(traces)}")
    return list(traces.values())


def capture(bench, output, model, previous, window, interval, status):
    stop_polling = threading.Event()
    poller = None

    def poll_window():
        last = previous
        while not stop_polling.is_set():
            began = time.monotonic()
            _, last = sample_metrics(output, model, "window", last)
            stop_polling.wait(max(0, began + interval - time.monotonic()))

    try:
        request_start = time.monotonic()
        event(output, "profile_start_requested")
        post("/start_profile", timeout=90)
        opened = time.monotonic()
        status["start_api_seconds"] = opened - request_start
        event(output, "profile_started", window_seconds=window)
        poller = threading.Thread(target=poll_window, daemon=True)
        poller.start()
        try:
            bench.wait(timeout=max(0, opened + window - time.monotonic()))
        except subprocess.TimeoutExpired:
            status["window_completed"] = True
    finally:
        stop_polling.set()
        request_stop = time.monotonic()
        if "start_api_seconds" in status:
            status["controller_window_seconds"] = request_stop - opened
        event(output, "profile_stop_requested")
        post("/stop_profile", timeout=180)
        status["stop_api_seconds"] = time.monotonic() - request_stop
        event(output, "profile_stopped")
        if poller is not None:
            poller.join(timeout=3)


def stop_client(bench):
    if bench.poll() is None:
        try:
            os.killpg(bench.pid, signal.SIGTERM)
            bench.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(bench.pid, signal.SIGKILL)
            bench.wait(timeout=10)


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    global API
    mode, folder, api, window, target, interval, required, timeout, separator, *command = sys.argv[1:]
    API = api
    if mode not in ("off", "slot") or separator != "--" or not command:
        raise ValueError("Expected mode, output, API, window, target, interval, samples, timeout, --, command")
    window, target, interval, required, timeout = map(int, (window, target, interval, required, timeout))
    if min(window, target, interval, timeout) <= 0 or required < 2:
        raise ValueError("Invalid profiling timing or load threshold")
    output = Path(folder)
    model = f"dsv4flash_longbench_dp8_4k_{mode}"
    profile_dir, workers = check_service_profile(output, mode)
    status = dict(mode=mode, valid=False, reason="incomplete", window_seconds=window,
                  window_completed=False, profile_dir=str(profile_dir))
    bench = None
    try:
        event(output, "warmup_start", mode=mode, requests=16)
        warm = warmup(mode)
        (output / "warmup_validated.json").write_text(json.dumps(warm, indent=2) + "\n")
        event(output, "warmup_validated", mode=mode, requests=len(warm))
        wait_idle(output, model)
        replay_start = time.monotonic()
        bench = subprocess.Popen(command, start_new_session=True)
        event(output, "aisbench_started", mode=mode, pid=bench.pid)
        gated, previous = wait_for_load(bench, output, model, target, required, interval, timeout)
        if gated is None:
            status["reason"] = "saturation/prefill gate not reached"
            event(output, "capture_skipped", reason=status["reason"])
        else:
            event(output, "gate_reached", metrics=gated)
            capture(bench, output, model, previous, window, interval, status)
            try:
                status["ranks"] = inspect_traces(profile_dir, workers)
            except (OSError, ValueError, KeyError) as exc:
                status["reason"] = str(exc)
            else:
                status["valid"] = status["window_completed"]
                status["reason"] = "ok" if status["valid"] else "AISBench ended before the full window"
        if bench.wait(timeout=max(0, replay_start + 7200 - time.monotonic())) != 0:
            status.update(valid=False, reason=f"AISBench exited {bench.returncode}")
            raise subprocess.CalledProcessError(bench.returncode, command)
        event(output, "aisbench_complete", mode=mode)
    except BaseException as exc:
        if status["reason"] in ("incomplete", "ok"):
            status.update(valid=False, reason=str(exc))
        raise
    finally:
        if bench is not None:
            stop_client(bench)
        (output / "CAPTURE_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    main()
