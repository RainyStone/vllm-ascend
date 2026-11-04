#!/usr/bin/env python3
"""Check natural-routing slot requests and dispatch/combine activity, then stop."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import re
import shutil
import signal
import subprocess
import time
import urllib.request

import run_experiment as r

PROMPTS = ("The capital of France is", "计算：1 + 1 = ")


def counter_records(text):
    records = []
    pattern = r"\[moonep-tstats dev=([^\]]+)\].*?dispatch: calls=(\d+).*?combine: calls=(\d+)"
    for line in text.splitlines():
        workers = re.findall(r"\(Worker_DP(\d+)_EP\d+ pid=\d+\)", line)
        matches = list(re.finditer(pattern, line))
        # Multiprocess stdout can interleave; use only unambiguous records.
        if len(workers) != 1 or len(matches) != 1:
            continue
        match = matches[0]
        values = {"dispatch": int(match.group(2)), "combine": int(match.group(3))}
        prefetch = re.search(r"prefetch: calls=(\d+)", line)
        if prefetch:
            values["prefetch"] = int(prefetch.group(1))
        records.append(("rank" + workers[0], values))
    return records


def counters(text):
    records = {}
    for rank, values in counter_records(text):
        previous = records.get(rank, {})
        records[rank] = {key: max(value, previous.get(key, 0)) for key, value in values.items()}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=r.Path, default=r.HERE / "config.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = r.load_config(args.config)
    plan = r.make_plan(cfg)
    case = next(case for case in plan["cases"] if case["mode"] == "slot")
    route = next(route for route in plan["routes"] if route["name"] == case["route"])
    env = r.service_env(cfg, case, route)
    for key in list(env):
        if key.startswith("VLLM_ASCEND_MOE_SIM_ROUTE"):
            env.pop(key)
    env.update({"MOONEP_BALANCE_STATS": "1", "MOONEP_BALANCE_STATS_EVERY": "43",
                "MOONEP_TIME_STATS": "1", "MOONEP_TIME_STATS_EVERY": "43"})
    if not args.execute:
        print("Natural routing; two short requests only; no performance matrix.")
        print(r.service_command(cfg, case))
        return
    r.preflight(cfg)
    with (r.HERE / f".port_{cfg['port']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        root = r.HERE / "data" / f"slot_requests_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        for path in (args.config, r.Path(__file__), r.HERE / "run_experiment.py"):
            shutil.copy2(path, root / path.name)
        report = {"purpose": "Natural-routing short-request functionality; no performance claims",
                  "config": cfg, "routing": "natural; SIM_ROUTE unset",
                  "started_at": r.timestamp(), "status": "starting", "requests": [],
                  "environment": {k: v for k, v in env.items() if k.startswith((
                      "MOONEP", "ASCEND_MOONEP", "VLLM_ASCEND_MOE_", "ASCEND_RT_"))
                                  or k == "VLLM_PLUGINS"}}
        r.write_json(root / "check.json", report)
        print("REQUEST_CHECK_RESULTS", root, flush=True)
        process = None

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted: {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        log_path = root / "service.log"
        try:
            r.wait_cards_free(cfg, root / "npu_before.txt")
            with log_path.open("w") as log:
                command = r.service_command(cfg, case)
                report["service_command"] = command
                process = subprocess.Popen(r.with_cann(cfg, command), env=env, cwd=r.HERE,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                report["service_pgid"] = process.pid
                r.write_json(root / "check.json", report)
                r.wait_ready(cfg, process, log_path)
                report["ready_at"] = r.timestamp()
                signature = r.worker_signature(cfg, root / "npu_ready.txt")
                if any(len(pids) != 1 for pids in signature.values()):
                    raise RuntimeError("Unexpected NPU process set.")
                report["worker_signature"] = signature
                report["counters_before"] = counters(log_path.read_text(errors="replace"))
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                for index, prompt in enumerate(PROMPTS, 1):
                    body = {"model": cfg["served_model"], "prompt": prompt, "max_tokens": 16,
                            "temperature": 0.0, "seed": cfg["server_seed"]}
                    r.write_json(root / f"request_{index}.json", body)
                    entry = {"index": index, "prompt": prompt}
                    report["requests"].append(entry)
                    r.write_json(root / "check.json", report)
                    try:
                        request = urllib.request.Request(
                            f'http://127.0.0.1:{cfg["port"]}/v1/completions',
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})
                        with opener.open(request, timeout=180) as response:
                            entry["http_status"] = response.status
                            result = json.load(response)
                        r.write_json(root / f"response_{index}.json", result)
                        entry.update(usage=result.get("usage"), choices=result.get("choices"))
                        if entry["http_status"] != 200 or not result.get("choices"):
                            raise RuntimeError("Missing completion choices or non-200 status.")
                        if not result["choices"][0].get("text", "").strip():
                            raise RuntimeError("Empty completion text.")
                        if result.get("usage", {}).get("completion_tokens", 0) <= 0:
                            raise RuntimeError("No generated tokens.")
                        entry["status"] = "passed"
                        r.assert_same_workers(cfg, signature, root / f"npu_request_{index}.txt")
                        print("REQUEST_OK", index, result["choices"][0]["text"], flush=True)
                    except BaseException as exc:
                        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        r.write_json(root / "check.json", report)
                time.sleep(1)
                text = log_path.read_text(errors="replace")
                after = counters(text)
                report["counters_after"] = after
                before = report["counters_before"]
                report["activity_deltas"] = {
                    device: {key: values[key] - before.get(device, {}).get(key, 0)
                             for key in ("dispatch", "combine")}
                    for device, values in after.items()}
                if len(after) != 8 or any(
                    min(values.values()) <= 0 for values in report["activity_deltas"].values()):
                    raise RuntimeError("Missing positive dispatch/combine evidence on all eight ranks.")
                if "[sim-route" in text:
                    raise RuntimeError("Unexpected synthetic routing log.")
                if "MOONEP_SLOT_MODE=1" not in text:
                    raise RuntimeError("Worker slot-mode audit missing.")
                report["status"] = "passed"
                print("MOONEP_ACTIVITY_CONFIRMED", report["activity_deltas"], flush=True)
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                r.stop_owned_group(process)
                if process is not None:
                    r.wait_cards_free(cfg, root / "npu_after.txt")
            except BaseException as exc:
                report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                report["ended_at"] = r.timestamp()
                r.write_json(root / "check.json", report)
                signal.signal(signal.SIGTERM, previous)
                print("REQUEST_CHECK_STATUS", report["status"], flush=True)


if __name__ == "__main__":
    main()
