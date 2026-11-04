#!/usr/bin/env python3
"""Validate one slot startup and shut it down; never sends benchmark requests."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import shutil
import signal
import subprocess
import time
from pathlib import Path

import run_experiment as r

STARTUP_ERROR_MARKERS = (
    "torch.OutOfMemoryError:", "AssertionError:", "ValueError:", "WorkerProc failed to start.",
    "Engine core initialization failed.", "Worker failed with error",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=r.HERE / "config_s248_kv55.json")
    parser.add_argument("--batched-tokens", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = r.load_config(args.config)
    plan = r.make_plan(cfg)
    case = next(case for case in plan["cases"] if (
        case["mode"] == "slot" and case["batched_tokens"] == args.batched_tokens))
    route = next(route for route in plan["routes"] if route["name"] == case["route"])
    command = r.service_command(cfg, case)
    if not args.execute:
        print("Startup only; no requests or benchmark. Command:", command)
        return
    r.preflight(cfg)
    with (r.HERE / f".port_{cfg['port']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        root = r.HERE / "data" / f"startup_{stamp}_B{args.batched_tokens}"
        root.mkdir(parents=True, exist_ok=False)
        for path in (args.config, Path(__file__), r.HERE / "run_experiment.py"):
            shutil.copy2(path, root / path.name)
        state = {"purpose": "startup validation only, zero benchmark requests",
                 "started_at": r.timestamp(), "config": cfg, "case": case, "route": route,
                 "service_command": command, "ready": False, "status": "starting"}
        r.write_json(root / "startup.json", state)
        print("STARTUP_RESULTS", root, flush=True)
        process = None

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted: {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            r.wait_cards_free(cfg, root / "npu_before.txt")
            env = r.service_env(cfg, case, route)
            state["plugins"] = env["VLLM_PLUGINS"]
            with (root / "service.log").open("w") as log:
                process = subprocess.Popen(r.with_cann(cfg, command), env=env, cwd=r.HERE,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                state["service_pgid"] = process.pid
                r.write_json(root / "startup.json", state)
                deadline = time.monotonic() + cfg["ready_timeout_seconds"]
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited: {process.returncode}")
                    text = (root / "service.log").read_text(errors="replace")
                    failure = next((marker for marker in STARTUP_ERROR_MARKERS if marker in text), None)
                    if failure:
                        raise RuntimeError(f"Terminal startup error in service.log: {failure}")
                    try:
                        models = r.read_models(cfg)
                    except (OSError, ValueError, KeyError):
                        models = {}
                    if any(model.get("id") == cfg["served_model"] for model in models.get("data", [])):
                        signature = r.worker_signature(cfg, root / "npu_ready.txt")
                        if any(len(pids) != 1 for pids in signature.values()):
                            raise RuntimeError("Unexpected NPU process set.")
                        state.update(ready=True, models=models, worker_signature=signature,
                                     ready_at=r.timestamp(), status="ready")
                        r.write_json(root / "startup.json", state)
                        print("STARTUP_READY", root, flush=True)
                        break
                    time.sleep(r.READINESS_POLL_SECONDS)
                else:
                    raise TimeoutError("Startup readiness timed out.")
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                r.stop_owned_group(process)
                if process is not None:
                    r.wait_cards_free(cfg, root / "npu_after.txt")
                if state["ready"]:
                    state["status"] = "passed_and_stopped"
            except BaseException as exc:
                state.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                state["ended_at"] = r.timestamp()
                r.write_json(root / "startup.json", state)
                signal.signal(signal.SIGTERM, previous)
                print("STARTUP_STATUS", state["status"], flush=True)


if __name__ == "__main__":
    main()
