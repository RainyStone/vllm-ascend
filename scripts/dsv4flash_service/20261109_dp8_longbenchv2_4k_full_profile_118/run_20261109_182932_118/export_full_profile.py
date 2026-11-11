#!/usr/bin/env python3
"""Export per-rank trace views from scratch copies of the original captures."""
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OFFLINE = ROOT.parents[1] / "20261106_dp16_optimized_retest/offline_analyse_profile.py"
EXPORT = ROOT / "exported"

def export(rank_path):
    mode = rank_path.parent.name
    out = EXPORT / mode / rank_path.name
    out.mkdir(parents=True, exist_ok=True)
    target = out / "trace_view.json"
    if target.is_file() and target.stat().st_size > 0:
        return f"{mode}/{rank_path.name} already exported"
    with (out / "export.log").open("w", buffering=1) as log:
        print(f"RAW {rank_path}", file=log)
        with tempfile.TemporaryDirectory(prefix="moonep_full_trace_", dir="/dev/shm") as tmp:
            copy = Path(tmp) / rank_path.name
            shutil.copytree(rank_path, copy)
            print("OFFLINE_EXPORT_START", file=log)
            subprocess.run([sys.executable, str(OFFLINE), str(copy), "1"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            traces = list(copy.rglob("trace_view.json"))
            if len(traces) != 1 or traces[0].stat().st_size == 0:
                raise RuntimeError(f"invalid trace views for {rank_path}: {traces}")
            partial = target.with_suffix(".json.partial")
            shutil.copyfile(traces[0], partial)
            partial.replace(target)
            print(f"OFFLINE_EXPORT_DONE {target.stat().st_size}", file=log)
    return f"{mode}/{rank_path.name} {target.stat().st_size} bytes"

def main():
    limit = time.monotonic() + 30 * 60
    while time.monotonic() < limit:
        state = json.loads((ROOT / "STATE.json").read_text())
        if state["stage"] == "runs_complete":
            break
        time.sleep(5)
    else:
        raise TimeoutError(state)
    ranks = []
    for mode in ("off", "slot"):
        raw = sorted((ROOT / "profiling" / mode).glob("dp*_ascend_pt"))
        if len(raw) != 8 or not all(list(p.glob("profiler_info_*.json")) for p in raw):
            raise RuntimeError(f"raw capture inventory incomplete: {mode} {raw}")
        ranks.extend(raw)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(export, path): path for path in ranks}
        for future in concurrent.futures.as_completed(futures):
            try:
                print("EXPORTED", future.result(), flush=True)
            except Exception as exc:
                failures.append(str(futures[future]))
                print("EXPORT_FAILED", futures[future], repr(exc), flush=True)
    if failures:
        raise RuntimeError(failures)

if __name__ == "__main__":
    main()
