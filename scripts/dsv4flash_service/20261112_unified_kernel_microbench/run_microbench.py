#!/usr/bin/env python3
"""Run local operator A/B in separate processes on physical NPU 0 of 124."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[3] / "ascend-moonep"
lock = open("/tmp/xzx_moonep_unified_micro_npu0.lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
order = ["split", "unified", "unified", "split", "split", "unified"]
status_file = ROOT / "RUN_STATUS.json"
records = json.loads(status_file.read_text()) if status_file.exists() else []
for index, variant in enumerate(order, 1):
    done = next((row for row in records if row["index"] == index), None)
    if done is not None:
        assert done["variant"] == variant and done["returncode"] == 0
        saved = json.loads(Path(done["output"]).read_text())
        assert saved["finished"] and saved["correctness_pass"]
        print("KEEP completed", index, variant, flush=True)
        continue
    # Driver accounting may briefly retain an exited process as <INVALID>.
    # Never run until it reports no process; a live task remains protected.
    for attempt in range(16):
        info = subprocess.check_output(["npu-smi", "info"], text=True)
        (ROOT / "raw" / f"npu_smi_before_{index:02d}_{attempt:02d}.txt").write_text(info)
        if re.search(r"No running processes found in NPU 0\b", info):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Physical NPU 0 remains occupied; do not stop other work")
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1")
    paths = [SDK / "ascend_moonep/lib", SDK / "3rdparty/shmem/install/shmem/lib",
             SDK / "3rdparty/shmem/install/memfabric_hybrid/lib",
             Path("/usr/local/Ascend/cann-9.2.0/lib64"),
             Path("/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64")]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, paths)) + ":" + env.get("LD_LIBRARY_PATH", "")
    name = f"{index:02d}_{variant}"
    target = ROOT / "raw" / (name + ".json")
    print("START", name, flush=True)
    start = time.time()
    with (ROOT / "raw" / (name + ".log")).open("w") as log:
        process = subprocess.run(
            ["python", "-u", str(ROOT / "bench_unified_kernels.py"),
             "--variant", variant, "--output", str(target)],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300,
        )
    row = {"index": index, "variant": variant, "returncode": process.returncode,
           "seconds": time.time() - start, "output": str(target),
           "machine": "141.61.45.124", "physical_npu": 0}
    records.append(row)
    (ROOT / "RUN_STATUS.json").write_text(json.dumps(records, indent=2))
    print("END", row, flush=True)
    if process.returncode:
        print((ROOT / "raw" / (name + ".log")).read_text()[-5000:], flush=True)
        raise SystemExit(process.returncode)
    data = json.loads(target.read_text())
    assert data["finished"] and data["correctness_pass"]
(ROOT / "raw/npu_smi_after.txt").write_text(
    subprocess.check_output(["npu-smi", "info"], text=True))
print("ALL MICROBENCH RUNS FINISHED", flush=True)
