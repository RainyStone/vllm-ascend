#!/usr/bin/env python3
"""Preview a frozen plan by default. Workloads require an explicit --execute."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys

import analyze
import runtime
import study

SOURCE_FILES = ("runtime.py", "study.py", "plan_sweep.py", "run_sweep.py",
                "analyze.py", "settings.json", "EXPERIMENT.md")


def source_hashes():
    return {name: runtime.sha256(study.HERE / name) for name in SOURCE_FILES}


def environment_signature(cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    runtime.inventory(cfg, directory)
    record = study.read(directory / "environment.json")
    record.pop("at", None)
    if any(not item["head"] for item in record["repos"].values()):
        raise RuntimeError("Cannot determine repository versions.")
    record["pip_freeze_sha256"] = runtime.sha256(directory / "pip_freeze.txt")
    relevant = {k: v for k, v in os.environ.items() if k.startswith((
        "VLLM_", "MOONEP", "ASCEND_", "HCCL_", "GLOO_", "TP_", "OMP_"))
        or k in ("LD_LIBRARY_PATH", "PYTHONPATH")}
    record["ambient_runtime_environment_sha256"] = study.digest(relevant)
    version_files = [Path(cfg["cann_env"]), Path("/usr/local/Ascend/driver/version.info"),
                     Path("/usr/local/Ascend/ascend-toolkit/latest/share/info/runtime/version.info")]
    record["cann_driver_files"] = {str(p): runtime.sha256(p) for p in version_files if p.is_file()}
    workspace = Path(cfg["workspace"])
    binaries = []
    for package in (workspace / "ascend-moonep/ascend_moonep",
                    workspace / "vllm-ascend/vllm_ascend"):
        binaries.extend(package.glob("*cpython-311-aarch64*.so"))
        binaries.extend(package.glob("lib/*.so"))
        binaries.extend(package.glob("_cann_ops_custom/vendors/custom_transformer/op_api/lib/*.so"))
    record["binary_sha256"] = {str(path): runtime.sha256(path) for path in sorted(set(binaries))}
    if not record["binary_sha256"]:
        raise RuntimeError("Missing compiled extension provenance.")
    return study.digest(record)


def archived_route(root, descriptor):
    path = root / "sim" / descriptor["filename"]
    if not path.is_file() or runtime.sha256(path) != descriptor["sha256"]:
        raise ValueError("Archived route content changed.")
    return {key: value for key, value in descriptor.items() if key != "text"} | {"path": str(path)}


def archive_partial_pair(root, pair):
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    for case in pair["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]):
            raise ValueError("Invalid case ID.")
        path = (root / case["id"]).resolve()
        if path.parent != root.resolve():
            raise ValueError("Case path escapes session.")
        if path.exists():
            destination = root / "attempts" / stamp / case["id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)  # Preserve both old modes; never mix half-pairs across retries.


def execute(plan, resume=None):
    if study.settings().get("superseded_by"):
        raise RuntimeError("Superseded design; use ../20261103_dp8_bisect/run_bisect.py")
    config = plan["settings"]
    if not config.get("design_confirmed"):
        raise RuntimeError("Design is awaiting user confirmation.")
    if study.settings() != config:
        raise RuntimeError("Current settings differ from plan; regenerate the plan.")
    cfg = config["runtime"]
    runtime.preflight(cfg)
    model = study.read(Path(cfg["model"]) / "config.json")
    if model.get("n_routed_experts") != 256 or model.get("num_experts_per_tok") != 6:
        raise RuntimeError("Model geometry differs from the routing family.")
    lock_path = study.HERE.parent / f".dp8_crossover_port_{cfg['port']}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume is None:
            stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            root = study.HERE / "data" / f"{plan['phase']}_{stamp}"
            root.mkdir(parents=True, exist_ok=False)
            runtime.write_json(root / "plan.json", plan)
            for name in SOURCE_FILES:
                shutil.copy2(study.HERE / name, root / name)
            for descriptor in plan["routes"]:
                study.write_new(root / "sim" / descriptor["filename"], descriptor["text"])
            manifest = {"sources": source_hashes(),
                        "environment_signature": environment_signature(cfg, root),
                        "plan_sha256": plan["plan_sha256"]}
            runtime.write_json(root / "manifest.json", manifest)
        else:
            root = Path(resume).resolve()
            if study.read(root / "plan.json") != plan:
                raise ValueError("Resume plan differs from archived plan.")
            manifest = study.read(root / "manifest.json")
            if manifest["sources"] != source_hashes():
                raise RuntimeError("Scripts/settings changed; resume refused. Start a new session.")
            stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            current = environment_signature(cfg, root / "resume_checks" / stamp)
            if current != manifest["environment_signature"]:
                raise RuntimeError("Git/package/model/extension environment changed; resume refused.")
        print("RESULTS", root, flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            routes = {item["name"]: item for item in plan["routes"]}
            for index, pair in enumerate(plan["pairs"], 1):
                if analyze.collect_pair(root, pair, plan)["status"] == "complete":
                    print("SKIP_COMPLETE_PAIR", pair["id"], flush=True)
                    continue
                archive_partial_pair(root, pair)
                print(f"PAIR {index}/{len(plan['pairs'])} {pair['id']} {pair['mode_order']}", flush=True)
                for case in pair["cases"]:
                    descriptor = routes[case["route"]]
                    runtime.run_case(cfg, case, archived_route(root, descriptor), root)
        finally:
            signal.signal(signal.SIGTERM, previous)
            analyze.summarize(root)
            print("SUMMARY", root / "REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Frozen plan JSON; default plans/coarse.json")
    parser.add_argument("--resume", type=Path, help="Same-environment session; repeats incomplete whole pairs")
    parser.add_argument("--execute", action="store_true", help="Actually run services and benchmarks")
    args = parser.parse_args()
    if study.settings().get("superseded_by"):
        parser.error("Superseded by ../20261103_dp8_bisect; old 1.00-2.50 plan is archived, not executable.")
    if args.resume:
        plan = study.read(args.resume / "plan.json")
        if args.plan and study.read(args.plan) != plan:
            parser.error("Explicit plan differs from resume session.")
    else:
        plan = study.read(args.plan or study.HERE / "plans/coarse.json")
    study.validate_plan(plan)
    if not args.execute:
        print(json.dumps({"phase": plan["phase"], "ratios": plan["ratios"],
                          "pairs_per_ratio": plan["independent_pairs_per_ratio"],
                          "service_starts": plan["service_starts"],
                          "benchmark_calls": plan["benchmark_calls"],
                          "mode_orders": [pair["mode_order"] for pair in plan["pairs"]],
                          "workloads_started": False}, indent=2))
        return
    execute(plan, args.resume)


if __name__ == "__main__":
    main()
