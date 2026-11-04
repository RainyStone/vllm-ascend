#!/usr/bin/env python3
"""Prepare or run the rear-four-card MoonEP routing-skew experiment.

No arguments / --plan: print commands only. Only --execute launches workloads.
Standard library only; all vLLM imports happen in child processes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROUTE_FILES = (
    ("uniform", "sim_route_uniform.txt"),
    ("4hot", "sim_route_4hot.txt"),
    ("3hot1r", "sim_route_3hot_1rank.txt"),
    ("4hot1r", "sim_route_4hot_1rank.txt"),
)
ROUNDS_SEED_BASE = 9000
HARDWARE_WAIT_SECONDS = 5
TERMINATE_GRACE_SECONDS = 90
READINESS_POLL_SECONDS = 5
CAP_SLACK_PERCENT = 0.01


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    cfg = read_json(path)
    if cfg["cards"] != [4, 5, 6, 7]:
        raise ValueError("This experiment is restricted to physical cards [4,5,6,7].")
    if cfg["modes"][0] != "off" or len(cfg["modes"]) != 2:
        raise ValueError("modes must be [off, slot] or [off, owner].")
    if cfg["modes"][1] not in ("slot", "owner"):
        raise ValueError("Unknown MoonEP mode.")
    if cfg.get("first_mode", "off") not in cfg["modes"]:
        raise ValueError("first_mode must be one of the compared modes.")
    for key in ("modes", "routes", "input_lengths", "batched_tokens"):
        if not cfg[key] or len(cfg[key]) != len(set(cfg[key])):
            raise ValueError(f"{key} must contain unique entries.")
    if not isinstance(cfg["slot_optimizations"], bool):
        raise ValueError("slot_optimizations must be a JSON boolean.")
    if cfg["rounds"] <= cfg["discard_rounds"] or cfg["discard_rounds"] < 0:
        raise ValueError("No measured rounds remain after discarding warmup rounds.")
    for key in ("num_prompts", "max_concurrency", "rounds", "output_length",
                "kv_cache_bytes", "max_model_len", "ready_timeout_seconds",
                "bench_timeout_seconds", "release_timeout_seconds",
                "idle_hbm_limit_mb", "min_dispatch_tokens", "omp_num_threads"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["output_length"] != 1:
        raise ValueError("This prefill experiment fixes output_length=1.")
    if not 0 < cfg["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1.")
    if not 1024 <= cfg["port"] <= 65535:
        raise ValueError("Invalid port.")
    for key in ("input_lengths", "batched_tokens"):
        if any(not isinstance(x, int) or x <= 0 for x in cfg[key]):
            raise ValueError(f"Invalid {key}.")
    if max(cfg["input_lengths"]) + cfg["output_length"] > cfg["max_model_len"]:
        raise ValueError("Input plus output exceeds max_model_len.")
    if min(cfg["batched_tokens"]) < cfg["max_concurrency"]:
        raise ValueError("batched_tokens cannot be below max_concurrency.")
    for key in ("workspace", "model", "python", "cann_env"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path.")
    return cfg


def route_info(name, model_cfg):
    route_files = dict(ROUTE_FILES)
    if name not in route_files:
        raise ValueError(f"Unknown route: {name}")
    path = (HERE.parent / "sim" / route_files[name]).resolve()
    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    pairs = [line for line in lines if line]
    experts = model_cfg["n_routed_experts"]
    top_k = model_cfg["num_experts_per_tok"]
    if experts != 256 or top_k != 6 or experts % 4:
        raise ValueError("These routing files require E=256, K=6, EP=4.")
    shares = [None] * experts
    for entry in pairs:
        expert, share = entry.split(":")
        expert, share = int(expert), float(share)
        if not 0 <= expert < experts or shares[expert] is not None:
            raise ValueError(f"Duplicate/out-of-range expert in {path}")
        if not math.isfinite(share) or not 0 <= share <= 100 / top_k + CAP_SLACK_PERCENT:
            raise ValueError(f"Illegal share in {path}")
        shares[expert] = share
    remaining = 100 - math.fsum(v for v in shares if v is not None)
    count = shares.count(None)
    if remaining < -1e-6 or not count:
        raise ValueError(f"Invalid sparse route {path}")
    shares = [remaining / count if v is None else v for v in shares]
    per_rank = experts // 4
    rank_shares = [math.fsum(shares[i:i + per_rank])
                   for i in range(0, experts, per_rank)]
    return {"name": name, "path": str(path), "sha256": sha256(path),
            "rank_route_percent": rank_shares,
            "home_rank_peak_to_mean": max(rank_shares) / 25.0}


def make_plan(cfg):
    model_cfg = read_json(Path(cfg["model"]) / "config.json")
    if model_cfg.get("model_type") != "deepseek_v4":
        raise ValueError("Expected DeepSeek V4 model.")
    if model_cfg.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("This experiment expects the FP8 checkpoint, not BF16.")
    routes = [route_info(name, model_cfg) for name in cfg["routes"]]
    cases = []
    index = 0
    for route in routes:
        for length in cfg["input_lengths"]:
            for budget in cfg["batched_tokens"]:
                # Adjacent mode pairs, alternating which mode goes first.
                initial = cfg["modes"] if cfg.get("first_mode", "off") == "off" else list(reversed(cfg["modes"]))
                order = initial if index % 2 == 0 else list(reversed(initial))
                for mode in order:
                    cases.append({"id": f'{route["name"]}_L{length}_B{budget}_{mode}',
                                  "route": route["name"], "mode": mode,
                                  "input_length": length, "batched_tokens": budget})
                index += 1
    return {"config": cfg, "routes": routes, "cases": cases,
            "service_starts": len(cases),
            "benchmark_calls": len(cases) * (cfg["rounds"] + 1),
            "seed_rule": "9000 + input_length + round; warmup seed 999"}


def service_env(cfg, case, route):
    # Avoid stale inherited settings changing either side of the comparison.
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith(("MOONEP", "ASCEND_MOONEP", "VLLM_ASCEND_MOE_"))
                   or k in ("VLLM_TOKENIZER_BATCH_WAIT_MS", "VLLM_TORCH_PROFILER_DIR",
                            "VLLM_PROFILER_CONFIG", "ASCEND_VISIBLE_DEVICES"))}
    for key in ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY"):
        env.pop(key, None)
    ws = Path(cfg["workspace"])
    plugin = ws / "vllm-ascend"
    package = ws / "ascend-moonep" / "ascend_moonep"
    vendor = plugin / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "VLLM_PLUGINS": ",".join((
            "ascend", "ascend_kv_connector", "ascend_model_loader",
            "ascend_service_profiling", "ascend_model",
        )),
        "VLLM_USE_V1": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, cfg["cards"])),
        "HCCL_IF_IP": cfg["interface_ip"],
        "GLOO_SOCKET_IFNAME": cfg["interface"],
        "TP_SOCKET_IFNAME": cfg["interface"],
        "HCCL_SOCKET_IFNAME": cfg["interface"],
        "HCCL_BUFFSIZE": "1024",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": str(cfg["omp_num_threads"]),
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONPATH": ":".join(map(str, (ws / "vllm", plugin, ws / "ascend-moonep"))),
        "ASCEND_CUSTOM_OPP_PATH": str(vendor),
        "VLLM_ASCEND_MOE_SIM_ROUTE": "@" + route["path"],
        "VLLM_ASCEND_MOE_SIM_ROUTE_SEED": str(cfg["route_seed"]),
        "VLLM_ASCEND_MOE_SIM_ROUTE_EVERY": "0",
        "VLLM_ASCEND_MOE_SIM_ROUTE_TABLE": "0",
        "NO_PROXY": "127.0.0.1,localhost," + cfg["interface_ip"],
        "no_proxy": "127.0.0.1,localhost," + cfg["interface_ip"],
    })
    libs = [str(vendor / "op_api/lib")]
    if case["mode"] != "off":
        env.update({
            "MOONEP": "1",
            "MOONEP_SLOT_MODE": "1" if case["mode"] == "slot" else "0",
            "MOONEP_SHMEM_BOOTSTRAP": "ipport",
            "MOONEP_DISABLE_UDMA": "1",
            "MOONEP_REAL_T": "1",
            "MOONEP_MIN_DISPATCH_TOKENS": str(cfg["min_dispatch_tokens"]),
            "ASCEND_MOONEP_BUILD_DIR": str(package),
        })
        libs.insert(0, str(package / "lib"))
        if case["mode"] == "slot" and cfg["slot_optimizations"]:
            env.update({"ASCEND_MOONEP_PREFETCH_RELAXED": "1",
                        "ASCEND_MOONEP_NO_RECONCILE": "1",
                        "ASCEND_MOONEP_SKIP_DUP": "1"})
    env["LD_LIBRARY_PATH"] = ":".join(libs + [
        x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x])
    return env


def service_command(cfg, case):
    return [cfg["python"], "-m", "vllm.entrypoints.cli.main", "serve", cfg["model"],
            "--host", "127.0.0.1", "--port", str(cfg["port"]),
            "--api-server-count", "1", "--data-parallel-size", "4",
            "--tensor-parallel-size", "1", "--enable-expert-parallel",
            "--enforce-eager", "--enable-chunked-prefill",
            "--additional-config", json.dumps({"enable_moonep": case["mode"] != "off"}),
            "--served-model-name", cfg["served_model"], "--seed", str(cfg["server_seed"]),
            "--max-num-seqs", str(cfg["max_concurrency"]),
            "--max-num-batched-tokens", str(case["batched_tokens"]),
            "--max-model-len", str(cfg["max_model_len"]),
            "--kv-cache-memory-bytes", str(cfg["kv_cache_bytes"]),
            "--trust-remote-code", "--gpu-memory-utilization",
            str(cfg["gpu_memory_utilization"])]


def bench_command(cfg, case, directory, round_index):
    warmup = round_index == 0
    prompts = 2 if warmup else cfg["num_prompts"]
    length = 1024 if warmup else case["input_length"]
    seed = 999 if warmup else ROUNDS_SEED_BASE + case["input_length"] + round_index
    return [cfg["python"], "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai", "--endpoint", "/v1/completions",
            "--base-url", f'http://127.0.0.1:{cfg["port"]}',
            "--model", cfg["model"], "--served-model-name", cfg["served_model"],
            "--tokenizer", cfg["model"], "--dataset-name", "random",
            "--num-prompts", str(prompts), "--max-concurrency",
            str(2 if warmup else cfg["max_concurrency"]),
            "--random-input-len", str(length), "--random-output-len", "1",
            "--random-range-ratio", "0.0", "--request-rate", "inf",
            "--ignore-eos", "--seed", str(seed),
            "--percentile-metrics", "ttft,e2el", "--metric-percentiles", "50,90,99",
            "--save-result", "--result-dir", str(directory),
            "--result-filename", f"round_{round_index}.json"]


def with_cann(cfg, command):
    # Positional arguments preserve quoting; the shell never interpolates model paths.
    return ["bash", "-c", 'source "$1" >&2 || exit; shift; exec "$@"',
            "dsv4flash", cfg["cann_env"], *command]


def parse_npu(text):
    devices, processes = {}, {}
    current, in_processes = None, False
    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_processes = True
            continue
        if in_processes:
            match = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
            if match:
                card, pid = map(int, match.groups())
                processes.setdefault(card, set()).add(pid)
            continue
        fields = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(fields) >= 3 and fields[0].isdigit() and fields[1].startswith("Ascend"):
            current = int(fields[0])
            devices[current] = {"health": fields[2]}
        elif current is not None:
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if pairs:
                used, total = map(int, pairs[-1])
                if total > 0:
                    devices[current].update(used_mb=used, total_mb=total)
    return devices, processes


def capture_npu(path=None):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                            timeout=60, check=True)
    if path is not None:
        Path(path).write_text(result.stdout + result.stderr)
    return parse_npu(result.stdout)


def wait_cards_free(cfg, path):
    deadline = time.monotonic() + cfg["release_timeout_seconds"]
    while True:
        devices, processes = capture_npu(path)
        for card in cfg["cards"]:
            if card not in devices or "used_mb" not in devices[card]:
                raise RuntimeError(f"Cannot parse npu-smi for card {card}; refusing launch.")
            if devices[card]["health"] != "OK":
                raise RuntimeError(f"Card {card} is unhealthy.")
        busy = {card: devices[card] for card in cfg["cards"]
                if processes.get(card) or devices[card]["used_mb"] >= cfg["idle_hbm_limit_mb"]}
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Rear cards are not free; no workload was stopped: {busy}")
        time.sleep(HARDWARE_WAIT_SECONDS)


def port_free(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def preflight(cfg):
    port_free(cfg["port"])
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run inside the xzx_moonep container.")
    for key in ("python", "cann_env"):
        if not Path(cfg[key]).is_file():
            raise RuntimeError(f'Missing {key}: {cfg[key]}')
    if not Path("/sys/class/net", cfg["interface"]).exists():
        raise RuntimeError("Configured network interface does not exist.")
    # Verify IP belongs to the configured interface without requiring the ip utility.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        name = cfg["interface"].encode()
        address = socket.inet_ntoa(fcntl.ioctl(
            sock.fileno(), 0x8915, struct.pack("256s", name[:15]))[20:24])
    if address != cfg["interface_ip"]:
        raise RuntimeError(f'Interface IP changed: {address} != {cfg["interface_ip"]}')
    model = Path(cfg["model"])
    index = read_json(model / "model.safetensors.index.json")
    for name in set(index["weight_map"].values()):
        shard = model / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty model shard: {shard}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise RuntimeError(f"Missing tokenizer file: {name}")


def read_models(cfg):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{cfg["port"]}/v1/models', timeout=3) as response:
        return json.load(response)


def wait_ready(cfg, process, log_path=None):
    deadline = time.monotonic() + cfg["ready_timeout_seconds"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited early: {process.returncode}")
        if log_path is not None:
            text = Path(log_path).read_text(errors="replace")
            markers = ("torch.OutOfMemoryError:", "AssertionError:", "ValueError:",
                       "WorkerProc failed to start.", "Engine core initialization failed.",
                       "Worker failed with error")
            if any(marker in text for marker in markers):
                raise RuntimeError("Terminal startup failure; inspect service.log")
        try:
            models = read_models(cfg)
            if any(model.get("id") == cfg["served_model"] for model in models["data"]):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(READINESS_POLL_SECONDS)
    raise TimeoutError("Service readiness timeout; inspect service.log")


def stop_owned_group(process):
    if process is None:
        return
    # Only the session/process group created by this runner is ever signalled.
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader; check whether workers still exist in its group.
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def validate_result(path, expected_prompts):
    result = read_json(path)
    if result.get("completed") != expected_prompts or result.get("failed", 0) != 0:
        raise ValueError(f"Failed/incomplete requests in {path}")
    for key in ("duration", "mean_ttft_ms", "mean_e2el_ms", "total_token_throughput",
                "total_input_tokens", "total_output_tokens"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite {key} in {path}")
        if value <= 0:
            raise ValueError(f"Nonpositive {key} in {path}")
    if result["total_output_tokens"] != expected_prompts:
        raise ValueError(f"Expected exactly one output token per request: {path}")
    return result


def worker_signature(cfg, path):
    devices, processes = capture_npu(path)
    for card in cfg["cards"]:
        if card not in devices or devices[card]["health"] != "OK" or not processes.get(card):
            raise RuntimeError(f"Missing/unhealthy worker on card {card}")
    return {str(card): sorted(processes.get(card, set())) for card in cfg["cards"]}


def assert_same_workers(cfg, signature, path):
    current = worker_signature(cfg, path)
    if current != signature:
        raise RuntimeError("NPU process set changed; invalidate this run and inspect NPU snapshots.")


def run_case(cfg, case, route, root):
    directory = root / case["id"]
    directory.mkdir(exist_ok=False)
    env = service_env(cfg, case, route)
    command = service_command(cfg, case)
    metadata = {"case": case, "route": route, "status": "starting", "started_at": timestamp(),
                "service_command": command, "bench_commands": [],
                "environment": {k: v for k, v in env.items() if (
                    k.startswith(("MOONEP", "ASCEND_MOONEP", "VLLM_ASCEND_MOE_", "HCCL"))
                    or k in ("ASCEND_RT_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH",
                             "ASCEND_CUSTOM_OPP_PATH", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
                             "PYTORCH_NPU_ALLOC_CONF", "VLLM_WORKER_MULTIPROC_METHOD",
                             "VLLM_PLUGINS"))}}
    write_json(directory / "metadata.json", metadata)
    process = None
    try:
        port_free(cfg["port"])
        wait_cards_free(cfg, directory / "npu_before.txt")
        with (directory / "service.log").open("w") as log:
            process = subprocess.Popen(with_cann(cfg, command), env=env, cwd=HERE,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            metadata["service_pgid"] = process.pid
            write_json(directory / "metadata.json", metadata)
            wait_ready(cfg, process, directory / "service.log")
            signature = worker_signature(cfg, directory / "npu_ready.txt")
            metadata["worker_signature"] = signature
            # Exactly one steady-state worker per card for DP4/TP1; fail on extra jobs.
            if any(len(pids) != 1 for pids in signature.values()):
                raise RuntimeError("Expected one vLLM worker per selected card; possible interference.")
            for round_index in range(cfg["rounds"] + 1):
                assert_same_workers(cfg, signature, directory / f"npu_round_{round_index}_before.txt")
                bench = bench_command(cfg, case, directory, round_index)
                metadata["bench_commands"].append(bench)
                write_json(directory / "metadata.json", metadata)
                with (directory / f"bench_{round_index}.log").open("w") as bench_log:
                    subprocess.run(with_cann(cfg, bench), env=env, cwd=HERE,
                                   stdout=bench_log, stderr=subprocess.STDOUT,
                                   timeout=cfg["bench_timeout_seconds"], check=True)
                validate_result(directory / f"round_{round_index}.json",
                                2 if round_index == 0 else cfg["num_prompts"])
                if process.poll() is not None:
                    raise RuntimeError("Service died during benchmark.")
                assert_same_workers(cfg, signature, directory / f"npu_round_{round_index}_after.txt")
            metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            stop_owned_group(process)
            if process is not None:
                wait_cards_free(cfg, directory / "npu_after.txt")
        except BaseException as exc:
            metadata["status"] = "failed"
            metadata["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metadata["ended_at"] = timestamp()
            write_json(directory / "metadata.json", metadata)


def inventory(cfg, root):
    inventory = {"at": timestamp(), "repos": {}}
    for repo in ("vllm", "vllm-ascend", "ascend-moonep"):
        path = Path(cfg["workspace"]) / repo
        inventory["repos"][repo] = {}
        for name, args in (("head", ["rev-parse", "HEAD"]),
                           ("status", ["status", "--short", "--untracked-files=no"])):
            result = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                    text=True, timeout=60)
            inventory["repos"][repo][name] = result.stdout.strip()
    result = subprocess.run([cfg["python"], "-m", "pip", "freeze"], capture_output=True,
                            text=True, timeout=60)
    (root / "pip_freeze.txt").write_text(result.stdout + result.stderr)
    inventory["model_config_sha256"] = sha256(Path(cfg["model"]) / "config.json")
    inventory["weight_index_sha256"] = sha256(Path(cfg["model"]) / "model.safetensors.index.json")
    write_json(root / "environment.json", inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true", help="Print plan only (default).")
    group.add_argument("--execute", action="store_true", help="Actually launch services and benchmarks.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    plan = make_plan(cfg)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        route_map = {route["name"]: route for route in plan["routes"]}
        for case in plan["cases"]:
            print(case["id"], shlex.join(service_command(cfg, case)))
            print("  route:", route_map[case["route"]]["path"])
        return
    preflight(cfg)
    # One session for this port, even when two copies are launched concurrently.
    with (HERE / f".port_{cfg['port']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = HERE / "data" / dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime(
            "%Y%m%d_%H%M%S_%f")
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "plan.json", plan)
        shutil.copy2(args.config, root / "config.json")
        for name in ("run_experiment.py", "summarize.py", "EXPERIMENT.md"):
            shutil.copy2(HERE / name, root / name)
        for route in plan["routes"]:
            shutil.copy2(route["path"], root / Path(route["path"]).name)
        inventory(cfg, root)
        print(f"Results: {root}", flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        old_term = signal.signal(signal.SIGTERM, interrupted)
        try:
            route_map = {route["name"]: route for route in plan["routes"]}
            for index, case in enumerate(plan["cases"], 1):
                print(f"[{index}/{len(plan['cases'])}] {case['id']}", flush=True)
                run_case(cfg, case, route_map[case["route"]], root)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            subprocess.run([sys.executable, str(HERE / "summarize.py"), str(root)], check=True)


if __name__ == "__main__":
    main()
