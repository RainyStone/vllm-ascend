"""MoE A/B test: baseline EP comm vs MoonEP EP comm on the same model.

Two ways to run.

1) Single process, tensor parallel (original behaviour):

       TP=2 MOONEP=0 python scripts/moe_ab_test.py
       TP=2 MOONEP=1 python scripts/moe_ab_test.py

2) Data parallel (this same file acts as its own launcher, one process per DP
   rank; needed on A5, where DeepSeek-V4-Flash cannot use TP at all because the
   DSA sparse attention wants 64 or 128 q heads *per rank*):

       DP=2 CARDS=6,7 MOONEP=0 python scripts/moe_ab_test.py
       DP=2 CARDS=6,7 MOONEP=1 python scripts/moe_ab_test.py

   The parent only sets VLLM_DP_* / ASCEND_RT_VISIBLE_DEVICES for each child;
   `data_parallel_size` is deliberately NOT passed to LLM() -- single-process DP
   raises in vllm/entrypoints/llm.py, the env branch of ParallelConfig is what
   feeds the multi-process mode.

   Owner mode is the default. Slot (balance) mode is one more env var; it mirrors
   the local expert rows, so the KV pool shrinks by ~epn*12.75MiB*layers/card
   (~34 GiB at EP4, ~17 GiB at EP8):

       DP=4 CARDS=4,5,6,7 MOONEP=1 MOONEP_SLOT_MODE=1 MOONEP_SLOT_STATS=1 \
           MODEL=<dir> python scripts/moe_ab_test.py

   Leave MOONEP_B unset (the library requires B == E/R) and MOONEP_SLOT_QUANT
   unset on a quantized checkpoint (=0 makes the MoE layer raise, by design).

Env:
  MODEL=<dir>          model path (default /mnt/weight/Qwen3-30B-A3B)
  MOONEP=1             enable_moonep leg (needs enable_expert_parallel)
  NOEP=1               baseline leg WITHOUT expert parallelism (all-gather MoE)
  TP=<int>             tensor parallel size (EP spans the TP group on Ascend)
  DP=<int>             data parallel size; >1 takes the launcher path below
  CARDS=6,7            DP only: the WHOLE card group, handed to every child as
                       ASCEND_RT_VISIBLE_DEVICES (the engine slices it by local
                       dp rank; handing out a single card raises IndexError)
  AB_SLICE=1           DP only: rank r takes PROMPTS[r::DP] (canonical DP split).
                       Default is "every rank runs the full list", which keeps
                       each rank's output comparable with the single-process run
  AB_STATS=1           DP only: also dump the MoE comm-method trace
                       (VLLM_ASCEND_MOE_PAR_STATS) to /tmp/moe_par_<leg>_<tag>
  AB_TIMEOUT=<sec>     DP only: per-child wait budget, default 1800; a child that
                       overruns is killed by pid (hung DP workers do not exit on
                       their own and keep ~90 GB of HBM)
  GMU=<float>          gpu_memory_utilization (default 0.90); slot mode mirrors
                       the expert weights, so lower it if the profile step OOMs
  DEVICES=...          single process: passed via ASCEND_RT_VISIBLE_DEVICES outside

Writes /tmp/ab_{moonep|baseline}_{tag}.json              (single process)
       /tmp/ab_{moonep|baseline}_{tag}_dp{r}.json        (DP, one per rank)
       /tmp/ab_{moonep|baseline}_{tag}_dp_all.json       (DP, merged)
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

MODEL = os.environ.get("MODEL", "/mnt/weight/Qwen3-30B-A3B")
TAG = os.path.basename(MODEL.rstrip("/"))
MOONEP = os.environ.get("MOONEP", "0") == "1"
NOEP = os.environ.get("NOEP", "0") == "1"
LEG = "moonep" if MOONEP else "baseline"
TP = int(os.environ.get("TP", "1"))
DP = int(os.environ.get("DP", "1"))
DP_RANK = os.environ.get("VLLM_DP_RANK")  # only set inside DP children

# Values the DP launcher injects when the caller has not set them. Without these
# the first cross-engine collective of a DP+EP run hangs on this box
# (KERNEL_AICPU / halSqCqAllocate drvRetCode=17) -- see docs/cross_repo_fixes.md 5.
LAUNCH_ENV_DEFAULTS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "MOONEP_DISABLE_UDMA": "1",  # shmem init returns -3 with UDMA enabled here
    "HCCL_BUFFSIZE": "1024",
}

PROMPTS = [
    "Hello! My name is",
    "The future of AI is",
]


def out_path(rank=None):
    suffix = "" if rank is None else f"_dp{rank}"
    return f"/tmp/ab_{LEG}_{TAG}{suffix}.json"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_worker():
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    rank = int(os.environ.get("VLLM_DP_RANK", "0"))
    dp_size = int(os.environ.get("VLLM_DP_SIZE", "1"))
    dp = dp_size > 1
    prompts = PROMPTS
    if dp and os.environ.get("AB_SLICE", "0") == "1":
        prompts = [p for i, p in enumerate(PROMPTS) if i % dp_size == rank]
    if not prompts:  # a rank with no slice still has to join the collectives
        prompts = ["Placeholder"]
    who = f"{LEG} dp{rank}" if dp else LEG
    print(f"[{who}] card={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
          f"DP={dp_size} TP={TP} MOONEP={MOONEP} NOEP={NOEP} prompts={len(prompts)}",
          flush=True)

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        enable_expert_parallel=not NOEP,
        tensor_parallel_size=TP,
        max_model_len=4096,
        max_num_batched_tokens=512,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.90")),
        additional_config={"enable_moonep": MOONEP},
        # data_parallel_size is intentionally absent: the multi-process DP mode
        # reads VLLM_DP_* in ParallelConfig, passing it here raises.
    )
    params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    outputs = llm.generate(prompts, params)
    results = []
    for o in outputs:
        first = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        results.append({
            "dp_rank": rank if dp else None,
            "prompt": o.prompt,
            "token_ids": o.outputs[0].token_ids,
            "text": o.outputs[0].text,
            "top5_first": [(t, round(float(lp.logprob), 4)) for t, lp in first.items()],
        })
    path = out_path(rank if dp else None)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{who}] saved {path}", flush=True)
    for r in results:
        print(f"[{who}]   {r['prompt'][:40]!r} -> {len(r['token_ids'])} tok: {r['text'][:70]!r}",
              flush=True)
    return 0


def merge_and_report():
    """Merge the per-rank files and show the cross-rank spread per prompt."""
    merged = []
    for rank in range(DP):
        path = out_path(rank)
        if not os.path.exists(path):
            print(f"[dp] WARNING: rank {rank} produced no {path}", flush=True)
            continue
        with open(path) as f:
            merged.append({"dp_rank": rank, "results": json.load(f)})
    path = f"/tmp/ab_{LEG}_{TAG}_dp_all.json"
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[dp] merged {len(merged)}/{DP} rank files -> {path}", flush=True)

    if os.environ.get("AB_SLICE", "0") == "1" or not merged:
        return len(merged) == DP
    prompts = [r["prompt"] for r in merged[0]["results"]]
    for i, prompt in enumerate(prompts):
        per_rank = []
        for entry in merged:
            res = entry["results"][i]
            top1 = res["top5_first"][0] if res["top5_first"] else (None, None)
            per_rank.append((entry["dp_rank"], top1[0], top1[1]))
        spread = [p[2] for p in per_rank if p[2] is not None]
        delta = f" spread={max(spread) - min(spread):.4f}" if len(spread) > 1 else ""
        print(f"[dp] {prompt[:30]!r} top1(rank,token,logprob)={per_rank}{delta}", flush=True)
    return len(merged) == DP


def launch_dp():
    """Parent side: one child per DP rank, all env-driven, then merge."""
    if not os.environ.get("CARDS"):
        sys.exit("DP>1 needs CARDS=<whole group> so every child sees all of its cards, "
                 "e.g. CARDS=6,7")
    injected = {k: v for k, v in LAUNCH_ENV_DEFAULTS.items() if k not in os.environ}
    if injected:
        print(f"[dp] injecting {injected}", flush=True)
    if os.environ.get("AB_STATS", "0") == "1":
        injected.update({
            "VLLM_ASCEND_MOE_PAR_STATS": "1",
            "VLLM_ASCEND_MOE_PAR_STATS_DIR": f"/tmp/moe_par_{LEG}_{TAG}",
            "VLLM_ASCEND_MOE_PAR_STATS_TAG": f"{LEG}_{TAG}",
        })
    port = free_port()
    script = os.path.abspath(__file__)
    # cwd must not contain a `vllm/` source tree: the trailing empty segment of
    # PYTHONPATH resolves against cwd and would make `vllm` a namespace package.
    cwd = tempfile.gettempdir()
    print(f"[dp] launch DP={DP} TP={TP} {LEG} model={MODEL} cards={os.environ['CARDS']} "
          f"port={port}", flush=True)

    procs = []
    for rank in range(DP):
        env = os.environ.copy()
        env.update(injected)
        env.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(DP),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ["CARDS"],
        })
        procs.append(subprocess.Popen([sys.executable, script], cwd=cwd, env=env,
                                      start_new_session=True))

    timeout = float(os.environ.get("AB_TIMEOUT", "1800"))
    deadline = time.time() + timeout
    rc = 0
    for rank, proc in enumerate(procs):
        try:
            rc |= proc.wait(timeout=max(1.0, deadline - time.time())) or 0
        except subprocess.TimeoutExpired:
            print(f"[dp] rank {rank} (pid {proc.pid}) exceeded {timeout:.0f}s, killing",
                  flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = 1
    merge_and_report()
    print(f"[dp] {LEG} exit={'OK' if rc == 0 else rc}", flush=True)
    return rc


def main():
    if DP > 1 and not DP_RANK:
        return launch_dp()
    return run_worker()


if __name__ == "__main__":
    sys.exit(main())
