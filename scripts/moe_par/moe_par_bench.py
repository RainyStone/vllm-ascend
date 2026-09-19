"""Offline MoE load / PAR sweep: run a dataset through vLLM and dump per-rank routing load.

Each run writes one dump per rank into ``<stats-dir>/<tag>/`` and a ``bench.json`` next to
them. Feed those directories to ``analyze_moe_par.py`` to get the HTML report -- the exact
command is printed at the end of the run and stored in ``bench.json["analyze_cmd"]``.

``bench.json`` also carries the run's own performance numbers: overall throughput plus
per-request TTFT / TPOT / end-to-end, read off ``RequestOutput.metrics`` (see
:func:`request_latency`). The report renders them in a 性能 panel, per request and per run,
so the MoE communication legs can be compared on latency and not only on load.

Four things to know before editing this file:

* **One setting per process.** The recorder keeps module-level state (layer map, step
  counter, open file handle) that is never reset, so building a second ``LLM`` in the same
  process would append to the same dump under a stale layer map. Run this once per configuration.
* **The stats switch is latched at import.** ``moe_par_stats`` reads ``VLLM_ASCEND_MOE_PAR_*``
  when the module is first imported, so :func:`apply_stats_env` must run *before*
  ``import vllm`` inside :func:`main`. Do not hoist that import to the top of this file.
* **The engine's own stats must be asked for.** The offline ``LLM`` defaults to
  ``disable_log_stats=True``; with them off the engine allocates no request stats and every
  latency field reads as 0.0. :func:`main` passes ``disable_log_stats=False``.
* **A DP run is a parent process plus one child per rank.** The child is this same file
  re-executed with ``VLLM_DP_RANK`` in its environment (see :func:`launch_dp`); the parent
  never imports vllm, it only launches, releases and merges. ``data_parallel_size`` is
  deliberately *not* passed to ``LLM()`` -- single-process DP raises in
  ``vllm/entrypoints/llm.py``, the ``VLLM_DP_*`` env branch of ParallelConfig is what feeds
  the multi-process mode.

Example:
  ASCEND_RT_VISIBLE_DEVICES=1,6 python scripts/moe_par/moe_par_bench.py \\
      --tp 2 --batch-size 32 --max-num-batched-tokens 512 --max-tokens 256 \\
      --num-prompts 128 --tag tp2-b32-bt512

  # DeepSeek-V4-Flash cannot use TP (DSA sparse attention needs 64 q heads per rank and the
  # checkpoint has 64 in total), so the only shape that runs is DP:
  python scripts/moe_par/moe_par_bench.py --dp 8 --cards 0,1,2,3,4,5,6,7 --tp 1 \\
      --moonep-mode slot --model /mnt/weight/A5-weights/DeepSeek-V4-Flash \\
      --dataset scripts/moe_par/gsm8k_test.jsonl --gpu-mem-util 0.80
"""
import argparse
import glob
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback

# This file lives in scripts/moe_par/, next to the analyzer it points at and the dataset it
# defaults to; the dumps land one level up in scripts/moe_par_stats/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "gsm8k_test.jsonl")
DEFAULT_MODEL = "/mnt/data/weight/Qwen3-Coder-30B-A3B-Instruct"

# The vendored, patched copy of the ascend_moonep python package. The sibling checkout
# one directory up is an older copy without acl_vmm_access_device_id; resolving to it
# breaks slot mode on a remapped card group (docs/cross_repo_fixes.md). Pinning it first
# on PYTHONPATH is what makes this checkout's fixes win, and the .so still comes from
# ASCEND_MOONEP_BUILD_DIR.
VENDORED_MOONEP = os.path.join(REPO_DIR, "third_party", "ascend_moonep")

# Values injected when the caller has not set them. Without these the first cross-engine
# collective of a DP+EP run hangs on this box (KERNEL_AICPU / halSqCqAllocate
# drvRetCode=17) or shmem init returns -3 -- see docs/cross_repo_fixes.md 5.
LAUNCH_ENV_DEFAULTS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "MOONEP_DISABLE_UDMA": "1",  # shmem init returns -3 with UDMA enabled here
    "HCCL_BUFFSIZE": "1024",
}

# Where the compiled ascend_moonep lives on this box. ASCEND_MOONEP_BUILD_DIR wins; the
# sibling-repo guess is what the verified DP runs used.
MOONEP_BUILD_CANDIDATES = [
    os.path.join(os.path.dirname(REPO_DIR), "ascend-moonep", "ascend_moonep"),
]

# Memory ledger lines, straight out of the worker (worker.py:593,759) and the engine core
# (kv_cache_utils.py). Parsed with re.search over the whole log and "last match wins":
# these lines are per-process, and the multiprocess formatter prefixes them with
# ``(Worker_DP3_EP3 pid=...)``, so nothing may be anchored to the start of a line.
LOG_PATTERNS = {
    "weights_gib": r"Actual usage: ([\d.]+) GiB for weights",
    # "<value> GiB for peak activation, <value> GiB for non-torch memory" -- the value
    # precedes the label, so anchoring on what follows it would read the next number.
    "peak_activation_gib": r"([\d.]+) GiB for peak activation",
    "non_torch_gib": r"([\d.]+) GiB for non-torch memory",
    "npugraph_gib": r"([\d.]+) GiB for NPU graph memory",
    "requested_gib": r"Desired GPU memory utilization is \([\d.]+, ([\d.]+) GiB\)",
    "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "load_weights_s": r"Loading model weights took ([\d.]+) G(?:i)?B",
}
# Slot mode's exit summary (moonep_comm.py:_dump_slot_stats). The M7 criterion is
# total_slot_rows > 0 and calls_with_slot_rows > 0 -- a slot run that never took a slot
# row is a run that silently fell back to the owner path.
SLOT_STATS_RE = re.compile(
    r"\[moonep-stats\] dispatch_calls=(\d+) calls_with_slot_rows=(\d+) "
    r"total_slot_rows=(\d+) total_prefetched_expert_slots=(\d+)")

# The wedge detector in _supervise: how much of a rank log to look at, and how long a
# traceback-tailed log may stay frozen before the rank counts as hung. Model loading is
# silent for ~100s but writes no traceback, so neither number needs to survive that.
LOG_TAIL_BYTES = 64 * 1024
WEDGE_GRACE = 60.0


# --------------------------------------------------------------------------- args -- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a dataset offline and dump per-rank MoE routing load for PAR analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("workload")
    g.add_argument("--dataset", default=DEFAULT_DATASET, help="jsonl file; one object per line")
    g.add_argument("--prompt-field", default="question", help="which json field holds the prompt")
    g.add_argument("--num-samples", "--num-prompts", dest="num_prompts", type=int, default=0,
                   help="how many dataset rows to run; 0 = the whole file (this is the default)")
    g.add_argument("--max-tokens", type=int, default=256, help="output length per request")
    g.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                   help="generate exactly --max-tokens tokens, so every run carries the same load "
                        "(this is the default)")
    g.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false", default=argparse.SUPPRESS,
                   help="stop at EOS instead (realistic, but output length then varies per run)")
    g.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy, reproducible")

    g = p.add_argument_group("batching")
    g.add_argument("--batch-size", "--max-num-seqs", dest="max_num_seqs", type=int, default=32,
                   help="decode concurrency, i.e. the batch size")
    g.add_argument("--max-num-batched-tokens", type=int, default=1024,
                   help="per-step token budget; this is what select_moe_comm_method keys on")

    g = p.add_argument_group("parallelism")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--tp", type=int, default=None,
                   help="tensor parallel size; EP spans this group (default: 4, or 1 whenever "
                        "--dp > 1 -- DP is the only shape this model runs, and leaving the "
                        "legacy 4 here would fail every DP leg with an AICPU 22007 message)")
    g.add_argument("--dp", type=int, default=1,
                   help="data parallel size; >1 takes the multi-process launcher path, one child "
                        "per rank, each with its own engine (the only shape DeepSeek-V4-Flash runs)")
    g.add_argument("--cards", default="",
                   help="DP only: the WHOLE card group, e.g. 0,1,2,3,4,5,6,7; handed to every child "
                        "as ASCEND_RT_VISIBLE_DEVICES, because the engine slices it by local rank")
    g.add_argument("--dp-slice", dest="dp_slice", action="store_true", default=None,
                   help="DP only: rank r runs questions[r::DP] instead of the whole list. On by "
                        "default: every rank carries the same load, so the aggregate throughput "
                        "shows DP scaling and no rank falls behind into dummy batches")
    g.add_argument("--no-dp-slice", dest="dp_slice", action="store_false", default=argparse.SUPPRESS,
                   help="DP only: every rank runs the full list (the aggregate then counts the "
                        "same prompts DP times, matching what the dumps record)")
    g.add_argument("--no-ep", dest="ep", action="store_false", default=True,
                   help="turn expert parallelism off (ep_size=1, the all-gather MoE run); "
                        "EP is ON unless you pass this")
    g.add_argument("--moonep", action="store_true",
                   help="enable_moonep + MOONEP=1 (owner mode); same as --moonep-mode owner")
    g.add_argument("--moonep-mode", choices=("off", "owner", "slot"), default=None,
                   help="MoonEP leg: off = plain all-to-all EP, owner = MOONEP_SLOT_MODE=0, "
                        "slot = MOONEP_SLOT_MODE=1 (mirrored expert rows). Sets both the engine "
                        "switch and the library env, so a stale shell value cannot leak across legs")
    g.add_argument("--max-model-len", type=int, default=40960)
    g.add_argument("--gpu-mem-util", type=float, default=0.80)
    g.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false", default=True,
                   help="allow graph capture; eager is ON unless you pass this")
    g.add_argument("--additional-config", default="",
                   help="extra JSON merged into additional_config, e.g. '{\"foo\": 1}'")

    g = p.add_argument_group("gsm8k scoring")
    g.add_argument("--score", action="store_true",
                   help="run the few-shot GSM8K scoring pass instead of the load pass: raw "
                        "completion prompts (no chat template), stop sequences, EOS honoured. "
                        "Its own process, so keep the dump clean with --no-stats")
    g.add_argument("--train-file", default="", help="5-shot examples; default: probe the local copies")
    g.add_argument("--test-file", default="",
                   help="questions to score; default is --dataset, which must then carry 'answer'")
    g.add_argument("--num-shots", type=int, default=5, help="few-shot examples in the prompt")
    g.add_argument("--score-max-tokens", type=int, default=256,
                   help="output budget per scored question (the reference evaluation uses 256)")

    g = p.add_argument_group("profiling")
    g.add_argument("--profile-dir", default="",
                   help="absolute directory for the torch_npu traces; requires --profile-only. The "
                        "recorder latches at import, so profiling in the same process as the load "
                        "pass would append profile steps to the load dump")
    g.add_argument("--profile-only", action="store_true",
                   help="skip the load pass and only run the start_profile()/stop_profile() window "
                        "over --num-prompts requests")
    g.add_argument("--profile-max-tokens", type=int, default=32,
                   help="output length inside the profiling window; short and fixed keeps every "
                        "rank in lockstep, which is what the export needs")

    g = p.add_argument_group("stats")
    g.add_argument("--stats-dir", default=os.path.join(SCRIPTS_DIR, "moe_par_stats"),
                   help="root for the dumps; they land in <stats-dir>/<tag>/ so runs cannot "
                        "collide (ignored when --dump-dir is given)")
    g.add_argument("--dump-dir", default="",
                   help="exact directory for this run's dumps, used verbatim with no <tag>/ "
                        "suffix; keep one tag per directory, because analyze_moe_par.py globs "
                        "the whole directory and would merge two tags into a single run")
    g.add_argument("--force", action="store_true",
                   help="write into --dump-dir even though it already holds dumps of another tag")
    g.add_argument("--tag", default="",
                   help="name for this run; it names the dump subdirectory and the --run name "
                        "in the report; default is derived from the settings")
    g.add_argument("--flush-every", type=int, default=16, help="steps buffered per host copy")
    g.add_argument("--no-stats", dest="stats", action="store_false", default=True,
                   help="run with collection off, for an overhead baseline; collection is ON "
                        "unless you pass this")
    g.add_argument("--out", default="", help="summary json path; default <stats-dir>/<tag>/bench.json")
    g.add_argument("--drain-timeout", type=float, default=30.0,
                   help="seconds to wait for the workers to finish flushing their dumps")

    g = p.add_argument_group("launch")
    g.add_argument("--timeout", type=float, default=3600.0,
                   help="DP only: per-run budget for the children; a child that overruns is killed "
                        "by process group (hung DP workers keep ~90 GB of HBM and never exit)")
    g.add_argument("--rendezvous-timeout", type=float, default=600.0,
                   help="DP only: how long a child waits for its peers before tearing its engine "
                        "down. A DP rank must not exit while a peer still generates -- the peer's "
                        "next cross-rank all-reduce would fail -- so the children hold each other "
                        "back until everyone is done")
    g.add_argument("--min-free-gib", type=float, default=60.0,
                   help="DP only: refuse to start unless every card in --cards has at least this "
                        "much free HBM (0 disables the check)")
    g.add_argument("--allow-tiny-kv", action="store_true",
                   help="do not treat a KV pool smaller than the workload it must serve as a "
                        "failure. A run whose pool cannot hold --max-model-len or one full batch "
                        "measures preemption, not MoE communication, so it is reported as failed")
    g.add_argument("--dry-run", action="store_true",
                   help="print the per-rank env and command for a manual launch; nothing starts")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    return finalize_args(args)


def finalize_args(args):
    """Resolve the flags that have a legacy form and reject the shapes that cannot run."""
    args.mode = args.moonep_mode or ("owner" if args.moonep else "off")
    args.moonep = args.mode != "off"
    if args.tp is None:  # see --tp: the legacy default is only sane without DP
        args.tp = 1 if args.dp > 1 else 4
    if args.dp_slice is None:
        args.dp_slice = args.dp > 1
    args.cards_list = [int(c) for c in args.cards.replace(" ", "").split(",") if c]

    # Paths are resolved here, once, against the directory the run was launched from. The DP
    # children are re-executions with cwd=/tmp (a `vllm/` source tree in the cwd would turn
    # `vllm` into a namespace package, see launch_dp), so a relative --dataset would resolve
    # against /tmp inside every rank and fail there instead of here.
    base = os.environ.get("VLLM_BENCH_CWD") or os.getcwd()
    for name in ("dataset", "model", "stats_dir", "dump_dir", "out", "profile_dir",
                 "train_file", "test_file"):
        value = getattr(args, name, "")
        if value and not os.path.isabs(value):
            setattr(args, name, os.path.normpath(os.path.join(base, value)))

    if args.dp > 1 and args.tp > 1:
        sys.exit("DP>1 with TP>1 cannot run this model: the DSA sparse attention needs 64 or "
                 "128 q heads per rank and DeepSeek-V4-Flash has 64 in total (AICPU 22007). "
                 "Use --tp 1 and let EP span the DP group.")
    if args.dp > 1 and len(args.cards_list) < args.dp:
        sys.exit(f"--dp {args.dp} needs --cards with at least {args.dp} cards "
                 f"(got {args.cards!r}); every child sees the whole group and slices it itself")
    if args.mode != "off" and not args.ep:
        sys.exit(f"--moonep-mode {args.mode} needs expert parallelism; drop --no-ep")
    if args.profile_only and not args.profile_dir:
        sys.exit("--profile-only needs --profile-dir")
    if args.profile_dir and not args.profile_only:
        sys.exit("--profile-dir needs --profile-only: the recorder latches its switch at import, "
                 "so profiling inside the load pass would append its steps to the load dump")
    if args.profile_only:
        args.stats = False
        # The window is one `--num-prompts` split across the DP ranks, and every rank has to
        # run the same number of prompts: the profiler's export is synchronous inside a
        # worker, so a rank with fewer steps exports while its peers are still stepping, and
        # a rank with no prompts at all exits mid-collective (see _profile_pass). Reject the
        # ragged shapes here, where the message can name the flag, rather than in eight
        # children at once.
        if args.dp > 1 and (args.num_prompts <= 0 or args.num_prompts % args.dp):
            sys.exit(f"--profile-only with --dp {args.dp} needs --num-prompts to be a "
                     f"positive multiple of {args.dp} (one window per rank); got "
                     f"{args.num_prompts}. The sweep passes --dp * --profile-prompts.")
    if args.max_tokens >= args.max_model_len:
        sys.exit(f"--max-tokens {args.max_tokens} leaves no room in --max-model-len "
                 f"{args.max_model_len}")
    if args.score and args.stats:
        print("[bench] WARNING: --score with collection on writes the scored answers into the "
              "same dump as the load pass; the MoE load then mixes a ragged, EOS-stopped batch "
              "in with the measured one. Pass --no-stats unless you really want that.", flush=True)
    return args


def make_tag(args):
    """A run name that already says what makes this configuration differ from its neighbours."""
    parts = [f"dp{args.dp}" if args.dp > 1 else f"tp{args.tp}"]
    parts.append("ep" if args.ep else "noep")
    # 'off' stays suffix-free so tags from before --moonep-mode existed still line up.
    if args.mode == "owner":
        parts.append("moonep")
    elif args.mode == "slot":
        parts.append("moonep-slot")
    parts.append(f"b{args.max_num_seqs}")
    parts.append(f"bt{args.max_num_batched_tokens}")
    parts.append(f"o{args.max_tokens}")
    return "-".join(parts)


# ------------------------------------------------------------------------ prompts -- #
def load_questions(path, field, limit):
    """Read up to ``limit`` prompts and report how many the file held in total."""
    rows = []
    total = None
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: not valid json ({exc})")
            if field not in obj:
                sys.exit(f"{path}:{lineno}: no field {field!r}; has {sorted(obj)}")
            rows.append(str(obj[field]))
            if limit and len(rows) >= limit:
                # Only worth a second pass once we know we stopped short.
                total = len(rows) + sum(1 for rest in fh if rest.strip())
                break
    if not rows:
        sys.exit(f"{path}: no usable rows")
    return rows, (len(rows) if total is None else total)


def build_prompts(tok, questions):
    """Chat-template each question; prompts go out at the dataset's own length."""
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        try:
            # return_dict=False is required: transformers 5.x returns a BatchEncoding
            # otherwise, and vLLM wants a plain list of ids.
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=False)
        except Exception:
            # No chat template on this tokenizer -- fall back to the raw question.
            ids = tok(question).input_ids
        prompts.append({"prompt_token_ids": ids})
    return prompts


# --------------------------------------------------------------------------- env -- #
def stats_tag(tag):
    """The recorder puts the tag *inside a filename* (``moe_par_<tag>_rank<r>.log``), so a
    sweep's ``<sweep-tag>/<leg>`` has to lose its separators or the open() fails with ENOENT
    and the run ends with no dumps at all. Only the environment sees this form; the
    directory layout keeps the original tag."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", tag).strip("-") or "run"


def apply_stats_env(args, tag, dump_dir):
    """Must run before ``import vllm`` -- the recorder reads these at import time."""
    os.environ["VLLM_ASCEND_MOE_PAR_STATS"] = "1" if args.stats else "0"
    os.environ["VLLM_ASCEND_MOE_PAR_STATS_DIR"] = dump_dir
    os.environ["VLLM_ASCEND_MOE_PAR_STATS_TAG"] = stats_tag(tag)
    os.environ["VLLM_ASCEND_MOE_PAR_FLUSH_EVERY"] = str(args.flush_every)
    # Workers must be able to run their exit hooks. A forked worker leaves through
    # os._exit(), which skips atexit, so the last unflushed steps of the dump -- and
    # with them the pad accounting, which only the exit hook used to write -- would
    # be lost. setdefault, not assignment: an explicit choice in the environment wins.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if os.environ["VLLM_WORKER_MULTIPROC_METHOD"] != "spawn":
        print(
            f"警告：VLLM_WORKER_MULTIPROC_METHOD={os.environ['VLLM_WORKER_MULTIPROC_METHOD']}"
            f"，worker 退出时不执行 atexit 钩子：dump 会缺最后不足 {args.flush_every} 步"
            "（对账自查会报出来）。除非确知后果，否则请用 spawn。",
            file=sys.stderr,
        )
    apply_moonep_env(args.mode)


def leg_env(mode):
    """The MOONEP_* overlay for one leg, as a dict; ``None`` means "remove this".

    The library loads libshmem off ``MOONEP`` and picks the dispatch shape off
    ``MOONEP_SLOT_MODE``; both are process-wide, so a value left in the shell by the
    previous leg would silently run the wrong configuration. The sweep builds each leg's
    environment from here, which is why this is a pure function and not just a mutation.

    ``MOONEP_B`` stays unset on purpose: the library requires B == E/R and its default is
    exactly that. ``MOONEP_SLOT_QUANT`` stays unset too -- it defaults to 1, while 0 makes
    the MoE layer raise on a quantized checkpoint (by design).
    """
    if mode == "off":
        return {"MOONEP": None, "MOONEP_SLOT_MODE": None, "MOONEP_SLOT_STATS": None}
    return {
        "MOONEP": "1",
        "MOONEP_SLOT_MODE": "1" if mode == "slot" else "0",
        # Only slot mode counts slot rows, and the M7 criterion is read back off the rank
        # logs -- the exit hook prints nothing without this.
        "MOONEP_SLOT_STATS": "1" if mode == "slot" else None,
    }


def apply_moonep_env(mode):
    """Set this process's MOONEP_* from :func:`leg_env`."""
    for key, value in leg_env(mode).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def find_moonep_build():
    """The compiled ascend_moonep, or None when this box has none at the known places."""
    if os.environ.get("ASCEND_MOONEP_BUILD_DIR"):
        return os.environ["ASCEND_MOONEP_BUILD_DIR"]
    for path in MOONEP_BUILD_CANDIDATES:
        if glob.glob(os.path.join(path, "_C.cpython-*.so")):
            os.environ["ASCEND_MOONEP_BUILD_DIR"] = path
            return path
    return None


def apply_launch_env(args):
    """Process-wide defaults, injected only where the caller left a hole.

    ``setdefault`` and not assignment: the point of these is to keep the verified recipe
    reachable without a wrapper script, not to override an operator who is deliberately
    varying one of them.
    """
    injected = {k: v for k, v in LAUNCH_ENV_DEFAULTS.items() if k not in os.environ}
    for key, value in injected.items():
        os.environ[key] = value
    if injected:
        print(f"[bench] injecting {injected}", flush=True)
    if os.environ["PYTORCH_NPU_ALLOC_CONF"] != LAUNCH_ENV_DEFAULTS["PYTORCH_NPU_ALLOC_CONF"]:
        print(f"[bench] NOTE: PYTORCH_NPU_ALLOC_CONF={os.environ['PYTORCH_NPU_ALLOC_CONF']} "
              f"(the verified DP recipe uses "
              f"{LAUNCH_ENV_DEFAULTS['PYTORCH_NPU_ALLOC_CONF']})", flush=True)

    vendored = os.path.join(VENDORED_MOONEP, "ascend_moonep")
    if os.path.isdir(vendored):
        # Keep whatever else the caller had, including the empty trailing segment that
        # PYTHONPATH normally carries -- only the order is ours to decide.
        existing = os.environ.get("PYTHONPATH", "")
        if not existing.startswith(VENDORED_MOONEP):
            os.environ["PYTHONPATH"] = VENDORED_MOONEP + os.pathsep + existing
        build = find_moonep_build()
        print(f"[bench] ascend_moonep: vendored copy first on PYTHONPATH, "
              f"build dir {build or '<not found> (import will likely fail)'}", flush=True)
    else:
        print(f"[bench] WARNING: no vendored ascend_moonep at {vendored}; the import may resolve "
              f"to an unpatched copy and break slot mode on a remapped card group", flush=True)


def check_moonep_import():
    """Resolve ``ascend_moonep`` in a child and make sure it is the vendored one.

    The failure this guards against is silent: the package imports fine either way and only
    fails much later, on a card, with a driver error. Cheap enough to run once per launch.
    """
    vendored = os.path.join(VENDORED_MOONEP, "ascend_moonep")
    if not os.path.isdir(vendored):
        return None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import ascend_moonep; print(ascend_moonep.__file__)"],
            env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[bench] WARNING: could not check the ascend_moonep import: {exc}", flush=True)
        return None
    resolved = ""
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.strip().endswith("__init__.py"):
            resolved = line.strip()
            break
    if proc.returncode != 0 or not resolved.startswith(VENDORED_MOONEP):
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        print(f"[bench] FATAL: ascend_moonep resolves to {resolved or '<nothing>'}, not the "
              f"vendored {vendored}", flush=True)
        for line in tail:
            print(f"[bench]   {line}", flush=True)
        print("[bench] The sibling checkout one level up is an older copy; on a card group other "
              "than 0..N that shows up as aclrtMemSetAccess 507899 in slot mode. Fix "
              "PYTHONPATH/ASCEND_MOONEP_BUILD_DIR, or delete the stale copy.", flush=True)
        sys.exit(2)
    print(f"[bench] ascend_moonep -> {resolved}", flush=True)
    return resolved


def card_free_gib():
    """Free HBM per card id, read off ``npu-smi info``. Empty when it cannot be read."""
    try:
        proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[bench] WARNING: npu-smi info failed ({exc}); cannot check free HBM", flush=True)
        return {}
    free = {}
    card = None
    for line in proc.stdout.splitlines():
        head = re.match(r"\|\s*(\d+)\s*\|\s*Ascend", line)
        if head:
            card = int(head.group(1))
            continue
        if card is None:
            continue
        # The line under the card id carries '<used> / <total>' twice: pages then HBM.
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if pairs:
            used, total = pairs[-1]
            if int(total) > 0:
                free[card] = (int(total) - int(used)) / 1024.0
            card = None
    return free


def check_cards_free(args, blocking=True):
    """Refuse to start a DP run on cards someone else is holding."""
    free = card_free_gib()
    if not free:
        return {}
    report = " ".join(f"{c}:{free.get(c, float('nan')):.1f}G" for c in args.cards_list)
    idle = [c for c, gib in free.items() if gib >= args.min_free_gib]
    print(f"[bench] free HBM per card (GiB): {report}; cards with >= {args.min_free_gib:.0f} GiB "
          f"free: {idle or 'none'}", flush=True)
    if args.min_free_gib <= 0:
        return free
    tight = {c: free.get(c) for c in args.cards_list
             if free.get(c) is not None and free[c] < args.min_free_gib}
    if tight and blocking:
        detail = ", ".join(f"card {c}: {gib:.1f} GiB free" for c, gib in sorted(tight.items()))
        sys.exit(f"refusing to start: {detail} (--min-free-gib {args.min_free_gib:.0f}). Someone "
                 f"else's process is on those cards. Cards that are free: "
                 f"{idle or 'none'} -- pick a smaller --dp over an idle group, or wait. To "
                 f"override the check, pass --min-free-gib 0.")
    return free


# ------------------------------------------------------------------------- dumps -- #
def drain(dump_dir, world, timeout):
    """Wait until every rank's dump exists and has stopped growing."""
    deadline = time.time() + timeout
    last, stable = None, 0
    while time.time() < deadline:
        files = sorted(glob.glob(os.path.join(dump_dir, "moe_par_*rank*.log")))
        cur = (len(files), tuple(os.path.getsize(f) for f in files))
        if len(files) >= world and cur == last:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        last = cur
        time.sleep(1.0)
    return sorted(glob.glob(os.path.join(dump_dir, "moe_par_*rank*.log")))


def existing_tags(directory):
    """Tags already sitting in a dump directory, read off each file's ``# meta`` line."""
    tags = set()
    for path in glob.glob(os.path.join(directory, "moe_par_*.log")):
        try:
            with open(path) as fh:
                head = fh.readline()
        except OSError:
            continue
        match = re.search(r"\btag=(\S+)", head)
        tags.add(match.group(1) if match else "?")
    return tags


def check_dump_dir(args, tag, dump_dir):
    """Refuse to mix two configurations in one directory -- the analyzer reads them as one run."""
    if not args.stats or not os.path.isdir(dump_dir):
        return
    prior = existing_tags(dump_dir)
    # Compared in the recorder's form: that is what the dumps carry in their header.
    tag = stats_tag(tag)
    strangers = prior - {tag}
    if strangers and not args.force:
        sys.exit(f"{dump_dir} already holds dumps of tag(s) {sorted(strangers)}. The analyzer "
                 f"reads every dump in a directory as one run, so mixing them would fuse two "
                 f"configurations into a single report. Use another --dump-dir, or --force.")
    if tag in prior:
        print(f"[bench] WARNING: {dump_dir} already holds tag {tag}; its dumps will be "
              f"overwritten", flush=True)


# --------------------------------------------------------------------- latency -- #
def request_latency(output, n_out):
    """This request's own timings, in ms, straight off the engine's clocks.

    Read from ``RequestOutput.metrics``, so these are measured rather than inferred
    from wall time. The names are vLLM's: TTFT is ``first_token_latency``, which
    vLLM computes from the arrival timestamp and therefore *includes* time spent
    queued; TPOT is the decode interval spread over the decode tokens, i.e. vLLM's
    ``mean_time_per_output_token``.

    Two clocks are in play, which is why end-to-end is a sum and not a difference:
    ``arrival_time`` is a frontend wall-clock stamp, while the engine-core
    timestamps (queued / scheduled / first token / last token) are monotonic ones
    taken in the engine's process. Subtracting across the two is meaningless, so
    end-to-end is built as TTFT + decode.

    Returns None when metrics are off. The offline ``LLM`` is the trap here: it
    defaults to ``disable_log_stats=True``, and with stats off the engine allocates
    no ``RequestStateStats`` at all, so every field would otherwise read as a
    plausible-looking 0.0. :func:`main` passes ``disable_log_stats=False`` for
    exactly this reason.
    """
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    decode_s = metrics.last_token_ts - metrics.first_token_ts
    latency = {
        "ttft_ms": 1000.0 * metrics.first_token_latency,
        "decode_ms": 1000.0 * decode_s,
        "e2e_ms": 1000.0 * (metrics.first_token_latency + decode_s),
        # Do not count the token prefill produced, exactly as vLLM's own TPOT does.
        "tpot_ms": 1000.0 * decode_s / (n_out - 1) if n_out > 1 else None,
    }
    if metrics.queued_ts > 0 and metrics.scheduled_ts > 0:
        latency["queue_ms"] = 1000.0 * (metrics.scheduled_ts - metrics.queued_ts)
        latency["prefill_ms"] = 1000.0 * (metrics.first_token_ts - metrics.scheduled_ts)
    return {k: (None if v is None else round(v, 3)) for k, v in latency.items()}


def percentile(values, p):
    """Linear-interpolated percentile of an already-sorted list; ``p`` in [0, 1]."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = p * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarise_perf(requests, generate_s):
    """Per-request latencies condensed to quantiles, plus this run's throughput."""
    perf = {
        "source": "RequestOutput.metrics",
        "measured": sum(1 for r in requests if r.get("latency")),
        "n_requests": len(requests),
        "quantiles": {},
    }
    for name in ("ttft_ms", "tpot_ms", "e2e_ms", "queue_ms", "prefill_ms", "decode_ms"):
        values = sorted(
            r["latency"][name]
            for r in requests
            if r.get("latency") and r["latency"].get(name) is not None
        )
        if not values:
            continue
        perf["quantiles"][name] = {
            "n": len(values),
            "mean": round(sum(values) / len(values), 3),
            "p50": round(percentile(values, 0.50), 3),
            "p90": round(percentile(values, 0.90), 3),
            "p99": round(percentile(values, 0.99), 3),
            "min": round(values[0], 3),
            "max": round(values[-1], 3),
        }
    prompt_tokens = sum(r["prompt_tokens"] for r in requests)
    output_tokens = sum(r["output_tokens"] for r in requests)
    if generate_s > 0:
        perf["throughput"] = {
            "input_tok_s": round(prompt_tokens / generate_s, 2),
            "output_tok_s": round(output_tokens / generate_s, 2),
            "total_tok_s": round((prompt_tokens + output_tokens) / generate_s, 2),
            "req_s": round(len(requests) / generate_s, 3),
        }
    return perf


def describe_dump(path):
    """"Read a dump's header and count its rows, so a silent failure is visible."""
    info = {"path": path, "bytes": os.path.getsize(path), "rows": 0, "steps": 0, "meta": {}}
    steps = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("# meta "):
                info["meta"].update(re.findall(r"(\w+)=(\S+)", line))
            elif line.startswith("step="):
                info["rows"] += 1
                steps.add(line.split(" ", 1)[0])
    info["steps"] = len(steps)
    return info


def parse_child_log(path):
    """The memory ledger and the failure flag for one rank, read off its captured stdout.

    Every number here is logged by the process that owns it (the worker for the memory
    ledger, the engine core for the KV pool, the MoonEP exit hook for the slot counts),
    which makes the rank log the only place where a per-rank memory question -- e.g. why
    two runs of the same configuration showed 2.9 GiB and 37.9 GiB of non-torch memory --
    can be answered.
    """
    if not os.path.exists(path):
        return {"log": path, "missing": True}
    with open(path, errors="replace") as fh:
        text = fh.read()
    out = {"log": path, "bytes": len(text)}
    for key, pattern in LOG_PATTERNS.items():
        found = re.findall(pattern, text)
        if not found:
            continue
        value = found[-1]
        if key == "kv_tokens":
            value = int(value.replace(",", ""))
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        out[key] = value
    slot = SLOT_STATS_RE.findall(text)
    if slot:
        calls, with_slots, rows, prefetch = slot[-1]
        out["slot_stats"] = {
            "dispatch_calls": int(calls),
            "calls_with_slot_rows": int(with_slots),
            "total_slot_rows": int(rows),
            "total_prefetched_expert_slots": int(prefetch),
        }
    errors = [line for line in text.splitlines() if " ERROR " in line or line.startswith("ERROR")]
    if errors:
        out["error_lines"] = len(errors)
        out["last_error"] = errors[-1][:400]
    if "Traceback (most recent call last)" in text:
        out["traceback"] = True
        # The exception line is the one fact worth extracting from a traceback whose frames
        # are three files deep in the recorder; "\b" keeps it from matching the logger's
        # own "(... pid=...)" prefix.
        raised = re.findall(r"^\S*(?:.*?\s)?([A-Z]\w*(?:Error|Exception)): (.{0,200})",
                            text, flags=re.MULTILINE)
        if raised:
            kind, detail = raised[-1]
            out["last_exception"] = f"{kind}: {detail}"
    return out


def kv_verdict(args, memory, requests):
    """Whether every rank's KV pool could serve the batch it was asked to serve.

    A pool below ``max_num_seqs * (prompt + output)`` cannot hold one full batch, and one
    below the longest request cannot hold that request; either way the engine makes progress
    by preempting, and the MoE communication numbers then describe a run that spent its time
    re-prefilling. Reported, and by default a failure.

    ``--max-model-len`` is reported but not part of the verdict: vLLM already refuses to
    start when the pool cannot cover it, and for a 1k-token gsm8k workload a 40k-token
    ``--max-model-len`` says nothing about whether this run preempted.
    """
    pools = [m["kv_tokens"] for m in memory.values() if isinstance(m.get("kv_tokens"), int)]
    if not pools:
        return None
    prompts = [r["prompt_tokens"] for r in requests] or [0]
    avg_prompt = sum(prompts) / len(prompts)
    longest = max(prompts) + args.max_tokens
    batch = args.max_num_seqs * (avg_prompt + args.max_tokens)
    required = max(longest, batch)
    verdict = {
        "worst_kv_tokens": min(pools),
        "required_tokens": int(required),
        "needed_for_model_len": args.max_model_len,
        "needed_for_full_batch": int(batch),
        "needed_for_longest_request": int(longest),
        "ok": min(pools) >= required,
    }
    if not verdict["ok"]:
        print(f"[bench] WARNING: smallest KV pool holds {verdict['worst_kv_tokens']} tokens but "
              f"this workload needs {verdict['required_tokens']} (full batch "
              f"{verdict['needed_for_full_batch']}, longest request "
              f"{verdict['needed_for_longest_request']}). The engine will preempt "
              f"instead of pipelining, so the load and latency numbers describe preemption. Raise "
              f"--gpu-mem-util, cut --max-num-batched-tokens or --max-tokens, or accept it with "
              f"--allow-tiny-kv.", flush=True)
    return verdict


# ------------------------------------------------------------------- the passes -- #
def _load_pass(args, llm, sampling_params, rank, dp_size, questions, total_rows, use_slice):
    """The measured pass: the dataset at its own length, dumped per rank."""
    tok = llm.get_tokenizer()
    prompts = build_prompts(tok, questions)
    print(f"[bench] rank {rank}: {len(prompts)} prompts, token length "
          f"{min(len(p['prompt_token_ids']) for p in prompts)}"
          f"-{max(len(p['prompt_token_ids']) for p in prompts)}", flush=True)

    params = sampling_params(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        seed=args.seed,
    )

    t0 = time.time()
    # use_tqdm=False: the bar writes to stderr and would interleave with the worker logs
    # the parent is capturing into one file per rank.
    outs = llm.generate(prompts, params, use_tqdm=False)
    generate_s = time.time() - t0

    requests = []
    for i, o in enumerate(outs):
        n_out = len(o.outputs[0].token_ids)
        requests.append({
            # With --dp-slice every rank owns disjoint questions, so the global index is
            # what makes the merged list, and the analyzer's reconciliation, line up.
            "idx": (rank + i * dp_size) if use_slice else i,
            "dp_rank": rank,
            "prompt_tokens": len(o.prompt_token_ids),
            "output_tokens": n_out,
            "finish_reason": o.outputs[0].finish_reason,
            "latency": request_latency(o, n_out),
        })
    prompt_tokens = sum(r["prompt_tokens"] for r in requests)
    output_tokens = sum(r["output_tokens"] for r in requests)
    finished = sum(1 for r in requests if r["finish_reason"] == "stop")

    return {
        "tag": args.tag,
        "rank": rank,
        "model": args.model,
        "dataset": os.path.abspath(args.dataset),
        "devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "dump_dir": args.dump_dir,
        "timing": {
            "generate_s": round(generate_s, 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "output_tok_s": round(output_tokens / generate_s, 2) if generate_s else None,
            "stopped_early": finished,
        },
        "requests": requests,
        "perf": summarise_perf(requests, generate_s),
        "num_prompts_total": int(total_rows),
    }


def _score_pass(args, llm, sampling_params, rank, dp_size, use_slice, gsm8k):
    """The GSM8K accuracy pass: reference prompts, reference extraction, one score per run."""
    prompts, labels, meta = gsm8k.build_prompts(
        num_questions=args.num_prompts, num_shots=args.num_shots,
        train_file=args.train_file, test_file=args.test_file or args.dataset)
    if args.num_prompts and len(prompts) < args.num_prompts:
        print(f"[bench] WARNING: asked for {args.num_prompts} questions, the test split has "
              f"{len(prompts)}", flush=True)

    # Slice the raw prompt strings, not the tokens: the scoring pass is a completion
    # prompt, so it must not go through the chat template the load pass uses.
    mine = list(range(len(prompts)))
    if use_slice:
        mine = mine[rank::dp_size]
    if not mine:
        sys.exit(f"rank {rank} got an empty slice of {len(prompts)} questions with "
                 f"--dp {dp_size}; nothing to score")

    params = sampling_params(
        temperature=args.temperature,
        max_tokens=args.score_max_tokens,
        ignore_eos=False,  # the reference stops at the answer, and so must we
        stop=list(gsm8k.STOP),
        seed=args.seed,
    )
    print(f"[bench] rank {rank}: scoring {len(mine)} of {len(prompts)} questions, "
          f"{meta['num_shots']}-shot, max_tokens={args.score_max_tokens}", flush=True)

    t0 = time.time()
    outs = llm.generate([prompts[i] for i in mine], params, use_tqdm=False)
    latency = time.time() - t0

    items = []
    for i, o in zip(mine, outs):
        text = o.outputs[0].text
        pred = gsm8k.get_answer_value(text)
        items.append({
            "idx": i,
            "pred": pred,
            "gold": labels[i],
            "ok": pred == labels[i],
            "n_tokens": len(o.outputs[0].token_ids),
            "finish_reason": o.outputs[0].finish_reason,
            "text": text[-300:],
        })
    n_tokens = sum(item["n_tokens"] for item in items)
    print(f"[bench] rank {rank}: {sum(1 for i in items if i['ok'])}/{len(items)} correct in "
          f"{latency:.1f}s", flush=True)
    return {
        "tag": args.tag,
        "rank": rank,
        "items": items,
        "latency_s": round(latency, 3),
        "output_tokens": n_tokens,
        "meta": meta,
    }


def _profile_pass(args, llm, sampling_params, rank, dp_size, questions):
    """The profiling window: a short, fixed, lockstep batch on every rank.

    ``stop_profile()`` exports the trace synchronously inside each worker, and a rank that
    is exporting executes no steps. If the batch were ragged the exporting rank would be
    waiting on collectives its peers have already left, so the window is pinned: same
    prompt count, same output length, greedy.

    ``questions`` is already this rank's slice of ``--num-prompts`` (the caller cuts it), so
    there is nothing left to slice here. Slicing a second time is how a profile leg dies:
    rank r of dp would take rows ``r::dp`` of rows ``r::dp``, which is empty for every rank
    but the first, and the rank that exits on an empty window takes the whole run with it --
    it closes its gloo sockets while its peers are still inside the startup collective, and
    they die with "Connection closed by peer". ``main`` therefore rejects a
    ``--num-prompts`` that does not divide evenly across the ranks.
    """
    tok = llm.get_tokenizer()
    mine = questions
    if not mine:
        sys.exit(f"rank {rank}: no prompts for the profiling window "
                 f"(--num-prompts must be a positive multiple of the DP size)")
    prompts = build_prompts(tok, mine)
    params = sampling_params(temperature=0.0, max_tokens=args.profile_max_tokens,
                             ignore_eos=True, seed=args.seed)
    trace_dir = os.path.abspath(args.profile_dir)
    print(f"[bench] rank {rank}: profiling {len(prompts)} prompts x "
          f"{args.profile_max_tokens} tokens -> {trace_dir}", flush=True)
    t0 = time.time()
    llm.start_profile(f"{args.tag}_r{rank}")
    outs = llm.generate(prompts, params, use_tqdm=False)
    llm.stop_profile()
    generate_s = time.time() - t0
    print(f"[bench] rank {rank}: profile window done in {generate_s:.1f}s "
          f"({len(outs)} requests)", flush=True)
    return {
        "tag": args.tag,
        "rank": rank,
        "profile_dir": trace_dir,
        "num_prompts": len(prompts),
        "max_tokens": args.profile_max_tokens,
        "generate_s": round(generate_s, 3),
    }


# ----------------------------------------------------------------- rendezvous -- #
def _marker_path(dump_dir, name):
    return os.path.join(dump_dir, "state", name)


def state_dir(dump_dir):
    return os.path.join(dump_dir, "state")


def clear_state(dump_dir):
    """Drop the previous run's markers; a stale ``go`` would release the next run early."""
    directory = state_dir(dump_dir)
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.startswith(("done_rank", "fail_rank")) or name == "go":
                try:
                    os.remove(os.path.join(directory, name))
                except OSError:
                    pass
    else:
        os.makedirs(directory, exist_ok=True)


def wait_for_marker(dump_dir, name, timeout):
    deadline = time.time() + timeout
    path = _marker_path(dump_dir, name)
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.5)
    return False


def child_rendezvous(args, dump_dir, rank, dp_size):
    """Hold this rank's engine alive until the parent says every rank is done.

    A DP engine core talks to its peers every 32 steps (``sync_dp_state``), and a peer that
    exits early turns the next of those into a failed all-reduce, which is how a DP run
    hangs instead of failing. The parent releases the group only once the last rank
    reports in, so the teardown at the end of this function cannot be seen by anyone.
    """
    os.makedirs(state_dir(dump_dir), exist_ok=True)
    open(_marker_path(dump_dir, f"done_rank{rank}"), "w").close()
    if not wait_for_marker(dump_dir, "go", args.rendezvous_timeout):
        print(f"[bench] rank {rank}: no release after {args.rendezvous_timeout:.0f}s "
              f"(a peer is probably stuck); tearing down anyway", flush=True)


# Flags that carry a filesystem path, mapped to the attribute holding its resolved value.
PATH_FLAGS = {
    "--dataset": "dataset", "--model": "model", "--stats-dir": "stats_dir",
    "--dump-dir": "dump_dir", "--out": "out", "--profile-dir": "profile_dir",
    "--train-file": "train_file", "--test-file": "test_file",
}


def child_argv(args, argv):
    """``argv`` with every path flag replaced by the path this process resolved it to.

    The children run with cwd=/tmp (and a handheld dry-run command runs from wherever the
    user happens to be), so a relative ``--dataset`` would be resolved somewhere else
    entirely -- in /tmp it resolves to a file that does not exist and every rank exits.
    Handing down what the parent resolved is also what makes the printed dry-run command
    copy-and-run safe from any directory.
    """
    out = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in PATH_FLAGS:
            out.append(token)
            out.append(getattr(args, PATH_FLAGS[token]))
            index += 2
            continue
        flag, sep, value = token.partition("=")
        if sep and flag in PATH_FLAGS and not os.path.isabs(value):
            out.append(f"{flag}={getattr(args, PATH_FLAGS[flag])}")
        elif token != "--dry-run":  # the parent's flag; forwarding it would run 8 dry runs
            out.append(token)
        index += 1
    return out


def launch_dp(args, argv, tag, dump_dir):
    """Parent side of a DP run: preflight, one child per rank, then merge."""
    script = os.path.abspath(__file__)
    dry = args.dry_run
    if args.dp_slice:
        # The parent refuses this, rather than the children: with fewer rows than ranks only
        # some of them get an empty slice, and the ones that do exit while their peers sit in
        # a collective waiting for a rank that is already gone.
        rows, _ = load_questions(args.dataset, args.prompt_field, 0)
        if len(rows) < args.dp:
            sys.exit(f"--dp {args.dp} with --dp-slice needs at least {args.dp} rows in "
                     f"{args.dataset} (it has {len(rows)}); use a smaller --dp or "
                     f"--no-dp-slice")
    if not dry:
        check_cards_free(args, blocking=True)
        os.makedirs(os.path.join(dump_dir, "log"), exist_ok=True)
        clear_state(dump_dir)
    else:
        check_cards_free(args, blocking=False)

    port = free_port()
    injected = dict(LAUNCH_ENV_DEFAULTS)
    child_args = child_argv(args, argv)
    children = []
    for rank in range(args.dp):
        env = os.environ.copy()
        env.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(args.dp),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "ASCEND_RT_VISIBLE_DEVICES": args.cards,
        })
        # cwd must not contain a `vllm/` source tree: the trailing empty segment of
        # PYTHONPATH resolves against cwd and would make `vllm` a namespace package.
        children.append({"rank": rank, "env": env, "cwd": tempfile.gettempdir()})

    if dry:
        print(f"\n[bench] DRY RUN -- DP={args.dp} cards={args.cards} tag={tag}\n"
              f"[bench] dumps would go to {dump_dir}", flush=True)
        print(f"[bench] inherited: ASCEND_MOONEP_BUILD_DIR="
              f"{os.environ.get('ASCEND_MOONEP_BUILD_DIR', '<unset>')}", flush=True)
        print(f"[bench] PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
        for key in sorted(injected):
            print(f"[bench] injected: {key}={injected[key]}", flush=True)
        cmd = " ".join([sys.executable, script] + child_args)
        for child in children:
            print(f"\n[bench] --- rank {child['rank']} (cd {child['cwd']})", flush=True)
            print(f"[bench]   ASCEND_RT_VISIBLE_DEVICES={args.cards}", flush=True)
            print(f"[bench]   VLLM_DP_RANK={child['rank']} VLLM_DP_RANK_LOCAL={child['rank']} "
                  f"VLLM_DP_SIZE={args.dp} VLLM_DP_MASTER_IP=127.0.0.1 "
                  f"VLLM_DP_MASTER_PORT={port}", flush=True)
            print(f"[bench]   {cmd}", flush=True)
        print(f"\n[bench] 手动跑：每条 leg 的 {args.dp} 条命令必须同时起（DP rank 之间会互相等）；"
              f"三条 leg 只差 --moonep-mode/--tag/--stats-dir\n", flush=True)
        return 0

    print(f"[bench] launching DP={args.dp} cards={args.cards} port={port} tag={tag}", flush=True)
    procs = []
    log_files = []
    for child in children:
        log = open(os.path.join(dump_dir, "log", f"rank{child['rank']}.log"), "w", buffering=1)
        log_files.append(log)
        # Never subprocess.PIPE: three processes' worth of logs with no reader fills the
        # 64 KB pipe buffer and stops the child mid-run.
        proc = subprocess.Popen([sys.executable, script] + child_args, cwd=child["cwd"],
                                env=child["env"], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        procs.append(proc)
        print(f"[bench]   rank {child['rank']} pid {proc.pid} -> "
              f"{os.path.relpath(log.name, dump_dir)}", flush=True)

    rc = _supervise(args, procs, dump_dir)
    for log in log_files:
        log.close()

    logs = {r: os.path.join(dump_dir, "log", f"rank{r}.log") for r in range(args.dp)}
    if args.score:
        return merge_score(args, tag, dump_dir, rc)
    if args.profile_only:
        return merge_profile(args, tag, dump_dir, rc, logs)
    return merge_dp(args, tag, dump_dir, rc, logs)


def _log_signature(path):
    """``(size, has_traceback)`` for one rank log, read from its tail only.

    A load pass writes ~500 KB per rank and this runs every poll, so reading the whole file
    would cost more than the poll itself -- and a fatal error is always the last thing a
    process writes, since nothing follows it.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return 0, False
    return size, "Traceback (most recent call last)" in tail


def _supervise(args, procs, dump_dir):
    """Release the ranks once the last one reports in, and kill the group on overrun.

    The release is what keeps a DP run alive: every child holds its engine open until it
    sees ``go``, so no rank can be torn down while a peer still generates. The kill is the
    last resort for a run that stopped reporting -- a hung DP worker holds ~90 GB of HBM
    and will not exit on its own. Only these children's process groups are ever signalled.

    There is a third exit, between "reported a failure" and "hit the timeout": a rank whose
    log already holds a traceback but which is still alive. That is what a dead engine core
    looks like from here -- ``EngineCore_DP{n}`` raising inside ``_initialize_kv_caches``
    (e.g. "No available memory for the cache blocks") takes its dp-rank parent down with
    it only if the parent notices, and the vLLM mp client blocks in the startup handshake
    instead. No ``fail_rank`` is ever written, so without this the parent would sit out its
    whole ``--timeout`` while the cards stay held. The traceback plus a log that has
    stopped growing is the signal; a rank that is merely slow keeps writing.
    """
    deadline = time.time() + args.timeout
    released = None
    # rank -> (first seen at, log size then), for a rank whose log has a traceback.
    frozen = {}
    wedged = []
    while released is None:
        done = [r for r in range(args.dp)
                if os.path.exists(_marker_path(dump_dir, f"done_rank{r}"))]
        failed = [r for r in range(args.dp)
                  if os.path.exists(_marker_path(dump_dir, f"fail_rank{r}"))]
        if failed:
            print(f"[bench] rank(s) {failed} reported a failure; releasing the group so the "
                  f"others can exit", flush=True)
            released = "failed"
        elif len(done) == args.dp:
            released = "done"
        elif time.time() > deadline:
            print(f"[bench] only {len(done)}/{args.dp} ranks reported in within "
                  f"{args.timeout:.0f}s; killing the process group", flush=True)
            released = "timeout"
        elif all(proc.poll() is not None for proc in procs):
            released = "exited"
        else:
            now = time.time()
            for rank, proc in enumerate(procs):
                if proc.poll() is not None or rank in wedged:
                    continue
                size, fatal = _log_signature(
                    os.path.join(dump_dir, "log", f"rank{rank}.log"))
                if not fatal:
                    continue
                entry = frozen.get(rank)
                if entry is None or entry[1] != size:
                    frozen[rank] = (now, size)  # traceback is new or the log grew: keep waiting
                elif now - entry[0] > WEDGE_GRACE:
                    wedged.append(rank)
            if wedged:
                print(f"[bench] rank(s) {wedged} logged a traceback and their log has not "
                      f"grown for {WEDGE_GRACE:.0f}s (a dp-rank blocks forever when its "
                      f"engine core dies during startup); releasing the group", flush=True)
                released = "wedged"
            else:
                time.sleep(0.5)

    if released == "timeout":
        for proc in procs:
            _killpg(proc)
    else:
        # Every rank that is still alive is waiting on this marker; without it they would
        # sit out their whole rendezvous timeout before failing.
        open(_marker_path(dump_dir, "go"), "w").close()
        # A wedged rank is stuck by evidence, not by guesswork: it has already printed its
        # traceback and stopped writing. Waiting out the teardown budget for it would only
        # hold its HBM longer, so it goes now while its peers get the normal window.
        for rank in wedged:
            print(f"[bench] killing wedged rank {rank} (pid {procs[rank].pid}) now", flush=True)
            _killpg(procs[rank])

    # Teardown gets its own budget: engine shutdown and, under --profile-only, the
    # synchronous trace export happen here, and a kill during the export loses the trace.
    # The failed case gets a much shorter one -- a rank that died during model load leaves
    # its peers blocked in a collective they will never leave, and there is no trace worth
    # waiting half an hour for.
    budget = 120.0 if released in ("failed", "wedged") else max(300.0, min(args.timeout, 1800.0))
    if released in ("failed", "wedged"):
        print(f"[bench] giving the other ranks {budget:.0f}s to unwind before killing them",
              flush=True)
    teardown = time.time() + budget
    rc = 0
    for rank, proc in enumerate(procs):
        try:
            code = proc.wait(timeout=max(1.0, teardown - time.time()))
        except subprocess.TimeoutExpired:
            print(f"[bench] rank {rank} (pid {proc.pid}) did not exit; killing", flush=True)
            _killpg(proc)
            proc.wait()
            code = 1
        if code:
            print(f"[bench] rank {rank} exited {code}", flush=True)
            rc |= code
    if released == "done" and rc == 0:
        print("[bench] all ranks exited cleanly", flush=True)
    return rc


def _killpg(proc):
    """Signal this child's own process group -- never another tenant's processes."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# --------------------------------------------------------------------- merging -- #
def merge_dp(args, tag, dump_dir, rc, logs):
    """One summary out of DP per-rank summaries, plus the per-rank memory ledger.

    Three things are not symmetric in DP and would be wrong if treated as such:

    * ``generate_s`` is the *slowest* rank's window, not a sum: the ranks run in parallel,
      so the run's wall time is the longest of them, and summing would divide the
      throughput by the rank count.
    * token counts *are* summed -- each rank really generated its own share, and the dumps
      next to this file record exactly that many.
    * ``requests`` keeps the ``(dp_rank, idx)`` pairs, because with ``--no-dp-slice`` every
      rank runs the same questions and a bare index would silently collide.
    """
    rank_summaries = []
    missing = []
    for rank in range(args.dp):
        path = os.path.join(dump_dir, f"bench_rank{rank}.json")
        if not os.path.exists(path):
            missing.append(rank)
            continue
        with open(path) as fh:
            rank_summaries.append(json.load(fh))
    if missing:
        print(f"[bench] WARNING: no summary from rank(s) {missing}; their share of the load is "
              f"missing from bench.json (check {os.path.join(dump_dir, 'log')})", flush=True)

    requests = [r for s in rank_summaries for r in s["requests"]]
    requests.sort(key=lambda r: (r["idx"], r["dp_rank"]))
    generate_s = max((s["timing"]["generate_s"] for s in rank_summaries), default=0.0)
    prompt_tokens = sum(r["prompt_tokens"] for r in requests)
    output_tokens = sum(r["output_tokens"] for r in requests)
    finished = sum(1 for r in requests if r["finish_reason"] == "stop")
    memory = {str(rank): parse_child_log(logs[rank]) for rank in range(args.dp)}
    dumps = [describe_dump(p) for p in drain(dump_dir, args.dp, args.drain_timeout)] \
        if args.stats else []

    summary = {
        "tag": tag,
        "model": args.model,
        "dataset": os.path.abspath(args.dataset),
        "config": {
            "dp": args.dp,
            "tp": args.tp,
            "ep": args.ep,
            "moonep": args.moonep,
            "moonep_mode": args.mode,
            "cards": args.cards,
            "batch_size": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "additional_config": {"enable_moonep": args.moonep},
            "dp_slice": args.dp_slice,
            # The analyzer's reconciliation divides by this when there are no per-request
            # records, so it has to be the global count -- not one rank's slice.
            "num_prompts": len(requests),
            "rows_in_dataset": max((int(s.get("num_prompts_total") or 0) for s in rank_summaries),
                                   default=0),
            "seed": args.seed,
        },
        "devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", args.cards),
        "dump_dir": dump_dir,
        "dumps": dumps,
        "timing": {
            # Labelled rather than raw: the run's wall clock is the slowest rank's window,
            # and the parent's own clock would be dominated by model loading.
            "generate_s": round(generate_s, 3),
            "generate_s_basis": "slowest rank",
            "per_rank_generate_s": {str(s["rank"]): s["timing"]["generate_s"]
                                    for s in rank_summaries},
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "output_tok_s": round(output_tokens / generate_s, 2) if generate_s else None,
            "stopped_early": finished,
        },
        "requests": requests,
        "perf": summarise_perf(requests, generate_s),
        "per_rank": {str(s["rank"]): {"timing": s["timing"], "perf": s["perf"]}
                     for s in rank_summaries},
        "memory": memory,
    }
    summary["kv_check"] = kv_verdict(args, memory, requests)
    if args.stats:
        summary["analyze_cmd"] = (
            f"python {os.path.join(SCRIPT_DIR, 'analyze_moe_par.py')} "
            f"--run {tag}={dump_dir} -o {tag}_report.html"
        )

    out = args.out or os.path.join(dump_dir, "bench.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    report_merged(tag, out, summary, rc, dump_dir)
    # Fewer dumps than ranks is a failed run, not a note: the load panels and the
    # reconciliation check are computed from the dumps, so a run that wrote none has
    # nothing to report on -- and exit 0 here would let the sweep treat it as good.
    short_dumps = args.stats and len(dumps) < args.dp
    return 1 if (rc or missing or short_dumps
                 or (summary["kv_check"] and not summary["kv_check"]["ok"]
                     and not args.allow_tiny_kv)) else 0


def report_merged(tag, out, summary, rc, dump_dir):
    """Print the DP run's headline, in the same shape as the single-process path."""
    perf = summary["perf"]
    q = perf["quantiles"]
    tp = perf.get("throughput") or {}
    print(f"[bench] tag={tag} dp={summary['config']['dp']} mode={summary['config']['moonep_mode']} "
          f"cards={summary['config']['cards']}", flush=True)
    print(f"[bench] {summary['timing']['output_tokens']} output tokens over the slowest rank's "
          f"{summary['timing']['generate_s']}s, {summary['timing']['stopped_early']}/"
          f"{len(summary['requests'])} hit EOS", flush=True)
    if tp:
        print(f"[bench] throughput in {tp['input_tok_s']} / out {tp['output_tok_s']} tok/s, "
              f"{tp['req_s']} req/s (aggregate over all ranks)", flush=True)
    if perf["measured"]:
        ttft, tpot = q["ttft_ms"], q.get("tpot_ms")
        print(f"[bench] TTFT ms: p50 {ttft['p50']} p90 {ttft['p90']} p99 {ttft['p99']}"
              + (f" | TPOT ms: p50 {tpot['p50']} p90 {tpot['p90']}" if tpot
                 else " | TPOT: n/a (no request produced a second token)")
              + f" (over {perf['measured']}/{perf['n_requests']} requests)", flush=True)
    for rank in sorted(summary["memory"], key=int):
        m = summary["memory"][rank]
        if m.get("missing"):
            print(f"[bench]   rank {rank}: no log", flush=True)
            continue
        print(f"[bench]   rank {rank}: weights {m.get('weights_gib', '?')} GiB, "
              f"non-torch {m.get('non_torch_gib', '?')} GiB, "
              f"KV {m.get('available_kv_gib', '?')} GiB / {m.get('kv_tokens', '?')} tokens"
              + (f", slot_rows {m['slot_stats']['total_slot_rows']}"
                 if m.get("slot_stats") else ""), flush=True)
        if m.get("traceback"):
            # Kept loud because a recorder that raises shows up as *missing dumps*, which
            # is easy to mistake for "the analyzer found nothing to plot".
            print(f"[bench]   rank {rank}: its log contains a traceback -- "
                  f"{m.get('last_exception') or m.get('last_error') or 'see the rank log'}",
                  flush=True)
    for d in summary["dumps"]:
        meta = d["meta"]
        print(f"[bench]   {os.path.basename(d['path'])}: {d['rows']} rows x {d['steps']} steps, "
              f"layers={meta.get('moe_layers')} experts={meta.get('global_experts')} "
              f"topk={meta.get('topk')} ep_size={meta.get('ep_size')}", flush=True)
    if len(summary["dumps"]) < summary["config"]["dp"]:
        print(f"[bench] WARNING: expected {summary['config']['dp']} dumps, found "
              f"{len(summary['dumps'])} -- check {os.path.join(dump_dir, 'log')} for "
              f"'[moe-par]' lines", flush=True)
    print(f"[bench] saved {out}", flush=True)
    if summary.get("analyze_cmd"):
        print(f"\n下一步，出报告:\n  {summary['analyze_cmd']}\n", flush=True)


def merge_score(args, tag, dump_dir, rc):
    """Union the per-rank answers and score once, over the whole question set.

    Averaging per-rank accuracies would weight a short slice as heavily as a long one, and
    the slices are only equal because 1319 happens to divide evenly.
    """
    import gsm8k_score

    items, meta, latency, tokens = [], {}, 0.0, 0
    for rank in range(args.dp):
        path = os.path.join(dump_dir, f"score_rank{rank}.json")
        if not os.path.exists(path):
            print(f"[bench] WARNING: rank {rank} scored nothing ({path} missing)", flush=True)
            continue
        with open(path) as fh:
            part = json.load(fh)
        items.extend(part["items"])
        meta = meta or part["meta"]
        latency = max(latency, part["latency_s"])
        tokens += part["output_tokens"]
    if not items:
        print("[bench] no scored answers; nothing to write", flush=True)
        return 1
    items.sort(key=lambda i: i["idx"])
    result = gsm8k_score.score([i["pred"] for i in items], [i["gold"] for i in items],
                               latency=latency, output_tokens=tokens, shots_meta=meta)
    payload = {"tag": tag, "config": {"dp": args.dp, "moonep_mode": args.mode,
                                      "cards": args.cards, "num_shots": args.num_shots,
                                      "max_tokens": args.score_max_tokens},
               "result": result, "items": items}
    out = os.path.join(dump_dir, "score.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[bench] accuracy {result['accuracy']:.3f} ({result['n_correct']}/{result['n']}), "
          f"invalid {result['invalid_rate']:.3f}, {result['num_shots']}-shot, saved {out}",
          flush=True)
    return 0 if rc == 0 else 1


def merge_profile(args, tag, dump_dir, rc, logs):
    """Record where the traces landed, and the slot counts of the profiled window."""
    ranks = []
    for rank in range(args.dp):
        path = os.path.join(dump_dir, f"profile_rank{rank}.json")
        if not os.path.exists(path):
            print(f"[bench] WARNING: rank {rank} wrote no profile record", flush=True)
            continue
        with open(path) as fh:
            ranks.append(json.load(fh))
    trace_dir = os.path.abspath(args.profile_dir)
    files = sorted(path for path in glob.glob(os.path.join(trace_dir, "**"), recursive=True)
                   if os.path.isfile(path))
    payload = {
        "tag": tag,
        "profile_dir": trace_dir,
        "config": {"dp": args.dp, "moonep_mode": args.mode, "cards": args.cards,
                   "max_tokens": args.profile_max_tokens, "num_prompts": args.num_prompts},
        "ranks": ranks,
        "memory": {str(rank): parse_child_log(logs[rank]) for rank in range(args.dp)},
        "trace_files": files,
    }
    out = os.path.join(dump_dir, "profile.json")
    write_json(out, payload)
    print(f"[bench] profile: {len(ranks)}/{args.dp} ranks reported, {len(files)} trace file(s) "
          f"under {trace_dir}, saved {out}", flush=True)
    for path in files[:8]:
        print(f"[bench]   {os.path.relpath(path, trace_dir)}", flush=True)
    print(f"[bench] 看 trace: tensorboard --logdir {trace_dir}", flush=True)
    return 0 if rc == 0 else 1


# --------------------------------------------------------------------------- main -- #
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    tag = args.tag or make_tag(args)
    dump_dir = args.dump_dir or os.path.join(args.stats_dir, tag)
    args.dump_dir = dump_dir
    args.tag = tag

    # Validate before the (slow) vllm import so a typo fails fast.
    if not os.path.exists(args.dataset):
        sys.exit(f"dataset not found: {args.dataset}")
    if args.max_num_seqs > args.max_num_batched_tokens:
        print(f"[bench] WARNING: --batch-size {args.max_num_seqs} exceeds "
              f"--max-num-batched-tokens {args.max_num_batched_tokens}; vLLM may reject this",
              flush=True)

    # A DP run is a parent that never imports vllm plus one child per rank; the children
    # are this same file with VLLM_DP_RANK in the environment.
    if args.dp > 1 and not os.environ.get("VLLM_DP_RANK"):
        apply_launch_env(args)
        if not args.dry_run:
            # A dry run must not be blocked by the state of the dump directory or the
            # cards -- it exists so the commands can be lifted out by hand.
            check_dump_dir(args, tag, dump_dir)
            check_moonep_import()
        return launch_dp(args, argv, tag, dump_dir)

    if args.dry_run:
        dry_run_single(args, argv, tag, dump_dir)
        return 0

    return run_worker(args)


def dry_run_single(args, argv, tag, dump_dir):
    """The single-process equivalent of the DP dry run: the env, then the command."""
    apply_launch_env(args)
    apply_moonep_env(args.mode)
    print(f"\n[bench] DRY RUN (single process) -- tag={tag}\n"
          f"[bench] dumps would go to {dump_dir}", flush=True)
    for key in sorted(LAUNCH_ENV_DEFAULTS):
        print(f"[bench]   {key}={os.environ.get(key, '<unset>')}", flush=True)
    print(f"[bench]   ASCEND_MOONEP_BUILD_DIR="
          f"{os.environ.get('ASCEND_MOONEP_BUILD_DIR', '<unset>')}", flush=True)
    print(f"[bench]   MOONEP={os.environ.get('MOONEP', '<unset>')} "
          f"MOONEP_SLOT_MODE={os.environ.get('MOONEP_SLOT_MODE', '<unset>')} "
          f"MOONEP_SLOT_STATS={os.environ.get('MOONEP_SLOT_STATS', '<unset>')}", flush=True)
    print(f"[bench]   PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
    # Printed without --dry-run, so the line can be copied and run for real.
    command = " ".join([sys.executable, os.path.abspath(__file__)]
                       + [a for a in argv if a != "--dry-run"])
    print(f"\n[bench]   {command}\n", flush=True)


def run_worker(args):
    """This rank's own run: build the engine, do the pass, hold for the peers, tear down."""
    rank = int(os.environ.get("VLLM_DP_RANK", "0"))
    dp_size = int(os.environ.get("VLLM_DP_SIZE", "1"))
    in_dp = dp_size > 1
    tag, dump_dir = args.tag, args.dump_dir

    if not in_dp:
        # In a DP run the parent has already checked, once; eight children printing the
        # same warning would only bury the rest of the log.
        check_dump_dir(args, tag, dump_dir)
    apply_stats_env(args, tag, dump_dir)
    apply_launch_env(args)

    from vllm import LLM, SamplingParams  # noqa: E402  -- must follow apply_stats_env

    questions, total_rows = load_questions(args.dataset, args.prompt_field, args.num_prompts)
    if in_dp and args.dp_slice:
        questions = questions[rank::dp_size]
        if not questions:
            # A placeholder here would be counted by the recorder and not by the requests,
            # which is exactly the mismatch the analyzer's reconciliation reports.
            raise SystemExit(f"rank {rank} got an empty slice of {total_rows} rows with "
                             f"--dp {dp_size}; use a smaller --dp or --no-dp-slice")

    print(f"[bench] tag={tag} rank={rank}/{dp_size} mode={args.mode}", flush=True)
    print(f"[bench] dataset={args.dataset} ({len(questions)} of {total_rows} rows)", flush=True)
    print(f"[bench] devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')} "
          f"dp={dp_size} tp={args.tp} ep={args.ep} moonep={args.mode} batch={args.max_num_seqs} "
          f"batched_tokens={args.max_num_batched_tokens} out_len={args.max_tokens}", flush=True)
    print(f"[bench] dumps -> {dump_dir}", flush=True)

    additional = {"enable_moonep": args.moonep}
    if args.additional_config:
        additional.update(json.loads(args.additional_config))

    llm_kwargs = dict(
        model=args.model,
        enforce_eager=args.enforce_eager,
        enable_expert_parallel=args.ep,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        additional_config=additional,
        # The offline LLM turns stats off unless asked (llm.py sets
        # disable_log_stats=True by default). With them off the engine never
        # allocates RequestStateStats, so RequestOutput.metrics comes back None and
        # there is no TTFT/TPOT to read at all. Turning them on costs per-iteration
        # bookkeeping in the engine and one periodic log line -- no device work, no
        # collectives, so it cannot perturb the load measurement.
        disable_log_stats=False,
    )
    if args.profile_dir:
        # Must be present at construction: the worker raises on profile() without it.
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": os.path.abspath(args.profile_dir),
            # Default True, and the Ascend wrapper maps it to the expensive with_modules
            # path; the MoE timeline does not need per-op python frames.
            "torch_profiler_with_stack": False,
        }

    rc = 0
    llm = None
    try:
        # Inside the try on purpose: a rank that dies while building its engine would
        # otherwise leave its peers blocked in a collective with no marker to notice, and
        # the parent only finds out when the whole run hits --timeout (an hour by default).
        llm = LLM(**llm_kwargs)
        world = llm.get_world_size()
        if args.score:
            import gsm8k_score
            partial = _score_pass(args, llm, SamplingParams, rank, dp_size, args.dp_slice,
                                  gsm8k_score)
            out = (os.path.join(dump_dir, f"score_rank{rank}.json") if in_dp
                   else os.path.join(dump_dir, "score.json"))
            write_json(out, partial)
            if not in_dp:
                n_ok = sum(1 for item in partial["items"] if item["ok"])
                print(f"[bench] accuracy {n_ok}/{len(partial['items'])} = "
                      f"{n_ok / max(1, len(partial['items'])):.3f}, saved {out}", flush=True)
        elif args.profile_only:
            partial = _profile_pass(args, llm, SamplingParams, rank, dp_size, questions)
            write_json(os.path.join(dump_dir, f"profile_rank{rank}.json")
                       if in_dp else os.path.join(dump_dir, "profile.json"), partial)
        else:
            partial = _load_pass(args, llm, SamplingParams, rank, dp_size, questions,
                                 total_rows, args.dp_slice)
            if in_dp:
                write_json(os.path.join(dump_dir, f"bench_rank{rank}.json"), partial)
            else:
                # Let the workers tear down and flush before we look at their files.
                del llm
                llm = None
                partial["dumps"] = ([describe_dump(p)
                                     for p in drain(dump_dir, world, args.drain_timeout)]
                                    if args.stats else [])
                partial["config"] = config_summary(args, additional, len(questions), total_rows)
                if args.stats:
                    partial["analyze_cmd"] = (
                        f"python {os.path.join(SCRIPT_DIR, 'analyze_moe_par.py')} "
                        f"--run {tag}={dump_dir} -o {tag}_report.html")
                out = args.out or os.path.join(dump_dir, "bench.json")
                write_json(out, partial)
                report_single(args, tag, out, partial, world)
    except BaseException:
        traceback.print_exc()
        rc = 1
        if in_dp:
            open(_marker_path(dump_dir, f"fail_rank{rank}"), "w").close()
    finally:
        if in_dp:
            # Always release the group, even on the failure path: a peer left waiting on a
            # rank that died would sit out its whole rendezvous timeout before failing too.
            try:
                child_rendezvous(args, dump_dir, rank, dp_size)
            except Exception:
                traceback.print_exc()
        if llm is not None:
            try:
                del llm
            except Exception:
                traceback.print_exc()
    return rc


def config_summary(args, additional, n_prompts, total_rows):
    return {
        "dp": args.dp,
        "tp": args.tp,
        "ep": args.ep,
        "moonep": args.moonep,
        "moonep_mode": args.mode,
        "batch_size": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "additional_config": additional,
        "num_prompts": n_prompts,
        "rows_in_dataset": total_rows,
        "seed": args.seed,
    }


def report_single(args, tag, out, summary, world):
    """The single-process summary's console report; unchanged from before DP existed."""
    perf = summary["perf"]
    q = perf["quantiles"]
    tp = perf.get("throughput") or {}
    timing = summary["timing"]
    print(f"[bench] {timing['output_tokens']} output tokens in {timing['generate_s']:.1f}s "
          f"({timing['output_tok_s']} tok/s), {timing['stopped_early']}/{len(summary['requests'])} "
          f"hit EOS (ignore_eos={args.ignore_eos})", flush=True)
    if tp:
        print(f"[bench] throughput in {tp['input_tok_s']} / out {tp['output_tok_s']} tok/s, "
              f"{tp['req_s']} req/s (over the whole generate() call)", flush=True)
    if perf["measured"]:
        ttft, tpot = q["ttft_ms"], q.get("tpot_ms")
        print(f"[bench] TTFT ms: p50 {ttft['p50']} p90 {ttft['p90']} p99 {ttft['p99']}"
              + (f" | TPOT ms: p50 {tpot['p50']} p90 {tpot['p90']}" if tpot
                 else " | TPOT: n/a (no request produced a second token)")
              + f" (over {perf['measured']}/{perf['n_requests']} requests)", flush=True)
        print("[bench] TTFT includes queue time: offline puts every prompt in at once, so the "
              "later ones wait for the earlier ones", flush=True)
    else:
        print(f"[bench] no per-request latency: vLLM returned no metrics "
              f"({perf['n_requests']} outputs, 0 measured) -- only token throughput is available. "
              "The offline LLM needs disable_log_stats=False for metrics to be collected.",
              flush=True)
    if args.stats:
        dumps = summary["dumps"]
        if len(dumps) < world:
            print(f"[bench] WARNING: expected {world} dumps, found {len(dumps)} -- check the "
                  f"worker logs for '[moe-par]' lines", flush=True)
        for d in dumps:
            meta = d["meta"]
            print(f"[bench]   {os.path.basename(d['path'])}: {d['rows']} rows x {d['steps']} steps, "
                  f"layers={meta.get('moe_layers')} experts={meta.get('global_experts')} "
                  f"topk={meta.get('topk')} ep_size={meta.get('ep_size')}", flush=True)
        print(f"[bench] saved {out}", flush=True)
        print(f"\n下一步，出报告:\n  {summary['analyze_cmd']}\n", flush=True)
        print("多次运行对比（每个配置单独跑一次，各自 --tag）:\n"
              f"  python {os.path.join(SCRIPT_DIR, 'analyze_moe_par.py')} "
              "--run a=<dir-a> --run b=<dir-b> -o report.html\n", flush=True)
    else:
        print(f"[bench] saved {out} (collection off)", flush=True)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
