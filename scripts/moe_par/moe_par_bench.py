"""Offline MoE load / PAR sweep: run a dataset through vLLM and dump per-rank routing load.

Each run writes one dump per rank into ``<stats-dir>/<tag>/`` and a ``bench.json`` next to
them. Feed those directories to ``analyze_moe_par.py`` to get the HTML report -- the exact
command is printed at the end of the run and stored in ``bench.json["analyze_cmd"]``.

``bench.json`` also carries the run's own performance numbers: overall throughput plus
per-request TTFT / TPOT / end-to-end, read off ``RequestOutput.metrics`` (see
:func:`request_latency`). The report renders them in a 性能 panel, per request and per run,
so the MoE communication legs can be compared on latency and not only on load.

Three things to know before editing this file:

* **One setting per process.** The recorder keeps module-level state (layer map, step
  counter, open file handle) that is never reset, so building a second ``LLM`` in the same
  process would append to the same dump under a stale layer map. Run this once per configuration.
* **The stats switch is latched at import.** ``moe_par_stats`` reads ``VLLM_ASCEND_MOE_PAR_*``
  when the module is first imported, so :func:`apply_stats_env` must run *before*
  ``import vllm`` inside :func:`main`. Do not hoist that import to the top of this file.
* **The engine's own stats must be asked for.** The offline ``LLM`` defaults to
  ``disable_log_stats=True``; with them off the engine allocates no request stats and every
  latency field reads as 0.0. :func:`main` passes ``disable_log_stats=False``.

Example:
  ASCEND_RT_VISIBLE_DEVICES=1,6 python scripts/moe_par_bench.py \\
      --tp 2 --batch-size 32 --max-num-batched-tokens 512 --max-tokens 256 \\
      --num-prompts 128 --tag tp2-b32-bt512
"""
import argparse
import glob
import json
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "gsm8k_test.jsonl")
DEFAULT_MODEL = "/mnt/data/weight/Qwen3-Coder-30B-A3B-Instruct"


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
    g.add_argument("--tp", type=int, default=4, help="tensor parallel size; EP spans this group")
    g.add_argument("--no-ep", dest="ep", action="store_false", default=True,
                   help="turn expert parallelism off (ep_size=1, the all-gather MoE run); "
                        "EP is ON unless you pass this")
    g.add_argument("--moonep", action="store_true",
                   help="enable_moonep: EP over SHMEM instead of the default all-to-all")
    g.add_argument("--max-model-len", type=int, default=40960)
    g.add_argument("--gpu-mem-util", type=float, default=0.80)
    g.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false", default=True,
                   help="allow graph capture; eager is ON unless you pass this")
    g.add_argument("--additional-config", default="",
                   help="extra JSON merged into additional_config, e.g. '{\"foo\": 1}'")

    g = p.add_argument_group("stats")
    g.add_argument("--stats-dir", default=os.path.join(SCRIPT_DIR, "moe_par_stats"),
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

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)

    return p.parse_args(argv)


def make_tag(args):
    """A run name that already says what makes this configuration differ from its neighbours."""
    parts = [f"tp{args.tp}"]
    parts.append("ep" if args.ep else "noep")
    if args.moonep:
        parts.append("moonep")
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
def apply_stats_env(args, tag, dump_dir):
    """Must run before ``import vllm`` -- the recorder reads these at import time."""
    os.environ["VLLM_ASCEND_MOE_PAR_STATS"] = "1" if args.stats else "0"
    os.environ["VLLM_ASCEND_MOE_PAR_STATS_DIR"] = dump_dir
    os.environ["VLLM_ASCEND_MOE_PAR_STATS_TAG"] = tag
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
    if args.moonep:
        # ascend-moonep loads libshmem off this variable; the engine switch is additional_config.
        os.environ["MOONEP"] = "1"
    else:
        # Make the CLI authoritative: a stale MOONEP=1 exported by the shell would load
        # libshmem while additional_config says enable_moonep=False.
        os.environ.pop("MOONEP", None)


# -------------------------------------------------------------------------- dumps -- #
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


# --------------------------------------------------------------------------- main -- #
def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or make_tag(args)
    dump_dir = args.dump_dir or os.path.join(args.stats_dir, tag)

    # analyze_moe_par.py globs a whole directory into one run, so a second tag in here would
    # silently merge two configurations into a single report.
    if args.stats and os.path.isdir(dump_dir):
        prior = existing_tags(dump_dir)
        strangers = prior - {tag}
        if strangers and not args.force:
            sys.exit(f"{dump_dir} already holds dumps of tag(s) {sorted(strangers)}. The analyzer "
                     f"reads every dump in a directory as one run, so mixing them would fuse two "
                     f"configurations into a single report. Use another --dump-dir, or --force.")
        if tag in prior:
            print(f"[bench] WARNING: {dump_dir} already holds tag {tag}; its dumps will be "
                  f"overwritten", flush=True)

    # Validate before the (slow) vllm import so a typo fails fast.
    if not os.path.exists(args.dataset):
        sys.exit(f"dataset not found: {args.dataset}")
    if args.max_tokens >= args.max_model_len:
        sys.exit(f"--max-tokens {args.max_tokens} leaves no room in --max-model-len "
                 f"{args.max_model_len}")
    if args.max_num_seqs > args.max_num_batched_tokens:
        print(f"[bench] WARNING: --batch-size {args.max_num_seqs} exceeds "
              f"--max-num-batched-tokens {args.max_num_batched_tokens}; vLLM may reject this",
              flush=True)

    apply_stats_env(args, tag, dump_dir)

    from vllm import LLM, SamplingParams  # noqa: E402  -- must follow apply_stats_env

    questions, total_rows = load_questions(args.dataset, args.prompt_field, args.num_prompts)
    print(f"[bench] tag={tag}", flush=True)
    print(f"[bench] dataset={args.dataset} ({len(questions)} of {total_rows} rows)", flush=True)
    print(f"[bench] devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')} tp={args.tp} "
          f"ep={args.ep} moonep={args.moonep} batch={args.max_num_seqs} "
          f"batched_tokens={args.max_num_batched_tokens} out_len={args.max_tokens}", flush=True)
    print(f"[bench] dumps -> {dump_dir}", flush=True)

    additional = {"enable_moonep": args.moonep}
    if args.additional_config:
        additional.update(json.loads(args.additional_config))

    llm = LLM(
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
    world = llm.get_world_size()
    tok = llm.get_tokenizer()
    prompts = build_prompts(tok, questions)
    print(f"[bench] world_size={world} prompt tokens "
          f"{min(len(p['prompt_token_ids']) for p in prompts)}"
          f"-{max(len(p['prompt_token_ids']) for p in prompts)}", flush=True)

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        seed=args.seed,
    )

    t0 = time.time()
    outs = llm.generate(prompts, params)
    generate_s = time.time() - t0

    requests = []
    for i, o in enumerate(outs):
        n_out = len(o.outputs[0].token_ids)
        requests.append({
            "idx": i,
            "prompt_tokens": len(o.prompt_token_ids),
            "output_tokens": n_out,
            "finish_reason": o.outputs[0].finish_reason,
            "latency": request_latency(o, n_out),
        })
    prompt_tokens = sum(r["prompt_tokens"] for r in requests)
    output_tokens = sum(r["output_tokens"] for r in requests)
    finished = sum(1 for r in requests if r["finish_reason"] == "stop")

    # Let the workers tear down and flush before we look at their files.
    del llm
    dumps = drain(dump_dir, world, args.drain_timeout) if args.stats else []

    summary = {
        "tag": tag,
        "model": args.model,
        "dataset": os.path.abspath(args.dataset),
        "config": {
            "tp": args.tp,
            "ep": args.ep,
            "moonep": args.moonep,
            "batch_size": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "additional_config": additional,
            "num_prompts": len(prompts),
            "rows_in_dataset": total_rows,
            "seed": args.seed,
        },
        "devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "dump_dir": dump_dir,
        "dumps": [describe_dump(p) for p in dumps],
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
    }

    out = args.out or os.path.join(dump_dir, "bench.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.stats:
        # -o is left relative so the report lands in whatever directory you run this from.
        summary["analyze_cmd"] = (
            f"python {os.path.join(SCRIPT_DIR, 'analyze_moe_par.py')} "
            f"--run {tag}={dump_dir} -o {tag}_report.html"
        )
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[bench] {output_tokens} output tokens in {generate_s:.1f}s "
          f"({summary['timing']['output_tok_s']} tok/s), {finished}/{len(requests)} hit EOS "
          f"(ignore_eos={args.ignore_eos})", flush=True)
    perf = summary["perf"]
    q = perf["quantiles"]
    tp = perf.get("throughput") or {}
    if tp:
        print(f"[bench] throughput in {tp['input_tok_s']} / out {tp['output_tok_s']} tok/s, "
              f"{tp['req_s']} req/s (over the whole generate() call)", flush=True)
    if perf["measured"]:
        # TTFT is always there when metrics were read. TPOT is missing when no request
        # produced a second token, e.g. --max-tokens 1: quote what exists, not a zero.
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
        if len(dumps) < world:
            print(f"[bench] WARNING: expected {world} dumps, found {len(dumps)} -- check the "
                  f"worker logs for '[moe-par]' lines", flush=True)
        for d in summary["dumps"]:
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


if __name__ == "__main__":
    main()
