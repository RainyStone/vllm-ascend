"""Three-leg MoE communication sweep, end to end: mc2 / moonep-owner / moonep-slot.

One command runs the same workload three times -- once with plain all-to-all EP, once with
MoonEP in owner mode, once with MoonEP in slot mode -- and then writes a single HTML report
plus a summary table. Every leg is a separate process tree driven by ``moe_par_bench.py``,
so each one gets a fresh engine, a fresh recorder (the stats switch latches at import) and
its own dump directory.

  python scripts/moe_par/moe_par_sweep.py --dp 8 --cards 0,1,2,3,4,5,6,7 \\
      --model /mnt/weight/A5-weights/DeepSeek-V4-Flash \\
      --dataset scripts/moe_par/gsm8k_test.jsonl --num-prompts 0

The three legs only differ in ``--moonep-mode``; the workload flags are written once here
and passed down unchanged, because a leg comparison is only meaningful if the batch shape
is identical. ``--dry-run`` prints the exact commands (with each leg's env overlay) and
starts nothing -- that is also how a manual launch is done: the printed lines are what the
sweep itself would have executed.

Layout under ``<stats-dir>/<sweep-tag>/``:

  mc2/ moonep-owner/ moonep-slot/      one bench dump directory per leg (bench.json inside)
  mc2-score/ ...                       the accuracy pass of each leg, when scoring is on
  mc2-prof/ ...                        the profiling pass of each leg, when profiling is on
  logs/<leg>.log                       the leg's full console output
  traces/<leg>/                        torch_npu traces (tensorboard --logdir)
  report.html  payload.json            the analyzer's report and its data
  sweep_summary.json  sweep_summary.md the comparison table

This file deliberately owns no launcher: the DP parent/child protocol lives in
``moe_par_bench.py`` and there is exactly one writer for each ``bench.json``.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# scripts/moe_par/, where the bench and the analyzer live; the dataset and the dumps are
# one level up in scripts/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import moe_par_bench as bench  # noqa: E402  -- shares the leg definitions and the tag rules

BENCH = os.path.join(SCRIPT_DIR, "moe_par_bench.py")
ANALYZE = os.path.join(SCRIPT_DIR, "analyze_moe_par.py")

# leg name -> --moonep-mode value. The names are what the report calls each run.
LEGS = [("mc2", "off"), ("moonep-owner", "owner"), ("moonep-slot", "slot")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the MoE communication legs back to back and report on all of them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model and workload")
    g.add_argument("--model", required=True, help="model directory, e.g. DeepSeek-V4-Flash")
    g.add_argument("--dataset", default=os.path.join(SCRIPT_DIR, "gsm8k_test.jsonl"))
    g.add_argument("--num-prompts", type=int, default=0, help="0 = the whole file")
    g.add_argument("--max-tokens", type=int, default=256, help="output length of the load pass")
    g.add_argument("--batch-size", "--max-num-seqs", dest="max_num_seqs", type=int, default=32)
    g.add_argument("--max-num-batched-tokens", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("parallelism")
    g.add_argument("--dp", type=int, default=8)
    g.add_argument("--cards", default="", help="card group; default is the first --dp cards")
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--gpu-mem-util", type=float, default=0.80)
    g.add_argument("--max-model-len", type=int, default=40960)
    g.add_argument("--min-free-gib", type=float, default=60.0)
    g.add_argument("--timeout", type=float, default=3600.0, help="per-leg budget, seconds")
    g.add_argument("--allow-tiny-kv", action="store_true",
                   help="do not treat a KV pool that cannot hold the batch as a failure")

    g = p.add_argument_group("passes")
    g.add_argument("--legs", default=",".join(name for name, _ in LEGS),
                   help=f"comma-separated subset of {[name for name, _ in LEGS]}")
    g.add_argument("--skip-score", dest="score", action="store_false", default=True,
                   help="skip the GSM8K accuracy pass (it is a separate pass and roughly "
                        "doubles the wall clock; ON unless you pass this)")
    g.add_argument("--score-max-tokens", type=int, default=256)
    g.add_argument("--num-shots", type=int, default=5)
    g.add_argument("--test-file", default="", help="scored questions; default --dataset")
    g.add_argument("--train-file", default="", help="5-shot examples; default: probe the box")
    g.add_argument("--score-batched-tokens", type=int, default=4096,
                   help="per-step budget for the scoring pass: its 5-shot prompts are ~2k "
                        "tokens, and 1024 would chunk every prefill")
    g.add_argument("--skip-profile", dest="profile", action="store_false", default=True,
                   help="skip the torch_npu profiling pass; ON unless you pass this")
    g.add_argument("--profile-prompts", type=int, default=8,
                   help="profiled requests PER RANK (the leg passes --dp*this as "
                        "--num-prompts, which the bench splits evenly across ranks); "
                        "8 per rank x 8 ranks x 32 tokens puts ~8 DP all-reduces in the window")
    g.add_argument("--profile-max-tokens", type=int, default=32)

    g = p.add_argument_group("output")
    g.add_argument("--stats-dir", default=os.path.join(SCRIPTS_DIR, "moe_par_stats"))
    g.add_argument("--sweep-tag", default="", help="default dp<N>-sweep-<timestamp>")
    g.add_argument("--force", action="store_true", help="write into existing dump directories")
    g.add_argument("--dry-run", action="store_true",
                   help="print every command and its env, then exit without running anything")
    return p.parse_args(argv)


# ------------------------------------------------------------------------- plans -- #
def resolve_cards(args):
    if args.cards:
        cards = [c for c in args.cards.replace(" ", "").split(",") if c]
    else:
        cards = [str(i) for i in range(args.dp)]
    if len(cards) < args.dp:
        sys.exit(f"--dp {args.dp} needs at least {args.dp} cards; got {len(cards)}")
    return cards


def base_cmd(args, cards, tag):
    """The flags every leg shares -- written once, so the legs cannot drift apart."""
    cmd = [
        sys.executable, BENCH,
        "--dp", str(args.dp), "--cards", ",".join(cards), "--tp", str(args.tp),
        "--model", args.model, "--dataset", args.dataset,
        "--num-prompts", str(args.num_prompts), "--max-tokens", str(args.max_tokens),
        "--batch-size", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--max-model-len", str(args.max_model_len),
        "--seed", str(args.seed), "--timeout", str(args.timeout),
        "--min-free-gib", str(args.min_free_gib),
        "--stats-dir", args.stats_dir, "--tag", tag,
    ]
    if args.allow_tiny_kv:
        cmd.append("--allow-tiny-kv")
    if args.force:
        cmd.append("--force")
    return cmd


def plan(args, cards):
    """Every command this sweep would run, with the env overlay each one needs."""
    sweep_dir = os.path.join(args.stats_dir, args.sweep_tag)
    legs = [name for name in args.legs.split(",") if name]
    known = dict(LEGS)
    unknown = [name for name in legs if name not in known]
    if unknown:
        sys.exit(f"unknown leg(s) {unknown}; known: {list(known)}")

    steps = []
    for name in legs:
        mode = known[name]
        leg_dir = os.path.join(sweep_dir, name)
        cmd = base_cmd(args, cards, os.path.join(args.sweep_tag, name)) + \
            ["--moonep-mode", mode]
        steps.append({
            "kind": "load", "leg": name, "mode": mode, "cmd": cmd,
            "env": bench.leg_env(mode), "dump_dir": leg_dir,
            "log": os.path.join(sweep_dir, "logs", f"{name}.log"),
        })
        if args.score:
            score_dir = os.path.join(sweep_dir, f"{name}-score")
            cmd = base_cmd(args, cards, os.path.join(args.sweep_tag, f"{name}-score")) + [
                "--moonep-mode", mode, "--score", "--no-stats",
                "--score-max-tokens", str(args.score_max_tokens),
                "--num-shots", str(args.num_shots),
                "--max-num-batched-tokens", str(max(args.score_batched_tokens,
                                                    args.max_num_batched_tokens)),
            ]
            if args.train_file:
                cmd += ["--train-file", args.train_file]
            if args.test_file:
                cmd += ["--test-file", args.test_file]
            steps.append({
                "kind": "score", "leg": name, "mode": mode, "cmd": cmd,
                "env": bench.leg_env(mode), "dump_dir": score_dir,
                "log": os.path.join(sweep_dir, "logs", f"{name}-score.log"),
            })
    for name in legs:
        if not args.profile:
            break
        mode = known[name]
        prof_dir = os.path.join(sweep_dir, f"{name}-prof")
        cmd = base_cmd(args, cards, os.path.join(args.sweep_tag, f"{name}-prof")) + [
            "--moonep-mode", mode, "--profile-only",
            "--profile-dir", os.path.join(sweep_dir, "traces", name),
            # --profile-prompts is per rank; the bench splits --num-prompts across the group,
            # so it needs the product or every rank past the first gets an empty window.
            "--num-prompts", str(args.profile_prompts * args.dp),
            "--profile-max-tokens", str(args.profile_max_tokens),
            "--no-stats",
        ]
        steps.append({
            "kind": "profile", "leg": name, "mode": mode, "cmd": cmd,
            "env": bench.leg_env(mode), "dump_dir": prof_dir,
            "log": os.path.join(sweep_dir, "logs", f"{name}-prof.log"),
        })
    return steps, sweep_dir, legs


# --------------------------------------------------------------------- preflight -- #
def nearest_existing(path):
    """The closest ancestor of ``path`` that exists, for a disk check on a fresh dir."""
    path = os.path.abspath(path)
    while path and not os.path.exists(path):
        path = os.path.dirname(path)
    return path or "."


def preflight(args, cards, legs, sweep_dir):
    """Everything that would waste an hour if it were found out later -- checked up front."""
    problems = []
    print(f"[sweep] model={args.model}", flush=True)
    if not os.path.isdir(args.model):
        sys.exit(f"model directory not found: {args.model}")
    if not os.path.exists(args.dataset):
        sys.exit(f"dataset not found: {args.dataset}")

    free = bench.card_free_gib()
    if free:
        report = " ".join(f"{c}:{free.get(int(c), float('nan')):.1f}G" for c in cards)
        print(f"[sweep] free HBM per card (GiB): {report}", flush=True)
        tight = [c for c in cards if int(c) in free and free[int(c)] < args.min_free_gib]
        if tight:
            problems.append(
                f"card(s) {tight} have less than {args.min_free_gib:.0f} GiB free; those cards "
                f"belong to someone else, so the leg would fail at startup. The cards that are "
                f"free: {[c for c, gib in free.items() if gib >= args.min_free_gib]}")
    else:
        print("[sweep] WARNING: could not read free HBM (npu-smi); the legs will check again",
              flush=True)

    print(f"[sweep] PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
    print(f"[sweep] ASCEND_MOONEP_BUILD_DIR="
          f"{os.environ.get('ASCEND_MOONEP_BUILD_DIR', '<unset>')}", flush=True)

    if args.score:
        try:
            import gsm8k_score
            prompts, labels, meta = gsm8k_score.build_prompts(
                num_questions=1, num_shots=args.num_shots,
                train_file=args.train_file, test_file=args.test_file or args.dataset)
            print(f"[sweep] scoring: {meta['num_shots']}-shot, train={meta['train_file']}, "
                  f"test={meta['test_file']}", flush=True)
            if meta["num_shots"] != args.num_shots:
                problems.append(f"asked for {args.num_shots}-shot but the train split gave "
                                f"{meta['num_shots']}; accuracy would not be comparable")
        except SystemExit as exc:
            problems.append(f"the scoring pass cannot build its prompts: {exc}")

    if args.max_num_seqs > args.max_num_batched_tokens:
        problems.append(f"--batch-size {args.max_num_seqs} exceeds --max-num-batched-tokens "
                        f"{args.max_num_batched_tokens}; vLLM may reject the engine config")

    if args.num_prompts and args.num_prompts < args.dp:
        problems.append(f"--num-prompts {args.num_prompts} < --dp {args.dp}: some ranks would "
                        f"get an empty slice, which is a hard error by design")

    if args.profile:
        traces = os.path.join(sweep_dir, "traces")
        usage = shutil.disk_usage(nearest_existing(os.path.dirname(traces)))
        print(f"[sweep] disk free under {os.path.dirname(traces)}: "
              f"{usage.free / 2**30:.1f} GiB", flush=True)
        if usage.free < 20 * 2**30:
            problems.append(f"only {usage.free / 2**30:.1f} GiB free where the traces go; a "
                            f"level-1 trace across {args.dp} ranks is GBs and a full disk fails "
                            f"the run mid-export")

    if problems:
        print("\n[sweep] preflight refused to start:", flush=True)
        for problem in problems:
            print(f"[sweep]   - {problem}", flush=True)
        sys.exit(2)
    print(f"[sweep] preflight ok: {len(legs)} leg(s) {legs}, "
          f"score={'on' if args.score else 'off'}, profile={'on' if args.profile else 'off'}",
          flush=True)


# ------------------------------------------------------------------ execution -- #
def run_step(step, sweep_dir, dry):
    env = os.environ.copy()
    for key, value in step["env"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    for key, value in bench.LAUNCH_ENV_DEFAULTS.items():
        env.setdefault(key, value)

    header = (f"{' '.join(f'{k}={v}' for k, v in step['env'].items() if v is not None)} "
              f"{' '.join(step['cmd'])}")
    print(f"\n[sweep] == {step['kind']} leg={step['leg']} mode={step['mode']}\n"
          f"[sweep] {header}", flush=True)
    if dry:
        return 0
    os.makedirs(os.path.dirname(step["log"]), exist_ok=True)
    started = time.time()
    # The child's output goes to the console and to its log at once: a leg takes minutes
    # and the interesting line is usually the last one before it wedges.
    with open(step["log"], "w", buffering=1) as log:
        proc = subprocess.Popen(step["cmd"], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    print(f"[sweep] == {step['kind']} leg={step['leg']} exit={rc} "
          f"({time.time() - started:.0f}s, log {step['log']})", flush=True)
    return rc


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def memory_span(bench_json):
    """min/max over ranks of each memory number, so one odd rank cannot hide."""
    out = {}
    memory = (bench_json or {}).get("memory") or {}
    for key in ("weights_gib", "peak_activation_gib", "non_torch_gib", "available_kv_gib",
                "kv_tokens", "load_weights_s"):
        values = [m[key] for m in memory.values() if isinstance(m.get(key), (int, float))]
        if values:
            out[key] = {"min": min(values), "max": max(values)}
    slot_rows = [m["slot_stats"]["total_slot_rows"]
                 for m in memory.values() if m.get("slot_stats")]
    slot_calls = [m["slot_stats"]["calls_with_slot_rows"]
                  for m in memory.values() if m.get("slot_stats")]
    if slot_rows:
        out["slot_rows_min"] = min(slot_rows)
        out["slot_rows_max"] = max(slot_rows)
        out["slot_calls_min"] = min(slot_calls)
    return out


def par_from_payload(payload, name):
    for run in (payload or {}).get("runs", []):
        if run.get("name") != name:
            continue
        steps = [v for v in (run.get("par_by_step") or []) if isinstance(v, (int, float))]
        agg = [v for v in (run.get("par_agg") or []) if isinstance(v, (int, float))]
        return {
            "par_whole": run.get("par_whole"),
            "par_by_step_mean": sum(steps) / len(steps) if steps else None,
            "par_agg_mean": sum(agg) / len(agg) if agg else None,
            "n_steps": run.get("n_steps"),
            "n_ranks": run.get("n_ranks"),
            "checks": run.get("checks"),
        }
    return None


def summarise(args, legs, sweep_dir, payload, leg_rc):
    rows = []
    for name in legs:
        leg_dir = os.path.join(sweep_dir, name)
        summary = read_json(os.path.join(leg_dir, "bench.json")) or {}
        score = read_json(os.path.join(sweep_dir, f"{name}-score", "score.json"))
        profile = read_json(os.path.join(sweep_dir, f"{name}-prof", "profile.json"))
        perf = (summary.get("perf") or {}).get("quantiles") or {}
        rows.append({
            "leg": name,
            "mode": dict(LEGS)[name],
            "dump_dir": leg_dir,
            "exit_ok": leg_rc.get(name, 0) == 0,
            "wall_s": (summary.get("timing") or {}).get("generate_s"),
            "output_tokens": (summary.get("timing") or {}).get("output_tokens"),
            "output_tok_s": ((summary.get("perf") or {}).get("throughput") or {}).get("output_tok_s"),
            "total_tok_s": ((summary.get("perf") or {}).get("throughput") or {}).get("total_tok_s"),
            "ttft_p50_ms": (perf.get("ttft_ms") or {}).get("p50"),
            "ttft_p90_ms": (perf.get("ttft_ms") or {}).get("p90"),
            "tpot_p50_ms": (perf.get("tpot_ms") or {}).get("p50"),
            "n_requests": len(summary.get("requests") or []),
            "kv_check": summary.get("kv_check"),
            "memory": memory_span(summary),
            "par": par_from_payload(payload, name),
            "accuracy": (score or {}).get("result", {}).get("accuracy"),
            "invalid_rate": (score or {}).get("result", {}).get("invalid_rate"),
            "profile": {"dir": (profile or {}).get("profile_dir"),
                        "traces": len((profile or {}).get("trace_files") or [])},
        })
    return rows


def write_markdown(rows, sweep_dir, args):
    def fmt(value, digits=2):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        f"# MoE 通信三配置对比（DP{args.dp}）",
        "",
        f"- model: `{args.model}`",
        f"- dataset: `{args.dataset}`（num_prompts={args.num_prompts or '全部'}，"
        f"batch={args.max_num_seqs}，max_tokens={args.max_tokens}）",
        f"- cards: `{args.cards}`，gpu_mem_util={args.gpu_mem_util}",
        "",
        "| leg | wall_s | out tok/s | TTFT p50 / p90 (ms) | TPOT p50 (ms) | 全程 PAR | "
        "weights/non-torch (GiB) | KV tokens | slot rows | 准确率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        memory = row["memory"]
        weights = memory.get("weights_gib") or {}
        non_torch = memory.get("non_torch_gib") or {}
        kv = memory.get("kv_tokens") or {}
        par = row["par"] or {}
        lines.append(
            f"| {row['leg']} | {fmt(row['wall_s'], 1)} | {fmt(row['output_tok_s'])} | "
            f"{fmt(row['ttft_p50_ms'], 1)} / {fmt(row['ttft_p90_ms'], 1)} | "
            f"{fmt(row['tpot_p50_ms'], 2)} | {fmt(par.get('par_whole'), 3)} | "
            f"{fmt(weights.get('max'))} / {fmt(non_torch.get('max'))} | "
            f"{kv.get('min', 'n/a')} | {memory.get('slot_rows_min', '-')} | "
            f"{fmt(row['accuracy'], 3)} |")
    lines += [
        "",
        "口径：wall_s 是**最慢 rank** 的 generate 窗口（不是父进程墙钟，模型加载占了大部分）；"
        "tok/s 是各 rank 输出 token 之和除以该窗口；PAR 来自 analyze_moe_par.py 的 payload"
        "（全程 PAR = 所有 rank 所有层的峰值/均值）。",
        "",
        "- `KV tokens` 是该 leg 中最小的 KV 池（tokens），`kv_check.ok=false` 说明这个池子装不下"
        "一个整批或一个满长度请求，引擎会靠抢占推进，负载与延迟数字描述的是抢占而不是 MoE 通信。",
        "- slot rows 只有 slot leg 会有：M7 判据是 `total_slot_rows > 0 && calls_with_slot_rows > 0`。",
        "",
        f"报告：`{os.path.join(sweep_dir, 'report.html')}`，原始数据："
        f"`{os.path.join(sweep_dir, 'payload.json')}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    args.sweep_tag = args.sweep_tag or f"dp{args.dp}-sweep-{time.strftime('%m%d-%H%M')}"
    cards = resolve_cards(args)
    steps, sweep_dir, legs = plan(args, cards)

    # The pinned env (vendored PYTHONPATH, the build dir, the three DSV4 defaults) is set
    # here so the whole sweep runs under one recipe and the children inherit it --
    # moe_par_bench.py re-asserts it per leg, and the dry run prints what it resolved to.
    bench.apply_launch_env(args)

    if args.dry_run:
        print(f"[sweep] DRY RUN -- {len(steps)} command(s) under {sweep_dir}", flush=True)
        preflight(args, cards, legs, sweep_dir)
        for step in steps:
            run_step(step, sweep_dir, dry=True)
        print(f"\n[sweep] 手动跑法：按上面的顺序逐条执行；同一条 leg 的命令是一条（DP 子进程由 "
              f"moe_par_bench.py 自己拉起）\n", flush=True)
        return 0

    preflight(args, cards, legs, sweep_dir)
    os.makedirs(os.path.join(sweep_dir, "logs"), exist_ok=True)

    failures = []
    leg_rc = {}
    skip_legs = set()
    for step in steps:
        if step["leg"] in skip_legs:
            continue
        rc = run_step(step, sweep_dir, dry=False)
        leg_rc[step["leg"]] = leg_rc.get(step["leg"], 0) or rc
        if rc != 0:
            failures.append(f"{step['kind']}:{step['leg']}")
            if step["kind"] == "load":
                # A leg whose load pass failed has no dump to analyze; its score and
                # profile runs would only add noise to the failure list.
                print(f"[sweep] leg {step['leg']} failed; skipping its remaining passes",
                      flush=True)
                skip_legs.add(step["leg"])
                if step["leg"] in legs:
                    legs.remove(step["leg"])

    report = os.path.join(sweep_dir, "report.html")
    payload_path = os.path.join(sweep_dir, "payload.json")
    payload = None
    runs = [f"{name}={os.path.join(sweep_dir, name)}" for name in legs
            if os.path.exists(os.path.join(sweep_dir, name, "bench.json"))]
    if runs:
        cmd = [sys.executable, ANALYZE]
        for spec in runs:
            cmd += ["--run", spec]
        cmd += ["-o", report, "--json", payload_path]
        print(f"\n[sweep] analyze: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            failures.append("analyze")
        payload = read_json(payload_path)
        print(f"[sweep] report -> {report}", flush=True)
    else:
        print("[sweep] no leg produced a bench.json; nothing to report", flush=True)
        failures.append("no-dumps")

    rows = summarise(args, legs, sweep_dir, payload, leg_rc)
    write_json(os.path.join(sweep_dir, "sweep_summary.json"),
               {"tag": args.sweep_tag, "args": vars(args), "legs": rows,
                "payload": payload_path if payload else None,
                "report": report if runs else None, "failures": failures})
    markdown = write_markdown(rows, sweep_dir, args)
    with open(os.path.join(sweep_dir, "sweep_summary.md"), "w") as fh:
        fh.write(markdown)
    print("\n" + markdown, flush=True)

    print(f"[sweep] 产物目录 {sweep_dir}", flush=True)
    if failures:
        print(f"[sweep] 有失败项: {failures}", flush=True)
    return 1 if failures else 0


def write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
