"""DP/EP decode benchmark with engine-side ground-truth timing.

Parent mode (no --worker): spawns DP subprocesses with VLLM_DP_* env, aggregates.
Worker mode (--worker R): builds the LLM, generates this rank's shard.

Why this differs from bench_dp_orig.py
--------------------------------------
The original measured one wall clock around llm.generate() and derived
prefill/decode throughput from a G1-vs-G128 delta. That is wrong in three ways:

  1. max_num_seqs was never set, so the engine used its own default (256 on
     vllm-ascend -- NPUPlatform.get_device_total_memory raises NotImplementedError,
     so vLLM's get_batch_defaults never sees the 84 GiB of HBM and falls into the
     "<70 GiB" branch). A 1024-seq run therefore ran in waves, and the G1 leg
     (n_seq=128) and G128 leg (n_seq=1024) had different concurrency, so the
     delta attributed prefill/decode incorrectly.

  2. LLM() forces disable_log_stats=True unless you pass it explicitly, so no
     engine stats existed at all. There was no way to see the real running batch.

  3. The first generate() pays one-time init costs with no warmup.

This version attaches a custom StatLogger to the LLMEngine. It is called once per
engine step in the same process as llm.generate(), and exposes:
    scheduler_stats.num_running_reqs / num_waiting_reqs / kv_cache_usage
    iteration_stats.iteration_timestamp            -> per-step wall time
    iteration_stats.num_generation_tokens          -> decode tokens this step
    iteration_stats.prompt_token_stats.computed    -> prefill tokens this step
    iteration_stats.time_to_first_tokens_iter      -> real TTFT samples
    iteration_stats.inter_token_latencies_iter     -> real ITL samples
    iteration_stats.finished_requests[].mean_time_per_output_token -> real TPoT

so prefill and decode are separated by what the engine actually did, not by a
subtraction between two differently-shaped runs.

Env: DP(=2) TP(=1) N_SEQ(default = MAX_NUM_SEQS) PROMPT_LEN(256) GEN_TOKENS(128)
     MAX_NUM_SEQS(256) MBT(max(N_SEQ,1024)) WARMUP(1) MODEL TAG VISIBLE
     SWEEP("")  comma-separated max_num_seqs list; runs each in its own engine
     STEP_DUMP(0) write the full per-step trace as jsonl
     MOONEP / MOONEP_DISABLE_UDMA / MOONEP_SHMEM_BOOTSTRAP ... (passed through)

Aggregate tok/s = sum(rank gen tokens) / max(rank generate wall).
"""
import enum
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass

PARA = (
    "The history of computing spans more than a century, from mechanical "
    "calculators through vacuum tubes and transistors to modern microprocessors. "
    "Each generation brought dramatic changes in speed, size, and accessibility. "
    "Today, computers are embedded in almost every aspect of daily life. "
)


# --------------------------------------------------------------------------
# percentile helpers
# --------------------------------------------------------------------------
def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summary(xs, scale=1.0):
    """Percentile summary. scale=1000 converts seconds -> ms."""
    if not xs:
        return {"n": 0}
    v = [x * scale for x in xs]
    return {
        "n": len(v),
        "mean": round(statistics.fmean(v), 3),
        "p50": round(_pct(v, 0.50), 3),
        "p90": round(_pct(v, 0.90), 3),
        "p99": round(_pct(v, 0.99), 3),
        "max": round(max(v), 3),
    }


def _finished_to_dict(f):
    """FinishedRequestStats is a dataclass; finish_reason is an IntEnum, so it
    would otherwise serialize as a bare int (0) instead of a readable name."""
    items = ([(fl.name, getattr(f, fl.name)) for fl in fields(f)]
             if is_dataclass(f) else list(vars(f).items()))
    d = {}
    for k, v in items:
        if isinstance(v, enum.Enum):
            d[k] = v.name
        elif v is None or isinstance(v, (bool, int, float, str)):
            d[k] = v
        else:
            d[k] = str(v)
    return d


# --------------------------------------------------------------------------
# engine-side step logger
# --------------------------------------------------------------------------
class StepLogger:
    """Per-engine-step ground truth.

    Instantiated in the LLMEngine process (the same one that runs llm.generate),
    so the worker can read .steps / .ttfts / .itls / .finished after the run.
    """

    def __init__(self, vllm_config=None, engine_index=0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.steps = []      # one dict per engine step
        self.ttfts = []      # seconds, one per first token
        self.itls = []       # seconds, one per inter-token gap
        self.finished = []   # one dict per finished request

    def record(self, scheduler_stats=None, iteration_stats=None,
               mm_cache_stats=None, engine_idx=0):
        st, it = scheduler_stats, iteration_stats
        rec = {
            "t": it.iteration_timestamp if it is not None else time.time(),
            "running": getattr(st, "num_running_reqs", -1) if st else -1,
            "waiting": getattr(st, "num_waiting_reqs", -1) if st else -1,
            "kv": round(getattr(st, "kv_cache_usage", 0.0) or 0.0, 4) if st else 0.0,
            "gen": 0, "pre": 0, "pre_total": 0, "cached": 0,
        }
        if it is not None:
            rec["gen"] = it.num_generation_tokens
            ps = it.prompt_token_stats
            rec["pre"] = ps.computed
            rec["pre_total"] = ps.total
            rec["cached"] = ps.cached_tokens
            rec["preempted"] = it.num_preempted_reqs
            self.ttfts.extend(it.time_to_first_tokens_iter)
            self.itls.extend(it.inter_token_latencies_iter)
            for f in it.finished_requests:
                self.finished.append(_finished_to_dict(f))
        self.steps.append(rec)

    def log_engine_initialized(self):  # required by the logger interface
        pass

    def log(self):
        pass

    def record_sleep_state(self, *a, **k):
        pass


def attach_step_logger(llm):
    """Attach a StepLogger to an already-built LLM.

    LLM.__init__ never forwards `stat_loggers` to LLMEngine.from_engine_args, so
    we append directly to the manager's logger list. Requires
    disable_log_stats=False, otherwise logger_manager is None.
    """
    engine = llm.llm_engine
    mgr = getattr(engine, "logger_manager", None)
    if mgr is None:
        raise RuntimeError(
            "logger_manager is None -- construct LLM(..., disable_log_stats=False) "
            "so the engine builds stat loggers")
    lg = StepLogger(getattr(engine, "vllm_config", None), 0)
    mgr.stat_loggers.append(lg)
    return lg


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyse(lg, wall_s, total_gen):
    """Turn the raw step trace into prefill/decode phase metrics.

    Duration attribution: iteration_timestamp is stamped when the step's stats
    are assembled (i.e. near the step's END). We charge each step the gap that
    precedes it, so a phase's duration is the sum of the latencies of the steps
    in it rather than (last_t - first_t), which would drop the final step and
    undercount a 10-step decode phase as 9 step-times.
    """
    steps = lg.steps
    is_dec = [s["gen"] > 0 for s in steps]                  # produced output tokens
    is_pre = [s.get("pre", 0) > 0 for s in steps]           # computed prompt tokens
    is_dec_only = [d and not p for d, p in zip(is_dec, is_pre)]
    is_pre_only = [p and not d for d, p in zip(is_dec, is_pre)]

    out = {"n_steps": len(steps),
           "n_decode_steps": sum(is_dec),
           "n_prefill_steps": sum(is_pre)}

    # gaps[i-1] is the latency of step i.
    # NOTE: with chunked prefill these overlap. A step that finishes a request's
    # prefill also emits that request's first token, so almost every prefill step
    # carries gen>0 and a strict "prefill-only" bucket comes out empty (this bit
    # the first version: prefill_s=0.0 while prefill_tokens=8128). We therefore
    # charge a step to decode if it produced tokens and to prefill if it computed
    # prompt tokens, and report the overlap separately instead of pretending the
    # phases are disjoint.
    dec_s = pre_s = mix_s = 0.0
    dec_only_s = pre_only_s = 0.0
    pure_dec_gaps = []
    for i in range(1, len(steps)):
        g = steps[i]["t"] - steps[i - 1]["t"]
        if is_dec[i]:
            dec_s += g
        if is_pre[i]:
            pre_s += g
        if is_dec[i] and is_pre[i]:
            mix_s += g
        elif is_dec_only[i]:
            dec_only_s += g
        elif is_pre_only[i]:
            pre_only_s += g
        if is_dec_only[i] and is_dec_only[i - 1]:
            pure_dec_gaps.append(g)

    dec_tokens = sum(s["gen"] for s in steps)
    pre_tokens = sum(s.get("pre", 0) for s in steps)

    dec_only_tokens = sum(s["gen"] for s, d in zip(steps, is_dec_only) if d)

    out["decode_s"] = round(dec_s, 4)              # any step that emitted tokens
    out["decode_tokens"] = dec_tokens
    out["decode_tok_s"] = round(dec_tokens / dec_s, 2) if dec_s > 0 else 0.0
    out["decode_only_s"] = round(dec_only_s, 4)    # steps with no prefill mixed in
    out["decode_only_tok_s"] = (round(dec_only_tokens / dec_only_s, 2)
                                if dec_only_s > 0 else 0.0)
    out["step_ms"] = _summary(pure_dec_gaps, 1000.0)
    out["prefill_s"] = round(pre_s, 4)             # any step that computed prompts
    out["prefill_tokens"] = pre_tokens
    out["prefill_tok_s"] = round(pre_tokens / pre_s, 2) if pre_s > 0 else 0.0
    out["prefill_only_s"] = round(pre_only_s, 4)
    out["mixed_s"] = round(mix_s, 4)               # decode+prefill in one step

    dec_idx = [i for i, d in enumerate(is_dec) if d]
    if dec_idx:
        out["running"] = _summary([steps[i]["running"] for i in dec_idx])
        out["kv_peak"] = round(max(steps[i]["kv"] for i in dec_idx), 4)
    else:
        out["running"] = {"n": 0}
        out["kv_peak"] = 0.0
    out["prompt_cached_tokens"] = sum(s.get("cached", 0) for s in steps)

    # --- whole-run ---
    out["wall_s"] = round(wall_s, 3)
    out["total_gen_tokens"] = total_gen
    out["gen_tok_s"] = round(total_gen / wall_s, 2) if wall_s > 0 else 0.0

    # --- per-request, from the engine's own accounting ---
    out["ttft_ms"] = _summary(lg.ttfts, 1000.0)
    out["itl_ms"] = _summary(lg.itls, 1000.0)
    tpot = [f["mean_time_per_output_token"] for f in lg.finished
            if f.get("mean_time_per_output_token")]
    out["tpot_ms"] = _summary(tpot, 1000.0)
    out["e2e_ms"] = _summary([f["e2e_latency"] for f in lg.finished
                              if f.get("e2e_latency")], 1000.0)
    out["n_finished"] = len(lg.finished)
    out["n_preempted"] = sum(s.get("preempted", 0) for s in steps)
    return out


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def run_worker(rank):
    from vllm import LLM, SamplingParams

    dp = int(os.environ.get("VLLM_DP_SIZE", "2"))
    tp = int(os.environ.get("TP", "1"))
    n_total = int(os.environ.get("N_SEQ", "256"))
    plen = int(os.environ.get("PROMPT_LEN", "256"))
    gen = int(os.environ.get("GEN_TOKENS", "128"))
    mbt = int(os.environ.get("MBT", "0")) or max(n_total, 1024)
    mns = int(os.environ.get("MAX_NUM_SEQS", "256"))
    mbt = max(mbt, mns)          # vLLM rejects max_num_batched_tokens < max_num_seqs
    unique = os.environ.get("UNIQUE_PROMPTS", "1") == "1"
    warmup = int(os.environ.get("WARMUP", "1"))
    tag = os.environ.get("TAG", "mc2")
    model = os.environ.get("MODEL", "/home/Qwen3-30B-A3B")
    moonep = os.environ.get("MOONEP", "0") == "1"

    n_local = n_total // dp + (1 if rank < n_total % dp else 0)
    # disable_log_stats=False is what makes logger_manager exist; LLM() would
    # otherwise force it to True and we would get no engine-side data at all.
    # enforce_eager=True keeps results comparable with every earlier run, but it
    # is also the prime suspect for the per-step floor this bench exposes: in
    # eager mode every decode step relaunches the whole MoE model from the host.
    # ENFORCE_EAGER=0 turns on the ACL graph path for an A/B on that floor.
    eager = os.environ.get("ENFORCE_EAGER", "1") == "1"
    llm = LLM(
        model=model,
        enforce_eager=eager,
        enable_expert_parallel=True,
        tensor_parallel_size=tp,
        max_model_len=int(os.environ.get("MAXLEN", "4096")),
        max_num_batched_tokens=mbt,
        max_num_seqs=mns,
        disable_log_stats=False,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.85")),
        additional_config={"enable_moonep": moonep},
    )
    lg = attach_step_logger(llm)

    tok = llm.get_tokenizer()
    base_ids = tok(PARA * (plen // 20 + 4)).input_ids[:max(plen - 3, 1)]

    # vLLM V1 enables prefix caching by default. With N identical prompts every
    # request after the first reuses the whole prefix, so prompt_token_stats
    # reports almost no *computed* tokens and the prefill phase is measured as
    # near-free. Give each request a distinct leading token so the cache cannot
    # hide prefill work. UNIQUE_PROMPTS=0 restores the old identical behaviour
    # (kept so results stay comparable with earlier runs).
    prompt_file = os.environ.get("PROMPT_FILE", "")
    if prompt_file:
        # diverse, per-request distinct prompts (a stand-in for GSM8K: this box
        # cannot reach the S3/HF dataset endpoints). Each DP rank takes a
        # different slice, like a real data-parallel server would.
        import json as _json

        with open(prompt_file) as fh:
            pool = [_json.loads(line)["text"] for line in fh if line.strip()]
        off = int(os.environ.get("PROMPT_OFFSET", str(rank * n_local)))
        prompts = [pool[(off + i) % len(pool)] for i in range(n_local)]
        prompts = [tok.decode(tok(p).input_ids[:plen]) for p in prompts]   # trim to prompt_len
        unique = True
    elif unique:
        prompts = [tok.decode([1000 + (i % 9000)] + base_ids) for i in range(n_local)]
    else:
        prompts = [tok.decode(base_ids)] * n_local
    params = SamplingParams(temperature=0.0, max_tokens=gen)

    if warmup:
        # warmup uses a different prompt so it cannot warm the real prompts' prefix
        wparams = SamplingParams(temperature=0.0, max_tokens=8)
        llm.generate([tok.decode([9999] + base_ids)], wparams)
        lg.steps.clear(); lg.ttfts.clear(); lg.itls.clear(); lg.finished.clear()

    t0 = time.time()
    outs = llm.generate(prompts, params)
    wall = time.time() - t0

    total_gen = sum(len(o.outputs[0].token_ids) for o in outs)
    firsts = {o.outputs[0].token_ids[0] for o in outs if o.outputs[0].token_ids}
    res = {
        "rank": rank, "n_local": n_local, "gen_tokens": gen,
        "max_num_seqs": mns, "max_num_batched_tokens": mbt,
        "total_gen_tokens": total_gen, "wall_s": round(wall, 3),
        "gen_tok_s": round(total_gen / wall, 2),
        "first_token_set": sorted(firsts)[:8], "num_distinct_first": len(firsts),
    }
    res.update(analyse(lg, wall, total_gen))
    res["unique_prompts"] = unique
    if res["prompt_cached_tokens"] > 0:
        print(f"[dp-bench r{rank}] WARNING: {res['prompt_cached_tokens']} prompt tokens "
              f"came from the prefix cache (computed={res['prefill_tokens']}); "
              f"prefill_tok_s is NOT a clean compute measurement. "
              f"Re-run with UNIQUE_PROMPTS=1", flush=True)
    print(f"[dp-bench r{rank}] {json.dumps(res)}", flush=True)
    with open(f"/tmp/bench_dp_rank{rank}.json", "w") as f:
        json.dump(res, f, indent=2)

    if os.environ.get("DUMP_TOKENS", "0") == "1":
        # exact greedy token ids, for cross-route output comparison (same prompts
        # every leg, so a differing sequence means the model itself differs)
        path = f"/tmp/bench_tokids_{tag}_rank{rank}.json"
        with open(path, "w") as f:
            json.dump([o.outputs[0].token_ids for o in outs], f)
        print(f"[dp-bench r{rank}] token ids -> {path}", flush=True)

    if os.environ.get("STEP_DUMP", "0") == "1":
        path = f"/tmp/bench_steps_{tag}_b{mns}_gen{gen}_rank{rank}.jsonl"
        with open(path, "w") as f:
            for s in lg.steps:
                f.write(json.dumps(s) + "\n")
        print(f"[dp-bench r{rank}] steps -> {path}", flush=True)

    time.sleep(5)  # let the engine drain / atexit timers flush


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------
def _merge(rank_res, key):
    """Merge a percentile dict across ranks by pooling the raw n (weighted mean
    for the means; the p-values we approximate by pooling max/p90 across ranks).
    We keep it simple: report the worst-case (max) of each rank's p90/p99 and
    the mean of the means, since a serving SLO is set by the slowest rank."""
    vals = [r[key] for r in rank_res if r.get(key, {}).get("n")]
    if not vals:
        return {"n": 0}
    return {
        "n": sum(v["n"] for v in vals),
        "mean": round(statistics.fmean(v["mean"] for v in vals), 3),
        "p50": round(max(v["p50"] for v in vals), 3),
        "p90": round(max(v["p90"] for v in vals), 3),
        "p99": round(max(v["p99"] for v in vals), 3),
        "max": round(max(v["max"] for v in vals), 3),
    }


def _run_one(dp, tag, n_total, gen, mns, base_env):
    for r in range(dp):
        try:
            os.remove(f"/tmp/bench_dp_rank{r}.json")
        except FileNotFoundError:
            pass
    from vllm.utils.network_utils import get_open_port

    env = dict(base_env)
    env.update({
        "VLLM_DP_SIZE": str(dp),
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(get_open_port()),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "MAX_NUM_SEQS": str(mns),
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("VISIBLE", "0,1"),
    })
    procs = []
    for r in range(dp):
        e = dict(env)
        e["VLLM_DP_RANK"] = str(r)
        e["VLLM_DP_RANK_LOCAL"] = str(r)
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker", str(r)], env=e))
    rcs = [p.wait() for p in procs]

    ranks = []
    for r in range(dp):
        try:
            with open(f"/tmp/bench_dp_rank{r}.json") as f:
                ranks.append(json.load(f))
        except FileNotFoundError:
            pass
    if len(ranks) != dp or any(rc != 0 for rc in rcs):
        print(f"[dp-bench] {tag} B={mns}: INCOMPLETE rcs={rcs} got={len(ranks)}/{dp}",
              flush=True)
        return None

    tot_gen = sum(r["total_gen_tokens"] for r in ranks)
    max_wall = max(r["wall_s"] for r in ranks)
    # aggregate decode throughput: pool tokens, divide by the slowest rank's
    # decode-phase duration (phase windows are per-rank, so this is a lower bound)
    dec_s = max(r.get("decode_s", 0) or 0 for r in ranks)
    dec_tok = sum(r.get("decode_tokens", 0) for r in ranks)
    pre_s = max(r.get("prefill_s", 0) or 0 for r in ranks)
    pre_tok = sum(r.get("prefill_tokens", 0) for r in ranks)

    res = {
        # --- keys kept for report3.py / analyze_dp2ep2.py compatibility ---
        "tag": tag, "dp": dp, "tp": int(os.environ.get("TP", "1")), "ep": dp,
        "n_seq": n_total, "gen_tokens": gen,
        "moonep": os.environ.get("MOONEP", "0"),
        "rank_walls": [r["wall_s"] for r in ranks],
        "total_gen_tokens": tot_gen, "max_rank_wall_s": max_wall,
        "agg_gen_tok_s": round(tot_gen / max_wall, 2),
        "rank_res": ranks,
        # --- new: the numbers that were previously inferred wrong ---
        "max_num_seqs": mns,
        "max_num_batched_tokens": ranks[0].get("max_num_batched_tokens"),
        "agg_decode_tok_s": round(dec_tok / dec_s, 2) if dec_s > 0 else 0.0,
        "agg_prefill_tok_s": round(pre_tok / pre_s, 2) if pre_s > 0 else 0.0,
        "decode_tokens": dec_tok, "prefill_tokens": pre_tok,
        "unique_prompts": ranks[0].get("unique_prompts"),
        "prompt_cached_tokens": sum(r.get("prompt_cached_tokens", 0) for r in ranks),
        "n_finished": sum(r.get("n_finished", 0) for r in ranks),
        "tpot_ms": _merge(ranks, "tpot_ms"),
        "itl_ms": _merge(ranks, "itl_ms"),
        "ttft_ms": _merge(ranks, "ttft_ms"),
        "step_ms": _merge(ranks, "step_ms"),
        "running": _merge(ranks, "running"),
        "n_decode_steps": max((r.get("n_decode_steps", 0) for r in ranks), default=0),
        "n_preempted": sum(r.get("n_preempted", 0) for r in ranks),
    }
    print(f"[dp-bench] {tag}: {json.dumps({k: res[k] for k in ('dp','tp','ep','n_seq','gen_tokens','max_num_seqs','rank_walls','total_gen_tokens','agg_gen_tok_s')})}", flush=True)
    return res


def main():
    dp = int(os.environ.get("DP", "2"))
    tag = os.environ.get("TAG", "mc2")
    gen = int(os.environ.get("GEN_TOKENS", "128"))
    sweep = [int(x) for x in os.environ.get("SWEEP", "").replace(" ", "").split(",") if x]
    base_mns = int(os.environ.get("MAX_NUM_SEQS", "256"))
    if not sweep:
        sweep = [base_mns]

    base_env = dict(os.environ)
    # pin concurrency to the batch we are sizing unless N_SEQ was given
    explicit_nseq = os.environ.get("N_SEQ")
    rows = []
    for mns in sweep:
        n_total = int(explicit_nseq) if explicit_nseq else mns
        e = dict(base_env)
        e["N_SEQ"] = str(n_total)
        e["MAX_NUM_SEQS"] = str(mns)
        print(f"[dp-bench] === max_num_seqs={mns} n_seq={n_total} gen={gen} ===", flush=True)
        r = _run_one(dp, tag, n_total, gen, mns, e)
        if r is not None:
            rows.append(r)

    if not rows:
        sys.exit(1)

    suffix = f"_b{sweep[0]}" if len(sweep) == 1 else "_sweep"
    out = f"/tmp/bench_dp2ep2_{tag}_g{gen}{suffix}.json"
    with open(out, "w") as f:
        json.dump(rows if len(rows) > 1 else rows[0], f, indent=2)
    print(f"[dp-bench] saved {out}", flush=True)

    if len(rows) > 1:
        print(f"\n{'B':>5} {'decode tok/s':>13} {'T_step ms':>10} {'p90':>8} "
              f"{'TPoT ms':>9} {'p90':>8} {'run':>6} {'prefill tok/s':>14}", flush=True)
        for r in rows:
            print(f"{r['max_num_seqs']:>5} {r['agg_decode_tok_s']:>13} "
                  f"{r['step_ms'].get('p50',0):>10} {r['step_ms'].get('p90',0):>8} "
                  f"{r['tpot_ms'].get('p50',0):>9} {r['tpot_ms'].get('p90',0):>8} "
                  f"{r['running'].get('p50',0):>6} {r['agg_prefill_tok_s']:>14}", flush=True)
        target = float(os.environ.get("TPOT_TARGET_MS", "20"))
        ok = [r for r in rows if r["tpot_ms"].get("p90", 1e9) <= target]
        if ok:
            best = max(ok, key=lambda r: r["max_num_seqs"])
            print(f"\n[dp-bench] TPoT p90 <= {target}ms holds up to "
                  f"max_num_seqs={best['max_num_seqs']} "
                  f"(p90={best['tpot_ms']['p90']}ms, decode={best['agg_decode_tok_s']} tok/s)",
                  flush=True)
        else:
            print(f"\n[dp-bench] no sweep point met TPoT p90 <= {target}ms", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        run_worker(int(sys.argv[2]))
    else:
        main()
