"""Online-serving T sweep: fire N concurrent requests, read TTFT + engine steps.

Companion to serve_moonep.sh. Each request draws a DIFFERENT GSM8K prompt, so
prompt lengths vary naturally (38..225 tokens, p50 69) instead of every request
replaying one fixed string. T is therefore MEASURED, never assumed:

    T (per-rank MoE tokens) = (prompt tokens this burst) / DP

taken from the server's own `vllm:prompt_tokens_total` delta, cross-checked
against a client-side tokenization of exactly the prompts sent.

Fresh prompts every burst
-------------------------
`vllm serve` runs with prefix caching on. Replaying the same prompts across
sweep points (or across `--reps`) would let the engine serve them from cache and
the prefill would look free. So a cursor walks a shuffled pool and every burst
takes the next slice; the requests within a burst are still distinct.

  python3 serve_probe.py --port 8000 --dp 2 --sweep 2,8,32,128,232,464,928
"""
import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import statistics
import time

import numpy as np

import aiohttp

PROMPT_FILE = "/home/tmp/prompts_gsm8k_raw.jsonl"
TOKENIZER = "/home/Qwen3-30B-A3B"
SHUFFLE_SEED = 20260920


def _load_pool(path=None):
    pool = []
    with open(path or PROMPT_FILE, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pool.append(json.loads(line)["text"])
    random.Random(SHUFFLE_SEED).shuffle(pool)
    return pool


_TOK = None


def _token_counts(prompts):
    """Token length of each prompt, so the client knows exactly what it sent."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer

            _TOK = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] tokenizer unavailable ({exc}); relying on server counts")
            _TOK = False
    if _TOK is False:
        return []
    return [len(_TOK(p).input_ids) for p in prompts]


async def _one(session, url, prompt, out, max_tokens=1, routes=None):
    body = {"model": "qwen3", "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body) as r:
            data = await r.json()
        # Append (latency, generated) -- with max_tokens=1 this latency IS TTFT;
        # with a long cap it is end-to-end and `generated` says how much decode
        # got mixed into the same engine steps.
        usage = data.get("usage") or {}
        out.append((time.perf_counter() - t0,
                    int(usage.get("completion_tokens") or 0)))
        if routes is not None:
            s = (data.get("choices") or [{}])[0].get("routed_experts")
            if s:
                routes.append(np.load(io.BytesIO(base64.b64decode(s))))
    except Exception as exc:  # noqa: BLE001
        out.append((float("nan"), 0))
        print(f"  request failed: {exc}")


async def _burst(base, prompts, max_tokens=1, routes=None, warm=0):
    """Fire `prompts` concurrently, optionally after warming the connection pool.

    Warm and burst share ONE ClientSession on purpose: aiohttp tears its
    connector (and every kept-alive socket) down when the session exits, so a
    warm-up in its own session would open sockets that the burst can never
    reuse. The warm requests are trivial 1-token ones -- the goal is sockets,
    not work -- and they are fired before the caller reads its metrics
    baseline, so their prompt tokens stay outside the measured window.
    """
    out = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        url = f"{base}/v1/completions"
        if warm > 0:
            junk = []
            await asyncio.gather(*[_one(s, url, "hi", junk, 1, None)
                                   for _ in range(warm)])
            await asyncio.sleep(0.3)     # let the warm-up drain out of the engine

        # Baseline AFTER the warm-up: its prompt tokens must not be charged to
        # the burst. Read from here (not from main) so the sockets above are
        # still inside this session.
        async with aiohttp.ClientSession() as probe:
            async with probe.get(f"{base}/metrics") as r:
                txt = await r.text()
        base_steps = _steps(txt)
        base_tok = _metric_sum(txt, "prompt_tokens_total")

        t0 = time.perf_counter()
        await asyncio.gather(*[_one(s, url, p, out, max_tokens, routes)
                               for p in prompts])
        return (time.perf_counter() - t0, out, base_steps, base_tok)


async def _metrics(base):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{base}/metrics") as r:
            return await r.text()


def _metric_sum(text, name):
    return sum(float(m.group(1))
               for m in re.finditer(rf'vllm:{name}\{{[^}}]*\}}\s+([0-9.e+]+)', text))


def _steps(text):
    """Sum of iteration_tokens_total_count over engines = engine steps taken."""
    return _metric_sum(text, "iteration_tokens_total_count")


def _imbalance(arrays, num_experts, world_size):
    """PER-LAYER load skew from the router's own expert assignment.

    Granularity is the whole point, and an earlier version of this got it wrong.
    MoonEP's planner runs PER LAYER, and one layer's expert histogram is exactly
    what it balances, so that is the unit to measure. Summing a request's counts
    across its 48 layers first produces a number no single layer ever
    experiences -- it averages the skew away (measured on the same requests:
    p50 0.52 per layer vs 0.18 aggregated). The spread ACROSS layers is signal,
    not noise: some layers really are far more skewed than others, and the worst
    ones are what cost time. Hence a distribution, not a median.

    Matches the offline MOONEP_BALANCE_STATS definition: rows_all is per layer
    and the ratio is max/mean over ranks.

    NOTE on what this does and does not mean. The field is the ROUTER output and
    both MoE routes share one router, so this is the skew of the WORKLOAD -- the
    input both routes face -- not how well a route copes with it. The difference
    between routes shows up in latency, not here.

      imb_*  max/mean of the rows each rank will receive, expert e -> rank e/(E/R)
      top_*  share of routed slots taken by the hottest 10% of experts
    """
    imb, top = [], []
    k = max(1, int(round(num_experts * 0.10)))
    for arr in arrays:
        for layer in range(arr.shape[1]):
            counts = np.bincount(arr[:, layer, :].reshape(-1).astype(np.int64),
                                 minlength=num_experts)
            total = counts.sum()
            if total == 0:
                continue
            rows = counts.reshape(world_size, num_experts // world_size).sum(axis=1)
            if rows.mean() > 0:
                imb.append(rows.max() / rows.mean())
            top.append(np.sort(counts)[::-1][:k].sum() / total)
    if not imb:
        return None
    imb.sort()
    top.sort()

    def q(v, p):
        return v[min(len(v) - 1, int(len(v) * p))]

    return {"imb_p50": q(imb, .5), "imb_p90": q(imb, .9), "imb_max": imb[-1],
            "top_p50": q(top, .5), "top_p90": q(top, .9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sweep", default="2,8,32,128,232,464,928",
                    help="request counts; T is measured, not derived from these")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--routes", action="store_true",
                    help="decode routed_experts and report load skew "
                         "(server must run with ROUTES=1; inflates TTFT)")
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--prompt-file", default=None,
                    help="prompt pool (one {\"text\": ...} per line); default is "
                         "the GSM8K pool. Use prompts_long.jsonl for a single "
                         "long context stitched from GSM8K.")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="1 = pure prefill, latency IS TTFT. A large value lets "
                         "generation run to EOS, which mixes decode into the same "
                         "engine steps (PD mixed); latency then becomes end-to-end "
                         "and is labelled E2E, and `gen` shows the decode share. "
                         "routed_experts covers prompt+generated either way.")
    ap.add_argument("--warm", type=int, default=0,
                    help="connections to pre-open before each measured burst")
    ap.add_argument("--offset", type=int, default=0,
                    help="start the pool cursor here. The cursor is per-process "
                         "and always starts at 0, so a second invocation replays "
                         "the first one's prompts -- and the server's prefix cache "
                         "serves them (measured 98%% of prompt tokens), which turns "
                         "the prefill you meant to time into a cache-hit step. "
                         "Advance the offset per invocation to keep bursts fresh.")
    ap.add_argument("--mm", type=float, default=None,
                    help="target peak-to-average ratio (max/mean over ranks) for "
                         "this burst. 1.0 = balanced/off. Written to the control "
                         "file the server's shim polls (MOONEP_ROUTER_SKEW_FILE), "
                         "so skew can be swept without restarting servers. "
                         "CEILING = R/w (w = MOONEP_SKEW_WIDTH, default 1): 2.0 at "
                         "EP=2, 4.0 at EP=4, 8.0 at EP=8. Use this, not a raw "
                         "fraction -- the same fraction means different skew at "
                         "different EP, which made earlier sweeps incomparable.")
    ap.add_argument("--skew-file", default="/home/tmp/router_skew.txt",
                    help="control file the servers poll for the target max/mean")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    if args.mm is not None:
        if not 1.0 <= args.mm <= 64.0:
            raise SystemExit(f"--mm must be >= 1.0 (1.0 = off), got {args.mm}")
        with open(args.skew_file, "w") as fh:
            fh.write(f"{args.mm:.4f}\n")
        ep = int(os.environ.get("MOONEP_SKEW_EP_SIZE", "2"))
        w = int(os.environ.get("MOONEP_SKEW_WIDTH", "1"))
        note = ""
        if args.mm > ep / w:
            note = f"  [warn] 超过可达上限 R/w={ep / w:.1f}，会被 clamp 到 frac=1"
        print(f"[probe] target max/mean <- {args.mm:.3f} ({args.skew_file})"
              f"  R={ep} w={w}{note}", flush=True)
        time.sleep(1.0)        # let both servers' poll interval elapse

    pool = _load_pool(args.prompt_file)
    src = args.prompt_file or PROMPT_FILE
    print(f"prompt pool: {len(pool)} prompts from {src}", flush=True)
    cursor = args.offset

    lat_name = "TTFT" if args.max_tokens == 1 else "E2E"
    if args.max_tokens != 1:
        print(f"[probe] max_tokens={args.max_tokens} -> generation runs to EOS; "
              f"latency is END-TO-END (TTFT + decode), not TTFT", flush=True)
    hdr = (f"{'N_req':>6s} {'T/rank':>7s} {'len_min':>8s} {'len_p50':>8s} "
           f"{'len_max':>8s} {lat_name + '_p50':>9s} {lat_name + '_max':>9s} "
           f"{'wall_s':>8s} {'steps':>6s} {'tok/step':>9s} {'gen':>5s}")
    if args.routes:
        hdr += (f" {'imb_p50':>8s} {'imb_p90':>8s} {'imb_max':>8s}"
                f" {'top_p50':>8s} {'top_p90':>8s}")
    print(hdr, flush=True)

    for n in (int(x) for x in args.sweep.split(",") if x):
        if cursor + n * args.reps > len(pool):
            print(f"{n:6d}  skip: prompt pool exhausted "
                  f"({cursor}+{n * args.reps} > {len(pool)})", flush=True)
            continue
        best = None
        for _ in range(args.reps):
            # fresh prompts each burst: no prefix-cache reuse across bursts
            prompts = pool[cursor:cursor + n]
            cursor += n

            routes = [] if args.routes else None
            # _burst warms the pool, then reads its own baseline (inside the
            # same session, so the warmed sockets are actually reused) and only
            # then fires the burst.
            wall, rec, m0, p0 = asyncio.run(
                _burst(base, prompts, args.max_tokens, routes, args.warm))
            m1 = _steps(asyncio.run(_metrics(base)))
            p1 = _metric_sum(asyncio.run(_metrics(base)), "prompt_tokens_total")

            gen = [g for x, g in rec if x == x]
            lat = [x for x, _ in rec if x == x]
            if not lat:
                continue
            lens = _token_counts(prompts) or [0] * n
            rec = {
                "ttft_p50": statistics.median(lat) * 1e3,
                "ttft_max": max(lat) * 1e3,
                "wall": wall,
                "steps": m1 - m0,
                # server-side ground truth, per rank
                "T_server": (p1 - p0) / args.dp,
                # what the client believes it sent, per rank
                "T_client": sum(lens) / args.dp,
                "len_min": min(lens), "len_p50": statistics.median(lens),
                "len_max": max(lens),
                "gen": statistics.mean(gen) if gen else 0,
            }
            if args.routes:
                stats = _imbalance(routes, args.num_experts, args.dp)
                if stats:
                    rec.update(stats)
            if best is None or rec["ttft_p50"] < best["ttft_p50"]:
                best = rec
        if best is None:
            print(f"{n:6d}  FAILED", flush=True)
            continue

        # prompt tokens (server-measured) + generated tokens (client-measured);
        # with a long max_tokens the generated side dominates.
        tok_total = best["T_server"] * args.dp + best.get("gen", 0) * n
        note = ""
        if best["T_client"] and abs(best["T_client"] - best["T_server"]) > args.dp:
            note = f"  [warn: client {best['T_client']:.0f} vs server {best['T_server']:.0f}]"
        row = (f"{n:6d} {best['T_server']:7.0f} {best['len_min']:8.0f} "
               f"{best['len_p50']:8.0f} {best['len_max']:8.0f} "
               f"{best['ttft_p50']:9.1f} {best['ttft_max']:9.1f} {best['wall']:8.3f} "
               f"{best['steps']:6.0f} "
               f"{tok_total / max(best['steps'], 1):9.1f} "
               f"{best.get('gen', 0):5.0f}")
        if args.routes:
            na = float('nan')
            row += (f" {best.get('imb_p50', na):8.3f} {best.get('imb_p90', na):8.3f}"
                    f" {best.get('imb_max', na):8.3f}"
                    f" {best.get('top_p50', na):8.3f} {best.get('top_p90', na):8.3f}")
        print(row + note, flush=True)


if __name__ == "__main__":
    main()
