"""Is a burst split into several engine steps because of ARRIVAL or CAPACITY?

Samples /metrics every ~5 ms during one burst:
  vllm:num_requests_waiting      queued, not yet scheduled
  vllm:num_requests_running      in the current batch
  vllm:iteration_tokens_total_count   engine steps so far (all engines)

Reading:
  * If `waiting` is already at N when the first step fires, the whole burst was
    there -- any split after that is the SCHEDULER's doing (levers: MBT,
    max_num_seqs, KV blocks).
  * If `waiting` climbs as the burst lands (and steps tick up in between), the
    split is the CLIENT's arrival spread -- the engine does not wait for a batch
    to assemble, so no server flag can fix it. Levers are client-side: pre-open
    the connections and send faster.
  * If the first step's token count is already near MBT, it is a genuine
    CAPACITY limit and MBT is the lever.

  python3 why_steps.py --port 8000 --n 128 --dp 2
"""
import argparse
import asyncio
import json
import re
import random
import time

import aiohttp

PROMPT_FILE = "/home/tmp/prompts_gsm8k_raw.jsonl"


def _per_engine(text, name):
    """{engine_label: value} -- per-engine, so uneven DP load is visible."""
    out = {}
    for m in re.finditer(rf'vllm:{name}\{{([^}}]*)\}}\s+([0-9.e+]+)', text):
        e = re.search(r'engine="(\d+)"', m.group(1))
        out[int(e.group(1)) if e else -1] = float(m.group(2))
    return out


def _parse(text):
    run = _per_engine(text, "num_requests_running")
    st = _per_engine(text, "iteration_tokens_total_count")
    w = sum(_per_engine(text, "num_requests_waiting").values())
    return (w, run, st)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--interval", type=float, default=0.005)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    pool = [json.loads(l)["text"] for l in open(PROMPT_FILE) if l.strip()]
    random.Random(7).shuffle(pool)
    prompts = pool[:args.n]

    samples = []

    async def sampler(sess, stop):
        while not stop.is_set():
            try:
                async with sess.get(f"{base}/metrics") as r:
                    w, run, st = _parse(await r.text())
                samples.append((time.perf_counter(), w, run, st))
            except Exception:
                pass
            await asyncio.sleep(args.interval)

    async def fire(sess, p):
        async with sess.post(f"{base}/v1/completions",
                             json={"model": "qwen3", "prompt": p,
                                   "max_tokens": 1, "temperature": 0.0}) as r:
            await r.json()

    conn = aiohttp.TCPConnector(limit=0)
    to = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(connector=conn, timeout=to) as s:
        stop = asyncio.Event()
        task = asyncio.create_task(sampler(s, stop))
        await asyncio.sleep(0.1)
        t0 = time.perf_counter()
        await asyncio.gather(*[fire(s, p) for p in prompts])
        t1 = time.perf_counter()
        stop.set()
        await task

    print(f"N={args.n} dp={args.dp}  burst wall={(t1 - t0) * 1e3:.0f} ms  "
          f"samples={len(samples)}")
    engs = sorted({e for *_, run, _ in samples for e in run})
    hdr = " ".join(f"{'run_e' + str(e):>8s}" for e in engs)
    print(f"{'t_ms':>7s} {hdr} {'steps_e0':>9s} {'steps_e1':>9s} {'sum':>6s}")
    first = samples[0][0]
    prev = None
    for ts, _w, run, st in samples:
        key = (tuple(sorted(run.items())), tuple(sorted(st.items())))
        if key != prev:
            runs = " ".join(f"{run.get(e, 0):8.0f}" for e in engs)
            s0, s1 = st.get(0, 0), st.get(1, 0)
            print(f"{(ts - first) * 1e3:7.1f} {runs} {s0:9.0f} {s1:9.0f} "
                  f"{s0 + s1:6.0f}")
            prev = key
    print(f"peak waiting = {max(s[1] for s in samples):.0f} / N={args.n}")
    first_st = samples[0][3]
    last_st = samples[-1][3]
    for e in engs:
        print(f"engine {e}: steps = {last_st.get(e, 0) - first_st.get(e, 0):.0f}")


if __name__ == "__main__":
    asyncio.run(main())
