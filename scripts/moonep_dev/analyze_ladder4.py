"""TTFT + per-layer communication time for the 4-card (one UB clique) ladder.

Companion to ``run_ladder_4card.sh``. Reads the two artifacts that script
leaves behind:

  /tmp/bench_dp2ep2_lad4_<rung>_<route>_g1_b256.json   TTFT, steps, T
  /home/tmp/lad4_<rung>_<route>.log                    [moonep-tstats] dump

Two accounting traps this script exists to avoid:

  * ``disp_buf`` is a self-referential accumulator (``_tstats("disp_buf", _t1)``
    measures from the previous call's ``_t1``), so it equals the whole dispatch
    and is NOT a sub-phase. Only ``disp_pre/disp_pad/disp_tpe/disp_lib/disp_post``
    decompose, and even those are nested inside ``dispatch`` -- do not sum them.
  * ``ag`` legs never enter moonep_comm, so they have no tstats at all. Compare
    them on TTFT and wall time; per-layer comm for the all-gather route needs
    msprof.

  python3 scripts/moonep_dev/analyze_ladder4.py
"""
import glob
import json
import re
import statistics

RUNGS = list("ABCDEFGH")
PROMPT = 69
DP = 4


def ttft():
    out = {}
    for f in glob.glob("/tmp/bench_dp2ep2_lad4_*_g1_b*.json"):
        d = json.load(open(f))
        m = re.match(r"lad4_([A-Z])_(slot|ag)$", d.get("tag", ""))
        if not m:
            continue
        rung, route = m.groups()
        r0 = (d.get("rank_res") or [{}])[0] or {}
        out[(rung, route)] = {
            "ttft": (d.get("ttft_ms") or {}).get("p50"),
            "wall": d.get("max_rank_wall_s"),
            "tok": d.get("prefill_tokens", 0),
            "steps": r0.get("n_prefill_steps"),
            "T": (d.get("n_seq", 0) // DP) * PROMPT,
        }
    return out


def tstats():
    """Per-phase host wall totals, slowest rank, from the MOONEP_TIME_STATS dump."""
    out = {}
    for f in glob.glob("/home/tmp/lad4_*.log"):
        m = re.search(r"lad4_([A-Z])_(slot|ag)\.log$", f)
        if not m:
            continue
        txt = open(f, errors="ignore").read()
        i = txt.find("[moonep-tstats")
        if i < 0:
            continue
        line = txt[i : txt.find("\n", i)]
        vals = {k: (int(c), float(t))
                for k, c, t in re.findall(r"(\w+): calls=(\d+) total=([0-9.]+)s", line)}
        if vals.get("dispatch", (0, 0))[0]:
            out[(m.group(1), m.group(2))] = vals
    return out


def main():
    t, s = ttft(), tstats()

    print("=" * 96)
    print("TTFT (ms, p50) -- 4 cards, one UB clique, no cross-socket hop")
    print("=" * 96)
    print(f"{'rung':>4s} {'T':>6s} {'steps':>6s} {'slot':>10s} {'ag':>10s} "
          f"{'slot/ag':>9s} {'slot wall':>10s} {'ag wall':>9s}")
    for r in RUNGS:
        a, b = t.get((r, "slot"), {}), t.get((r, "ag"), {})
        if not a and not b:
            continue
        sa, sb = a.get("ttft"), b.get("ttft")
        ratio = f"{sa / sb:.2f}" if sa and sb else "-"
        print(f"{r:>4s} {a.get('T', b.get('T', 0)):6d} "
              f"{a.get('steps', b.get('steps', 0)):6} "
              f"{sa or 0:10.1f} {sb or 0:10.1f} {ratio:>9s} "
              f"{a.get('wall', 0):10.3f} {b.get('wall', 0):9.3f}")

    print()
    print("=" * 96)
    print("Communication time per layer (ms, host wall, slowest rank) -- slot route")
    print("=" * 96)
    if not s:
        print("  no [moonep-tstats] lines found")
        return
    print(f"{'rung':>4s} {'calls':>6s} {'dispatch':>9s} {'disp_lib':>9s} "
          f"{'combine':>8s} {'prefetch':>9s} | {'total':>7s}  {'x48 layers':>10s}")
    for r in RUNGS:
        v = s.get((r, "slot"))
        if not v:
            continue
        n = v["dispatch"][0]
        disp = v["dispatch"][1] / n * 1e3
        lib = v.get("disp_lib", (0, 0))[1] / n * 1e3
        cb = v.get("combine", (0, 0))[1] / n * 1e3
        pf = v.get("prefetch", (0, 0))[1] / n * 1e3
        print(f"{r:>4s} {n:6d} {disp:9.3f} {lib:9.3f} {cb:8.3f} {pf:9.3f} | "
              f"{disp + cb + pf:7.3f}  {(disp + cb + pf) * 48:10.1f}")


if __name__ == "__main__":
    main()
