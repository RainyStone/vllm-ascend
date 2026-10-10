"""Hit rate of an *early* plan predicted during attention (MOONEP_EARLY_PLAN=1).

Idea under test: this layer's MoE input (post-attention) is close to the
previous layer's output (pre-attention) because of the residual stream, so
running this layer's router on the pre-attention hidden -- i.e. while this
layer's attention is still computing -- should predict which experts will be hot,
hence which experts the planner will mirror (the "redundant experts"), giving the
prefetch something to do off the critical path.

Inputs (per rank):
  /tmp/moonep_earlyplan_<tag>_rank<r>.json   {"early": [hist...], "real": [hist...]}
  /tmp/moonep_planstats_<tag>_rank<r>.json   [[experts_to_copy[rank]] per layer]

Reported per metric: predicted-from-early vs predicted-from-the-real-routing
(ceiling), plus two baselines: the previous step's real routing (lag = layers per
step) and the previous layer's real routing.
"""
import json
import sys
from collections import Counter

TAG = sys.argv[1] if len(sys.argv) > 1 else "earlyplan"
E, R = 128, 4
EPN = E // R


def load(name):
    out = {}
    for r in range(R):
        try:
            out[r] = json.load(open(f"/tmp/{name}_{TAG}_rank{r}.json"))
        except Exception:
            pass
    return out


def main():
    early = load("moonep_earlyplan")
    plans = load("moonep_planstats")
    if not early:
        print("no early-plan dumps found")
        return
    n = min(len(v["early"]) for v in early.values())
    n = min(n, min(len(v["real"]) for v in early.values()))
    print(f"ranks={sorted(early)} layers={n}")

    def global_hist(key, i):
        tot = Counter()
        for r, d in early.items():
            h = d[key][i]
            for e, c in enumerate(h):
                if c:
                    tot[e] += c
        return tot

    def hot_remote(hist, rank, k=1):
        return [e for e, _ in hist.most_common(20)
                if not (rank * EPN <= e < (rank + 1) * EPN)][:k]

    hits = {m: Counter() for m in ("early", "real", "prev_step", "prev_layer")}
    totals = Counter()
    for r in sorted(early):
        rows = plans.get(r, [])
        for i in range(n):
            if i < 48 or i >= len(rows):
                continue                                    # skip warmup / prefill edges
            target = {e for e in rows[i] if e >= 0}
            if not target:
                continue
            totals["layers"] += 1
            cand = {
                "early": hot_remote(global_hist("early", i), r, 1),
                "real": hot_remote(global_hist("real", i), r, 1),
                "prev_step": hot_remote(global_hist("real", i - 48), r, 1)
                if i >= 48 else [],
                "prev_layer": hot_remote(global_hist("real", i - 1), r, 1) if i else [],
            }
            for name, c in cand.items():
                if c and c[0] in target:
                    hits[name][r] += 1
    print(f"layers with a real mirror: {totals['layers']}")
    for name in ("early", "real", "prev_step", "prev_layer"):
        rates = [hits[name][r] / max(1, sum(1 for _ in range(totals['layers']))) for r in sorted(early)]
        tot = sum(hits[name].values()) / max(1, totals["layers"])
        print(f"  {name:11s} top-1 remote expert matches the plan: {tot:.3f} "
              f"(per rank {[f'{x:.2f}' for x in rates]})")


if __name__ == "__main__":
    main()
