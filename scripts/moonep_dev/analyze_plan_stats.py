"""How stable and predictable is the MoonEP planner's slot assignment?

This is the gate for the two overlap ideas:

  * speculative prefetch (start the remote GET one layer early, during
    attention): needs the *next* layer's (slot -> expert) mapping to be
    predictable, and a repair path when it is not;
  * a local cache of frequently mirrored experts: needs a slot to keep the
    same expert across steps, otherwise the cached bytes land in the wrong
    slot and are useless.

Metrics, all computed from MOONEP_PLAN_STATS=1 dumps (one row per layer, each a
list of B expert ids, -1 = unused slot):

  lag L          layers per step (auto-detected by the best self-match lag)
  exact@L        slot i holds the same expert as L layers ago (cache hit rate)
  expert@L       the *set* of mirrored experts is the same, slot index ignored
  exact@1        same slot/expert as the immediately preceding layer
  usage          mean number of used slots per layer
"""
import json
import statistics as st
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "planstat"


def load(tag):
    out = {}
    for r in range(4):
        try:
            out[r] = json.load(open(f"/tmp/moonep_planstats_{tag}_rank{r}.json"))
        except Exception:
            pass
    return out


def main():
    per_rank = load(TAG)
    print(f"ranks: {sorted(per_rank)}  layers: {[len(v) for _, v in sorted(per_rank.items())]}")
    for r, rows in sorted(per_rank.items()):
        n = len(rows)
        if n < 100:
            continue
        width = len(rows[0])

        def exact(a, b):
            return sum(1 for x, y in zip(a, b) if x == y and x >= 0) / max(
                sum(1 for x in a if x >= 0), 1)

        def expertset(a, b):
            sa, sb = {x for x in a if x >= 0}, {x for x in b if x >= 0}
            return len(sa & sb) / max(len(sa | sb), 1)

        candidates = range(40, 60)
        lag = max(candidates, key=lambda L: st.mean(
            [exact(rows[i], rows[i - L]) for i in range(L, min(n, 4000))]))
        epl = st.mean([exact(rows[i], rows[i - lag]) for i in range(lag, n)])
        epl_experts = st.mean([expertset(rows[i], rows[i - lag]) for i in range(lag, n)])
        e1 = st.mean([exact(rows[i], rows[i - 1]) for i in range(1, n)])
        used = st.mean([sum(1 for x in row if x >= 0) for row in rows])
        ever = st.mean([1.0 if any(x >= 0 for x in row) else 0.0 for row in rows])
        print(f"  r{r}: layers={n} B={width} detected_layers_per_step={lag} "
              f"exact@L={epl:.3f} expert@L={epl_experts:.3f} exact@1={e1:.3f} "
              f"used_slots/layer={used:.2f} layers_with_slots={ever:.2%}")

        # which experts would a cache hold?
        from collections import Counter
        c = Counter(x for row in rows for x in row if x >= 0)
        print(f"        top cached experts: {c.most_common(6)}")
        # per-slot stability: how often does a given slot keep its expert?
        slot_stable = []
        for s in range(width):
            seq = [row[s] for row in rows]
            pairs = [(seq[i], seq[i - lag]) for i in range(lag, n)]
            hit = sum(1 for a, b in pairs if a == b and a >= 0)
            tot = sum(1 for a, b in pairs if a >= 0 or b >= 0)
            if tot:
                slot_stable.append((s, hit / tot, sum(1 for a, _ in pairs if a >= 0) / len(pairs)))
        slot_stable.sort(key=lambda t: -t[1])
        print("        most stable slots (slot, keep_rate, used_rate): "
              + ", ".join(f"({s},{k:.2f},{u:.2f})" for s, k, u in slot_stable[:5]))


if __name__ == "__main__":
    main()
