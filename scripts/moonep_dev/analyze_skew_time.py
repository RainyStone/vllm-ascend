"""Does intra-rank group skew cost grouped-GEMM time?

x: the *own* group skew of a rank in a layer = max(group)/mean(group)
y: that layer's synchronised MLP wall time

Confound: layer time also scales with total rows. Control by restricting to the
decode plateau (a narrow row band) where the total is nearly constant.
"""
import json
import statistics as st
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "rowtime_mc2"


def load_rows(tag):
    out = {}
    for r in range(4):
        try:
            out[r] = json.load(open(f"/tmp/moe_rows_{tag}_rank{r}.json"))
        except Exception:
            pass
    return out


def load_times(tag):
    try:
        return json.load(open(f"/tmp/moe_rowtimes_{tag}.json"))
    except Exception:
        return []


def corr(xs, ys):
    if len(xs) < 10:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


rows = load_rows(TAG)
times = dict(load_times(TAG))          # call index -> layer time (one rank's file)
print(f"rows: {sorted(rows)}  times: {len(times)}")

best = None
for r, d in sorted(rows.items()):
    sums, mx, nz = d["sum"], d["max"], d["nz"]
    n = min(len(sums), max(times) + 1 if times else 0)
    lo, hi = st.median(sums[:n]) * 0.9, st.median(sums[:n]) * 1.1   # decode plateau
    xs, ys, raw = [], [], []
    for i in range(n):
        if i not in times or not nz[i]:
            continue
        mean_g = sums[i] / nz[i]
        if mean_g <= 0:
            continue
        sk = mx[i] / mean_g
        raw.append((sk, times[i], sums[i]))
        if lo <= sums[i] <= hi:                  # near-constant total rows
            xs.append(sk)
            ys.append(times[i])
    if len(xs) < 50:
        continue
    c = corr(xs, ys)
    slope = (st.mean([y for x, y in zip(xs, ys) if x > 1.5] or [0]) -
             st.mean([y for x, y in zip(xs, ys) if x < 1.2] or [0]))
    print(f"  r{r}: plateau layers={len(xs)} skew range {min(xs):.2f}-{max(xs):.2f} "
          f"time {min(ys)*1e3:.2f}-{max(ys)*1e3:.2f}ms corr={c:+.3f} "
          f"mean_time(skew>1.5)={st.mean([y for x, y in zip(xs, ys) if x > 1.5] or [float('nan')])*1e3:.2f}ms "
          f"(skew<1.2)={st.mean([y for x, y in zip(xs, ys) if x < 1.2] or [float('nan')])*1e3:.2f}ms")
    if best is None or abs(c) > abs(best[1]):
        best = (r, c, len(xs), slope)

if best:
    print(f"\nbest pairing: rank {best[0]} corr={best[1]:+.3f} on {best[2]} plateau layers")
    print(f"time difference (skew>1.5 minus skew<1.2) = {best[3]*1e3:+.3f} ms")
