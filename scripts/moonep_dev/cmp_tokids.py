"""Compare greedy token ids across benchmark legs (same prompts every leg).

Usage: python3 /tmp/cmp_tokids.py tok_mc2 tok_owner tok_slot
Reports, per rank, how many prompt sequences are bit-identical between the
first leg (reference) and each other leg, and where the first divergence is.
"""
import json
import sys
from collections import Counter


def load(tag: str):
    out = {}
    for rank in range(8):
        try:
            with open(f"/tmp/bench_tokids_{tag}_rank{rank}.json") as f:
                out[rank] = json.load(f)
        except FileNotFoundError:
            pass
    return out


def main(tags):
    legs = {t: load(t) for t in tags}
    for t, d in legs.items():
        print(f"{t}: ranks={sorted(d)} prompts/rank={[len(v) for _, v in sorted(d.items())]}")

    ref_tag = tags[0]
    ref = legs[ref_tag]
    for tag in tags[1:]:
        leg = legs[tag]
        print(f"\n=== {ref_tag} vs {tag}")
        for rank in sorted(set(ref) & set(leg)):
            a, b = ref[rank], leg[rank]
            n = min(len(a), len(b))
            same_full = sum(1 for i in range(n) if a[i] == b[i])
            same_first = sum(1 for i in range(n) if a[i] and b[i] and a[i][0] == b[i][0])
            div = next((i for i in range(n) if a[i] != b[i]), None)
            pos = None
            if div is not None:
                x, y = a[div], b[div]
                pos = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]),
                           min(len(x), len(y)))
            lens_a = Counter(len(x) for x in a)
            lens_b = Counter(len(x) for x in b)
            print(f"  r{rank}: n={n} identical={same_full} ({same_full / max(n, 1):.0%}) "
                  f"same_first_tok={same_first} first_diverge_prompt={div} at_pos={pos}")
            print(f"        gen len hist: {ref_tag}={dict(sorted(lens_a.items()))} "
                  f"{tag}={dict(sorted(lens_b.items()))}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["tok_mc2", "tok_owner", "tok_slot"])
