"""Aggregate a torch_npu profiler dump by kernel: where does the device time go?

Usage: python3 analyze_profile.py /tmp/prof_mc2 /tmp/prof_owner [--top 14]

Reads every <worker>/<...>_ascend_pt/kernel_details.csv under each root, sums the
device duration per kernel name, and prints the top kernels per route plus a
coarse grouping (attention / moe-gemm / comm / other).
"""
import collections
import csv
import glob
import sys

GROUPS = (
    ("moe-comm-moonep", ("moonep", "prefetch", "planning", "dispatch_weight", "combine_weight")),
    ("moe-comm-mc2", ("MoeDistribute", "moe_distribute", "AlltoAll", "alltoall")),
    ("moe-gemm", ("GroupedMatmul", "grouped_matmul", "GMM", "SwiGLU", "swiglu")),
    ("attention", ("FlashAttention", "flash_attention", "FIA", "attention", "PromptFlash")),
    ("collective", ("AllReduce", "AllGather", "ReduceScatter", "Broadcast", "Hccl", "hccl")),
    ("norm/rope/act", ("RmsNorm", "RMSNorm", "Rope", "rope", "Add", "Mul", "Silu", "Sigmoid")),
    ("copy/mem", ("copy", "Copy", "Memcpy", "Transpose", "Cast", "Contiguous")),
)


COL_ALIASES = {
    "name": ("Name", "Kernel Name", "kernel_name", "Op Name", "op_name", "Task Name"),
    "dur": ("Duration(us)", "Duration(μs)", "duration", "Task Duration(us)",
            "task_duration", "Total Time(us)", "aicore_time(us)", "Duration(us)"),
}


def _col(row, key):
    for cand in COL_ALIASES[key]:
        if cand in row and row[cand] not in (None, ""):
            return row[cand]
    return None


def load(root):
    """Accepts both the vLLM/torch_npu profiler layout (kernel_details.csv) and
    an msprof dump (mindstudio_profiler_output/op_summary_*.csv)."""
    tot, cnt = collections.Counter(), collections.Counter()
    files = (glob.glob(f"{root}/**/kernel_details.csv", recursive=True)
             + glob.glob(f"{root}/**/op_summary*.csv", recursive=True))
    for f in files:
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                name = _col(row, "name") or "?"
                raw = _col(row, "dur") or "0"
                try:
                    dur = float(str(raw).strip() or 0)
                except ValueError:
                    continue
                tot[name] += dur
                cnt[name] += 1
    return tot, cnt, len(files)


def group_of(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def main():
    roots = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 14
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    for root in roots:
        tot, cnt, nfiles = load(root)
        if not tot:
            print(f"{root}: no kernel_details.csv found ({nfiles} dirs scanned)")
            continue
        total_us = sum(tot.values())
        print(f"\n=== {root}: {len(tot)} distinct kernels, {nfiles} trace dirs, "
              f"{total_us / 1e6:.2f} s device time")
        g = collections.Counter()
        for name, us in tot.items():
            g[group_of(name)] += us
        print("  by group:")
        for grp, us in g.most_common():
            print(f"    {grp:18s} {us / 1e6:8.3f} s  {us / total_us:6.1%}")
        print(f"  top {top} kernels:")
        for name, us in tot.most_common(top):
            print(f"    {us / 1e6:8.3f} s  n={cnt[name]:6d}  {name[:78]}")


if __name__ == "__main__":
    main()
