"""Build a prompt pool that AMPLIFIES cross-rank expert-load imbalance.

Why: bench_dp's synthetic path repeats one fixed paragraph, so every rank's
router sees nearly identical input and the per-rank expert histograms match --
cross-rank imbalance is ~0 and MoonEP's only selling point (load balancing)
has nothing to do. Measured that way MoonEP looks 2.5x slower than all-gather
at prefill (TTFT 3599 vs 1445 ms), which says nothing about the mechanism.

This pool gives each DP rank its own *domain* instead, with lengths held
equal so the difference is pure expert affinity, not token count:

    index 0 -> python source        index 4 -> math word problems
    index 1 -> C++ / AscendC        index 5 -> shell + system text
    index 2 -> Chinese prose        index 6 -> JSON / config dumps
    index 3 -> markdown / tables    index 7 -> English prose

bench_dp maps rank r to pool[off + i] with off = rank * n_local, so with
N_SEQ = DP every rank draws exactly one entry and the domains stay disjoint.

  python3 scripts/moonep_dev/make_imbalance_prompts.py --out /home/tmp/prompts_imbalance.jsonl
"""
import argparse
import json
import os
import sys

from transformers import AutoTokenizer

# (name, list of source files) -- first readable source wins, repeated to length.
DOMAINS = [
    ("python", ["/home/vllm-ascend-moonep/vllm_ascend/ops/fused_moe/moonep_comm.py",
                "/home/vllm-ascend-moonep/vllm_ascend/ops/fused_moe/moonep_balance.py"]),
    ("cpp", ["/home/ascend-moonep/kernels/planning.cpp",
             "/home/ascend-moonep/kernels/moonep_dispatch.cpp"]),
    ("chinese", ["/home/vllm-ascend-moonep/docs/dev_results_2026-09-16.md"]),
    # distinct source from `chinese` on purpose: two ranks sharing a corpus would
    # share a router histogram and halve the imbalance signal we are trying to
    # create.
    ("report", ["/home/model_analysis/HY3.0_memory_analysis_W8A8C8_DP1TP4_4NPU.html",
                "/home/model_analysis/Hy3-analysis-report.html"]),
    ("math", []),                        # filled from the GSM8K pool
    ("shell", ["/tmp/run_multi.sh", "/tmp/run_2card.sh",
               "/etc/profile", "/etc/bash.bashrc"]),
    ("json", []),                        # filled from benchmark result JSONs
    ("english", []),                     # filled from the PARA constant
]

PARA = (
    "The history of computing spans more than a century, from mechanical "
    "calculators through vacuum tubes and transistors to modern microprocessors. "
)


def read_sources(paths):
    parts = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                parts.append(fh.read())
        except OSError:
            continue
    return "\n".join(parts)


def math_text():
    out = []
    try:
        with open("/home/tmp/prompts_gsm8k_raw.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line)["text"])
    except OSError:
        pass
    return "\n\n".join(out)


def json_text():
    import glob
    out = []
    for f in sorted(glob.glob("/tmp/bench_dp*.json"))[:6]:
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(fh.read())
        except OSError:
            continue
    return "\n".join(out)


def repeat_to_tokens(tok, text, target):
    """Repeat `text` until it tokenizes to roughly `target` tokens."""
    if not text.strip():
        return None
    ids = tok(text).input_ids
    if not ids:
        return None
    reps = max(1, target // len(ids) + 1)
    return tok.decode((ids * reps)[:target])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/tmp/prompts_imbalance.jsonl")
    ap.add_argument("--tokens", type=int, default=11000,
                    help="target token length per prompt (kept equal on purpose)")
    ap.add_argument("--tokenizer", default="/home/Qwen3-30B-A3B")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    bodies = {
        "math": math_text(),
        "json": json_text(),
        "english": PARA,
    }
    for name, paths in DOMAINS:
        if name not in bodies:
            bodies[name] = read_sources(paths)

    rows = []
    for name, _ in DOMAINS:
        prompt = repeat_to_tokens(tok, bodies.get(name, ""), args.tokens)
        if prompt is None:
            print(f"  [skip] {name}: no source text", file=sys.stderr)
            continue
        n = len(tok(prompt).input_ids)
        rows.append((name, n, prompt))
        print(f"  {name:9s} {n:6d} tokens")

    with open(args.out, "w", encoding="utf-8") as fh:
        for _, _, prompt in rows:
            fh.write(json.dumps({"text": prompt}) + "\n")
    print(f"wrote {len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
