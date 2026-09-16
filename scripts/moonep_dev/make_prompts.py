"""Build the GSM8K prompt file used by the dev-branch measurements.

One canonical GSM8K question per line (zero-shot, no CoT in the prompt), shuffled
with a fixed seed so every leg sees the same prompts in the same order:

    python3 make_prompts.py [--gsm8k /home/tmp/gsm8k/gsm8k] [--out /home/tmp/prompts_gsm8k_raw.jsonl]

Measured (Qwen3-30B-A3B tokenizer, 300 samples): prompt p50=69, p90=107,
max=225 tokens; the CoT answers are p50=110, p90=196, max=413 tokens.

bench_dp.py reads it via PROMPT_FILE; each rank takes PROMPT_OFFSET = rank *
n_local consecutive lines, so the four ranks never share a prompt.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", default="/home/tmp/gsm8k/gsm8k")
    ap.add_argument("--out", default="/home/tmp/prompts_gsm8k_raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for name in ("test.jsonl", "train.jsonl"):
        for line in open(f"{args.gsm8k}/{name}"):
            line = line.strip()
            if line:
                rows.append(json.loads(line)["question"].strip())
    random.seed(args.seed)
    random.shuffle(rows)

    with open(args.out, "w") as fh:
        for q in rows:
            fh.write(json.dumps(
                {"text": q + "\n\nSolve it step by step and give the final numeric answer."}) + "\n")
    print(f"wrote {len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
