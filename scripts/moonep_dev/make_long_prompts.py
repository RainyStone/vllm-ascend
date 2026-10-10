"""Concatenate GSM8K questions into prompts of an EXACT token length.

The dev prompt pool is one question per line (~69 tokens, p50). Comparing T
ladders wants a single long context instead, so this walks the GSM8K pool and
joins questions until the target length is reached, cutting the last question
mid-way if needed.

The length is verified by re-encoding the decoded text: decode->encode does not
always round-trip, so the generator trims token by token until it does. Without
that check the prompt the server sees is not the length you asked for.

  python3 make_long_prompts.py --tokens 16592 --count 4 --out /home/tmp/prompts_long.jsonl
"""
import argparse
import json
import random

from transformers import AutoTokenizer

GSM8K = "/home/tmp/prompts_gsm8k_raw.jsonl"
TOKENIZER = "/home/Qwen3-30B-A3B"


def build(tok, pool, target, seed):
    """One prompt whose RE-ENCODED token count is exactly `target`.

    decode->encode is not a round trip (tokenizer merges/splits at boundaries),
    so the id list that is `target` long can decode to text that re-encodes to a
    different length. Re-encoded length is monotone in how much of the id stream
    is kept, so binary-search the prefix instead of trimming one token at a time
    (which can never come back up to `target`).
    """
    items = list(pool)
    random.Random(seed).shuffle(items)
    ids = []
    want = target + 2048          # spare, so the search always has room
    for text in items:
        piece = tok(text)["input_ids"]
        if not piece:
            continue
        room = want - len(ids)
        if room <= 0:
            break
        ids.extend(piece[:room])

    def reencoded(n):
        return len(tok(tok.decode(ids[:n]))["input_ids"])

    lo, hi = 1, len(ids)
    while lo <= hi:
        mid = (lo + hi) // 2
        n = reencoded(mid)
        if n == target:
            return tok.decode(ids[:mid])
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # the search is on a monotone quantity but not necessarily step-free;
    # scan a small window around the landing point before giving up.
    for mid in range(max(1, lo - 40), min(len(ids), lo + 40)):
        if reencoded(mid) == target:
            return tok.decode(ids[:mid])
    raise RuntimeError(f"no prefix re-encodes to exactly {target} tokens "
                       f"(stream={len(ids)}, search landed at lo={lo})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16592)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--out", default="/home/tmp/prompts_long.jsonl")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--gsm8k", default=GSM8K)
    ap.add_argument("--seed", type=int, default=1000,
                    help="base seed; prompt i shuffles the pool with seed+i. "
                         "Change it to mint a pool whose prompts share no "
                         "prefix with an earlier pool -- the server's prefix "
                         "cache would otherwise serve them from cache.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    pool = [json.loads(l)["text"] for l in open(args.gsm8k, encoding="utf-8")
            if l.strip()]
    print(f"pool: {len(pool)} GSM8K questions", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.count):
            text = build(tok, pool, args.tokens, seed=args.seed + i)
            n = len(tok(text)["input_ids"])
            fh.write(json.dumps({"text": text}) + "\n")
            assert n == args.tokens, (n, args.tokens)
            print(f"  prompt {i}: {n} tokens "
                  f"(head {text[:48]!r}...)", flush=True)
    print(f"wrote {args.count} prompts -> {args.out}")


if __name__ == "__main__":
    main()
