"""Long-prefill A/B: MC2 vs MoonEP-owner vs MoonEP-slot on Qwen3-30B-A3B.

Env: MOONEP=0/1, MOONEP_SLOT_MODE=1, MOONEP_DISABLE_UDMA=1 for moonep legs.
Measures prefill seconds (max_tokens=1) for prompt lengths 1000/30000/40960.
"""
import json
import os
import time

import torch  # noqa: F401
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/home/Qwen3-30B-A3B")
MOONEP = os.environ.get("MOONEP", "0") == "1"
SLOT = os.environ.get("MOONEP_SLOT_MODE", "0") == "1"
TAG = f"{'moonep-slot' if SLOT else ('moonep' if MOONEP else 'mc2')}"
OUT = f"/tmp/bench_long_{TAG}.json"
LENGTHS = [int(os.environ.get("BENCH_LEN", "30000"))]

PARA = (
    "The history of computing spans more than a century, from mechanical "
    "calculators through vacuum tubes and transistors to modern microprocessors. "
    "Each generation brought dramatic changes in speed, size, and accessibility. "
    "Today, computers are embedded in almost every aspect of daily life. "
)


def main():
    print(f"[bench] TP={os.environ.get('TP')} ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
          f"MOONEP={os.environ.get('MOONEP')} UDMA_OFF={os.environ.get('MOONEP_DISABLE_UDMA')}", flush=True)
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        enable_expert_parallel=True,
        tensor_parallel_size=int(os.environ.get("TP", "4")),
        max_model_len=40960,
        max_num_batched_tokens=int(os.environ.get("MBT", "5120")),
        gpu_memory_utilization=float(os.environ.get("GMU", "0.80")),
        additional_config={"enable_moonep": MOONEP},
    )
    tok = llm.get_tokenizer()
    results = {}
    for L in LENGTHS:
        text = PARA * (L // 20 + 2)
        ids = tok(text).input_ids[:L]
        prompt = tok.decode(ids)
        params = SamplingParams(temperature=0.0, max_tokens=1)
        t0 = time.time()
        outs = llm.generate([prompt], params)
        dt = time.time() - t0
        first = outs[0].outputs[0].token_ids[0]
        results[str(L)] = {
            "prefill_s": round(dt, 3),
            "prefill_tok_s": round(L / dt, 1),
            "first_token": first,
        }
        print(f"[{TAG}] L={L}: {results[str(L)]}", flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{TAG}] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
