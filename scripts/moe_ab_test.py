"""MoE A/B test: baseline EP comm vs MoonEP EP comm on the same model.

Env:
  MODEL=<dir>          model path (default /home/Qwen3-30B-A3B)
  MOONEP=1             enable_moonep leg (needs enable_expert_parallel)
  NOEP=1               baseline leg WITHOUT expert parallelism (all-gather MoE)
  TP=<int>             tensor parallel size (EP spans the TP group on Ascend)
  DEVICES=...          passed via ASCEND_RT_VISIBLE_DEVICES outside
Writes /tmp/ab_{moonep|baseline}_{tag}.json
"""
import json
import os

import torch  # noqa: F401
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/home/Qwen3-30B-A3B")
TAG = os.path.basename(MODEL.rstrip("/"))
MOONEP = os.environ.get("MOONEP", "0") == "1"
NOEP = os.environ.get("NOEP", "0") == "1"
OUT = f"/tmp/ab_{'moonep' if MOONEP else 'baseline'}_{TAG}.json"

PROMPTS = [
    "Hello! My name is",
    "The future of AI is",
]


def main():
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        enable_expert_parallel=not NOEP,
        tensor_parallel_size=int(os.environ.get("TP", "4")),
        max_model_len=4096,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.90,
        additional_config={"enable_moonep": MOONEP},
    )
    params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    outputs = llm.generate(PROMPTS, params)
    results = []
    for o in outputs:
        first = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        results.append({
            "prompt": o.prompt,
            "token_ids": o.outputs[0].token_ids,
            "text": o.outputs[0].text,
            "top5_first": [(t, round(float(lp.logprob), 4)) for t, lp in first.items()],
        })
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{'moonep' if MOONEP else 'baseline'}] saved {OUT}", flush=True)
    for r in results:
        print(f"  {r['prompt'][:40]!r} -> {len(r['token_ids'])} tok: {r['text'][:70]!r}", flush=True)


if __name__ == "__main__":
    main()
