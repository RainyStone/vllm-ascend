#!/usr/bin/env python3
"""CPU-only full-prompt token-length audit; never starts a model service."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
DATASET = WORKSPACE / "benchmark/ais_bench/datasets/LongBench-v2/data.json"
MODEL = Path("/mnt/share/weight/DeepSeek-V4-Flash")
TEMPLATE_FILE = WORKSPACE / "benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tree = ast.parse(TEMPLATE_FILE.read_text())
    prompts = [n.value.value for n in ast.walk(tree)
               if isinstance(n, ast.keyword) and n.arg == "prompt"
               and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    assert len(prompts) == 1
    template = prompts[0]
    rows = json.loads(DATASET.read_text())
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    records = []
    with (ROOT / "length_audit.csv").open("w", newline="") as out:
        fields = ["index", "id", "domain", "length_label", "prompt_tokens", "prompt_characters"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            prompt = template.format(**row)
            count = len(tok.encode(prompt, add_special_tokens=True))
            record = dict(index=i, id=row["_id"], domain=row["domain"],
                          length_label=row["length"], prompt_tokens=count,
                          prompt_characters=len(prompt))
            records.append(record)
            writer.writerow(record)
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"AUDIT {i + 1}/{len(rows)}", flush=True)
    values = sorted(r["prompt_tokens"] for r in records)
    config = json.loads((MODEL / "config.json").read_text())
    summary = dict(samples=len(rows), minimum=values[0], median=values[len(values)//2],
                   p90=values[int(len(values)*0.9)], maximum=values[-1],
                   model_max_position_embeddings=config["max_position_embeddings"],
                   counts_at_input_caps={str(cap):sum(v <= cap for v in values)
                       for cap in [8192, 16384, 32768, 65536, 131072, 1048575]},
                   template=template, tokenizer_add_special_tokens=True,
                   dataset_sha256=hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                   tokenizer_sha256=hashlib.sha256((MODEL / "tokenizer.json").read_bytes()).hexdigest(),
                   note="Native LongBench-v2 template; plain completions API; no chat template; tokenizer-only CPU audit.")
    (ROOT / "length_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print("LENGTH_SUMMARY", json.dumps({k:v for k,v in summary.items() if k != "template"}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
