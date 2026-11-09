#!/usr/bin/env python3
"""Prepare all 503 samples, truncating only the middle of overlong contexts.

Draft: direct execution is disabled until the scripts are reviewed.
No existing dataset, service code or aisbench configuration is modified.
"""
import ast
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer

SCRIPT_REVIEWED = True
ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
SOURCE = WORKSPACE / "benchmark/ais_bench/datasets/LongBench-v2/data.json"
MODEL = Path("/mnt/share/weight/DeepSeek-V4-Flash")
TEMPLATE_FILE = WORKSPACE / "benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
INPUT_CAP = 131072
EXPECTED_SOURCE_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"

def render(template, row, context):
    # One substitution pass: braces contained in context are never reformatted.
    return template.format(**dict(row, context=context))

def truncate_context(row, template, count, cap):
    original = row["context"]
    full = render(template, row, original)
    full_tokens = count(full)
    if full_tokens <= cap:
        return dict(row), full, full_tokens, len(original), 0, full_tokens
    empty = render(template, row, "")
    if count(empty) > cap:
        raise ValueError(f"Question/options/instructions alone exceed cap: {row.get('_id')}")
    # Slice original Unicode characters, rather than decoding partial token byte
    # sequences. Retained text is byte-for-byte original after UTF-8 encoding.
    # Token counts at BPE joins need not be strictly monotone; every accepted
    # candidate is checked, so the result is valid without claiming exact filling.
    lo, hi = 0, len(original) - 1
    best_context, best_prompt, best_tokens, best_head, best_tail = "", empty, count(empty), 0, 0
    while lo <= hi:
        keep = (lo + hi) // 2
        head, tail = (keep + 1) // 2, keep // 2
        context = original[:head] + (original[-tail:] if tail else "")
        prompt = render(template, row, context)
        tokens = count(prompt)
        if tokens <= cap:
            best_context, best_prompt, best_tokens = context, prompt, tokens
            best_head, best_tail = head, tail
            lo = keep + 1
        else:
            hi = keep - 1
    prepared = dict(row, context=best_context)
    assert count(best_prompt) == best_tokens <= cap
    assert all(prepared[k] == row[k] for k in row if k != "context")
    assert len(best_context) < len(original)
    return prepared, best_prompt, best_tokens, best_head, best_tail, full_tokens

def main():
    if not SCRIPT_REVIEWED:
        raise SystemExit("Draft only: review scripts before preparing data.")
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SOURCE_SHA256
    rows = json.loads(raw)
    assert len(rows) == 503 and len({r["_id"] for r in rows}) == 503
    tree = ast.parse(TEMPLATE_FILE.read_text())
    prompts = [n.value.value for n in ast.walk(tree)
               if isinstance(n, ast.keyword) and n.arg == "prompt"
               and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    assert len(prompts) == 1
    template = prompts[0]
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    output, samples = [], []
    for i, row in enumerate(rows):
        prepared, prompt, tokens, head, tail, before = truncate_context(row, template, count, INPUT_CAP)
        output.append(prepared)
        samples.append(dict(
            data_id=i, source_index=i, id=row["_id"], prompt_tokens=tokens,
            original_prompt_tokens=before, truncated=prepared["context"] != row["context"],
            retained_head_characters=head, retained_tail_characters=tail,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            original_prompt_sha256=hashlib.sha256(render(template, row, row["context"]).encode()).hexdigest(),
        ))
        print(f"PREPARE {i+1}/503 original={before} prepared={tokens}", flush=True)
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    target = directory / "longbenchv2_all503_midcut128k.json"
    manifest_path = directory / "full503_manifest.json"
    assert not target.exists() and not manifest_path.exists(), "Do not overwrite prepared evidence"
    data = json.dumps(output, ensure_ascii=False, indent=2).encode()
    manifest = dict(
        source=str(SOURCE), source_sha256=EXPECTED_SOURCE_SHA256,
        prepared=str(target), prepared_sha256=hashlib.sha256(data).hexdigest(),
        tokenizer_sha256=hashlib.sha256((MODEL / "tokenizer.json").read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(TEMPLATE_FILE.read_bytes()).hexdigest(),
        policy="all503_context_middle_truncate_original_unicode_no_marker",
        selected=503, excluded=0, input_cap=INPUT_CAP, output_tokens=1,
        max_model_len=INPUT_CAP+1, samples=samples,
        truncated_samples=sum(s["truncated"] for s in samples),
    )
    target.write_bytes(data)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("DATA_READY", target, "samples=503")

if __name__ == "__main__":
    main()
