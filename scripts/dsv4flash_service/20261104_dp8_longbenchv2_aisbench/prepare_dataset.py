#!/usr/bin/env python3
"""Create the approved full-text subset on CPU, preserving original order."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/LongBench-v2/data.json")
MODEL = Path("/mnt/share/weight/DeepSeek-V4-Flash")
INPUT_CAP = 131072
MAX_MODEL_LEN = 131073
EXPECTED_SAMPLES = 316

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = json.loads((ROOT / "length_summary.json").read_text())
    assert sha256(SOURCE) == summary["dataset_sha256"]
    assert sha256(MODEL / "tokenizer.json") == summary["tokenizer_sha256"]
    model_config = json.loads((MODEL / "config.json").read_text())
    assert INPUT_CAP + 1 == MAX_MODEL_LEN <= model_config["max_position_embeddings"]
    records = json.loads(SOURCE.read_text())
    audit = list(csv.DictReader((ROOT / "length_audit.csv").open()))
    assert len(records) == len(audit) == 503
    selected, mapping = [], []
    for index, (record, entry) in enumerate(zip(records, audit)):
        assert record["_id"] == entry["id"] and index == int(entry["index"])
        tokens = int(entry["prompt_tokens"])
        if tokens <= INPUT_CAP:
            mapping.append(dict(data_id=len(selected), source_index=index,
                                id=record["_id"], prompt_tokens=tokens,
                                domain=record["domain"], difficulty=record["difficulty"]))
            selected.append(record)
    assert len(selected) == EXPECTED_SAMPLES
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    output = data_dir / "longbenchv2_full_le128k.json"
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    # This checks every field in every kept sample: no context truncation or editing.
    assert json.loads(output.read_text()) == [records[m["source_index"]] for m in mapping]
    manifest = dict(
        source=str(SOURCE), source_sha256=summary["dataset_sha256"],
        prepared=str(output), prepared_sha256=sha256(output),
        tokenizer_sha256=summary["tokenizer_sha256"],
        policy="filter_full_prompt_length_preserve_original_text_and_order",
        input_cap=INPUT_CAP, output_tokens=1, max_model_len=MAX_MODEL_LEN,
        selected=len(selected), excluded=len(records)-len(selected),
        shortest_prompt_tokens=min(m["prompt_tokens"] for m in mapping),
        longest_prompt_tokens=max(m["prompt_tokens"] for m in mapping),
        samples=mapping,
    )
    (data_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    assert sha256(SOURCE) == summary["dataset_sha256"]
    print("PREPARED", output)
    print("SELECTED", len(selected), "EXCLUDED", len(records)-len(selected))
    print("INPUT_TOKENS", manifest["shortest_prompt_tokens"], manifest["longest_prompt_tokens"])
    print("SHA256", manifest["prepared_sha256"])

if __name__ == "__main__":
    main()
