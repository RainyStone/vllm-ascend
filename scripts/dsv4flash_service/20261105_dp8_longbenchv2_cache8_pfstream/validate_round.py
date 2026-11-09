#!/usr/bin/env python3
"""Check retained aisbench request details; never starts a service."""
import json
import hashlib
import sys
from pathlib import Path

DATA_ABBR = "longbenchv2-full-le128k"

def rows_from_payload(payload):
    if isinstance(payload, list):
        for entry in payload:
            yield from rows_from_payload(entry)
    elif isinstance(payload, dict):
        if "success" in payload and "id" in payload:
            yield payload
        else:
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    yield from rows_from_payload(value)

def validate(result_dir, manifest_path):
    result_dir = Path(result_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    expected = {int(row["data_id"]): row for row in manifest["samples"]}
    files = [f for f in result_dir.rglob(DATA_ABBR + "_details.jsonl")
             if "warmup" not in str(f.relative_to(result_dir)).lower()]
    if not files:
        raise ValueError("No raw aisbench request details found; retain the entire work_dir.")
    records = []
    for path in files:
        # JSONL records are separated by physical LF, not Unicode line separators
        # that may legitimately occur inside JSON strings in LongBench contexts.
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.extend(rows_from_payload(json.loads(line)))
    errors, ids = [], []
    for row in records:
        index = int(row["id"])
        ids.append(index)
        if row.get("success") is not True:
            errors.append(f"id={index}: request failed: {row.get('error_info')}")
        if index not in expected:
            errors.append(f"Unexpected id={index}")
            continue
        actual_in = int(row.get("input_tokens", 0))
        actual_out = int(row.get("output_tokens", 0))
        wanted = expected[index]["prompt_tokens"]
        wanted_hash = expected[index].get("prompt_sha256")
        raw_input = row.get("input")
        if wanted_hash is not None:
            if not isinstance(raw_input, str) or hashlib.sha256(raw_input.encode()).hexdigest() != wanted_hash:
                errors.append(f"id={index}: actual prompt text differs from original full-text template")
        if actual_in != wanted:
            errors.append(f"id={index}: service input_tokens={actual_in}, tokenizer audit={wanted}; stop and reconcile tokenization.")
        if actual_in > manifest["input_cap"]:
            errors.append(f"id={index}: prompt exceeds input cap")
        if actual_out != 1:
            errors.append(f"id={index}: output_tokens={actual_out}, expected 1")
    if len(ids) != len(expected) or set(ids) != set(expected) or len(set(ids)) != len(ids):
        errors.append(f"Expected {len(expected)} distinct ids, got {len(ids)} rows / {len(set(ids))} ids")
    for failed in result_dir.rglob("*_failed.jsonl"):
        if failed.stat().st_size:
            errors.append(f"Failed-request file is nonempty: {failed}")
    report = dict(valid=not errors, requests=len(records), expected=len(expected),
                  details=[str(f) for f in files], errors=errors)
    (result_dir.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValueError("\n".join(errors[:25]))
    print(f"VALID_ROUND: {len(records)} successful full-text requests; every output=1; input token counts match audit.")
    return report

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: validate_round.py RESULT_DIR SELECTION_MANIFEST")
    validate(sys.argv[1], sys.argv[2])
    # A live three-round shell cannot safely be rewritten mid-loop. Preserve
    # its completed round 1, then use a controlled exit to prevent round 2.
    gate = Path(__file__).with_name("SWITCH_TO_CROSSOVER.json")
    if gate.exists():
        change = json.loads(gate.read_text())
        round_dir = Path(sys.argv[1]).resolve().parent
        if round_dir.name == "round_1" and round_dir.parent == Path(change["client_dir"]):
            change["round1_validated"] = True
            gate.write_text(json.dumps(change, indent=2) + "\n")
            print("USER_SCHEDULE_CHANGE: valid Off1 retained; stop batch before Off2.", flush=True)
            raise SystemExit(75)
