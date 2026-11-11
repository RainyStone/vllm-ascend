"""Validate one full GSM8K round and derive TTFT percentiles from retained timing data."""
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.db_utils import (
    load_all_numpy_from_db,
)

DATA_ABBR = "gsm8k-test"
CONCURRENCY = 16


def number(value):
    return float(str(value).replace(",", "").split()[0])


def validate(result_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    expected = {sample["data_id"]: sample for sample in manifest["samples"]}
    files = [path for path in result_dir.rglob(f"{DATA_ABBR}_details.jsonl")
             if "warmup" not in str(path.relative_to(result_dir)).lower()]
    if len(files) != 1:
        raise ValueError(f"Expected one formal details file, found {files}")
    details = files[0]
    with details.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indices = [int(row.get("data_id", row["id"])) for row in rows]
    if set(indices) != set(expected) or len(indices) != len(expected):
        raise ValueError(f"Expected {len(expected)} distinct test ids, found {len(set(indices))}")
    if any(path.stat().st_size for path in result_dir.rglob("*_failed.jsonl")):
        raise ValueError("A failed-request file is nonempty")

    timings = {}
    ttft_ms = []
    for row in rows:
        index = int(row.get("data_id", row["id"]))
        sample = expected[index]
        if row.get("success") is not True or int(row.get("output_tokens", 0)) != 1:
            raise ValueError(f"Request {index} failed or did not produce exactly one token")
        if int(row.get("input_tokens", 0)) != sample["prompt_tokens"]:
            raise ValueError(f"Input token count mismatch for request {index}")
        if hashlib.sha256(row["input"].encode()).hexdigest() != sample["prompt_sha256"]:
            raise ValueError(f"Question text changed for request {index}")
        time_points = row["time_points"]
        if isinstance(time_points, dict):
            db_path = (details.parent / "db_data" / row["db_name"]).resolve()
            if db_path not in timings:
                with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
                    timings[db_path] = load_all_numpy_from_db(connection)
            time_points = timings[db_path][time_points["__db_ref__"]]
        if len(time_points) < 2 or not np.all(np.isfinite(time_points[:2])):
            raise ValueError(f"Invalid timing data for request {index}")
        delta_ms = float(time_points[1] - time_points[0]) * 1000
        if delta_ms < 0:
            raise ValueError(f"Negative TTFT for request {index}")
        ttft_ms.append(delta_ms)

    summary = json.loads((details.parent / f"{DATA_ABBR}.json").read_text())
    input_tokens = sum(sample["prompt_tokens"] for sample in expected.values())
    totals = {
        "Total Requests": len(rows),
        "Success Requests": len(rows),
        "Failed Requests": 0,
        "Total Input Tokens": input_tokens,
        "Total Generated Tokens": len(rows),
        "Max Concurrency": CONCURRENCY,
    }
    for name, total in totals.items():
        if number(summary[name]["total"]) != total:
            raise ValueError(f"Unexpected summary {name}: {summary[name]['total']}")
    report = dict(valid=True, requests=len(rows), input_tokens=input_tokens,
                  output_tokens=len(rows), max_concurrency=CONCURRENCY,
                  mean_ttft_ms=float(np.mean(ttft_ms)),
                  p95_ttft_ms=float(np.percentile(ttft_ms, 95)),
                  input_tps=number(summary["Input Token Throughput"]["total"]),
                  raw_details=str(details), dataset_sha256=manifest["dataset_sha256"])
    (result_dir.parent / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VALID_ROUND", json.dumps(report))


if __name__ == "__main__":
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
