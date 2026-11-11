"""Capture Slot at the Off capture's observed replay progress."""
import json
import math
import signal
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT))
import profile_window as profile

OFF_COMPLETED_REQUESTS = 159
OFF_IN_SERVICE_AT_START_REQUEST = 14
read_prefill_metrics = profile.read_metrics


def read_metrics(text, model):
    ranks = read_prefill_metrics(text, model)
    for family in profile.text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != "vllm:generation_tokens_total" or sample.labels.get("model_name") != model:
                continue
            rank = int(sample.labels["engine"])
            value = sample.value
            if rank not in ranks or "generation_tokens" in ranks[rank] or not math.isfinite(value):
                raise ValueError(f"Unexpected generation counter: {sample}")
            if value < 0 or value != int(value):
                raise ValueError(f"Expected an integer generation counter: {sample}")
            ranks[rank]["generation_tokens"] = int(value)
    if any("generation_tokens" not in row for row in ranks.values()):
        raise ValueError("Missing generation counter for one or more DP engines")
    return ranks


def wait_for_load(bench, output, model, _target, _required, interval, timeout):
    log = output / "metrics_samples.jsonl"
    initial = [json.loads(line) for line in log.read_text().splitlines()][-1]
    if initial["phase"] != "before_replay" or initial["in_service"] != 0:
        raise ValueError("Missing idle metrics after the 16 warmup requests")
    baseline = sum(row["generation_tokens"] for row in initial["ranks"].values())
    profile.event(output, "match_off_progress", generation_baseline=baseline,
                  target_completed_requests=OFF_COMPLETED_REQUESTS,
                  min_in_service=OFF_IN_SERVICE_AT_START_REQUEST)
    previous = None
    deadline = time.monotonic() + timeout
    while bench.poll() is None and time.monotonic() < deadline:
        began = time.monotonic()
        row, previous = profile.sample_metrics(output, model, "match_gate", previous)
        if previous is not None:
            generated = sum(r["generation_tokens"] for r in previous.values()) - baseline
            if generated < 0:
                raise ValueError("Generation counter reset")
            if generated >= OFF_COMPLETED_REQUESTS and row["in_service"] >= OFF_IN_SERVICE_AT_START_REQUEST:
                return dict(row, trigger="match_off_progress", generated_requests=generated), previous
        time.sleep(max(0, min(began + interval, deadline) - time.monotonic()))
    return None, previous


if __name__ == "__main__":
    profile.read_metrics = read_metrics
    profile.wait_for_load = wait_for_load
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, profile.interrupt)
    profile.main()
