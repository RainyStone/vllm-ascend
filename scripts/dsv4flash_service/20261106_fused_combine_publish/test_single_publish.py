#!/usr/bin/env python3
"""Model-free byte correctness, tail guards and local publication timing."""
import argparse
import json
from pathlib import Path
import statistics

import torch
import torch_npu

from ascend_moonep.combine_publish import CombinePublisher

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
torch.npu.set_device(0)
publisher = CombinePublisher(args.library)
checks = []


def raw(tensor):
    return tensor.contiguous().view(torch.uint8).cpu()


def check(source_rows, destination_rows, hidden, values):
    # Guard both ends; source is arbitrary BF16 bits, including valid NaN payloads.
    source = torch.randn(source_rows, hidden, device="npu", dtype=torch.bfloat16)
    if source_rows:
        source.view(torch.int16).reshape(-1)[::37] = -1
    before = raw(source)
    guard_elements = 16
    backing = torch.full((destination_rows * hidden + 2 * guard_elements,),
                         19.0, device="npu", dtype=torch.bfloat16)
    destination = backing[guard_elements:-guard_elements].view(destination_rows, hidden)
    counts = None if values is None else torch.tensor(values, device="npu", dtype=torch.int64)
    publisher.publish(source, destination, counts)
    torch.npu.synchronize()
    valid = source_rows if values is None else max(0, min(source_rows, sum(values)))
    expected = torch.zeros(destination_rows, hidden, dtype=torch.bfloat16)
    expected[:valid].copy_(source.cpu()[:valid])
    actual = raw(destination)
    expected_bits = expected.view(torch.uint8)
    if not torch.equal(actual, expected_bits):
        bad = torch.nonzero(actual.reshape(-1) != expected_bits.reshape(-1)).flatten()
        diagnostic = dict(source_rows=source_rows, destination_rows=destination_rows,
                          hidden=hidden, counts=values, valid=valid, bad_bytes=bad.numel(),
                          first_indices=bad[:16].tolist(),
                          actual=actual.reshape(-1)[bad[:16]].tolist(),
                          expected=expected_bits.reshape(-1)[bad[:16]].tolist())
        Path(args.output).with_suffix(".failure.json").write_text(json.dumps(diagnostic, indent=2))
        raise AssertionError(diagnostic)
    assert torch.equal(raw(source), before)
    assert torch.all(backing[:guard_elements].cpu() == 19)
    assert torch.all(backing[-guard_elements:].cpu() == 19)
    checks.append(dict(source_rows=source_rows, destination_rows=destination_rows,
                       hidden=hidden, counts=values, valid_rows=valid, passed=True))


for case in [
    (0, 17, 4096, None),
    (17, 17, 4096, [5, 2, 0]),
    (17, 65, 4096, None),
    (65, 65, 17, [19, 2]),
    (65, 65, 17, None),
    (17, 65, 31, [1]),
    (65, 65, 4095, [64]),
    (37, 65, 4096, []),
    (65, 65, 4096, [-3, 1]),
    (65, 65, 4096, [100]),
    (49152, 49152, 4096, [49146]),
]:
    check(*case)
    print("PUBLISH_CASE_PASSED", checks[-1], flush=True)

# Reuse exactly the same count storage across calls; catch stale scalar GM reads.
source = torch.randn(65, 4096, device="npu", dtype=torch.bfloat16)
destination = torch.empty_like(source)
counts = torch.empty(2, device="npu", dtype=torch.int64)
for valid in (65, 0, 1, 17, 64, 63, 65, 0):
    counts.copy_(torch.tensor([valid, 0], dtype=torch.int64, device="npu"))
    publisher.publish(source, destination, counts)
    torch.npu.synchronize()
    expected = torch.zeros_like(source.cpu())
    expected[:valid].copy_(source.cpu()[:valid])
    assert torch.equal(raw(destination), expected.view(torch.uint8))
checks.append(dict(stage="repeated_count_storage", passed=True))

# Only publication is timed here; this is not an end-to-end model benchmark.
rows, hidden = 49152, 4096
source = torch.randn(rows, hidden, device="npu", dtype=torch.bfloat16)
destination = torch.empty_like(source)
counts = torch.tensor([rows - 6], device="npu", dtype=torch.int64)
row_ids = torch.arange(rows, device="npu", dtype=torch.int64)


def old_publish():
    valid = counts.sum().to(torch.int64)
    clean = torch.where((row_ids < valid).unsqueeze(-1), source, 0.0)
    destination.copy_(clean)


def new_publish():
    publisher.publish(source, destination, counts)


for _ in range(5):
    old_publish()
    new_publish()
torch.npu.synchronize()
timings = {"old": [], "fused": []}
for iteration in range(20):
    # Alternate ordering to reduce order bias.
    for name, fn in ([("old", old_publish), ("fused", new_publish)] if iteration % 2 == 0
                     else [("fused", new_publish), ("old", old_publish)]):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings[name].append(start.elapsed_time(end))
summary = {name: dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                      p95_ms=sorted(values)[18]) for name, values in timings.items()}
summary["mean_improvement_pct"] = (
    1 - summary["fused"]["mean_ms"] / summary["old"]["mean_ms"]
) * 100
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(dict(passed=True, model_loaded=False, checks=checks,
                                 rows=rows, hidden=hidden, timing_kind="NPU events; publication only",
                                 timings=timings, summary=summary), indent=2))
print("SINGLE_PUBLISH_PASSED", json.dumps(summary), flush=True)
