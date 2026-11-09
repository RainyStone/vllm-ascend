#!/usr/bin/env python3
"""Two-rank MTE/UDMA regression and timing, without a model."""
import argparse
import datetime as dt
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu
import torch.distributed as dist

from ascend_moonep.api import Buffer
from ascend_moonep.combine_publish import CombinePublisher
from ascend_moonep.shmem_runtime import ShmemRuntime

parser = argparse.ArgumentParser()
for key in ("backend", "output", "library", "bootstrap-ip"):
    parser.add_argument("--" + key, required=True)
parser.add_argument("--shmem-port", required=True, type=int)
args = parser.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)
rank, world, local = [int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
assert world == 2
torch.npu.set_device(local)
device = torch.device("npu", local)
dist.init_process_group("gloo", timeout=dt.timedelta(seconds=180))
S, K, H, E, B = 8192, 6, 4096, 32, 16
buffer = Buffer(S=S, H=H, K=K, E=E, B=B, num_ep_ranks=world,
                group=dist.group.WORLD, use_udma=args.backend == "udma")
ShmemRuntime.init_with_ip_port(
    group=dist.group.WORLD, ip_port=f"tcp://{args.bootstrap_ip}:{args.shmem_port}",
    enable_udma=args.backend == "udma", qp_count=2 if args.backend == "udma" else 0,
    prefetch_overlap=args.backend == "udma",
)
buffer._ensure_shmem_views()
ctx = buffer._require_ctx()
assert not ctx["_planning_use_udma"]
assert ctx["_combine_use_udma"] == (args.backend == "udma")
buffer._combine_publisher = CombinePublisher(args.library)

import ascend_moonep.launchers as launchers

native_calls = []
native_name = "CombineUrma" if args.backend == "udma" else "Combine"
native = getattr(torch.classes.AscendMoonEP, native_name)()


class Recorder:
    def __getattr__(self, name):
        method = getattr(native, name)
        if not name.startswith("compute"):
            return method
        def wrapped(*values, **kwargs):
            native_calls.append(dict(operator=native_name, method=name))
            return method(*values, **kwargs)
        return wrapped


if args.backend == "udma":
    launchers._COMBINE_UDMA = Recorder()
else:
    launchers._COMBINE = Recorder()
torch.manual_seed(118 + rank)
hidden = torch.randn(S, H, dtype=torch.bfloat16, device=device) * 0.25
row_ids = torch.arange(S * K, dtype=torch.int64, device=device)
zero = torch.zeros((), dtype=torch.bfloat16, device=device)
checks = []
outdir = Path(args.output)
outdir.mkdir(parents=True, exist_ok=True)


def sync():
    torch.npu.synchronize()
    dist.barrier()


def bits(tensor):
    return tensor.contiguous().view(torch.uint8).cpu()


def mismatch_count(actual, expected):
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


for pattern, tokens in (("balanced", 1), ("hot", 17), ("remote_hot", 63),
                        ("balanced", 8191), ("hot", 8192)):
    if pattern == "balanced":
        topk = (torch.arange(S * K).view(S, K) + rank * 16) % E
    elif pattern == "hot":
        topk = torch.arange(K)[None, :].expand(S, K).clone()
    else:
        topk = ((1 - rank) * 16 + torch.arange(K))[None, :].expand(S, K).clone()
    topk = topk.int().to(device)
    topk[tokens:].fill_(-1)
    histogram = torch.bincount(topk[:tokens].flatten().long(), minlength=E).int()
    expert_output, _, cu, plan = buffer.dispatch(
        hidden, topk_experts_sk=topk, tokens_per_expert=histogram, num_tokens=tokens,
    )
    expert_output = expert_output.clone()
    all_counts = torch.cat((cu[:1], cu[1:] - cu[:-1])).long()
    counts = torch.cat((all_counts[rank * 16:(rank + 1) * 16], all_counts[E:]))
    valid = int(counts.sum().cpu())
    expert_output[valid:].fill_(float("nan"))
    weights = torch.zeros(S, 8, dtype=torch.float32, device=device)
    weights[:tokens, :K].copy_(torch.rand(tokens, K, device=device))
    clean = torch.where((row_ids < counts.sum()).unsqueeze(-1), expert_output, zero)
    old, fused = [], []
    for mode in ("old", "fused"):
        for repeat in range(3):
            result = buffer.combine_weighted(
                plan, clean if mode == "old" else expert_output, weights, num_tokens=tokens,
                fused_publish=mode == "fused", row_counts=counts if mode == "fused" else None,
            )[0][:tokens].clone()
            sync()
            (old if mode == "old" else fused).append(result)
    # Read the independent identity-expert reference only after kernel samples.
    hidden_cpu = hidden[:tokens].float().cpu()
    weights_cpu = weights[:tokens].cpu()
    expected = torch.zeros_like(hidden_cpu)
    for k in range(K):
        expected.add_(hidden_cpu * weights_cpu[:, k:k+1])
    expected = expected.to(torch.bfloat16)
    old_cpu, fused_cpu = [x.cpu() for x in old], [x.cpu() for x in fused]
    record = dict(pattern=pattern, tokens=tokens, valid_rows=valid,
                  old_mismatches_to_cpu=[mismatch_count(x, expected) for x in old_cpu],
                  fused_mismatches_to_cpu=[mismatch_count(x, expected) for x in fused_cpu],
                  old_repeat_equal=all(torch.equal(bits(old_cpu[0]), bits(x)) for x in old_cpu[1:]),
                  fused_repeat_equal=all(torch.equal(bits(fused_cpu[0]), bits(x)) for x in fused_cpu[1:]))
    weights[:tokens, :K].fill_(1.0 / K)
    constant = {}
    expected_constant = torch.zeros_like(hidden_cpu)
    for k in range(K):
        expected_constant.add_(hidden_cpu * (1.0 / K))
    expected_constant = expected_constant.to(torch.bfloat16)
    for mode in ("old", "fused"):
        counts_bad = []
        for repeat in range(2):
            result = buffer.combine_weighted(
                plan, clean if mode == "old" else expert_output, weights, num_tokens=tokens,
                fused_publish=mode == "fused", row_counts=counts if mode == "fused" else None,
            )[0][:tokens].clone()
            sync()
            counts_bad.append(mismatch_count(result.cpu(), expected_constant))
        constant[mode] = counts_bad
    record["constant_weight_mismatches_to_cpu"] = constant
    checks.append(record)
    print("UDMA_NUMERICAL_DIAGNOSTIC", rank, json.dumps(record), flush=True)
    (outdir / f"rank{rank}_numeric_diagnostic.json").write_text(json.dumps(checks, indent=2))
    del old, fused, old_cpu, fused_cpu
record = dict(status="diagnostic_completed", rank=rank, backend=args.backend,
              model_loaded=False, cross_node=False, checks=checks)
(outdir / f"rank{rank}.json").write_text(json.dumps(record, indent=2))
sync()
ShmemRuntime.finalize()
dist.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
