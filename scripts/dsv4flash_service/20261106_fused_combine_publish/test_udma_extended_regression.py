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

# Candidate is process-local: verify ELF symbol resolution, not just environment.
import ctypes
class DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]
candidate_path = Path(os.environ["MOONEP_UDMA_TEST_LIBRARY"]).resolve()
global_symbols = ctypes.CDLL(None)
abi = global_symbols.moonep_udma_weight_reuse_fix_abi
abi.restype = ctypes.c_int
expected_abi = int(os.environ.get("MOONEP_UDMA_TEST_ABI", "1"))
assert abi() == expected_abi
libdl = ctypes.CDLL("libdl.so.2")
libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
proof = {"candidate_path": str(candidate_path), "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(), "abi": abi(), "symbols": {}}
for symbol in ["moonep_udma_combine_weighted_kernel_do", "moonep_udma_weight_reuse_fix_abi"]:
    fn = getattr(global_symbols, symbol)
    info = DlInfo()
    assert libdl.dladdr(ctypes.cast(fn, ctypes.c_void_p), ctypes.byref(info)) != 0
    resolved = Path(info.dli_fname.decode()).resolve()
    assert resolved == candidate_path, (symbol, resolved)
    proof["symbols"][symbol] = str(resolved)
print("CANDIDATE_RESOLUTION", json.dumps(proof), flush=True)

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
S, K, H, E, B = 32768, 6, 128, 32, 16
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
proof_path = Path(args.output) / f"rank{rank}_candidate.json"
proof_path.write_text(json.dumps(proof, indent=2))
outdir = Path(args.output)
outdir.mkdir(parents=True, exist_ok=True)


def sync():
    torch.npu.synchronize()
    dist.barrier()


def bits(tensor):
    return tensor.contiguous().view(torch.uint8).cpu()


def mismatch_count(actual, expected):
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


import numpy as np
for pattern, tokens in (("balanced", 1), ("remote", 17), ("remote", 63),
                        ("balanced", 65), ("hot", 1025),
                        ("balanced", 16385), ("remote", 32768)):
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=device) * 0.25
    if pattern == "balanced":
        topk_cpu = (torch.arange(S*K).view(S, K) + rank*16) % E
    elif pattern == "hot":
        topk_cpu = torch.arange(K)[None, :].expand(S, K).clone()
    else:
        topk_cpu = ((1-rank)*16 + torch.arange(K))[None, :].expand(S, K).clone()
    topk = topk_cpu.int().to(device)
    topk[tokens:].fill_(-1)
    histogram = torch.bincount(topk[:tokens].flatten().long(), minlength=E).int()
    expert_output, _, cu, plan = buffer.dispatch(
        hidden, topk_experts_sk=topk, tokens_per_expert=histogram, num_tokens=tokens)
    all_counts = torch.cat((cu[:1], cu[1:] - cu[:-1])).long()
    counts = torch.cat((all_counts[rank*16:(rank+1)*16], all_counts[E:]))
    valid = int(counts.sum().cpu())
    meta = dict(dst=plan.dst[:tokens*K].cpu().numpy(),
                experts=topk_cpu[:tokens].numpy().reshape(-1))
    peers = [None] * world
    dist.all_gather_object(peers, meta)
    # Give every expert a distinct power-of-two response within a top-K set.
    # Map by the original source routes, so redundant expert placement is included.
    scales = np.full(S*K, np.nan, dtype=np.float32)
    for peer in peers:
        raw = np.where(peer["dst"] >= 0, peer["dst"], ~peer["dst"])
        owned = raw // (S*K) == rank
        offsets = raw[owned] % (S*K)
        values = np.exp2(peer["experts"][owned] % 8).astype(np.float32)
        order = np.argsort(offsets)
        si, sv = offsets[order], values[order]
        if len(si) > 1:
            assert np.all((si[1:] != si[:-1]) | (sv[1:] == sv[:-1]))
        prior = scales[offsets]
        assert np.all(np.isnan(prior) | (prior == values))
        scales[offsets] = values
    assert np.all(np.isfinite(scales[:valid])), "Missing expert response mapping"
    scale_npu = torch.from_numpy(scales).to(device=device, dtype=torch.bfloat16)
    expert_output = expert_output * scale_npu[:, None]
    expert_output[valid:].fill_(float("nan"))
    clean = torch.where((row_ids < counts.sum()).unsqueeze(-1), expert_output, zero)
    hidden_cpu = hidden[:tokens].float().cpu()
    factors = torch.exp2((topk_cpu[:tokens] % 8).float())
    response_cpu = [(hidden_cpu * factors[:, k:k+1]).to(torch.bfloat16).float()
                    for k in range(K)]
    for repeat in range(3):
        weights = torch.zeros(S, 8, dtype=torch.float32, device=device)
        weights[:tokens, :K].copy_(torch.rand(tokens, K, device=device))
        weights_cpu = weights[:tokens].cpu()
        expected = torch.zeros_like(hidden_cpu)
        for k in range(K):
            expected.add_(response_cpu[k] * weights_cpu[:, k:k+1])
        expected = expected.to(torch.bfloat16)
        errors = {}
        for mode in ("old", "fused", "trimmed"):
            source = clean if mode == "old" else expert_output
            if mode == "trimmed":
                source = source[:valid]
            result = buffer.combine_weighted(
                plan, source, weights, num_tokens=tokens,
                fused_publish=mode != "old", row_counts=counts if mode != "old" else None)[0][:tokens].clone()
            sync()
            errors[mode] = mismatch_count(result.cpu(), expected)
        rec = dict(pattern=pattern, tokens=tokens, repeat=repeat,
                   distinct_expert_responses=True, changed_weights=True,
                   crosses_token_window=(tokens+31)//32 > 512, mismatches=errors)
        checks.append(rec)
        (outdir / f"rank{rank}_extended_checks.json").write_text(json.dumps(checks, indent=2))
        print("EXTENDED_NUMERICAL", rank, json.dumps(rec), flush=True)
        assert not any(errors.values()), rec
record = dict(status="passed", rank=rank, backend=args.backend, S=S, H=H, K=K,
              cross_node=False, model_loaded=False, checks=checks, proof=proof)
(outdir / f"rank{rank}.json").write_text(json.dumps(record, indent=2))
sync()
ShmemRuntime.finalize()
dist.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
