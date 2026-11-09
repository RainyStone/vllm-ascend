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
status = "failed"
timings = {"old": [], "fused": []}
outdir = Path(args.output)
outdir.mkdir(parents=True, exist_ok=True)


def bits(tensor):
    return tensor.contiguous().view(torch.uint8).cpu()


def sync():
    torch.npu.synchronize()
    dist.barrier()


try:
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
        valid = int(counts.sum().cpu())  # Test reference only, never the hot implementation.
        expert_output[valid:].fill_(float("nan"))
        weights = torch.zeros(S, 8, dtype=torch.float32, device=device)
        weights[:tokens, :K].copy_(torch.rand(tokens, K, device=device))
        clean = torch.where((row_ids < counts.sum()).unsqueeze(-1), expert_output, zero)
        reference = buffer.combine_weighted(
            plan, clean, weights, num_tokens=tokens,
        )[0][:tokens].clone()
        sync()
        expected = torch.zeros(tokens, H, dtype=torch.float32)
        hidden_cpu = hidden[:tokens].float().cpu()
        weights_cpu = weights[:tokens].cpu()
        for k in range(K):
            expected.add_(hidden_cpu * weights_cpu[:, k:k+1])
        expected = expected.to(torch.bfloat16)
        assert torch.equal(bits(reference.cpu()), bits(expected)), ("CPU", rank, pattern, tokens)
        fused = buffer.combine_weighted(
            plan, expert_output, weights, num_tokens=tokens,
            fused_publish=True, row_counts=counts,
        )[0][:tokens].clone()
        sync()
        assert torch.equal(bits(reference), bits(fused)), ("full", rank, pattern, tokens)
        assert torch.all(ctx["hidden_buf_local"][valid:].cpu() == 0)
        trimmed = buffer.combine_weighted(
            plan, expert_output[:valid].contiguous(), weights, num_tokens=tokens,
            fused_publish=True,
        )[0][:tokens].clone()
        sync()
        assert torch.equal(bits(reference), bits(trimmed)), ("trimmed", rank, pattern, tokens)
        checks.append(dict(pattern=pattern, tokens=tokens, valid_rows=valid,
                           full_bitwise_equal=True, trimmed_bitwise_equal=True,
                           nan_tail_zero=True))
        print("COMBINE_CHECK", rank, json.dumps(checks[-1]), flush=True)
        if tokens == S:
            def old_call():
                clean = torch.where(
                    (row_ids < counts.sum().to(torch.int64)).unsqueeze(-1), expert_output, zero,
                )
                return buffer.combine_weighted(plan, clean, weights, num_tokens=tokens)[0]

            def fused_call():
                return buffer.combine_weighted(
                    plan, expert_output, weights, num_tokens=tokens,
                    fused_publish=True, row_counts=counts,
                )[0]

            for _ in range(5):
                old_call()
                fused_call()
            sync()
            for iteration in range(10):
                order = [("old", old_call), ("fused", fused_call)]
                if iteration % 2:
                    order.reverse()
                for name, fn in order:
                    dist.barrier()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    result = fn()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end))
                    del result
    assert native_calls and all(x["method"] == "compute_weighted" for x in native_calls)
    status = "passed"
finally:
    libs = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and ("moonep" in line or "libshmem" in line)}
    summary = {name: dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values))
               for name, values in timings.items() if values}
    record = dict(status=status, rank=rank, world=world, backend=args.backend,
                  model_loaded=False, cross_node=False, planning_backend="mte",
                  checks=checks, native_calls=native_calls, timings=timings, summary=summary,
                  candidate_proof=proof, mapped_libraries={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                    for name in libs})
    (outdir / f"rank{rank}.json").write_text(json.dumps(record, indent=2))
    if status == "passed":
        sync()
        ShmemRuntime.finalize()
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()
print("COMBINE_PUBLISH_PASSED", rank, args.backend, flush=True)
