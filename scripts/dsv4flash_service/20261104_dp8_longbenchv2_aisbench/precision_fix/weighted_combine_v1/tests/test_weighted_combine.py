"""Isolated weighted-combine operator correctness and full-chain timing.
No model, tokenizer or vLLM server is loaded. torch.distributed uses gloo only.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--shmem-port", type=int, required=True)
parser.add_argument("--performance", action="store_true")
parser.add_argument("--check-skip-dup-guard", action="store_true")
parser.add_argument("--installed-smoke", action="store_true")
args = parser.parse_args()
if args.check_skip_dup_guard:
    os.environ["ASCEND_MOONEP_SKIP_DUP"] = "0"
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=8))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
records, timings = [], []

# Initialize the native oracle before SHMEM peer VMM mappings.
torch_npu.npu_moe_token_unpermute(
    torch.zeros(6, 16, dtype=torch.bfloat16, device=device),
    torch.arange(6, dtype=torch.int32, device=device),
    probs=torch.ones(1, 6, dtype=torch.float32, device=device),
    restore_shape=(1, 16))
torch.npu.synchronize()

import ascend_moonep
from ascend_moonep.api import Buffer
from ascend_moonep.planning import allocate_planning_outputs
from ascend_moonep.shmem_runtime import ShmemRuntime

H = 4096
buffers = {}
for S, K in [(128, 6), (8192, 6), (16, 1), (128, 8), (128, 16)]:
    buffers[S, K] = Buffer(
        S=S, H=H, K=K, E=256, B=256 // world, num_ep_ranks=world,
        group=dist.group.WORLD, use_udma=False)
ShmemRuntime.init_with_ip_port(
    group=dist.group.WORLD, ip_port=f"tcp://127.0.0.1:{args.shmem_port}",
    enable_udma=False, qp_count=0)
for b in buffers.values():
    b._ensure_shmem_views()
torch.npu.synchronize()

def weights_for(t, k, pe, K):
    nums = ((t[:, None] * 13 + torch.arange(K, device=device)[None, :] * 7
             + pe[:, None] * 3) % 31 + 1).float()
    return nums / nums.sum(dim=1, keepdim=True)

def values(rows, pes, version=0):
    # BF16 representable, signed, nonconstant across rows/channels/ranks.
    channels = torch.arange(H, device=device)[None, :]
    ints = ((rows[:, None] * 17 + channels * 11 + pes[:, None] * 23
             + version * 37) % 509 - 254)
    return (ints.float() / 128).to(torch.bfloat16)

def sync_all():
    torch.npu.synchronize()
    dist.barrier()

def check(name, actual, reference):
    a, b = actual.float(), reference.float()
    delta = a - b
    finite = bool(torch.isfinite(a).all().item())
    item = dict(name=name, rank=rank, elements=a.numel(), finite=finite,
                mismatches=int((actual != reference).sum().item()),
                max_abs=float(delta.abs().max().item()),
                rmse=float(delta.square().mean().sqrt().item()))
    records.append(item)
    gathered = [None] * world
    dist.all_gather_object(gathered, item)
    if rank == 0:
        print(json.dumps(dict(check=name, max_abs=max(x["max_abs"] for x in gathered),
                              mismatches=sum(x["mismatches"] for x in gathered)),
                         allow_nan=False), flush=True)
    # Exact BF16 agreement is the controlled-test acceptance requirement.
    if any(x["mismatches"] or not x["finite"] for x in gathered):
        raise AssertionError(f"{name}: mismatch; see rank result files")

def setup_case(S, K, T, kind="normal", version=0):
    b = buffers[S, K]
    ctx = b._require_ctx()
    plan, _ = allocate_planning_outputs(ctx)
    t = torch.arange(S, device=device)
    k = torch.arange(K, device=device)
    rows = t[:, None] * K + k[None, :]
    pes = (rank + k[None, :]).expand(S, K) % world
    raw = pes * (S * K) + rows
    if kind == "alias" and K > 1:
        raw[:, 1] = -raw[:, 0] - 1
    if kind == "invalid":
        raw[:, -1] = world * S * K + 17
    plan.dst.copy_(raw.flatten().int())
    plan.experts_to_copy.fill_(-1)
    plan.dup_groups.zero_()
    plan.dup_loffs.zero_()
    plan.dup_counts.zero_()

    stride = (K + 7) // 8 * 8
    w = torch.full((S, stride), float("nan"), device=device, dtype=torch.float32)
    w[:, :K] = weights_for(t, None, torch.full_like(t, rank), K)
    if kind == "zero":
        w[:, :K] = 0
    rows_dest = torch.arange(S * K, device=device)
    source_pes = (rank - rows_dest % K) % world
    hidden = values(rows_dest, torch.full_like(rows_dest, rank), version)
    real_by_rank = [T] * world
    if kind == "unequal":
        real_by_rank = [max(1, S // (2 ** (r % 4))) for r in range(world)]
        T = real_by_rank[rank]
    if kind in ("unequal", "padding"):
        real = torch.tensor(real_by_rank, device=device)[source_pes]
        hidden[rows_dest // K >= real] = float("nan")
        w[T:] = float("nan")
    # External multiplication in the old chain happens at destination ranks.
    dest_w = weights_for(rows_dest // K, None, source_pes, K).gather(
        1, (rows_dest % K)[:, None]).squeeze(1)
    if kind == "zero":
        dest_w.zero_()
    return b, plan, hidden, w, raw[:T], dest_w, T

def reference_for(raw, w, S, K, version=0):
    T = raw.shape[0]
    decoded = torch.where(raw < 0, -raw - 1, raw)
    valid = (decoded >= 0) & (decoded < world * S * K)
    decoded = decoded.clamp(0, world * S * K - 1)
    ref_rows = values(decoded.flatten() % (S*K),
                      decoded.flatten() // (S*K), version)
    ref_rows = ref_rows.view(T, K, H)
    ref_rows[~valid] = 0
    weights = w[:T, :K].contiguous()
    native = torch_npu.npu_moe_token_unpermute(
        ref_rows.reshape(T*K, H),
        torch.arange(T*K, dtype=torch.int32, device=device),
        probs=weights, restore_shape=(T, H))
    exact = torch.zeros(T, H, dtype=torch.float32, device=device)
    old_exact = torch.zeros_like(exact)
    for j in range(K):
        product = ref_rows[:, j].float() * weights[:, j:j+1]
        exact.add_(product)
        old_exact.add_(product.to(torch.bfloat16).float())
    return native, exact.to(torch.bfloat16), old_exact.to(torch.bfloat16), ref_rows

def correctness(S, K, T, kind="normal", version=0):
    name = f"S{S}_K{K}_T{T}_{kind}_v{version}"
    b, plan, hidden, w, raw, _, T = setup_case(S, K, T, kind, version)
    native, exact, old_exact, ref_rows = reference_for(raw, w, S, K, version)
    # All local writers must finish before any reader kernel enters SHMEM.
    sync_all()
    result = b.combine_weighted(plan, hidden, w, num_tokens=T)[0][:T]
    torch.npu.synchronize()
    check(name + "_vs_native", result, native)
    check(name + "_vs_fp32", result, exact)
    if kind == "normal":
        sync_all()
        legacy = b.combine(plan, hidden, num_tokens=T)[0][:T]
        plain = ref_rows[:, 0].float().clone()
        for j in range(1, K):
            plain.add_(ref_rows[:, j].float())
        check(name + "_legacy", legacy, plain.to(torch.bfloat16))
        records.append(dict(name=name + "_old_rounding_evidence", rank=rank,
                            mismatches=int((old_exact != native).sum().item()),
                            elements=native.numel()))
    # Immediate scratch reuse on the same stream with changed source weights.
    w[:, :K].mul_(0.5)
    sync_all()
    second = b.combine_weighted(plan, hidden, w, num_tokens=T)[0][:T]
    check(name + "_scratch_reuse", second, (exact.float() * 0.5).to(torch.bfloat16))
    # Exactly representable 0.5 scaling preserves the BF16 rounding above.
    del result, native, exact, old_exact, second, ref_rows

def rejection_checks():
    b, plan, hidden, w, raw, _, T = setup_case(128, 6, 16)
    cases = [
        ("zero_real_tokens", lambda: b.combine_weighted(plan, hidden, w, num_tokens=0)),
        ("bad_weights_dtype", lambda: b.combine_weighted(plan, hidden, w.to(torch.bfloat16), num_tokens=T)),
        ("bad_weights_stride", lambda: b.combine_weighted(plan, hidden, w[:, :6].contiguous(), num_tokens=T)),
        ("nonalias_zero_copy", lambda: b.combine_weighted(plan, hidden, w, num_tokens=T, zero_copy=True)),
    ]
    for name, fn in cases:
        try:
            fn()
        except (ValueError, TypeError, RuntimeError, NotImplementedError) as exc:
            records.append(dict(name=name, rank=rank, rejected=True,
                                message=str(exc)[:250]))
        else:
            raise AssertionError(f"expected rejection: {name}")
    sync_all()

def perf_case(S, K, T, kind="normal"):
    b, plan, hidden, w, raw, dest_w, T = setup_case(S, K, T, kind)
    def old():
        weighted = (hidden * dest_w[:, None]).to(torch.bfloat16)
        return b.combine(plan, weighted, num_tokens=T)[0]
    route_input = w[:T, :K].contiguous()
    def new():
        w[:T, :K].copy_(route_input)
        return b.combine_weighted(plan, hidden, w, num_tokens=T)[0]
    for _ in range(10):
        old(); new()
    sync_all()
    samples = {"old": [], "new": []}
    peak = {}
    for batch in range(5):
        order = ["old", "new"] if batch % 2 == 0 else ["new", "old"]
        for name in order:
            fn = old if name == "old" else new
            sync_all()
            torch.npu.reset_peak_memory_stats()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                out = fn()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end) / 20
            peak[name] = max(peak.get(name, 0), torch.npu.max_memory_allocated())
            gathered = [None] * world
            dist.all_gather_object(gathered, milliseconds)
            samples[name].append(max(gathered))
            del out
    med = {name: statistics.median(vals) for name, vals in samples.items()}
    item = dict(S=S, K=K, T=T, kind=kind, rank=rank,
                max_rank_samples_ms=samples, medians_ms=med,
                regression_percent=(med["new"] / med["old"] - 1) * 100,
                peak_allocated_bytes=peak)
    timings.append(item)
    if rank == 0:
        print(json.dumps(dict(performance=item)), flush=True)


def planning_round_trip(T=97):
    """Exercise a real native planner/dispatch plan before weighted combine."""
    S, K = 128, 6
    b = buffers[S, K]
    t = torch.arange(S, device=device)
    ids = ((t[:, None] * 13 + torch.arange(K, device=device)[None, :] * 7
            + rank * 3) % 256).int()
    # The plan intentionally includes padded inputs; REAL_T bounds must exclude them.
    counts = torch.bincount(ids[:T].flatten().long(), minlength=256).int()
    hidden = values(t, torch.full_like(t, rank), version=2)
    hidden[T:] = float("nan")
    w = torch.zeros(S, 8, dtype=torch.float32, device=device)
    w[:, :K] = weights_for(t, None, torch.full_like(t, rank), K)
    sync_all()
    shard, _, _, plan = b.dispatch(hidden, route_weights_sk=w[:, :K].contiguous(),
                                  topk_experts_sk=ids,
                                  tokens_per_expert=counts, num_tokens=T)
    torch.npu.synchronize()
    assert int(plan.dup_counts[0].item()) == 0, "SKIP_DUP produced a source-dup prologue"
    expected_rows = hidden[:T, None, :].expand(T,K,H).contiguous()
    native = torch_npu.npu_moe_token_unpermute(
        expected_rows.reshape(T*K,H),
        torch.arange(T*K,dtype=torch.int32,device=device),
        probs=w[:T,:K].contiguous(),restore_shape=(T,H))
    sync_all()
    result = b.combine_weighted(plan, shard, w, num_tokens=T)[0][:T]
    check("real_planning_dispatch_weighted_roundtrip", result, native)

try:
    maps = Path("/proc/self/maps").read_text().splitlines()
    libs = sorted({x.split()[-1] for x in maps if ".so" in x and
                   ("moonep" in x or "libshmem" in x)})
    records.append(dict(name="loaded_libraries", rank=rank, package=ascend_moonep.__file__, libraries=libs))
    if args.check_skip_dup_guard:
        b, plan, hidden, w, raw, _, T = setup_case(128, 6, 16)
        sync_all()
        try:
            b.combine_weighted(plan, hidden, w, num_tokens=T)
        except RuntimeError as exc:
            if "requires ASCEND_MOONEP_SKIP_DUP=1" not in str(exc):
                raise
            records.append(dict(name="SKIP_DUP_off_rejected",rank=rank,passed=True,message=str(exc)))
        else:
            raise AssertionError("SKIP_DUP=0 was accepted")
    elif args.installed_smoke:
        correctness(128, 6, 16)
        planning_round_trip()
        rejection_checks()
    else:
        for S, K, T, kind in [
            (128,6,1,"normal"), (128,6,16,"normal"), (128,6,128,"normal"),
            (8192,6,8192,"normal"), (8192,6,1024,"padding"),
            (8192,6,8192,"unequal"),
            (128,6,128,"alias"), (128,6,128,"invalid"), (128,6,128,"zero"),
            (16,1,16,"normal"), (128,8,128,"normal"), (128,16,128,"normal")]:
            correctness(S, K, T, kind)
        correctness(128, 6, 128, version=1)
        planning_round_trip()
        rejection_checks()
        if args.performance:
            for S, K, T, kind in [(128,6,16,"normal"), (128,6,128,"normal"),
                                   (8192,6,8192,"normal"), (8192,6,8192,"unequal")]:
                perf_case(S,K,T,kind)
    sync_all()
    status = "passed"
except Exception:
    import traceback
    status = "failed"
    records.append(dict(name="failure", rank=rank, traceback=traceback.format_exc()))
    raise
finally:
    (output / f"rank{rank}.json").write_text(json.dumps(
        dict(status=locals().get("status", "aborted"), rank=rank, world=world,
             records=records, timings=timings), indent=2, allow_nan=False))
    if locals().get("status") == "passed":
        sync_all()
        ShmemRuntime.finalize()
        for b in buffers.values():
            b.destroy()
        dist.destroy_process_group()
if rank == 0:
    print("OPERATOR_TESTS_PASSED", flush=True)
