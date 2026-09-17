"""Standalone planning-kernel microbenchmark -- no vLLM, no engine, ~20 s per run.

Why: in the engine, one planning_kernel call per layer costs ~280 us on the
device (msprof), which is 20% of the slot path's device time, and the pipe
counters say only ~10% of that is actual vec/scalar/mte issue -- the rest is
stalled. To optimize it we need a loop we can iterate on in seconds instead of
minutes, and a clean time-vs-S curve to separate fixed overhead from work that
scales with the route count N = S*K.

Reuses the 4-rank ip-port harness from test/mte/test_planning.py (no
torch.distributed, no Buffer, no model weights).

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 scripts/moonep_dev/bench_planning.py --first-device 4 \
      --sweep 16,64,256,1024,4096 --iters 50

Each sweep point prints two numbers:
  iso  -- one call alone on the stream, device-synced (what the engine sees)
  seq  -- N calls back-to-back, one sync (amortized; hides cross-call overlap)
"""
import argparse
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
# The test harness lives in the external ascend-moonep checkout (the vendored
# copy under third_party/ is the Python layer only).
_ROOT = os.environ.get("ASCEND_MOONEP_SRC", "/home/ascend-moonep")
for _p in (_ROOT, os.path.join(_ROOT, "test"), os.path.join(_ROOT, "test", "mte")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_moonep import ShmemInstance, ShmemRuntime  # noqa: E402
from ascend_moonep.launchers import planning_workspace_sizes  # noqa: E402
from ascend_moonep.buffer_c import create_sym_dist_tensor  # noqa: E402
from ascend_moonep.planning import allocate_planning_outputs  # noqa: E402
from test_planning import (  # noqa: E402
    IpPortPlanningOp,
    compute_planning,
    make_local_topk_case,
)


def _ctx_for(rank, args, shmem, S, mem_size):
    """One context per S: both the local and the symmetric workspace scale with S."""
    R = int(args.ep_size)
    E = int(args.num_experts)
    K = int(args.num_topk)
    NvS = S * K
    local_elements, symmetric_bytes = planning_workspace_sizes(S, K, E, NvS, R, args.core_num)
    if symmetric_bytes > mem_size // 2:
        raise RuntimeError(f"S={S}: symmetric workspace {symmetric_bytes} > heap budget")
    workspace = create_sym_dist_tensor(
        [symmetric_bytes], torch.uint8, torch.npu.current_device()
    )
    device = f"npu:{torch.npu.current_device()}"
    return {
        "rank": rank,
        "R": R,
        "E": E,
        "S": S,
        "K": K,
        "B": E // R,
        "NvS": NvS,
        "weights_buf": torch.empty(0, dtype=torch.int32, device=device),
        "planning_workspace": torch.empty(local_elements, dtype=torch.int32, device=device),
        "kernels": {"planning": IpPortPlanningOp(workspace, symmetric_bytes, args.core_num)},
    }


def _time_barrier(iters, block_dim):
    """Cost of one bare cross-rank aclshmemx_barrier_all_vec (amortized over a
    kernel that contains nothing but that call, so launch overhead drops out).
    This is the floor planning pays 3x per call."""
    from ascend_moonep._C import launch_device_barrier_bench
    launch_device_barrier_bench(5, block_dim)
    torch_npu.npu.synchronize()
    s = torch_npu.npu.Event(enable_timing=True)
    e = torch_npu.npu.Event(enable_timing=True)
    s.record()
    launch_device_barrier_bench(iters, block_dim)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def _time_call(ctx, topk, tpe, cu, plan, iters):
    """(isolated us/call, amortized us/call) with device events."""
    s = torch_npu.npu.Event(enable_timing=True)
    e = torch_npu.npu.Event(enable_timing=True)
    for _ in range(5):  # warmup
        compute_planning(ctx, topk, tpe, cu, plan)
    torch_npu.npu.synchronize()

    iso = []
    for _ in range(iters):
        s.record()
        compute_planning(ctx, topk, tpe, cu, plan)
        e.record()
        torch_npu.npu.synchronize()
        iso.append(s.elapsed_time(e) * 1e3)

    s.record()
    for _ in range(iters):
        compute_planning(ctx, topk, tpe, cu, plan)
    e.record()
    torch_npu.npu.synchronize()
    seq = s.elapsed_time(e) * 1e3 / iters

    iso.sort()
    return iso[len(iso) // 2], iso[int(len(iso) * 0.9)], seq


def run_rank(local_rank, args):
    rank = local_rank
    device_id = int(args.first_device) + local_rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["MOONEP_ASCEND_OP_WS_BYTES"] = "0"
    torch.npu.set_device(device_id)

    sweep = [int(x) for x in args.sweep.split(",")]
    R = int(args.ep_size)
    E = int(args.num_experts or (R * 2))
    K = int(args.num_topk)

    # One heap for the whole sweep, sized for the largest S.
    s_max = max(sweep)
    mem_size = ShmemRuntime.compute_mem_size(
        S=s_max, H=8, K=K, E=E, H_prime=8, num_ep_ranks=R,
        op_ws_bytes=0, num_reduce_buffers=0, num_prefetch_slots=0,
    )
    shmem = ShmemInstance(rank, R, f"tcp://127.0.0.1:{args.shmem_port}")
    shmem.init(mem_size)
    # Bare cross-rank barrier. MUST run in its own process (--barrier): the
    # barrier is a stateful generation counter, and a stray loop of them
    # desynchronises the peers' generations -- a run that did both here saw the
    # following planning calls jump 95 -> 5953 us.
    if args.barrier:
        bl = _time_barrier(max(8, args.iters), args.core_num)
        if rank == 0:
            print(f"bare aclshmemx_barrier_all_vec: {bl:.1f} us/barrier "
                  f"(R={R}, block_dim={args.core_num})")
        ShmemRuntime.finalize()
        shmem.destroy()
        return

    if rank == 0:
        print(f"shmem up: R={R} E={E} K={K} core_num={args.core_num} heap={mem_size/2**20:.1f} MiB")
        print(f"{'S':>6s} {'N=S*K':>8s} {'iso_p50':>9s} {'iso_p90':>9s} {'seq':>9s}  {'us/route':>9s}")
    torch_npu.npu.synchronize()

    try:
        for S in sweep:
            ctx = _ctx_for(rank, args, shmem, S, mem_size)
            topk = make_local_topk_case(
                args.case, rank, R, S, K, E, args.seed
            ).to(device="npu", dtype=torch.int32).contiguous()
            tpe = torch.zeros(E, device="npu", dtype=torch.int32)
            plan, cu = allocate_planning_outputs(ctx)
            p50, p90, seq = _time_call(ctx, topk, tpe, cu, plan, args.iters)
            if rank == 0:
                print(f"{S:6d} {S*K:8d} {p50:9.1f} {p90:9.1f} {seq:9.1f}  {p50/(S*K):9.4f}")
            torch_npu.npu.synchronize()
    finally:
        torch_npu.npu.synchronize()
        if rank == 0:
            print(f"done at {time.strftime('%H:%M:%S')}")
        ShmemRuntime.finalize()
        shmem.destroy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ep-size", type=int, default=4)
    p.add_argument("--first-device", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--core-num", type=int, default=32, help="planning block dim, [8, 56]")
    p.add_argument("--sweep", default="16,64,256,1024,4096", help="S values (tokens per rank)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--barrier", action="store_true",
                   help="measure the bare cross-rank barrier instead of planning "
                        "(own process only; see the comment in run_rank)")
    p.add_argument("--case", default="random", choices=("balanced", "skewed", "duplicate", "random"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shmem-port", type=int, default=28991)
    args = p.parse_args()
    print(f"launching {args.ep_size} ranks on npu {args.first_device}..{args.first_device+args.ep_size-1}")
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()
