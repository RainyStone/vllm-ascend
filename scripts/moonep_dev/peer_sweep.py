"""Per-peer SHMEM MTE put bandwidth, swept one peer at a time.

Why: moonep_dispatch measured 9.2 GB/s/rank at T=11000 while the same bulk-put
code hit 119 GB/s/rank at T=2000, and cutting the put count 31x changed nothing.
This probe found the actual cause -- the machine's two UB clusters and the
cross-socket PCIe path between them:

  rank 0 (its first 3 peers are UB, the rest cross-socket), GB/s per rank:
    P1=45.2  P2=90.3  P3=135.2  P4=87.1  P5=20.3  P6=9.1  P7=7.2

Each UB peer gives ~45 GB/s and they ADD UP; each PCIe peer gives only 5-7 GB/s,
and mixing the two into one round-robin collapses the total. `npu-smi info -t
topo` shows it directly: NPU0-3 and NPU4-7 are UB cliques, everything between
them is SYS (PCIe + cross-NUMA).

Dependency warning
------------------
This script drives `ascend_moonep._C.launch_put_bench`, which is a thin binding
over a `device_put_bench_kernel` that was written for this investigation and
then DELIBERATELY DISCARDED (see docs/dev_results_2026-09-20.md section 9): the
bulk-transfer path it was built alongside was disproved, and the kernel source
was never committed, so it is not recoverable from git.

Rebuilding it is small -- a 32-block AscendC kernel whose per-core loop issues
`aclshmemx_mte_put_nbi(dst, src, ub, kPutBenchPoolBytes, put_bytes, peer, ev)`
into one of `depth` staging buffers sliced out of a 192 KB UB pool, with the
peer chosen either per-put or bound to the core. The bindings it needs are:

  launch_put_bench(src, sym, put_bytes, n_puts, stride_bytes, n_peers,
                   in_flight, burst, depth, block_dim, bind_peer_to_core,
                   dst_bytes)

The measurements above are preserved in the doc either way.

  python3 scripts/moonep_dev/peer_sweep.py           # 8 ranks, cards 0-7
  BD=1 python3 scripts/moonep_dev/peer_sweep.py      # single core per rank
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import torch.multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ascend_moonep._C import launch_put_bench  # noqa: F401
except ImportError:
    sys.exit(
        "launch_put_bench is unavailable: the device_put_bench_kernel it binds "
        "was discarded with the bulk-transfer path (see this file's docstring "
        "and docs/dev_results_2026-09-20.md section 9)."
    )

from ascend_moonep import ShmemInstance, ShmemRuntime  # noqa: E402
from ascend_moonep.buffer_c import create_sym_dist_tensor  # noqa: E402

PB, NP, DEPTH = 8192, 2000, 8
BD = int(os.environ.get("BD", "32"))
PORT = int(os.environ.get("PORT", "29097"))


def run(rank, port):
    torch.npu.set_device(rank)
    sh = ShmemInstance(rank, 8, f"tcp://127.0.0.1:{port}")
    sh.init(512 << 20)
    torch_npu.npu.synchronize()
    sym = create_sym_dist_tensor([64 << 20], torch.uint8, torch.npu.current_device())
    src = torch.ones(65536, dtype=torch.uint8, device=f"npu:{rank}")
    out = []
    for peers in (1, 2, 3, 4, 5, 6, 7):
        best = 1e18
        for _ in range(3):
            launch_put_bench(src, sym, PB, NP, PB, peers, 2, 0, DEPTH, BD, 0, 64 << 20)
            torch_npu.npu.synchronize()
            s = torch_npu.npu.Event(enable_timing=True)
            e = torch_npu.npu.Event(enable_timing=True)
            s.record()
            launch_put_bench(src, sym, PB, NP, PB, peers, 2, 0, DEPTH, BD, 0, 64 << 20)
            e.record()
            e.synchronize()
            best = min(best, s.elapsed_time(e) * 1e3)
        # every core runs n_puts puts, so the byte total scales with block_dim
        out.append(f"P{peers}={PB * NP * BD / best / 1e3:6.1f}")
    os.write(1, (f"rank{rank} GB/s: " + "  ".join(out) + "\n").encode())
    torch_npu.npu.synchronize()
    ShmemRuntime.finalize()
    sh.destroy()


if __name__ == "__main__":
    mp.spawn(run, args=(PORT,), nprocs=8, join=True)
