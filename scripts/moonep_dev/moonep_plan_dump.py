"""Standalone 4-rank MoonEP plan dump: what does slot mode actually do?

Runs the real planning/dispatch path (no vLLM engine) with a *skewed* routing
table of the production geometry (E=128, K=8, R=4, epn=32) and prints the plan
that comes back: per-expert shard counts, slot-segment counts, dst histogram,
experts_to_copy, zero_fill_ranges, and the global demand histogram.

The decisive comparison is

    owner shard count for expert e   vs   global demand for expert e

computed from plan.dst + topk.  If the owner's count is smaller, slot mode
*moves* rows off the owner; if equal, the slot rows are *additional* work.

Also checks numerics of the owner path (compute only cu[0:E) groups, as the
integration does with MOONEP_SLOT_MODE=0) against a full duplicated-weight
reference, under the same skewed routing.

Run (cards 4-7):
  export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
  PYTHONPATH=/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep \
    torchrun --nproc_per_node=4 /tmp/moonep_plan_dump.py
"""
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend  # noqa: F401
import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc

E, K, H, F, R = 128, 8, 256, 64, 4
S = int(os.environ.get("PLAN_S", "64"))       # static capacity = pooled rung
T = int(os.environ.get("PLAN_T", "32"))       # real tokens on this rank
SKEW = float(os.environ.get("PLAN_SKEW", "0.6"))  # fraction of entries to hot experts
HOT = [4, 20, 36, 68, 84, 100, 116, 124]      # 2 per rank, like real MoE hot experts


def _skewed_topk(rank: int) -> torch.Tensor:
    """[T, K] expert ids, top-k of a skewed router score (distinct per token).

    p(e) = SKEW/|HOT| for the hot experts, (1-SKEW)/(E-|HOT|) otherwise, and the
    K highest scores win -- exactly the structure a real top-k router produces
    (no repeated expert within a token).
    """
    g = torch.Generator(device="cpu").manual_seed(1000 + rank)
    p = torch.full((E,), (1.0 - SKEW) / (E - len(HOT)), dtype=torch.float64)
    p[torch.tensor(HOT)] = SKEW / len(HOT)
    u = torch.rand(T, E, generator=g, dtype=torch.float64).clamp_min(1e-12)
    score = u.log() / p                                  # Gumbel-top-k over p
    return score.topk(K, dim=1).indices.to(torch.int32)


def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    Rr = dist.get_world_size()
    assert Rr == R, Rr
    # Keep the full visible list and bind an absolute device id, so CANN's
    # in-process logical id equals the physical id: shmem's buffer-property
    # validation compares the two and fails with -100 otherwise
    # (README limitation #7).
    dev_id = int(os.environ.get("PLAN_DEV_BASE", "0")) + rank
    torch.npu.set_device(dev_id)
    dev = f"npu:{dev_id}"
    epn = E // R

    afc._moonep_tokens_capacity = S
    afc.get_moonep_tokens_capacity = lambda: S

    class _FakeEp:
        world_size = R
        rank_in_group = rank
        device_group = dist.group.WORLD

    ep = _FakeEp()
    mc.get_ep_group = lambda: ep
    mc.get_tensor_model_parallel_world_size = lambda: 1
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn)

    # capture the plan out of Buffer.dispatch
    import ascend_moonep.api as api
    orig = api.Buffer.dispatch
    captured = {}

    def wrapped(self, *a, **kw):
        out = orig(self, *a, **kw)
        cu = out[2]
        plan = out[3]
        torch.npu.synchronize()
        captured["cu"] = cu.to(torch.int64).cpu()
        captured["plan"] = plan
        captured["dst"] = plan.dst.to(torch.int64).cpu()
        captured["etc"] = plan.experts_to_copy[rank].to(torch.int64).cpu()
        captured["zfr"] = plan.zero_fill_ranges.to(torch.int64).cpu()
        captured["dupc"] = plan.dup_counts.to(torch.int64).cpu()
        captured["rs"] = plan.remote_stats.to(torch.int64).cpu()
        return out

    api.Buffer.dispatch = wrapped

    topk = _skewed_topk(rank).to(dev)
    torch.manual_seed(7 + rank)
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w = torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25

    torch.manual_seed(1234)
    w13_full = (torch.randn(E, H, 2 * F, dtype=torch.bfloat16) * 0.8).to(dev)
    w2_full = (torch.randn(E, F, H, dtype=torch.bfloat16) * 0.8).to(dev)
    w13_local = w13_full[rank * epn:(rank + 1) * epn].contiguous()
    w2_local = w2_full[rank * epn:(rank + 1) * epn].contiguous()

    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_token_dispatch_input, build_mlp_compute_input)
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput, MoEWeights
    from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp

    def _fei():
        return MoEFusedExpertsInput(
            hidden_states=x, topk_weights=w, topk_ids=topk,
            weights=MoEWeights(w1=w13_local, w2=w2_local),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False)

    dispatcher = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
    disp_out = dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=_fei()))
    mlp_in = build_mlp_compute_input(fused_experts_input=_fei(),
                                     token_dispatch_output=disp_out, use_fusion_ops=False)
    rows, _ = unified_apply_mlp(mlp_compute_input=mlp_in)
    out = dispatcher.token_combine(rows, disp_out.combine_metadata)
    torch.npu.synchronize()

    # reference with full duplicated weights
    guE = torch.einsum("sh,ehf->sef", x.float(), w13_full.float())
    actE = torch.nn.functional.silu(guE[..., :F]) * guE[..., F:]
    yE = torch.einsum("tef,efh->teh", actE, w2_full.float())
    y = yE.gather(1, topk.long().unsqueeze(-1).expand(T, K, H))
    ref = (y * w.unsqueeze(-1)).sum(dim=1)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()

    cu = captured["cu"]
    dst = captured["dst"]
    counts = cu - torch.cat((cu.new_zeros(1), cu[:-1]))     # len E+B
    owned = counts[:E]
    slots = counts[E:]
    B = int(captured["etc"].numel())

    # global demand per expert (all ranks, incl. pads), from topk + dst
    topk_c = topk.to(torch.int64).cpu()
    print(f"[r{rank}] ==== S={S} T={T} B={B} skew={SKEW} err_rel={err:.2e} "
          f"{'OWNER-PATH-OK' if err < 3e-2 else 'OWNER-PATH-WRONG'} ====", flush=True)
    print(f"[r{rank}] rows: N=S*K={S * K} owner_shard[0:E].sum={int(owned.sum())} "
          f"slots[E:E+B].sum={int(slots.sum())} pad_entries={(S - T) * K}", flush=True)
    print(f"[r{rank}] slots non-zero: "
          f"{int((slots > 0).sum())}/{B} counts={slots[slots > 0].tolist()}", flush=True)
    print(f"[r{rank}] all_counts(nonzero in my shard)="
          f"{{ {', '.join(f'{i}:{int(v)}' for i, v in enumerate(owned) if v > 0)} }}", flush=True)
    print(f"[r{rank}] experts_to_copy={captured['etc'].tolist()}", flush=True)
    print(f"[r{rank}] slot zero_fill_ranges={captured['zfr'][E:].tolist()}", flush=True)
    print(f"[r{rank}] dup_counts={captured['dupc'].tolist()} remote_stats="
          f"{captured['rs'].tolist()}", flush=True)

    # dst histogram: real entries (i < T*K) vs pad entries
    real_dst = dst[:T * K]
    pad_dst = dst[T * K:]

    def _hist(d):
        h = [int((d == r).sum()) for r in range(R)]
        neg = int((d < 0).sum())
        return h, neg

    rd, rneg = _hist(real_dst)
    pd, pneg = _hist(pad_dst)
    print(f"[r{rank}] dst: real rows={real_dst.numel()} hist={rd} neg={rneg} | "
          f"pad rows={pad_dst.numel()} hist={pd} neg={pneg}", flush=True)

    # demand per expert (global, from routing incl. pads) vs what the owner holds
    flat_ids = topk_c.reshape(-1)
    demand = torch.zeros(E, dtype=torch.int64, device=dev)
    demand.scatter_add_(0, flat_ids.to(dev), torch.ones_like(flat_ids, device=dev))
    # pads route to the rank's own first expert
    demand[rank * epn] += (S - T) * K
    try:
        dist.all_reduce(demand)                  # global demand, same on all ranks
    except Exception as exc:  # noqa: BLE001
        print(f"[r{rank}] all_reduce int64 failed ({exc}); retry as float32", flush=True)
        df = demand.float()
        dist.all_reduce(df)
        demand = df.long()
    demand = demand.cpu()

    my = slice(rank * epn, (rank + 1) * epn)
    d_my = demand[my]
    o_my = owned[my]
    print(f"[r{rank}] experts {rank * epn}..{(rank + 1) * epn - 1}: "
          f"global_demand={int(d_my.sum())} owner_shard={int(o_my.sum())} "
          f"kept={int((d_my - o_my).sum())} (kept>0 = rows the owner no longer computes)", flush=True)
    order = torch.argsort(d_my, descending=True)[:6]
    for j in order.tolist():
        e = rank * epn + j
        print(f"[r{rank}]   expert {e}: demand={int(demand[e])} "
              f"owner_shard={int(owned[e])} diff={int(owned[e] - demand[e])}"
              f"{' [HOT]' if e in HOT else ''}", flush=True)
    # which remote experts did I mirror?
    for i, e in enumerate(captured["etc"].tolist()):
        if e >= 0:
            print(f"[r{rank}]   slot[{i}] mirrors expert {e} (owner rank {e // epn}), "
                  f"demand={int(demand[e])}, my slot rows={int(slots[i])}", flush=True)


if __name__ == "__main__":
    main()
