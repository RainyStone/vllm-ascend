"""Multi-hot-expert skew probe in slot mode: does the Ascend planner ever put
rows of MORE THAN B distinct remote experts on one rank (overflow beyond the
B=epn prefetch slots)? R=3, E=12 (epn=B=4, foreign max = 8 > B). Checks
plan.remote_stats, foreign cu groups, and full numeric slot-path validation.

Run: MOONEP_SLOT_MODE=1 torchrun --nproc_per_node=3 python /tmp/probe_slots_multi.py
"""
import os
import sys
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc


def main():
    assert os.environ.get("MOONEP_SLOT_MODE", "0") == "1"
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"
    assert R == 3

    afc._moonep_tokens_capacity = 512
    mc.get_ep_group = lambda: types.SimpleNamespace(
        world_size=R, rank_in_group=rank, device_group=dist.group.WORLD)
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)

    E, K, H, F = 12, 2, 2048, 128  # epn=4, B=4; 2F=256
    epn = E // R
    T = 384
    B = epn

    torch.manual_seed(1234)
    w13_full = (torch.randn(E, H, 2 * F, dtype=torch.bfloat16) * 0.8).to(dev)
    w2_full = (torch.randn(E, F, H, dtype=torch.bfloat16) * 0.8).to(dev)
    w13_local = w13_full[rank * epn:(rank + 1) * epn].contiguous()
    w2_local = w2_full[rank * epn:(rank + 1) * epn].contiguous()
    layer = types.SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_local, requires_grad=False),
        w2_weight=torch.nn.Parameter(w2_local, requires_grad=False),
    )
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn, layer=layer)

    # multi-hot skew: rank r hammers the experts of the NEXT two ranks
    # (distinct remote experts = 2*epn = 8 > B=4) with strong bias.
    hot_rem = [((rank + 1) % R) * epn + i for i in range(epn)] + \
              [((rank + 2) % R) * epn + i for i in range(epn)]
    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    g = torch.Generator(device=dev).manual_seed(5 + rank)
    rnd = torch.rand(T, K, device=dev, generator=g)
    topk_u = ((s + 3 * k) % E)
    # 40% -> hot_rem[0], 25% -> hot_rem[1], rest uniform
    topk = torch.where(rnd < 0.40, torch.full_like(topk_u, hot_rem[0]), topk_u)
    topk = torch.where((rnd >= 0.40) & (rnd < 0.65),
                       torch.full_like(topk, hot_rem[1]), topk)
    topk = topk.to(torch.int32).contiguous()
    x = (torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 1.0)
    w = (torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25)

    # reference (full tables, valid wherever rows are computed)
    guE = torch.einsum("sh,ehf->sef", x.float(), w13_full.float())
    gateE, upE = guE[..., :F], guE[..., F:]
    actE = torch.nn.functional.silu(gateE) * upE
    yE = torch.einsum("tef,efh->teh", actE, w2_full.float())
    y = yE.gather(1, topk.to(torch.int64).unsqueeze(-1).expand(T, K, H))
    ref = (y * w.unsqueeze(-1)).sum(dim=1)

    dispatcher = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
    rt = mc._MOONEP_RUNTIME
    rt._ensure_initialized()
    state = rt._state_by_w13[id(layer.w13_weight)]

    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_token_dispatch_input
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput, MoEWeights
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
    from vllm_ascend.ops.fused_moe.moonep_balance import run_slot_forward

    fei = MoEFusedExpertsInput(
        hidden_states=x, topk_weights=w, topk_ids=topk,
        weights=MoEWeights(w1=w13_local, w2=w2_local),
        routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                 mc2_mask=None, apply_router_weight_on_input=False),
        quant=MoEQuantParams(), dynamic_eplb=False,
    )
    mc._SLOT_TLS.state = state
    try:
        disp = dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=fei))
        meta = disp.combine_metadata
        plan = meta.plan
        cu = plan._cu_seqlens if hasattr(plan, "_cu_seqlens") else None
        rs = getattr(plan, "remote_stats", None)
        rs_v = rs.cpu().tolist() if rs is not None else None
        e2c = plan.experts_to_copy[rank].cpu().tolist()
        gl = meta.group_list
        n_home = int(gl[:epn].sum().item())
        n_slot = int(gl[epn:].sum().item())
        # foreign-group rows: from cu counts outside home & slot ranges
        foreign = None
        if cu is not None:
            counts = (cu[1:] - cu[:-1]).cpu().tolist() if cu.numel() > 1 else None
            if counts is not None:
                owned = set(range(rank * epn, (rank + 1) * epn))
                foreign = sum(c for i, c in enumerate(counts[:E]) if i not in owned)
        print(f"[r{rank}] hot_rem={hot_rem}", flush=True)
        print(f"[r{rank}] remote_stats={rs_v} experts_to_copy={e2c} "
              f"rows home={n_home} slot={n_slot} foreign_rows={foreign} "
              f"(B={B})", flush=True)
        mlp_rows = run_slot_forward(
            state=meta.state, buffer=meta.buffer, plan=meta.plan,
            hidden_trim=disp.hidden_states, group_list=meta.group_list,
            fused_experts_input=fei)
        torch.npu.synchronize()
        out = dispatcher.token_combine(mlp_rows, meta)
        torch.npu.synchronize()
    finally:
        mc._SLOT_TLS.state = None

    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < 3e-2
    print(f"[r{rank}] slot multi-hot err rel={rel:.5f} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
