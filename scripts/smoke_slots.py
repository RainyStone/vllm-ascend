"""Slot-mode (load-balance) harness: skewed routing must activate prefetch
slots; dispatch->prefetch->[epn+B] alias GEMM->combine must reproduce the
full-table einsum reference. R=2, E=8, H=256, K=2, S=512(cap), T=384.

Run: MOONEP_SLOT_MODE=1 torchrun --nproc_per_node=2 python /tmp/smoke_slots.py
"""
import os
import sys
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc

fake_ep = None


def main():
    global fake_ep
    assert os.environ.get("MOONEP_SLOT_MODE", "0") == "1", "set MOONEP_SLOT_MODE=1"
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"
    assert R == 2

    afc._moonep_tokens_capacity = 512

    class _FakeEp:
        world_size = R
        rank_in_group = rank
        device_group = dist.group.WORLD

    fake_ep = _FakeEp()
    mc.get_ep_group = lambda: fake_ep
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)

    E, K, H, F = 8, 2, 2048, 128
    epn = E // R
    T = 384  # < S: padded

    torch.manual_seed(1234)
    w13_full = (torch.randn(E, H, 2 * F, dtype=torch.bfloat16) * 0.8).to(dev)
    w2_full = (torch.randn(E, F, H, dtype=torch.bfloat16) * 0.8).to(dev)
    w13_local = w13_full[rank * epn:(rank + 1) * epn].contiguous()
    w2_local = w2_full[rank * epn:(rank + 1) * epn].contiguous()

    # fake "layer" whose w13/w2 are this rank's local rows (registered in the
    # runtime registry so ensure_runtime builds + publishes its balance state)
    layer = types.SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_local, requires_grad=False),
        w2_weight=torch.nn.Parameter(w2_local, requires_grad=False),
    )
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn, layer=layer)

    # skewed routing: 50% of rows -> hot expert owned by the OTHER rank
    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    hot = (rank ^ 1) * epn
    g = torch.Generator(device=dev).manual_seed(5 + rank)
    rnd = torch.rand(T, K, device=dev, generator=g)
    topk = ((s + 3 * k) % E)
    topk = torch.where(rnd < 0.5, torch.full_like(topk, hot), topk).to(torch.int32).contiguous()
    torch.manual_seed(7 + rank)
    x = (torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 1.0)
    w = (torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25)

    # reference: full-table compute per (s,k), weighted K-sum
    guE = torch.einsum("sh,ehf->sef", x.float(), w13_full.float())
    gateE, upE = guE[..., :F], guE[..., F:]
    actE = torch.nn.functional.silu(gateE) * upE
    yE = torch.einsum("tef,efh->teh", actE, w2_full.float())
    y = yE.gather(1, topk.to(torch.int64).unsqueeze(-1).expand(T, K, H))
    ref = (y * w.unsqueeze(-1)).sum(dim=1)

    # slot path through the real dispatcher + balance state
    dispatcher = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
    rt = mc._MOONEP_RUNTIME
    rt._ensure_initialized()
    state = rt._state_by_w13[id(layer.w13_weight)]
    print(f"[r{rank}] balance state built; experts_to_copy type ok", flush=True)

    mc._SLOT_TLS.state = state
    try:
        from vllm_ascend.ops.fused_moe.moe_runtime_args import (
            build_mlp_compute_input, build_token_dispatch_input,
        )
        from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
            MoEFusedExpertsInput, MoEWeights,
        )
        from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams

        fei = MoEFusedExpertsInput(
            hidden_states=x, topk_weights=w, topk_ids=topk,
            weights=MoEWeights(w1=w13_local, w2=w2_local),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False,
        )
        disp = dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=fei))
        meta = disp.combine_metadata
        n_slot = int(meta.group_list[epn:].sum().item()) if meta.group_list is not None else 0
        n_home = int(meta.group_list[:epn].sum().item()) if meta.group_list is not None else 0
        print(f"[r{rank}] rows home={n_home} slot={n_slot} total={n_home + n_slot}", flush=True)
        from vllm_ascend.ops.fused_moe.moonep_balance import run_slot_forward

        mlp_rows = run_slot_forward(
            state=meta.state, buffer=meta.buffer, plan=meta.plan,
            token_dispatch_output=disp, fused_experts_input=fei,
            use_fusion_ops=False,
        )
        torch.npu.synchronize()
        out = dispatcher.token_combine(mlp_rows, meta)
        torch.npu.synchronize()
    finally:
        mc._SLOT_TLS.state = None

    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < 3e-2
    print(f"[r{rank}] slot-mode err max={err:.4f} rel={rel:.5f} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
