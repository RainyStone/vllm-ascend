"""real-T correctness harness (no engine): for each routing mode, run the full
slot chain (dispatch -> prefetch -> [epn+B] alias GEMM -> combine) and check it
against the full-table einsum reference; save outputs for cross-tag comparison.

Modes (both run in one process, per --tag):
  base : moderate skew, homogeneous T (rank0-agnostic), like smoke_slots
  skew : extreme expert skew (75% of rows on one hot expert) + per-rank
         heterogeneous T ([384, 64, 1][:R]) -- the DP-imbalance shape

Cross-tag comparison (host side, see compare step in the run matrix):
  same tokens under different rungs (512 vs 1024 via --cap + MOONEP_MIN_DISPATCH_TOKENS)
  and under MOONEP_REAL_T 1 vs 0 must produce identical outputs.

Run (R=2 or 3; cards via ASCEND_RT_VISIBLE_DEVICES):
  MOONEP_SLOT_MODE=1 ASCEND_MOONEP_BUILD_DIR=<...> PYTHONPATH=vendored:repo \
  torchrun --nproc_per_node=3 scripts/verify_real_t.py --cap 512 --tag c512rt1 \
      --out /tmp/real_t_check
"""
import argparse
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc


def run_mode(rank, R, dev, E, K, H, F, cap, mode, rt, layer, weights_full):
    """One dispatch+MLP+combine pass; returns (out, ref, info) as CPU tensors."""
    w13_full, w2_full = weights_full
    epn = E // R
    if mode == "base":
        T = 384 if R == 2 else 320
    else:  # skew: heterogeneous per-rank T (rank order: busy -> idle)
        T = [384, 64, 1][:R][rank]
    assert T <= cap

    torch.manual_seed(1234 + rank)
    # routing: base = moderate skew on the OTHER rank's hot expert,
    #         skew = 75% of every rank's rows onto expert 1 (owned by rank 1)
    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    hot = ((rank ^ 1) * epn) if mode == "base" else (1 * epn)
    g = torch.Generator(device=dev).manual_seed(5 + rank + (17 if mode == "skew" else 0))
    rnd = torch.rand(T, K, device=dev, generator=g)
    topk = ((s + 3 * k) % E)
    p_hot = 0.5 if mode == "base" else 0.75
    topk = torch.where(rnd < p_hot, torch.full_like(topk, hot), topk).to(torch.int32).contiguous()
    torch.manual_seed(7 + rank + (23 if mode == "skew" else 0))
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w = torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25

    # reference: full-table einsum per (s,k), weighted K-sum
    guE = torch.einsum("sh,ehf->sef", x.float(), w13_full.float())
    gateE, upE = guE[..., :F], guE[..., F:]
    actE = torch.nn.functional.silu(gateE) * upE
    yE = torch.einsum("tef,efh->teh", actE, w2_full.float())
    y = yE.gather(1, topk.to(torch.int64).unsqueeze(-1).expand(T, K, H))
    ref = (y * w.unsqueeze(-1)).sum(dim=1)

    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_token_dispatch_input,
    )
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEFusedExpertsInput, MoEWeights,
    )
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
    from vllm_ascend.ops.fused_moe.moonep_balance import run_slot_forward

    state = rt._state_by_w13[id(layer.w13_weight)]
    mc._SLOT_TLS.state = state
    try:
        dispatcher = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
        fei = MoEFusedExpertsInput(
            hidden_states=x, topk_weights=w, topk_ids=topk,
            weights=MoEWeights(w1=layer.w13_weight, w2=layer.w2_weight),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False,
        )
        disp = dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=fei))
        meta = disp.combine_metadata
        gl = meta.group_list
        n_slot = int(gl[epn:].sum().item()) if gl is not None else 0
        n_home = int(gl[:epn].sum().item()) if gl is not None else 0
        rung = int(meta.buffer._require_ctx()["S"])
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
    info = dict(mode=mode, T=T, rung=rung, home=n_home, slot=n_slot, rel=rel,
                real_t=bool(mc._REAL_T))
    print(f"[r{rank}] {mode}: T={T} rung={rung} home={n_home} slot={n_slot} "
          f"rel_err={rel:.5f} real_T={info['real_t']}", flush=True)
    return out.cpu(), ref.cpu(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/tmp/real_t_check")
    args = ap.parse_args()

    assert os.environ.get("MOONEP_SLOT_MODE", "0") == "1", "set MOONEP_SLOT_MODE=1"
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"
    assert R in (2, 3), f"this harness targets R=2/3, got {R}"

    afc._moonep_tokens_capacity = args.cap
    mc.get_ep_group = lambda: types.SimpleNamespace(
        world_size=R, rank_in_group=rank, device_group=dist.group.WORLD)
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)

    E, K, H, F = (8, 2, 2048, 128) if R == 2 else (12, 2, 2048, 128)
    epn = E // R
    torch.manual_seed(1234)
    w13_full = (torch.randn(E, H, 2 * F, dtype=torch.bfloat16) * 0.8).to(dev)
    w2_full = (torch.randn(E, F, H, dtype=torch.bfloat16) * 0.8).to(dev)
    layer = types.SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_full[rank * epn:(rank + 1) * epn].contiguous(),
                                      requires_grad=False),
        w2_weight=torch.nn.Parameter(w2_full[rank * epn:(rank + 1) * epn].contiguous(),
                                     requires_grad=False),
    )
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn, layer=layer)
    rt = mc._MOONEP_RUNTIME
    rt._ensure_initialized()

    os.makedirs(args.out, exist_ok=True)
    payload = {}
    for mode in ("base", "skew"):
        out, ref, info = run_mode(rank, R, dev, E, K, H, F, args.cap, mode, rt,
                                  layer, (w13_full, w2_full))
        payload[mode] = dict(out=out, ref=ref, info=info)
    torch.save(payload, f"{args.out}/{args.tag}.r{rank}.pt")
    bad = [m for m, d in payload.items() if d["info"]["rel"] >= 3e-2]
    print(f"[r{rank}] {args.tag}: saved; reference check "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
