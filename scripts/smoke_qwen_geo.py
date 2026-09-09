"""4-rank synthetic numerics smoke test for the MoonEP comm method.

Exercises, on each rank: TokenDispatcherWithMoonEP.token_dispatch ->
unified_apply_mlp (real npu_grouped_matmul MLP) -> token_combine, then
compares against a per-rank reference that computes every routed expert
locally with full duplicated weight tables (same values, same kernels).

Run (from anywhere, NPUs 0-3):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  PYTHONPATH=/home/vllm-ascend-moonep torchrun --nproc_per_node=4 python /tmp/smoke_moonep_ep.py
"""
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

# --- shadow the editable vllm_ascend install with the moonep repo ---
import vllm_ascend  # noqa: F401
import vllm_ascend.ascend_forward_context as afc

# --- patch global state accessors the integration reads (no full engine) ---
import vllm_ascend.ops.fused_moe.moonep_comm as mc

fake_ep = None


def main():
    global fake_ep
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"
    assert R == 2

    # S capacity (static Buffer size)
    afc._moonep_tokens_capacity = 512
    assert afc.get_moonep_tokens_capacity() == 512

    class _FakeEp:
        world_size = R
        rank_in_group = rank
        device_group = dist.group.WORLD

    fake_ep = _FakeEp()
    mc.get_ep_group = lambda: fake_ep
    mc.get_tensor_model_parallel_world_size = lambda: 1
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)
    # PrepareAndFinalizeWithAll2All reads vllm parallel state for tp group —
    # bypassed: we do not call prepare/finalize in this smoke (tp=1 no-ops).

    # --- geometry: E=8, epn=2, K=2, H=16, F=32, T=12, S=16 ---
    E, K, H, F = 128, 8, 2048, 384
    epn = E // R
    T = 384  # < S: padding exercised
    assert H % 16 == 0

    # Register fake "layers" so the runtime pool materializes (K=2 main
    # geometry + K=1 pure-local geometry used in check D).
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn)
    mc.register_moonep_layer(E=E, K=1, H=H, nle=epn)

    # Deterministic routing: expert = (s + 3k) % E (same topk on all ranks,
    # like a real step; per-rank tokens differ in value only).
    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    topk = ((s + 3 * k) % E).to(torch.int32).contiguous()
    torch.manual_seed(7 + rank)
    x = (torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 1.0)
    w = (torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25)

    # Full duplicated weight tables for the *reference* on every rank.
    torch.manual_seed(1234)
    w13_full = (torch.randn(E, H, 2 * F, dtype=torch.bfloat16) * 0.8).to(dev)
    w2_full = (torch.randn(E, F, H, dtype=torch.bfloat16) * 0.8).to(dev)
    # Local tables = the owned rows of the same tensors (as in real EP).
    w13_local = w13_full[rank * epn:(rank + 1) * epn].contiguous()
    w2_local = w2_full[rank * epn:(rank + 1) * epn].contiguous()

    # --- reference: for each (s,k) row select expert e = topk[s,k] and compute
    # swiglu(x[s] @ w13[e]) @ w2[e], then the weighted K-sum. Per-expert acts
    # are e-insensitive to k, so compute once per token and gather.
    guE = torch.einsum("sh,ehf->sef", x.float(), w13_full.float())  # [T,E,2F]
    gateE, upE = guE[..., :F], guE[..., F:]
    actE = torch.nn.functional.silu(gateE) * upE  # [T,E,F]
    yE = torch.einsum("tef,efh->teh", actE, w2_full.float())  # [T,E,H]
    y = yE.gather(1, topk.to(torch.int64).unsqueeze(-1).expand(T, K, H))  # [T,K,H]
    ref = (y * w.unsqueeze(-1)).sum(dim=1)  # [T,H] fp32

    # --- MoonEP path ---
    dispatcher = mc.TokenDispatcherWithMoonEP(
        top_k=K, num_experts=E, num_local_experts=epn
    )
    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_token_dispatch_input
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput, MoEWeights
    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_mlp_compute_input
    from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp

    # Build the same inputs the runner would: hidden [T,H] bf16
    disp_out = dispatcher.token_dispatch(
        build_token_dispatch_input(
            fused_experts_input=MoEFusedExpertsInput(
                hidden_states=x,
                topk_weights=w,
                topk_ids=topk,
                weights=MoEWeights(w1=w13_local, w2=w2_local),
                routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0, mc2_mask=None, apply_router_weight_on_input=False),
                quant=MoEQuantParams(),
                dynamic_eplb=False,
            )
        )
    )
    mlp_in = build_mlp_compute_input(
        fused_experts_input=MoEFusedExpertsInput(
            hidden_states=x,
            topk_weights=w,
            topk_ids=topk,
            weights=MoEWeights(w1=w13_local, w2=w2_local),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0, mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(),
            dynamic_eplb=False,
        ),
        token_dispatch_output=disp_out,
        use_fusion_ops=False,
    )
    # (A) no-GEMM reference through the same comm path: rows = dispatched x,
    # weighted and combined -> out[s] must equal sum_k w[s,k] * x[s]
    no_gemm_out = dispatcher.token_combine(disp_out.hidden_states, disp_out.combine_metadata)
    ref_no_gemm = (x.float().unsqueeze(1) * w.unsqueeze(-1)).sum(dim=1)
    err_ng = (no_gemm_out.float() - ref_no_gemm).abs().max().item()

    mlp_rows, _ = unified_apply_mlp(mlp_compute_input=mlp_in)
    out = dispatcher.token_combine(mlp_rows, disp_out.combine_metadata)
    torch.npu.synchronize()
    print(f"[r{rank}] (A) noGEMM err={err_ng:.5f} (expect ~0)", flush=True)
    # (D) pure-local configuration: K=1, every token routes to the single
    # expert owned by this rank (e = rank). Zero cross-rank traffic: this
    # isolates MLP row/expert semantics from the comm content.
    torch.manual_seed(99 + rank)
    xd = (torch.randn(384, H, device=dev, dtype=torch.bfloat16) * 1.0)
    wd = torch.rand(384, 1, device=dev, dtype=torch.float32) * 0.5 + 0.25
    topkd = torch.full((384, 1), rank * epn, dtype=torch.int32, device=dev)
    eD = rank * epn
    guD = torch.einsum("sh,ehf->sef", xd.float(), w13_full.float())[:, eD, :]
    gateD, upD = guD[..., :F], guD[..., F:]
    actD = torch.nn.functional.silu(gateD) * upD
    refD = torch.einsum("sf,fh->sh", actD, w2_full[eD].float()) * wd.float()
    d = mc.TokenDispatcherWithMoonEP(top_k=1, num_experts=E, num_local_experts=epn)
    dd = d.token_dispatch(build_token_dispatch_input(
        fused_experts_input=MoEFusedExpertsInput(
            hidden_states=xd, topk_weights=wd, topk_ids=topkd,
            weights=MoEWeights(w1=w13_local, w2=w2_local),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False)))
    mlpD = unified_apply_mlp(mlp_compute_input=build_mlp_compute_input(
        fused_experts_input=MoEFusedExpertsInput(
            hidden_states=xd, topk_weights=wd, topk_ids=topkd,
            weights=MoEWeights(w1=w13_local, w2=w2_local),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False),
        token_dispatch_output=dd, use_fusion_ops=False))
    outD = d.token_combine(mlpD[0], dd.combine_metadata)
    torch.npu.synchronize()
    print(f"[r{rank}] (D) pure-local err={(outD.float() - refD).abs().max().item():.5f}", flush=True)

    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < 3e-2
    print(f"[r{rank}] moonep out err max={err:.5f} rel={rel:.5f} shape={tuple(out.shape)} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[r{rank}] out[0]={out[0][:6].tolist()}", flush=True)
    print(f"[r{rank}] ref[0]={ref[0][:6].tolist()}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
