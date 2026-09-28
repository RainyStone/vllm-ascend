"""顺序判据：MoonEP 的跨设备对称映射 与 AICPU custom 调度器启动 谁先谁后。

背景（2026-09-22 定位）：驱动不允许"进程里已经有跨设备 task group 映射"之后再起
AICPU **custom** 调度器（`svm_attach_task_group` / `halMemBindSibling` 会失败，
dmesg 报 `Task group map dst_devid should be consistent with src_devid`）。
vllm-ascend 的 A5 attention metadata 算子（`csrc/attention/*_metadata`，以
`add_modules_sources_aicpu` 编译）就走这个调度器；调度器起不来 => 算子永不执行 =>
host 卡在随后的 `.item()`。

本脚本用 2 卡（R=2）把顺序拆开验证：
  * ``--when after``  —— 先做 MoonEP 初始化（= 建立跨设备映射），再跑一个最小
    metadata 算子。预期：驱动拒绝、算子不返回（需要在 shell 里套 timeout）。
  * ``--when before`` —— 先跑同一个算子（= 调度器先起来），再做 MoonEP 初始化。
    预期：两边都成功。
  * ``--when none``   —— 只跑 MoonEP 路径（对照，已知 PASS）。

Run:
  cd $REPO
  ASCEND_RT_VISIBLE_DEVICES=0,1 MOONEP_DISABLE_UDMA=1 MOONEP_SLOT_MODE=1 \
  ASCEND_MOONEP_BUILD_DIR=$WS/ascend-moonep/ascend_moonep \
  PYTHONPATH=$REPO/third_party/ascend_moonep:$REPO:$PYTHONPATH \
  timeout 240 torchrun --nproc_per_node=2 scripts/repro_aicpu_vs_moonep_order.py --when after
"""

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401  显式带进 torch.ops._C_ascend
import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc


def launch_aicpu_metadata_op(dev: str, tag: str) -> None:
    """一个最小的 AICPU-custom 算子（DSv4 prefill 用的那个 metadata），1 token。"""
    lens = torch.ones(1, dtype=torch.int32, device=dev)
    print(f"[repro] {tag}: launching aicpu metadata op ...", flush=True)
    torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=lens.clone(),
        actual_seq_lengths_key=lens.clone(),
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=512,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=4,
        device=dev,
    )
    print(f"[repro] {tag}: launched, syncing ...", flush=True)
    torch.npu.synchronize()
    print(f"[repro] {tag}: aicpu op OK", flush=True)


def build_moonep(rank: int, R: int, dev: str, warmup_at: str | None = None) -> None:
    """与 scripts/smoke_slots.py 同构的最小 MoonEP slot 建链 + slot 前向。

    ``warmup_at`` 用来试不同插入点：
      * ``top`` —— 建 Buffer 之前（= _initialize_locked 入口）
      * ``mid`` —— 建完 Buffer、collective init（建立跨设备映射）之前
    """
    afc._moonep_tokens_capacity = 512
    mc.get_ep_group = lambda: types.SimpleNamespace(
        world_size=R, rank_in_group=rank, device_group=dist.group.WORLD)
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)

    E, K, H, F = 8, 2, 2048, 128
    epn = E // R
    T = 384

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

    if warmup_at == "top":
        launch_aicpu_metadata_op(dev, f"r{rank} warmup-top(before-buffers)")

    if warmup_at == "mid":
        # 建完 Buffer、collective init 之前：把 init_with_ip_port 包一层，
        # 正好落在 _initialize_locked 里"建立跨设备映射"的前一步。
        import ascend_moonep

        orig_init = ascend_moonep.ShmemRuntime.init_with_ip_port

        def _patched_init(*a, **kw):
            launch_aicpu_metadata_op(dev, f"r{rank} warmup-mid(before-collective-init)")
            return orig_init(*a, **kw)

        ascend_moonep.ShmemRuntime.init_with_ip_port = staticmethod(_patched_init)

    # 这一步就是"建立跨设备映射"：collective shmem init 把每个 rank 的 VMM
    # chunk 映射进所有 rank 的 VA 空间。
    rt = mc._MOONEP_RUNTIME
    rt._ensure_initialized()
    mt = getattr(rt, "_states", None)
    print(f"[repro] r{rank}: moonep initialized (states={0 if not mt else len(mt)})", flush=True)

    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    g = torch.Generator(device=dev).manual_seed(5 + rank)
    rnd = torch.rand(T, K, device=dev, generator=g)
    topk = ((s + 3 * k) % E)
    topk = torch.where(rnd < 0.5, torch.full_like(topk, (rank ^ 1) * epn), topk)
    topk = topk.to(torch.int32).contiguous()
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w = torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25

    dispatcher = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
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
    # token_dispatch 从 _SLOT_TLS 里取 balance state 塞进 combine_metadata，
    # 所以必须在 dispatch 之前设好（smoke_slots.py 的写法）。
    mc._SLOT_TLS.state = state
    try:
        disp = dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=fei))
        meta = disp.combine_metadata
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
    print(f"[repro] r{rank}: moonep slot forward OK (out {tuple(out.shape)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--when", choices=["none", "before", "after", "top", "mid"], default="after")
    args = ap.parse_args()

    assert os.environ.get("MOONEP_SLOT_MODE", "0") == "1", "set MOONEP_SLOT_MODE=1"
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"
    assert R == 2, "this reproducer is written for R=2"

    torch.npu.synchronize()
    if args.when == "before":
        # 每个 rank 各自把自己进程的 AICPU custom 调度器拉起来，再建跨设备映射。
        launch_aicpu_metadata_op(dev, f"r{rank} warmup-before-moonep")

    build_moonep(rank, R, dev, warmup_at=args.when if args.when in ("top", "mid") else None)

    if args.when in ("after", "before", "top", "mid"):
        # before/top/mid 模式下这次是"修复后"的真实算子：调度器已经起来，
        # peer 映射也已存在。
        launch_aicpu_metadata_op(dev, f"r{rank} aicpu-after-moonep")

    print(f"[repro] r{rank}: DONE when={args.when}", flush=True)


if __name__ == "__main__":
    main()
