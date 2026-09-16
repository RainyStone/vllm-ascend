"""Can a MoonEP Buffer be created *after* ShmemRuntime init? (lazy-rung probe)

Initializes the integration path the normal way (register layer -> first
token_dispatch materializes the pool + collective shmem init), then constructs
one extra Buffer with a capacity that was NOT in the pool (48) and runs a
dispatch on it. PASS = it initialized and the shard's per-expert counts sum to
the expected N = S*K.

Run:  bash /tmp/run_lazyprobe.sh
"""
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend  # noqa: F401
import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moonep_comm as mc

E, K, H, R = 8, 2, 256, 4
S_BASE = 512
S_NEW = 48


def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == R
    dev_id = int(os.environ.get("PLAN_DEV_BASE", "0")) + rank
    torch.npu.set_device(dev_id)
    dev = f"npu:{dev_id}"
    epn = E // R

    afc._moonep_tokens_capacity = S_BASE
    afc.get_moonep_tokens_capacity = lambda: S_BASE

    class _FakeEp:
        world_size = R
        rank_in_group = rank
        device_group = dist.group.WORLD

    ep = _FakeEp()
    mc.get_ep_group = lambda: ep
    mc.get_tensor_model_parallel_world_size = lambda: 1
    mc._EXTRA_CTX = types.SimpleNamespace(capturing=False)
    mc.register_moonep_layer(E=E, K=K, H=H, nle=epn)

    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_token_dispatch_input
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEFusedExpertsInput, MoEWeights

    T = 32
    torch.manual_seed(7 + rank)
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w = torch.rand(T, K, device=dev, dtype=torch.float32) * 0.5 + 0.25
    s = torch.arange(T, device=dev).view(-1, 1)
    k = torch.arange(K, device=dev).view(1, -1)
    topk = ((s + 3 * k) % E).to(torch.int32).contiguous()
    w13 = torch.zeros(epn, H, 2 * 32, device=dev, dtype=torch.bfloat16)
    w2 = torch.zeros(epn, 32, H, device=dev, dtype=torch.bfloat16)

    def _fei():
        return MoEFusedExpertsInput(
            hidden_states=x, topk_weights=w, topk_ids=topk,
            weights=MoEWeights(w1=w13, w2=w2),
            routing=MoERoutingParams(expert_map=None, global_redundant_expert_num=0,
                                     mc2_mask=None, apply_router_weight_on_input=False),
            quant=MoEQuantParams(), dynamic_eplb=False)

    d = mc.TokenDispatcherWithMoonEP(top_k=K, num_experts=E, num_local_experts=epn)
    d.token_dispatch(build_token_dispatch_input(fused_experts_input=_fei()))  # materializes pool + shmem init
    torch.npu.synchronize()
    pool = mc._MOONEP_RUNTIME._buffers[(E, K, H)]
    print(f"[r{rank}] pool after first dispatch: {sorted(pool)} caps", flush=True)

    # --- the actual probe: a Buffer for a capacity the pool never built ---
    from ascend_moonep import Buffer

    try:
        buf = Buffer(S_NEW, H, K, E, R, token_padding=1, B=epn,
                     group=dist.group.WORLD, explicitly_destroy=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[r{rank}] LAZY-CREATE-FAILED at construction: {type(exc).__name__}: {exc}", flush=True)
        return

    N = S_NEW * K
    xs = torch.randn(S_NEW, H, device=dev, dtype=torch.bfloat16)
    ws = torch.rand(S_NEW, K, device=dev, dtype=torch.float32)
    tpe = torch.histc(xs.new_zeros(1).to(torch.float32), bins=E).to(torch.int32)
    topk_s = torch.zeros(S_NEW, K, dtype=torch.int32, device=dev)
    tpe = torch.zeros(E, dtype=torch.int32, device=dev)
    tpe[:] = 0
    tpe[0] = (S_NEW - 1) * K
    tpe[1] = K
    topk_s[:, :] = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    try:
        _h, _w, cu, _plan = buf.dispatch(xs, ws, topk_s, tpe, zero_copy=False, inter_rank_sync=True)
        torch.npu.synchronize()
        cu64 = cu.to(torch.int64)
        counts = cu64 - torch.cat((cu64.new_zeros(1), cu64[:-1]))
        ssum = int(counts.sum().item())
        print(f"[r{rank}] LAZY-CREATE-OK: dispatch ran, cu sum={ssum} expected N={N} "
              f"{'PASS' if ssum == N else 'MISMATCH'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[r{rank}] LAZY-CREATE-FAILED at dispatch: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            buf.destroy()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
