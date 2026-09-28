#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 MC2 dispatch 探针：定位 off 运行（标准 MC2 all-to-all）561000 的根因。

背景
----
off 运行在启动 profile 阶段的第一层 MoE 挂：
    npu_moe_distribute_dispatch_v2 → call aclnnMoeDistributeDispatchV4 failed, error code is 561000
host plog（/root/ascend/log/debug/plog/）里的真根因：
    [mc2_context.cpp:160][GetHcclCommLink] No matching communication protocol found
                                          in HCCL links protocol is 6
    [mc2_context.cpp:333][CreatMc2Context] Get HCCL communication resource failed
6 = COMM_PROTOCOL_UB_MEM（见 CANN 头 hccl/hcomm_res_defs.h）。libopapi.so 里 MC2 的选择逻辑是：

    CheckProtocolSupport: "start CheckProtocolSupport, layerNum is %u"
                          "Get HCCL layers success, netLayerNum is: %u"
                          "HCCL communication layerNum is %u,so set protocol to UB_MEM"
                          取不到链路时报 "No matching communication protocol found ... protocol is %d"

即**通信域跨层（layerNum > 1）时 MC2 强制要 UB_MEM 协议**；UB 链路没建立起来（卡上
`8F180A00 UB Link Establishment Failure Major Alarm`）就取不到 UB_MEM 链路 → 561000。

本探针不加载模型、不依赖 vllm / moonep，只起 N 卡 HCCL 通信域再打一次 MC2 dispatch：
  1. 先 all_reduce 一次，证明通信域本身可用（排除 HCCL 整体故障）；
  2. 再按 off 运行 `token_dispatcher.get_dispatch_mc2_kwargs` 的真实形态调 MC2 dispatch。
预期判别：N=8 跨两个 UB 域（0-3 / 4-7，跨域走 SYS）→ layerNum>1 → 复现 561000；
          N=4 落在同一个 UB 域内 → layerNum=1 → 不走 UB_MEM，应当通过。

用法（8 卡，复现 off 挂）：
  cd <repo>
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_GLOBAL_LOG_LEVEL=1 \
  python -m torch.distributed.run --nproc_per_node=8 --master_port=29601 \
      scripts/repro_mc2_ub_link.py
同一 UB 域内 4 卡（对照，应当通过）：
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.run --nproc_per_node=4 ...
跨 UB 域 4 卡（对照，应当复现）：
  ASCEND_RT_VISIBLE_DEVICES=0,1,4,5  python -m torch.distributed.run --nproc_per_node=4 ...

通信域相关的环境变量（HCCL_IF_IP / 网卡 / HCCL_BUFFSIZE）脚本里给了与
`scripts/dsv4flash_service/start_service_mode.sh` 相同的默认值，可用环境变量覆盖。
"""
import os
import sys
import traceback

# 这些环境变量必须在 torch_npu / HCCL 加载前设好（与 start_service_mode.sh 的 off 运行一致）
os.environ.setdefault("HCCL_IF_IP", "90.90.93.34")          # 本机 ens6f1np1 的 IP
os.environ.setdefault("HCCL_SOCKET_IFNAME", "ens6f1np1")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "ens6f1np1")
os.environ.setdefault("TP_SOCKET_IFNAME", "ens6f1np1")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("OMP_PROC_BIND", "false")

import argparse  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402

EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0  # 与 vllm_ascend token_dispatcher 一致


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens-per-rank", type=int, default=64, help="每个 rank 送进 dispatch 的 token 数（global_bs = 该值 × world）")
    p.add_argument("--hidden-size", type=int, default=4096, help="DeepSeek-V4-Flash hidden_size")
    p.add_argument("--topk", type=int, default=6, help="num_experts_per_tok")
    p.add_argument("--experts", type=int, default=256, help="n_routed_experts + 冗余专家数")
    p.add_argument("--quant-mode", type=int, default=0, help="0=不量化（先只验协议）, 4=A5 mxfp 通信量化")
    p.add_argument("--y-dtype", default="", help="quant_mode 非 0 时的 y_dtype，如 float8_e4m3fn")
    p.add_argument("--skip-mc2", action="store_true", help="只跑 all_reduce 连通性，不调 MC2")
    return p.parse_args()


def log(msg, rank=None):
    prefix = "" if rank is None else f"[rank {rank}] "
    print(f"{prefix}{msg}", flush=True)


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)
    dev = torch.npu.current_device()
    log(f"init ok: world={world} local_rank={local_rank} device={dev} name={torch.npu.get_device_name(dev)}", rank)

    # ---- 步骤 1：all_reduce，证明通信域本身可用 ----
    t = torch.full((4,), float(rank + 1), dtype=torch.float32, device=f"npu:{dev}")
    dist.all_reduce(t)
    expect = world * (world + 1) / 2.0
    got = t[0].item()
    ar_ok = abs(got - expect) < 1e-3
    log(f"all_reduce ok={ar_ok} got={got} expect={expect}", rank)

    # ---- 取 HCCL 通信域名（off 运行里 MC2 的 group_ep / group_tp 都是它）----
    comm_name = ""
    try:
        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device("npu"))
        comm_name = backend.get_hccl_comm_name(rank)
    except Exception as e:  # noqa: BLE001
        log(f"取 comm name 失败（不致命，MC2 仍会试）: {type(e).__name__}: {e}", rank)
    all_names = ["" for _ in range(world)]
    dist.all_gather_object(all_names, comm_name)
    if rank == 0:
        log(f"group_ep 名字: {all_names}")
    log(f"comm_name={comm_name}", rank)

    if args.skip_mc2:
        dist.barrier()
        dist.destroy_process_group()
        return

    # ---- 步骤 2：按 off 运行的形态调 MC2 dispatch ----
    T, H, K = args.tokens_per_rank, args.hidden_size, args.topk
    torch.manual_seed(1234 + rank)
    kwargs = {
        "x": (torch.randn(T, H, dtype=torch.bfloat16, device=f"npu:{dev}") * 0.01),
        "expert_ids": torch.randint(0, args.experts, (T, K), dtype=torch.int32, device=f"npu:{dev}"),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": args.experts,
        "global_bs": T * world,
        "expert_token_nums_type": EXPERT_TOKEN_NUMS_TYPE_CUMSUM,
        "scales": None,
        "quant_mode": args.quant_mode,
        "group_ep": comm_name,
        "ep_world_size": world,
        "ep_rank_id": rank,
        # A5（need_extra_args）路径额外带的参数，off 运行同样会带
        "group_tp": comm_name,
        "tp_world_size": 1,
        "tp_rank_id": 0,
    }
    topk_w = torch.rand((T, K), dtype=torch.float32, device=f"npu:{dev}")
    kwargs["expert_scales"] = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.float32)
    if args.y_dtype:
        kwargs["y_dtype"] = getattr(torch, args.y_dtype)

    log(f"calling npu_moe_distribute_dispatch_v2: T={T} H={H} K={K} "
        f"quant_mode={args.quant_mode} global_bs={T * world} ep_rank={rank}/{world}", rank)

    err = ""
    shapes = ""
    try:
        out = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        shapes = " | ".join(str(tuple(o.shape)) if hasattr(o, "shape") else repr(o) for o in out[:7])
        log(f"MC2 dispatch OK: {shapes}", rank)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        log("MC2 dispatch FAILED:\n" + "".join(traceback.format_exc()), rank)

    # ---- 汇总（rank0 打印一行判定）----
    status = torch.tensor([0 if shapes else 1], dtype=torch.int32, device=f"npu:{dev}")
    gathered = [torch.zeros_like(status) for _ in range(world)]
    dist.all_gather(gathered, status)
    errs = ["" for _ in range(world)]
    dist.all_gather_object(errs, err)
    if rank == 0:
        codes = [int(x.item()) for x in gathered]
        log(f"==== 汇总: world={world} 失败 rank 数={sum(codes)}/{world} codes={codes}")
        for r, e in enumerate(errs):
            if e:
                log(f"  rank {r} 错误首行: {e.splitlines()[0][:300]}")
        log("判定: " + ("复现 off 挂（MC2 取不到 UB_MEM 链路）" if any(codes) else "MC2 dispatch 全部通过"))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
