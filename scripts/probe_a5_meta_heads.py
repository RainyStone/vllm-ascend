"""A5 稀疏注意力 metadata 算子的「每 rank q 头数」探针（不需要 vLLM 引擎，MB 级显存）。

背景见 docs/cross_repo_fixes.md 第 5 条：A5 上 `KvQuantSparseAttnSharedkvMetadata`
（AICPU）和消费者 `KvQuantSparseAttnSharedkv`（AscendC，host 侧校验）都要求每个 rank 的
q 头数是 64 或 128。DeepSeek-V4-Flash 是 64 头 + 权重 149 GB（单卡 HBM 98 GB）→ TP 只能=2
→ 每 rank 32 头 → 首次生成时挂：

    host plog : errcode:22007, msg:inner error, kernelName=KvQuantSparseAttnSharedkvMetadata
    device 日志: [kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp:103][CheckSingleParam]
                num_heads_q should be 64 or 128, but got 32

本脚本直接调该算子，按 num_heads_q 扫一遍，验证「只有校验挡人、算子本身是好的」，
以及失败与输入长度无关。

用法（挑一张空闲卡）：
    ASCEND_RT_VISIBLE_DEVICES=6 python scripts/probe_a5_meta_heads.py          # 默认 32/64/128
    ASCEND_RT_VISIBLE_DEVICES=6 HEADS=32 SEQ=300 python scripts/probe_a5_meta_heads.py
失败后立刻看设备侧日志（按次生成，不滚动）：
    /root/ascend/log/debug/device-<物理卡号>/device-<hostpid>_<时间戳>.log
"""
import os
import traceback

import torch
import torch_npu  # noqa: F401

import vllm_ascend  # noqa: F401
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
# A5 上 enable_custom_op() 是关的（上游 issue #7157），独立脚本要自己把它 import 进来
import vllm_ascend.vllm_ascend_C  # noqa: E402, F401

DEV = "npu:0"  # ASCEND_RT_VISIBLE_DEVICES 之后的逻辑 0
OP = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata

HEADS_KV = 1
HEAD_DIM = 512  # DSv4-Flash


def run(num_heads_q, seq):
    """按 dsa_v1.py 的 prefill 参数调一次 metadata 算子。"""
    cu = torch.tensor([0, seq], dtype=torch.int32, device=DEV)
    used = torch.tensor([seq], dtype=torch.int32, device=DEV)
    try:
        out = OP(
            num_heads_q=num_heads_q,
            num_heads_kv=HEADS_KV,
            head_dim=HEAD_DIM,
            kv_quant_mode=1,
            cu_seqlens_q=cu,
            cu_seqlens_ori_kv=cu,
            cu_seqlens_cmp_kv=None,
            seqused_q=used,
            seqused_kv=used,
            batch_size=1,
            max_seqlen_q=seq,
            max_seqlen_kv=seq,
            ori_topk=0,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device=DEV,
        )
        torch.npu.synchronize()
        print(
            f"[probe] num_heads_q={num_heads_q:4d} seq={seq:4d}  PASS  "
            f"out={tuple(out.shape)} nonzero={int((out != 0).sum().item())}",
            flush=True,
        )
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")[:150]
        print(f"[probe] num_heads_q={num_heads_q:4d} seq={seq:4d}  FAIL  {type(e).__name__}: {msg}",
              flush=True)
        traceback.print_exc()
        return False


def main():
    torch.npu.set_device(DEV)
    seq = int(os.environ.get("SEQ", "6"))
    heads = [int(h) for h in os.environ.get("HEADS", "32,64,128").split(",")]
    print(f"[probe] device={DEV} seq={seq} heads={heads} op={OP}", flush=True)
    for h in heads:
        if not run(h, seq):
            print("[probe] 失败后 device 侧日志："
                  "/root/ascend/log/debug/device-<物理卡号>/device-<hostpid>_*.log", flush=True)
            break


if __name__ == "__main__":
    main()
