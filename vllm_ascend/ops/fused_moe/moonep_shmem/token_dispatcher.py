#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""MoonEP shmem token 调度器：Buffer.dispatch / Buffer.combine 的 vllm 封装。

语义对齐 MindSpeed-MM 训练版 shmem_ep_forward 的 dispatch/combine 段：
  - dispatch：输入静态化为 [S, H]（不足 S 右侧补零，pad 槽 topk=0/route_w=0，
    前向贡献为 0），跨 rank 分发后返回按 [masters(epn) | slots(B)] 组序排列的
    payload、逐槽路由权重 w_nvs、cu_seqlens 与 plan。
  - combine：按 plan 逆散布并把同一 token 的 K 路结果求和，输出 [T, H]
    （slice 掉 pad 槽）。MoonEP 的 combine 无权（路由权重已在 GEMM 后逐槽
    乘上，见 comm_method.py）。
"""

from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
    MoEShmemCombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.moonep_shmem.runtime import (
    MoonEPLayerState,
    physical_tokens_per_expert,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatcher


class TokenDispatcherWithShmem(MoETokenDispatcher[MoEShmemCombineMetadata]):
    """基于 ascend-moonep Buffer 的 EP token 调度器（对称内存 + 单边 RMA）。

    与 All2AllV 调度器的差异：
      - 不做 expert_map/log2phy 映射——MoonEP 按**全局逻辑专家号**自行规划
        （planning），热点专家由 prefetch_weight 动态拉取副本到本地 B 槽；
      - 输入静态化为 [S, H]（Buffer 静态形状），输出组序为
        [masters(epn) | slots(B)] 而非 [num_local_experts]；
      - group_list 为本 rank 物理组计数 [epn+B]（int64, group_list_type=1）。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)
        assert self.num_local_experts > 0, "Expected at least one local expert"

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        moonep_state: MoonEPLayerState | None = None,
    ) -> MoETokenDispatchOutput[MoEShmemCombineMetadata]:
        assert moonep_state is not None, (
            "shmem dispatcher 需要 moonep_state（由 init_moonep_shmem_states 挂载）"
        )
        s = moonep_state
        hidden = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        # ---- 输入归一化：bf16 [T,H]、fp32 [T,K]、int32 [T,K] ----
        hidden = hidden.view(-1, hidden.shape[-1])
        if hidden.dtype != torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        hidden = hidden.contiguous()
        route_w = topk_weights.to(torch.float32).contiguous()
        topk = topk_ids.to(torch.int32).contiguous()

        T = hidden.shape[0]
        S = int(s.buffer._require_ctx()["S"])  # Buffer 静态 token 容量
        dev = hidden.device
        if T > S:
            raise RuntimeError(
                f"[moonep_shmem] 实际 token 数 {T} 超过 Buffer 容量 S={S}；"
                f"请调大 max_num_batched_tokens 或 shmem_moonep.max_tokens_per_rank。"
            )
        # MoonEP 要求静态 [S, H] 输入：变长 batch 右侧 pad（pad 槽 topk=0、
        # route_w=0，前向贡献为 0；combine 后 slice 掉）。pad 槽白算但不影响
        # 正确性，未来可在 Buffer 层做 skip_pad 优化。
        if T < S:
            hidden_p = torch.zeros(S, hidden.shape[-1], dtype=hidden.dtype, device=dev)
            hidden_p[:T] = hidden
            hidden = hidden_p
            topk_p = torch.zeros(S, topk.shape[-1], dtype=torch.int32, device=dev)
            topk_p[:T] = topk
            topk = topk_p
            rw_p = torch.zeros(S, route_w.shape[-1], dtype=torch.float32, device=dev)
            rw_p[:T] = route_w
            route_w = rw_p

        # 每专家 token 计数（全局逻辑专家号，供 MoonEP planning）
        tpe = torch.bincount(
            topk.reshape(-1), minlength=s.E
        ).to(torch.int32).contiguous()

        # ---- dispatch（inter_rank_sync 同时完成各 rank home 权重的发布可见）----
        h_nvs, w_nvs, cu_seqlens, plan = s.buffer.dispatch(
            hidden, route_w, topk, tpe,
            zero_copy=False, inter_rank_sync=True,
        )

        # [epn+B] 本 rank 物理组计数（masters 段 + 副本槽段），group_list_type=1
        group_list = physical_tokens_per_expert(
            cu_seqlens, s.rank, s.epn, s.E
        ).to(torch.int64)

        logger.debug(
            "[moonep_shmem] dispatch 完成: 层 %s T=%d S=%d 接收 %d 个 token-槽",
            s.layer_name, T, S, int(group_list.sum().item()),
        )
        return MoETokenDispatchOutput(
            hidden_states=h_nvs,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=MoEShmemCombineMetadata(
                plan=plan,
                num_actual_tokens=T,
                w_nvs=w_nvs,
                moonep_state=s,
            ),
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: MoEShmemCombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None, "shmem dispatcher 不支持 bias"
        s = combine_metadata.moonep_state
        # combine 无权（路由权重已逐槽乘上）：K 路求和并逆散布回 [S, H]
        out, _, _ = s.buffer.combine(
            plan=combine_metadata.plan,
            hidden_nvsh=hidden_states.contiguous(),
            route_weights_nvs=None,
            inter_rank_sync=True,
        )
        T = combine_metadata.num_actual_tokens
        return out[:T] if out.shape[0] > T else out
