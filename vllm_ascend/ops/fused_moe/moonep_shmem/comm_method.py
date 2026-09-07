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
"""MoonEP shmem MoE 通信方法：dispatch → prefetch_weight → GEMM → 逐槽加权 → combine。

对齐 MindSpeed-MM 训练版 shmem_ep_forward 的前向主流程（去 autograd 后）。
与基类 MoECommMethod.fused_experts 的差异：
  * 不做 log2phy/expert_map 映射——MoonEP 按全局逻辑专家号自行规划，
    热点专家由 prefetch_weight 动态复制到本地 B 槽（运行时负载均衡，
    与 EPLB 的周期性静态重排互斥）；
  * GEMM 权重为 [epn+B] 打包 VA 别名（home 段 + 副本槽段，零拷贝）；
  * 路由权重在 combine 前逐槽乘（MoonEP combine 无权），而非
    alltoall 路径的 npu_moe_token_unpermute(probs=...) 加权。

v1 约束：仅 bf16（量化路径预留分支点）、eager、单机 EP、
activation 仅支持 silu 系（swiglu）。
"""

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, MoECommMethod
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.moonep_shmem.runtime import MoonEPLayerState
from vllm_ascend.ops.fused_moe.moonep_shmem.token_dispatcher import (
    TokenDispatcherWithShmem,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All
from vllm_ascend.quantization.quant_type import QuantType


class ShmemCommImpl(MoECommMethod):
    """MoonEP 对称内存 MoE 通信方法（ascend-moonep 后端）。

    适用场景：enable_expert_parallel=True、EP>1、bf16 非量化、单机。
    与 EPLB（dynamic_eplb/num_redundant_experts）互斥——副本专家由
    prefetch_weight 每次前向动态建立，无需周期性重排。
    """

    def _get_token_dispatcher(self) -> TokenDispatcherWithShmem:
        return TokenDispatcherWithShmem(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self) -> PrepareAndFinalizeWithAll2All:
        # 复用 All2All 的 prepare/finalize：TP 维 pad/切分 + finalize 反操作
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> FusedExpertsResult:
        # ---- 守卫：v1 仅 bf16 非量化；量化扩展在此分支 ----
        if fused_experts_input.quant.quant_type != QuantType.NONE:
            # TODO(v2): W8A8 需确认 prefetch 按行拷贝 int8 + scale 张量同步；
            # W4A8/MXFP4 需处理 NZ/int4 打包格式。
            raise NotImplementedError(
                f"[moonep_shmem] v1 仅支持 bf16 非量化，got "
                f"{fused_experts_input.quant.quant_type}"
            )
        if fused_experts_input.dynamic_eplb:
            raise RuntimeError(
                "[moonep_shmem] 与 dynamic EPLB 互斥：MoonEP 每次前向动态建立"
                "副本专家，无需周期性重排。请关闭 eplb_config.dynamic_eplb 并将 "
                "num_redundant_experts 设为 0。"
            )
        if fused_experts_input.routing.log2phy is not None:
            raise RuntimeError("[moonep_shmem] 不应存在 log2phy 映射（EPLB 残留）")
        if fused_experts_input.routing.apply_router_weight_on_input:
            raise RuntimeError(
                "[moonep_shmem] 不支持 apply_router_weight_on_input："
                "路由权重在 combine 前逐槽乘（MoonEP combine 无权）"
            )
        if fused_experts_input.weights.w1_bias is not None or fused_experts_input.weights.w2_bias is not None:
            raise NotImplementedError("[moonep_shmem] v1 不支持 expert bias")

        state = fused_experts_input.moonep_state
        assert isinstance(state, MoonEPLayerState), (
            "[moonep_shmem] 缺少 moonep_state：请确认 model_runner 已在 "
            "load_model 内调用 init_moonep_shmem_states"
        )

        # ---- dispatch（pad→Buffer.dispatch，含 home 权重跨 rank 发布可见）----
        dispatch_out = self.token_dispatcher.token_dispatch(
            token_dispatch_input=build_token_dispatch_input(
                fused_experts_input=fused_experts_input
            ),
            moonep_state=state,
        )

        # ---- prefetch：把路由到的远程属主专家权重拉进本地 B 槽 ----
        state.buffer.prefetch_weight(
            plan=dispatch_out.combine_metadata.plan,
            full_gate_weight=state.full_w["gate"],
            full_up_weight=state.full_w["up"],
            full_down_weight=state.full_w["down"],
        )

        # ---- 专家 GEMM（组序 == compute_w 行序：[0:epn) home masters，
        #      [epn:epn+B) 副本槽；group_list 为 counts，group_list_type=1）----
        h_nvs = dispatch_out.hidden_states
        group_list = dispatch_out.group_list
        gate = torch_npu.npu_grouped_matmul(
            x=[h_nvs], weight=[state.compute_w["gate"]], bias=None,
            split_item=2, group_list_type=1, group_type=0, group_list=group_list,
        )[0]
        up = torch_npu.npu_grouped_matmul(
            x=[h_nvs], weight=[state.compute_w["up"]], bias=None,
            split_item=2, group_list_type=1, group_type=0, group_list=group_list,
        )[0]
        act_input = torch.cat([gate, up], dim=-1)
        inter = self._activate(act_input, fused_experts_input)
        # down 的 VMM 表为转置存储 [G, H, H_f]（物理），GEMM 传转置视图 [G, H_f, H]
        expert_out = torch_npu.npu_grouped_matmul(
            x=[inter], weight=[state.compute_w["down"].transpose(1, 2)], bias=None,
            split_item=2, group_list_type=1, group_type=0, group_list=group_list,
        )[0]

        # ---- 逐槽乘路由权重（fp32 精度），combine 本身无权 ----
        w_nvs = dispatch_out.combine_metadata.w_nvs
        expert_out = (expert_out.float() * w_nvs.unsqueeze(-1)).to(torch.bfloat16)

        # ---- combine（K 路求和 + 逆散布 + slice 掉 pad 槽）----
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=expert_out.contiguous(),
            combine_metadata=dispatch_out.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            group_list_type=1,
            # 物理组计数（masters+slots），仅供观测/调试；dynamic_eplb 关闭时
            # fused_moe.py 不会消费该字段
            expert_tokens=group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )

    @staticmethod
    def _activate(act_input: torch.Tensor, fused_experts_input: MoEFusedExpertsInput) -> torch.Tensor:
        """swiglu 激活（silu 系）；其余激活类型 v1 不支持。"""
        act_name = getattr(fused_experts_input.activation, "value", fused_experts_input.activation)
        if act_name not in ("silu", "swiglu"):
            raise NotImplementedError(
                f"[moonep_shmem] v1 仅支持 silu 系激活，got {act_name}"
            )
        limit = fused_experts_input.swiglu_limit
        if limit > 0:
            gate, up = act_input.chunk(2, dim=-1)
            gate.clamp_(max=limit)
            up.clamp_(min=-limit, max=limit)
        return torch_npu.npu_swiglu(act_input)


# 延迟注册入口（moe_comm_method.setup_moe_comm_method 调用），
# 避免未安装 ascend_moonep 时影响其他 comm method 的导入
def create_shmem_comm_impl(moe_config) -> ShmemCommImpl:
    from vllm_ascend.ops.fused_moe.moonep_shmem import check_ascend_moonep_available

    check_ascend_moonep_available()
    logger.info("[moonep_shmem] 注册 ShmemCommImpl（MoonEP 对称内存调度）")
    return ShmemCommImpl(moe_config)
