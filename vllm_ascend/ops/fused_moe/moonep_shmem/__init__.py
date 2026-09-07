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
"""MoonEP shared-memory EP 调度（ascend-moonep 后端）推理侧实现。

通过 ``additional_config = {"shmem_moonep": {"enabled": True}}`` 启用，
选择后 MoE 路由专家计算走 ascend-moonep ``Buffer``（AscendC kernel +
CANN VMM + aclshmem 对称堆）而非 HCCL alltoall，靠每次前向动态复制热点
专家（B 个运行时副本槽）实现完美负载均衡，与 EPLB 的周期性静态重排
互斥（启用时要求 dynamic_eplb=False 且 num_redundant_experts=0）。

本模块移植自 MindSpeed-MM 训练侧实现（shmem_ep_dispatcher.py，
commit 204a1311），裁剪了全部训练专属内容（autograd / reduce_grad /
梯度缓冲），推理侧权重静态，home 权重在加载后一次性 publish。
"""

from vllm_ascend.ops.fused_moe.moonep_shmem.runtime import (
    MoonEPLayerState,
    destroy_all_moonep_states,
    init_moonep_shmem_states,
    shmem_runtime_initialized,
)


def check_ascend_moonep_available() -> None:
    """检测 ascend-moonep 依赖是否可用，不可用时给出安装指引。"""
    try:
        import ascend_moonep  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "shmem_moonep 调度依赖 ascend-moonep（AscendC kernel + VMM 对称内存），"
            "当前环境未安装。请按以下步骤安装（需 urma_dev 分支，UDMA 传输需 "
            "CANN 9.2.0+）：\n"
            "  git clone <ascend-moonep仓库地址> && cd ascend-moonep\n"
            "  git checkout urma_dev && git submodule update --init --recursive\n"
            "  source /usr/local/Ascend/cann/set_env.sh\n"
            "  bash scripts/build.sh -soc_type Ascend950   # Ascend910B 可省略 -soc_type\n"
            "  pip install -e . --no-build-isolation --no-deps\n"
            "注意：同一进程不允许同时存在 cann-shmem wheel 与 ascend-moonep 自带的 "
            "libshmem.so（会触发多镜像报错）。"
        ) from e


__all__ = [
    "MoonEPLayerState",
    "check_ascend_moonep_available",
    "destroy_all_moonep_states",
    "init_moonep_shmem_states",
    "shmem_runtime_initialized",
]
