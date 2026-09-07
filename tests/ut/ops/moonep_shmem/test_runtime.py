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
"""MoonEP shmem 调度纯逻辑单元测试（不依赖 NPU 硬件与 ascend_moonep 库）。"""

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.ascend_config import ShmemMoonepConfig
from vllm_ascend.ops.fused_moe.moonep_shmem.runtime import (
    physical_tokens_per_expert,
    storage_shape,
)


class TestStorageShape(PytestBase):
    """down 投影转置存储约定：逻辑 [H_f,H] → 物理 [H,H_f]，其余不变。"""

    def test_down_transposed(self):
        assert storage_shape("down", 128, 256) == (256, 128)

    def test_gate_up_not_transposed(self):
        assert storage_shape("gate", 256, 128) == (256, 128)
        assert storage_shape("up", 256, 128) == (256, 128)


class TestPhysicalTokensPerExpert(PytestBase):
    """cu_seqlens → [masters(epn) | slots(B)] 物理组计数。

    例：E=8, R=4, epn=2, B=2；组 0..7 为全局专家段，组 8..9 为副本槽段。
    rank=1 的 masters 段为下标 [2, 4)。
    """

    def test_counts_split(self):
        # 各组 token 数: [1, 2, 3, 4, 5, 6, 7, 8 | 9, 10]
        counts = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cu_seqlens = torch.cumsum(counts, dim=0)
        result = physical_tokens_per_expert(cu_seqlens, rank=1, epn=2, E=8)
        # masters: 组 2,3 → [3, 4]；slots: 组 8,9 → [9, 10]
        assert result.tolist() == [3, 4, 9, 10]

    def test_zero_prefix(self):
        # 首项 diff 即 cu_seqlens[0]（前一项视为 0）
        counts = torch.tensor([0, 0, 5, 0, 0, 0, 0, 0, 1, 2])
        cu_seqlens = torch.cumsum(counts, dim=0)
        result = physical_tokens_per_expert(cu_seqlens, rank=0, epn=2, E=8)
        assert result.tolist() == [0, 0, 1, 2]


class TestShmemMoonepConfig(PytestBase):
    """配置解析与字段校验。"""

    def test_defaults(self):
        cfg = ShmemMoonepConfig(None)
        assert cfg.enabled is False
        assert cfg.num_slots_B is None
        assert cfg.token_padding == 1
        assert cfg.max_tokens_per_rank is None

    def test_enabled_with_overrides(self):
        cfg = ShmemMoonepConfig({"enabled": True, "num_slots_B": 4, "token_padding": 8})
        assert cfg.enabled is True
        assert cfg.num_slots_B == 4
        assert cfg.token_padding == 8

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="no attribute"):
            ShmemMoonepConfig({"bad_key": 1})

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError):
            ShmemMoonepConfig({"enabled": True, "num_slots_B": 0})
        with pytest.raises(ValueError):
            ShmemMoonepConfig({"enabled": True, "token_padding": 0})
        with pytest.raises(ValueError):
            ShmemMoonepConfig({"enabled": True, "max_tokens_per_rank": -1})
