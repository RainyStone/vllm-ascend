# SPDX-License-Identifier: Apache-2.0
"""Check the cache storage consumed by the A5 DSV4 attention operators."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_ascend.attention.dsa_attn_kv_plan import (
    get_dsa_attn_kv_plan,
    is_a5_bf16_kv_enabled,
    resolve_dsv4_cache_dtype,
    resolve_dsv4_cache_torch_dtype,
)
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.layer.attention.layer import DSAAttention
from vllm_ascend.utils import AscendDeviceType


def _config(cache_dtype):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=128),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def _on_a5():
    profile = get_hardware_profile(AscendDeviceType.A5)
    stack = ExitStack()
    for module in (
        "vllm_ascend.attention.dsa_attn_kv_plan",
        "vllm_ascend.models.layer.attention.layer",
    ):
        stack.enter_context(mock.patch(module + ".get_current_hardware_profile", return_value=profile))
    return stack


@pytest.mark.parametrize("requested", ["auto", "bfloat16", "bf16", "fp8", "fp8_e4m3", "float8_e4m3fn"])
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_a5_swa_and_compressed_storage_matches_attention(requested, compress_ratio):
    config = _config(requested)
    with _on_a5():
        dtype = resolve_dsv4_cache_torch_dtype(config)
        swa = object.__new__(AscendDeepseekV4SWACache)
        nn.Module.__init__(swa)
        swa.head_dim = 512
        swa.dtype = dtype
        swa.block_size = 128
        swa.window_size = 128
        swa.cache_config = config.cache_config
        swa_spec = swa.get_kv_cache_spec(config)

        compressed = object.__new__(DSAAttention)
        nn.Module.__init__(compressed)
        compressed.compress_ratio = compress_ratio
        compressed.head_size = 512
        compressed.n_local_heads = 32
        compressed.kv_cache_dtype = requested
        compressed_spec = compressed.get_kv_cache_spec(config)
        plan = get_dsa_attn_kv_plan(config, compress_ratio)

        bf16 = requested in ("auto", "bfloat16", "bf16")
        expected_dtype = torch.bfloat16 if bf16 else torch.float8_e4m3fn
        expected_width = 512 if bf16 else 608
        for spec in (swa_spec, compressed_spec):
            assert spec.dtype == expected_dtype
            assert spec.head_size == expected_width
        assert is_a5_bf16_kv_enabled(config) is bf16
        assert plan.uses_sparse_flash_mla is bf16
        assert plan.uses_kv_compress_epilog is not bf16
        # Resolution must not rewrite an engine-global request during spec creation.
        assert config.cache_config.cache_dtype == requested


@pytest.mark.parametrize("requested", ["auto", "bfloat16", "bf16", "fp8", "fp8_e4m3", "float8_e4m3fn"])
def test_a5_canonical_dtype_roundtrip_preserves_mode(requested):
    with _on_a5():
        before = _config(requested)
        canonical = resolve_dsv4_cache_dtype(requested, "bfloat16")
        after = _config(canonical)
        assert resolve_dsv4_cache_torch_dtype(before) == resolve_dsv4_cache_torch_dtype(after)
        assert is_a5_bf16_kv_enabled(before) == is_a5_bf16_kv_enabled(after)


@pytest.mark.parametrize("requested", ["fp8_e5m2", "fp8_inc", "fp8_ds_mla", "nvfp4_ds_mla"])
def test_a5_rejects_formats_without_an_e4m3_layout(requested):
    with _on_a5(), pytest.raises(ValueError, match="Unsupported DeepSeek-V4 KV cache dtype"):
        resolve_dsv4_cache_torch_dtype(_config(requested))


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
@pytest.mark.parametrize("requested", ["auto", "bfloat16", "fp8"])
def test_other_hardware_keeps_generic_storage_dtype(device_type, requested):
    with mock.patch(
        "vllm_ascend.attention.dsa_attn_kv_plan.get_current_hardware_profile",
        return_value=get_hardware_profile(device_type),
    ):
        config = _config(requested)
        expected = kv_cache_dtype_str_to_dtype(requested, config.model_config)
        assert resolve_dsv4_cache_torch_dtype(config) == expected


def test_turboquant_storage_policy_is_preserved():
    with _on_a5():
        config = _config("turboquant_4bit_nc")
        assert resolve_dsv4_cache_dtype(config.cache_config.cache_dtype, "bfloat16") == "turboquant_4bit_nc"
        assert resolve_dsv4_cache_torch_dtype(config) == torch.uint8
