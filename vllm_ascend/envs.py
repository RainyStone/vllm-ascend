#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

def _moonep_dispatch_dtype() -> str:
    value = os.getenv("VLLM_ASCEND_MOONEP_DISPATCH_DTYPE", "bf16").lower()
    if value not in ("bf16", "mxfp8"):
        raise ValueError("VLLM_ASCEND_MOONEP_DISPATCH_DTYPE must be bf16 or mxfp8")
    return value


def _moonep_prefetch_side_stream_overlap() -> str:
    value = os.getenv("MOONEP_PREFETCH_SIDE_STREAM_OVERLAP", "dispatch")
    if value not in ("none", "dispatch", "dispatch|home_gemm1"):
        raise ValueError(
            "MOONEP_PREFETCH_SIDE_STREAM_OVERLAP must be none, dispatch, "
            "or dispatch|home_gemm1"
        )
    return value


def _moonep_cached_redundant_expert_count() -> int:
    value = int(os.getenv("MOONEP_CACHED_REDUNDANT_EXPERT_COUNT", "0"))
    if value < 0:
        raise ValueError(f"MOONEP_CACHED_REDUNDANT_EXPERT_COUNT must be >= 0, got {value}")
    return value


# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # Persistent remote-expert cache capacity per EP rank and MoE layer.
    # Non-sensitive; integer >=0, default 0 disables caching. No E/R or Slot
    # capacity upper bound; available memory still limits actual allocation.
    "MOONEP_CACHED_REDUNDANT_EXPERT_COUNT": _moonep_cached_redundant_expert_count,
    # MoonEP prefetch overlap: none, dispatch (default), dispatch|home_gemm1.
    # Non-sensitive; set before worker startup. Home GEMM1 requires cache off
    # and an already-supported Home GEMM backend/operator library.
    "MOONEP_PREFETCH_SIDE_STREAM_OVERLAP": _moonep_prefetch_side_stream_overlap,
    # MoonEP activation transport: bf16 (default) or mxfp8 (Slot W4A8MXFP,
    # REAL_T only). Non-sensitive; no change to model weights or combine dtype.
    "VLLM_ASCEND_MOONEP_DISPATCH_DTYPE": _moonep_dispatch_dtype,
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_gmm_combine_decode` / `dispatch_ffn_combine`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Master switch for the per-rank MoE expert-load dump used by
    # scripts/moe_par/analyze_moe_par.py to compute the peak-to-average ratio offline.
    # "1" records, anything else leaves the recorder inert. Independent of
    # DYNAMIC_EPLB: the dump taps the same per-expert route counts but adds no
    # collective, so it is useful with EPLB off.
    "VLLM_ASCEND_MOE_PAR_STATS": lambda: bool(int(os.getenv("VLLM_ASCEND_MOE_PAR_STATS", "0"))),
    # Output directory for the dump; one file per rank is written there.
    "VLLM_ASCEND_MOE_PAR_STATS_DIR": lambda: os.getenv("VLLM_ASCEND_MOE_PAR_STATS_DIR", "/tmp/moe_par_stats"),
    # Label for this run, used in the dump filename so several legs (MC2, MoonEP,
    # all-gather) can be recorded side by side and compared offline.
    "VLLM_ASCEND_MOE_PAR_STATS_TAG": lambda: os.getenv("VLLM_ASCEND_MOE_PAR_STATS_TAG", "run"),
    # Steps buffered on device between host copies. Larger means fewer CPU-NPU
    # syncs and coarser durability. Must be >= 1.
    "VLLM_ASCEND_MOE_PAR_FLUSH_EVERY": lambda: int(os.getenv("VLLM_ASCEND_MOE_PAR_FLUSH_EVERY", "16")),
    # MoonEP small-step policy: when enable_moonep is on, keep the steps whose
    # SoC-default selection is MC2 (num_tokens <= mc2_tokens_capacity) on MC2 and
    # replace only the all-gather / all-to-all selections with SHMEM. Default
    # off: every step within the SHMEM capacity S goes to MoonEP. MC2 needs the
    # HCCL UB_MEM link, which a cross-UB-domain card group does not have
    # (aclnnMoeDistributeDispatchV4 561000), so enable this only on one UB domain.
    "MOONEP_KEEP_MC2": lambda: bool(int(os.getenv("MOONEP_KEEP_MC2", "0"))),
    # Redundant expert prefetch slots per EP rank, not the persistent cache size.
    # Unset is resolved per layer as E/EP; the current planner requires B == E/EP.
    # Runtime accepts integer values in [1, E/EP] before that planner check.
    # Non-sensitive. This does not change the SDK's B parameter.
    "MOONEP_REDUNDANT_EXPERT_SLOT_COUNT": lambda: os.getenv("MOONEP_REDUNDANT_EXPERT_SLOT_COUNT", None),
    # Synthetic expert-selection profile, for routing-skew performance experiments.
    # Replaces the router's top-k output wholesale in ops/fused_moe/experts_selector.py
    # (see ops/fused_moe/moe_sim_route.py). One share per expert, in percent, summing
    # to 100; the share of the *routes* (T*top_k slots), so a single expert cannot
    # exceed 100/top_k -- one token's top-k experts are always distinct. Empty means
    # off. Accepts a full list ("50,30,1,..."), a sparse list ("0:50,7:30", unlisted
    # experts split the remainder), or "@<path>" to read either form from a file.
    "VLLM_ASCEND_MOE_SIM_ROUTE": lambda: os.getenv("VLLM_ASCEND_MOE_SIM_ROUTE", ""),
    # RNG seed for the synthetic assignment. Same seed and same token count give a
    # bit-identical routing table, so legs stay comparable across runs. Default 0.
    "VLLM_ASCEND_MOE_SIM_ROUTE_SEED": lambda: int(os.getenv("VLLM_ASCEND_MOE_SIM_ROUTE_SEED", "0")),
    # Print the full per-expert table when the token count changes, then a one-line
    # reuse confirmation every N calls (0 disables the heartbeat). Default 43, the
    # MoE layer count, so the heartbeat lands once per step.
    "VLLM_ASCEND_MOE_SIM_ROUTE_EVERY": lambda: int(os.getenv("VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "43")),
    # "0" suppresses the full per-expert table, leaving only the per-rank summary line.
    # Use it on long served runs where the token count changes every step.
    "VLLM_ASCEND_MOE_SIM_ROUTE_TABLE": lambda: bool(int(os.getenv("VLLM_ASCEND_MOE_SIM_ROUTE_TABLE", "1"))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
