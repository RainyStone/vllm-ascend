# vllm-ascend-moonep

将 [MoonEP](https://gitcode.com/cann/moonep)（对称内存 MoE 专家并行通信）集成到
vllm-ascend：用 CANN SHMEM 对称内存的单边搬运，替换 MoE EP 中基于 HCCL 的
token dispatch/combine（MC2/all-to-all）。模型权重、路由、专家 GEMM、调度均保持
vllm-ascend 原样，只更换 MoE 的 token 通信层。

> 分支说明：`moonep` = 最新功能（owner + slot/平衡模式）；
> `checkpoint/mte-working` = MTE 路径在引擎内验证通过的稳定快照。

---

## 功能

| 模式 | 说明 | 启用 |
|---|---|---|
| **owner**（默认） | token 行按“专家属主”经 SHMEM 对称内存分发/归约，替换 HCCL MC2/all2all | `MOONEP=1`（引擎侧 additional config `enable_moonep`） |
| **slot / 平衡**（MindSpeed 风格） | 额外启用“冗余专家”预取槽：过热专家的行留在本地算，权重行 RDMA 预取进 B=epn 槽，负载均衡 | `MOONEP_SLOT_MODE=1`（需 bf16 无量化模型） |

两种模式共用同一通信内核；slot 模式增加对称权重镜像（home）+ 槽 + `[epn+B]` 别名 GEMM。

## 已验证（实测）

- owner & slot 均在 **vLLM 引擎内端到端跑通**（Qwen3-30B-A3B，TP2/EP2，0/1 两卡，MTE 内核），生成结果与原生 MC2 一致（首 token 相同 / 部分输出逐位相同，|Δlogp| ≤ ~0.1，bf16 噪声）。
- torchrun 数值 harness（E=128/K=8/H=2048/S=512 真实几何）：owner 0.5–0.8%、slot（重倾斜，槽生效）0.72–0.74% 相对误差。
- slot 30k prefill（2 卡，MTE）：5.49 s / 5465 tok/s；同配置 MC2：2.71 s / 11085 tok/s（正确性一致；性能差来自 MTE 兼容路径 + 无异步，见“限制”）。
- 真实路由统计（moe_ab 短生成）：912 次 MoE dispatch 中约 36%–52% 发生槽位平衡，每步预取 1 个远程专家（远低于 B=64 上限）。

## 环境搭建

### 依赖（本机验证版本）

- Ascend 950 硬件（Ascend950PR/A5），CANN 9.2.0，torch 2.10.0+cpu / torch_npu 2.10.0.post4
- vllm 0.23.0（`/vllm-workspace/vllm`，editable）与 vllm-ascend 0.23.0（editable），以及本仓库（editable）：
  ```bash
  pip install -e /vllm-workspace/vllm --no-build-isolation --no-deps
  pip install -e /vllm-workspace/vllm-ascend --no-build-isolation --no-deps
  pip install -e . --no-build-isolation --no-deps   # 本仓库
  ```
- [ascend-moonep](https://gitcode.com/cann/ascend-moonep)（`urma_dev` 分支，含 kernel 库 `ascend_moonep` 0.1.0）：
  ```bash
  cd ascend-moonep && git checkout urma_dev && git submodule update --init --recursive
  source /usr/local/Ascend/cann/set_env.sh        # 或你的 CANN set_env
  # 注意 x86_64 机器上 build.sh 默认找 aarch64 头文件，需显式指定：
  export ASCEND_MOONEP_RUNTIME_INCLUDE_DIR=/usr/local/Ascend/cann-9.2.0/x86_64-linux/pkg_inc/runtime/runtime
  bash scripts/build.sh -soc_type Ascend950        # A2/910B 可省略 -soc_type
  pip install -e . --no-build-isolation --no-deps  # 用训练/推理用的 python
  ```

### 运行必需的环境变量

```bash
source /usr/local/Ascend/cann/set_env.sh
unset MOONEP_SHMEM_IPPORT
export ASCEND_RT_VISIBLE_DEVICES=0,1             # 空闲的健康卡
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH   # 覆盖 stock vllm_ascend
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MOONEP=1 MOONEP_DISABLE_UDMA=1             # MoonEP + MTE 内核（引擎内当前可用路径）
```

## 使用

### 引擎内 A/B（离线 generate）

```bash
cd scripts
MODEL=/path/to/Qwen3-30B-A3B TP=2 python moe_ab_test.py        # MOONEP=1 → MoonEP
#   对照：不设 MOONEP（或 MOONEP=0）→ 原生 MC2/EP
#   slot 模式：export MOONEP_SLOT_MODE=1
#   输出：/tmp/ab_{moonep,mbaseline}_<模型名>.json
```

### 长序列 prefill 基准

```bash
cd scripts
MODEL=/path/to/Qwen3-30B-A3B TP=2 BENCH_LEN=30000 python bench_long.py
# 度量单条 30k prompt 的 prefill 时间/吞吐；默认 max_model_len=40960
```

### 数值 harness（torchrun，不需要 vLLM 引擎）

```bash
export MOONEP_SLOT_MODE=1            # 测 slot 平衡路径；去掉则测 owner
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 scripts/smoke_qwen_geo.py   # 真实几何 E128/K8/H2048/S512
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 scripts/smoke_slots.py      # slot+重倾斜数值
```

### 槽位使用统计

`MOONEP_SLOT_STATS=1` 会在进程退出时打印：dispatch 总次数、发生槽平衡的次数、槽内行数、预取的远程专家槽数。

## 关键配置/环境开关

| 开关 | 作用 |
|---|---|
| `MOONEP=1`（additional_config `enable_moonep`） | 启用 MoonEP comm（需 EP>1） |
| `MOONEP_DISABLE_UDMA=1` | 用 MTE 内核（**引擎内目前必须**，UDMA 见限制） |
| `MOONEP_SLOT_MODE=1` | slot/冗余专家平衡模式（仅 bf16 无量化） |
| `MOONEP_B` | 预取槽数（库要求 == E/R，其它值会报错） |
| `MOONEP_SLOT_STATS=1` | 槽位统计 |
| `VLLM_MOONEP_DEBUG=1` / `ASCEND_MOONEP_TRACE=1` | 调试打印（引擎/库内） |

## 限制与已知问题（重要）

1. **引擎内 UDMA 死锁**：UDMA（URMA）路径在 vllm 引擎进程模型下死锁（standalone torchrun 正常）。
   已定位到 HDK/驱动层：对称内存 fabric 需要 **HDK ≥ 26.0.RC1**（当前 25.7.rc1）。
   因此引擎内请用 `MOONEP_DISABLE_UDMA=1`（MTE），性能为兼容路径水平。
2. **NPU 对称内存可用性**：部分设备（长时间被占/驱动状态损坏的卡）会报
   `aclrtMemSetAccess ... 507899`（= ACL_ERROR_RT_DRV_INTERNAL_ERROR，驱动内部错误），
   需要设备 reset 或驱动升级；故障与 MoonEP 代码无关（MTE/UDMA 同样失败）。
3. **planning 要求 B == E/R**；GPU 原版“推理用小 B”在 Ascend 移植库上暂不支持。
4. **超容量/倾斜**：owner 模式在极端路由倾斜下可能丢行（见模块注释的容量包络）；
   计划中增加 `plan.remote_stats` 保护（越界自动回退原生 comm）。
5. 仅 bf16 无量化、无 dynamic EPLB/mix、tp=1..N 均可（EP 跨 TP 组）；需 `--enforce-eager`
   （MoonEP 集体内核不能进 CUDA 图）。
6. 当前引擎性能低于原生 MC2（MTE 兼容路径 + 无异步/流水）；性能目标依赖
   HDK 升级后的 UDMA 快路径与后续重叠优化。

## 代码结构

- `vllm_ascend/ops/fused_moe/moonep_comm.py` — MoonEP comm 方法（owner 模式、运行时池、选择逻辑）
- `vllm_ascend/ops/fused_moe/moonep_balance.py` — slot/平衡模式（权重镜像、预取、[epn+B] 别名 GEMM）
- 其余小改动：`ascend_config.py`（开关）、`ascend_forward_context.py`（MoECommType.SHMEM）、
  `worker/model_runner_v1.py`（load_model 末尾 wiring-time init）等。

## 测试脚本

`scripts/`：`moe_ab_test.py`（引擎 A/B）、`bench_long.py`（长 prefill）、
`smoke_qwen_geo.py` / `smoke_slots.py`（数值 harness）、`probe_slots_multi.py`（R3 多热专家探针）。
