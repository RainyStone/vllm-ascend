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
export PYTHONPATH=/path/to/this/repo/third_party/ascend_moonep:/path/to/this/repo:$PYTHONPATH
                                                 # vendored ascend-moonep 放最前(见代码结构),再覆盖 stock vllm_ascend
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MOONEP=1 MOONEP_SHMEM_BOOTSTRAP=ipport      # 引擎必须 ipport(config-store)引导,见限制#1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False  # URMA 必需,见限制#1②
export MOONEP_DISABLE_UDMA=1             # MTE 内核(无需 URMA 时;开 URMA 去掉此行)
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

## 性能优化点（dispatch/combine host 段，2026-09-10）

引擎内 MoonEP 单层开销的大头在 host 段：每层十几到二十个小算子（本机单个 elementwise
op ≈ 4-5µs host / ≈22µs 延迟，`int(x.sum().item())` ≈ 49/65µs），且逐层 `.item()` 同步把
host 时间钉在关键路径上。已做的优化（单位：每层每 rank，dp2ep2 / MTE / 1024 seq×256 prompt）：

| 优化 | 默认 | 实测 |
|---|---|---|
| `torch.bincount` → `torch.histc`：bincount 输出长度动态、需要 host 往返 | 开 | disp_tpe 0.30 → **0.02ms** |
| pad 尾部惰性写：只在 (T, 容量, pad 专家) 变化时写尾部 `[T:]`（`copy_` 每次覆盖 `[0:T]`） | 开 | disp_pad ≈0.03ms，每层省 3 次 launch |
| hot path 去掉 `.to()` 临时张量（`copy_` 自带类型转换） | 开 | — |
| **步长自适应容量池**（借鉴上游 vLLM `MoonEPBufferPool`）：init 预建 128/256/512/…/S_max 的 pow2 容量 Buffer，每步按 rank 对称步长（`forward_context.max_tokens_across_dp`）选最小容量 | 开 | 同码 A/B **+1.7%**，disp_buf 0.53→0.47ms |
| 库内输出复用 `reuse_outputs`：缓存每层的 32MB `empty_like` + planning 输出表 | 关（实验） | 待 A/B（目标把 disp_lib 0.16→~0.06ms） |
| 零同步 dispatch：不再读回 rows_total，MLP 直接吃完整 `[NvS,H]` + count-mode group_list | 关（实验） | dispatch 0.55→**0.29ms**、disp_post 0.33→**0.05ms**；但 combine 侧需设备侧 mask（+0.13ms）→ E2E 与同步版打平 |

累计（已验证口径：首 token 与 MC2 全一致 = 315）：dp2ep2 MTE G128 **4684 → 5469 tok/s（+16.8%）**、
prefill 131.8 → 154.4（+17.2%），相对 MC2 从 0.59 升到 **0.69**；tp2 同码复测 2982 tok/s（原 2581-2658）。
容量池与零同步路径的 N=1024 正确性复验被设备占用阻塞（冒烟规模已验证）。

**改这几处必须遵守的两条硬约束（踩过的坑）**：

1. **combine 的输入尾部必须是零**。combine 核会读整块 `[NvS,H]`，而 MLP 输出在
   `sum(group_list)` 之后是**未初始化内存**、权重 chunk 未用行是陈旧内存 —— 不清理就会
   *静默算错*（dp2ep2 下首 token 从全 315 变成十几个不同值，tp2 测不出来因为 tp2 每步
   T==capacity、pad 路径不激活）。清零要用设备侧 `where(row_ids < sum(group_list))`，
   不能用乘法（`NaN × 0 = NaN` 会漏）。
2. 1-D mask 广播到 `[NvS,H]` 必须 `.unsqueeze(-1)`，否则广播对齐到最后一维 →
   `size of tensor a (8192) must match b (2048)`，引擎挂到超时。
3. 计时口径：`MOONEP_TIME_STATS=1` 会打印 `[moonep-tstats]`（MoonEP 各子段）与
   `[tstats-baseline]`（MC2 融合算子）；MC2 是 host 入队时间（异步），MoonEP 含 kernel/同步，
   两者只能同侧比较。prefill/decode 拆分用 GEN=1 与 GEN=128 两条腿相减。

## 关键配置/环境开关

| 开关 | 作用 |
|---|---|
| `MOONEP=1`（additional_config `enable_moonep`） | 启用 MoonEP comm（需 EP>1） |
| `MOONEP_SHMEM_BOOTSTRAP=ipport` | shmem 引导方式（**引擎内必须 ipport**，默认已改；uid 走 HCCL 广播会死锁，见限制#1） |
| `MOONEP_SHMEM_IP_PORT` | ipport 引导的 config-store 端口（默认 28765） |
| `MOONEP_DISABLE_UDMA=1` | 用 MTE 内核（无需 URMA 时；开 URMA 则不要设） |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` | **开 URMA 时必需**（torch_npu expandable segments × URMA 死锁，见限制#1②） |
| `ASCEND_MOONEP_DISPATCH_USE_UDMA=0` | 仅关闭 dispatch 的 UDMA kernel（保留 UDMA 上下文，调测用） |
| `MOONEP_SLOT_MODE=1` | slot/冗余专家平衡模式（仅 bf16 无量化） |
| `MOONEP_B` | 预取槽数（库要求 == E/R，其它值会报错） |
| `MOONEP_SLOT_STATS=1` | 槽位统计 |
| `VLLM_MOONEP_DEBUG=1` / `ASCEND_MOONEP_TRACE=1` | 调试打印（引擎/库内） |
| `MOONEP_POOL=0` | 关闭步长自适应容量池（退回固定 `max_num_batched_tokens` 容量） |
| `MOONEP_POOL_CAPS=128,1024` | 手动指定容量池档位（默认 pow2 阶梯，≥128） |
| `MOONEP_NO_SYNC=1` | 零同步 dispatch（实验；需配合 combine 侧设备 mask，默认关） |
| `MOONEP_REUSE_OUT=1` | 库内 dispatch 输出复用（实验；默认关） |
| `MOONEP_TIME_STATS=1` | 分阶段计时，退出时打印 `[moonep-tstats]` |
| `MOONEP_CAP_TRACE=1` | 打印每步选中的 dispatch 容量 |

## 限制与已知问题（重要）

1. **引擎内 UDMA 路径已可用（2026-09-09 实测定案）**，两个坑：
   - **① shmem 引导**：默认 `uid` bootstrap 在引擎的 HCCL 组上做 UID 广播会触发
     设备任务异常（`UB_INLINE_WRITE` / `[HDCommunicate][Put] Invalid length` +
     `Device oom 0x7110012`）→ 死锁。**必须 `MOONEP_SHMEM_BOOTSTRAP=ipport`**
     （config-store，无 HCCL 广播）；代码默认已改 ipport。
   - **② torch_npu `expandable_segments` 分配器 × URMA 上下文死锁（根因已定）**：
     vllm-ascend 默认设 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；
     URMA 上下文（`enable_udma=True`）下，真实规模（数十 GB）设备分配后再做首个
     moonep 跨卡同步即设备侧死锁（双卡 100% util 自旋，无任务异常）。最小复现：
     引擎 spawn 链 + 28GB 分配 + expandable_segments:True + URMA on → 首 dispatch
     死锁；去掉 expandable_segments 即通过（HCCL 交错/流/进程链均非触发因子）。
     **引擎开 URMA 时必须 `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`**
     （vllm-ascend 尊重外部值）。实测：URMA on + ipport + 该 env → Qwen3-30B-A3B
     TP2/EP2 引擎 e2e 完整通过（输出落盘）。
     代价：无 expandable segments 时内存碎片/OOM 容忍度下降（vllm-ascend 默认开它
     的目的）；如需两全，须由 torch_npu/ascend-moonep 修 URMA 与 expandable
     segments 的 VA/页表兼容（或 HDK ≥ 26.0.RC1）。
2. **NPU 对称内存可用性**：部分设备（长时间被占/驱动状态损坏的卡）会报
   `aclrtMemSetAccess ... 507899`（= ACL_ERROR_RT_DRV_INTERNAL_ERROR，驱动内部错误），
   需要设备 reset 或驱动升级；故障与 MoonEP 代码无关（MTE/UDMA 同样失败）。
3. **planning 要求 B == E/R**；GPU 原版“推理用小 B”在 Ascend 移植库上暂不支持。
4. **超容量/倾斜**：owner 模式在极端路由倾斜下可能丢行（见模块注释的容量包络）；
   计划中增加 `plan.remote_stats` 保护（越界自动回退原生 comm）。
5. 仅 bf16 无量化、无 dynamic EPLB/mix、tp=1..N 均可（EP 跨 TP 组）；需 `--enforce-eager`
   （MoonEP 集体内核不能进 CUDA 图）。
6. **性能仍低于原生 MC2**（dp2ep2 下 MTE 约为 MC2 的 0.69，URMA 约 0.52；两者相对差距
   与 tp2 拓扑一致，说明瓶颈是每层 dispatch 的 host 段 + 无重叠，与拓扑无关）。已削掉的
   host 段见上文"性能优化点"；剩余大头是逐层 `.item()` 同步 + planning/dispatch 核的设备
   时间（`disp_post` ≈0.33ms）与库内 per-call host 开销（`disp_lib` ≈0.17ms）。下一步方向：
   库内输出复用/planning 输出表静态化、把 histogram 挪进 planning 核、planning 与 dispatch
   合核（省一次 launch + 一次全 rank 栅栏），以及分块双缓冲做真重叠。
7. **DP 拓扑下的 MoonEP 需要每个 rank 进程看到全部参与设备**（vLLM DP 默认会把
   `ASCEND_RT_VISIBLE_DEVICES` 收窄成单卡，MoonEP 的 VMM/shmem 初始化在收窄进程里会踩
   CANN 的 device id 语义问题：`aclrtMemSetAccess` 要物理号、`aclrtMallocPhysical` 要逻辑号）。
   库侧已修 `memSetAccess` 一半（见 `ascend_moonep/buffer_c.py: acl_vmm_access_device_id`）；
   实测 dp2ep2 需要保留全量可见 + 给 vllm-ascend 0.23.0 的 worker 补 DP 偏移（测试用
   shim：`/home/tmp/dpshim/sitecustomize.py`，`MOONEP_DP_FULL_VISIBLE=1`）。

## 代码结构

- `third_party/ascend_moonep/` — **vendored 的 ascend-moonep Python 层**（外部库无法推送改动，
  故随本仓库携带；编译产物 `_C*.so` 与 `lib/*.so` 仍取自外部构建目录，可用
  `ASCEND_MOONEP_BUILD_DIR` 指定）。**运行前把它放在 `PYTHONPATH` 最前面**:
  `export PYTHONPATH=<repo>/third_party/ascend_moonep:$PYTHONPATH`。
  携带的本地修改见 `third_party/ascend_moonep/README.md`（VMM device id 修复 +
  `dispatch(reuse_outputs=)`）。
- `vllm_ascend/ops/fused_moe/moonep_comm.py` — MoonEP comm 方法（owner 模式、运行时池、选择逻辑）
- `vllm_ascend/ops/fused_moe/moonep_balance.py` — slot/平衡模式（权重镜像、预取、[epn+B] 别名 GEMM）
- 其余小改动：`ascend_config.py`（开关）、`ascend_forward_context.py`（MoECommType.SHMEM）、
  `worker/model_runner_v1.py`（load_model 末尾 wiring-time init）等。

## 测试脚本

`scripts/`：`moe_ab_test.py`（引擎 A/B）、`bench_long.py`（长 prefill）、
`smoke_qwen_geo.py` / `smoke_slots.py`（数值 harness）、`probe_slots_multi.py`（R3 多热专家探针）。
