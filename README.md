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

> 数字来源：`Ascend 950PR / CANN 9.2.0 / torch_npu 2.10.0` 上的实测窗口（dp2ep2 = 2 卡 DP2×TP1 → EP=2）。
> 换机器、换驱动版本或换拓扑都需重新测量；跨测量窗口的绝对吞吐会漂移（本机实测约 ±8%），**对比必须在同一窗口内背靠背做**。

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

> 路径说明：本文出现的 `/vllm-workspace/...`、`/home/tmp/...`、`/home/ascend-moonep` 等是**编写机器**上的取值，
> 换机器请按实际路径替换（尤其 `ASCEND_CUSTOM_OPP_PATH`、vendored 目录、shim 路径）。

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

## 性能优化点与收益（dispatch/combine host 段，2026-09-10/11）

**一句话**：dp2ep2（1024 seq × 256 prompt、MTE）从 **4684 → ~6300–6800 tok/s**（同窗口 A/B **+45%**），
相对原生 MC2 从 **0.59 → 0.82**；URMA 从 3809 → ~5900（0.48 → **0.77**）。所有数字都以"每条序列的
首 token 与 MC2 一致（=315）"作为正确性口径。

原来的大头在 host 段：每层十几到二十个小算子（本机单个 elementwise op ≈ 4-5µs host / ≈22µs 延迟，
`int(x.sum().item())` ≈ 49/65µs），且**逐层 `.item()` 同步**把 host 时间钉在关键路径上。

### 逐项优化（问题 → 做法 → 收益；单位：每层每 rank）

| # | 问题 | 做法 | 收益 |
|---|---|---|---|
| 1 | `torch.bincount` 统计每专家 token 数需 host 往返（输出长度动态） | 换 `torch.histc`（定长输出；计数 ≤ S·K=8192 在 fp32 精确） | disp_tpe 0.30 → **0.02ms** |
| 2 | 每层无条件清空整块 pad 缓冲（hidden 16MB+） | 尾部惰性写：只在 (T, 容量, pad 专家) 变化时写 `[T:]`；`copy_` 每次覆盖 `[0:T]` | 每层省 3 次 launch，disp_pad ≈0.03ms |
| 3 | hot path 每层造 `.to()` 临时张量 | 直接 `copy_`（自带类型转换） | — |
| 4 | 固定按 `max_num_batched_tokens` 分派，短步长白跑 | **步长自适应容量池**（借鉴上游 vLLM `MoonEPBufferPool`）：init 预建 128/256/…/S_max 的 pow2 容量 Buffer，每步按 rank 对称步长（`forward_context.max_tokens_across_dp`）选最小容量 | 同码 A/B **+1.7%**，disp_buf 0.53→0.47ms |
| 5 | 库内每层重新分配 32MB `empty_like` + 7 张 planning 输出表 | `Buffer.dispatch(reuse_outputs=True)`：在 Buffer 上缓存（契约：仅本次到下次 dispatch 之间有效，尾部是陈旧数据，消费方只看 `group_list` 覆盖的行） | **+3.4%**，disp_lib 0.16 → **0.082ms** |
| 6 | 逐层 `.item()` 读回 `rows_total` 再切片，host 被同步钉死 | **零同步 dispatch**：MLP 直接吃完整 `[NvS,H]` + count-mode `group_list`（grouped matmul 只走 `sum(group_list)` 行），combine 输入尾部用设备侧 `where(row_ids < sum)` 清零 | **+20.5%**，dispatch 0.475→**0.30ms**，disp_post 0.33→**≈0** |
| 7 | 每次 dispatch/combine 都查 `torch_npu.npu.current_stream()`（barrier 是 no-op，不需要流） | 惰性查询（仅 trace 时取） | barrier 段 0.14 → **0.001ms** |

累计（dp2ep2 MTE G128，同窗口 A/B）：**5511 → 5698（+reuse）→ 6639（+no-sync）→ 6744 tok/s
（两个都开，+22.4%）**，相对当日基线 4684 为 **+44%**。

### 组件对照（MTE decode，每层每 rank，ms）

| 阶段 | 优化前 | 现在 |
|---|---|---|
| dispatch 合计 | 0.77 | **0.20** |
| ├ disp_pre（pad + tpe） | 0.35 | 0.064 |
| ├ disp_lib（库内 host） | 0.16 | 0.082 |
| └ disp_post（逐层同步） | 0.33 | 0.056 |
| combine（含尾部 mask） | 0.09 | 0.107 |
| **通信占 rank 时间** | 71% | **45%** |

### 三路线结果（同窗口，全部首 token 与 MC2 一致）

| 路线 | prefill tok/s | decode tok/s | 总 tok/s | vs MC2 | 单层 dispatch/combine (decode) |
|---|---|---|---|---|---|
| MC2 基线 | 50 675 | 11 014 | 7 719 | 1.00 | 0.093 / 0.064 |
| MoonEP-MTE | 44 110 | 8 785 | **6 318** | **0.82** | 0.204 / 0.107 |
| MoonEP-URMA | 35 900 | 8 749 | **5 913** | 0.77 | 0.169 / 0.158 |

**怎么复现**：`MOONEP_TIME_STATS=1` 跑 GEN=1 与 GEN=128 两条腿（prefill/decode 靠相减拆分），
用 `report3.py`（`/home/tmp/report3.py`）汇总三路线 × 相位的 dispatch/combine/compute 与吞吐。

**测量纪律（本机实测）**：同一份代码 40 分钟内的绝对吞吐会漂移 ~8%（6797 → 6131 tok/s，设备空闲、无租户；
用 `git stash` 回到旧提交对照确认不是代码回归），所以单次跑数不能跨窗口比较，A/B 必须同窗口背靠背；
本文所有 Δ 均为同窗口 A/B，小于 ~5% 的差异要在噪声里打折看。

**剩余瓶颈**：已经从"逐层同步"转到 **prefill 的 planning 核**（`disp_lib` prefill 0.31ms 基本是
launch 后等 planning 核跑完的设备时间，随 N·K 行数线性；dispatch 核 0.02ms、边界拷贝 0.02ms）与
URMA 上下文自身开销；prefill 通信占比 MTE 79% / URMA 97%。再往下要么把 planning 与 dispatch 合核、
要么把 histogram 挪进核 —— 都要改 AscendC 内核（item 2 单独做只值 ~0.02ms/层，约 1%）。

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

## 排障：报错 → 根因（每条都曾把人带偏）

| 报错 / 现象 | 根因 |
|---|---|
| `aclrtMemSetAccess(va=..., size=...) failed with 507899` | 见过两种：① DP 收窄可见设备后**物理 vs 逻辑 device id 混用**（见限制 #7，库侧已修一半）；② `PYTHONPATH` 里 **vendored 目录不在最前**，实际跑到了外部那份**不带修复的旧库** |
| `aclshmemx_init_attr_with_buffers failed with -100`，日志带 `Unsupported physical allocation properties ... locationId=...` | 同一个 id 空间问题的 **shmem 侧**：VMM 属性里是物理号、`aclrtGetDevice()` 是逻辑号，比较必然失败（修法与 dlsym 陷阱见限制 #7） |
| `ImportError ... '_build_info'` / 找不到 `vllm_ascend`、`ascend_moonep` | 仓库没装：三个 editable 包（vllm-ascend、本仓库）与 vendored 库路径没都挂上 |
| OOM / `Failed to apply for memory`，但 `npu-smi` 看着空闲 | ① 别的容器占着卡（`npu-smi` 看不见，只有 ACL 级 `torch.npu.mem_get_info` 才反映）；② 上一次 `kill -9` 的残留进程仍持卡 |
| `aclrtMallocPhysical(16777216, device_id=N) failed with 207001`（池初始化阶段） | **容量池档数过多**：每档除 hidden 块外还要为各算子分配 workspace（~10 个 16 MiB 级 VMM 物理块），几十档就把驱动的物理块吃光。降到 ≤32 档（`MOONEP_POOL_MAX_RUNGS`） |
| `cannot register a symmetric descriptor after buffer-backed shmem initialization` | 想在 shmem init 之后新建 `Buffer`（"懒建档"）——**库不支持**，对称堆在 init 时一次性定死；复现见 `scripts/moonep_dev/probe_lazy_buffer.py` |
| 参数改过之后跨路线 token 不再逐条一致 | **预期**：分组长度/tiling 变 ⇒ 数值末位差异 ⇒ 贪心解码分叉。跨实现不该要求 bit-exact，判据用任务准确率（见 `docs/dev_results_2026-09-16.md` §7） |
| `Free memory ... less than desired GPU memory utilization` | 同卡别的容器中途占回显存（`npu-smi` 之外也要看 ACL 级可用量）；自适应 GMU 会因此在运行中失效 |
| 崩溃日志是空的（kill 后无 traceback） | 别用 `… \| grep … \| tail -N` 跑长任务：`tail` 会缓冲到 EOF，进程被杀就什么都不落盘 |
| `HcclBroadcast ... port 16666` 等 HCCL 建链异常 | **未定根因**（个别机器只出现过一次、未复现）。遇到时先按 HCCL 错误码定位：`19 = HCCL_E_NETWORK`、`4 = 内部错误 / TLS 配置不一致`（曾见 0-3 卡 tls disabled、4-7 enabled 导致 8 卡建链失败） |

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
| `MOONEP_POOL_CAPS=128,1024` | 手动指定容量池档位（覆盖默认的"预算化档位表"） |
| `MOONEP_MIN_CAP=16` | 池最小档（默认 16；改大会抬高每步补零行） |
| `MOONEP_POOL_BUDGET_MB=2048` | 档位表的内存预算：预算越大档越密、补零行越少（只影响档位密度） |
| `MOONEP_POOL_MAX_RUNGS=32` | 档数上限。**受 VMM 物理块约束**（51 档会 `207001` 初始化失败，见排障） |
| `MOONEP_RELEASE_WEIGHTS=0` | 关闭"释放 vllm 原权重"（默认开：home chunk 接管后释放原张量，省 13.8 GB/卡） |
| `MOONEP_PAD_CUT=1` | 实验性：把补零行踢出 grouped GEMM。**实测错误，默认关**（见排障） |
| `MOONEP_NO_SYNC=0` | 关闭零同步 dispatch（默认开；改回逐层 `.item()` + 切片语义） |
| `MOONEP_REUSE_OUT=0` | 关闭库内 dispatch 输出复用（默认开） |
| `MOONEP_TIME_STATS=1` | 分阶段计时，退出时打印 `[moonep-tstats]` |
| `ASCEND_MOONEP_TIME_STATS=1` | 库内分段计时（barrier/planning 核/dispatch 核/边界拷贝，`[moonep-lib-tstats]`） |
| `MOONEP_CAP_TRACE=1` | 打印每步选中的 dispatch 容量 |
| `DUMP_TOKENS=1` | 导出贪心 token ids（`/tmp/bench_tokids_<TAG>_rank*.json`），用于跨路线对拍 |

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
   **另一半（shmem 侧）未修**：vendored shmem 的 `validate_user_buffer_property` 拿 VMM 句柄的
   `location.id`（物理号）与 `aclrtGetDevice()`（逻辑号）比较，收窄进程里必然不等 ⇒
   `aclshmemx_init_attr_with_buffers` 直接返回 `-100 ACLSHMEM_NOT_SUPPORTED`，日志伴随
   `Unsupported physical allocation properties ... locationId=...`。
   若要修：把该比较改成物理号，**取物理号必须用
   `dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId")`，不要用
   `DlAclApi::AclrtGetPhyDevIdByUserDevId`** —— 后者的函数指针在 `DlAclApi::LoadLibrary()`
   之前是空指针，补丁会**静默失效**；而报错行号会随改动位移，很容易误判成"已生效"。
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

`scripts/moonep_dev/`：dev 分支的负载均衡/容量测量工具集（harness、plan/懒建档探针、
token 对拍、组不均-耗时相关性分析）——见该目录 `README.md`；
结果记录见 `docs/dev_results_2026-09-16.md`。
