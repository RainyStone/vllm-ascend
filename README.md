# vllm-ascend-moonep

将 [MoonEP](https://gitcode.com/cann/moonep)（对称内存 MoE 专家并行通信）集成到
vllm-ascend：用 CANN SHMEM 对称内存的单边搬运，替换 MoE EP 中基于 HCCL 的
token dispatch/combine（MC2/all-to-all）。选中 MoonEP/SHMEM 时统一使用 Slot 负载均衡，
按计划预取远端专家权重，并通过现有专家 GEMM 和融合加权归并完成计算。

> 分支说明：`moonep` = 最新功能（MoonEP 固定 Slot/平衡路径）；
> `checkpoint/mte-working` = MTE 路径在引擎内验证通过的稳定快照。

---

## 功能

| 模式 | 说明 | 启用 |
|---|---|---|
| **原生 Off** | 保留原生 MC2/all2all 等通信选择，不启用 MoonEP | additional config `enable_moonep=false` |
| **MoonEP Slot / 平衡**（MindSpeed 风格） | 冗余专家预取槽接收计划中的远端权重，本地计算被重分配的行并进行融合加权归并 | additional config `enable_moonep=true` 且选中 SHMEM；保留现有 Slot 模型/布局及 eager 限制 |

MoonEP 的 vLLM 服务路径只保留 Slot：对称权重镜像（home）+ 槽 + `[epn+B]` 别名 GEMM。SDK 独立算子接口保留；`MOONEP_KEEP_MC2` 的原生小步通信选择不变。

迁移说明：`MOONEP_SLOT_MODE` 已移除。以前 MoonEP 开启时未设置或设为 0 会选择 owner-only，现在实际选中 SHMEM 时统一走 Slot，不再回退 owner-only。原生 Off 继续使用 `enable_moonep=false`；dispatch dtype、副流、持久缓存及融合 publish 等独立参数和默认值不变。历史实验脚本、结果及 owner 验证记录保留原样，不代表当前仍提供 owner-only 服务模式。

`MOONEP_FORCE_ALLGATHER` 已移除，固定采用原默认值 0 的正常选路：A5 小步满足条件时走 MC2，其余按并行规模与 top-k 选择 AllGather 或 All-to-All，必要的原生 AllGather 分支继续保留。MoonEP 的 SHMEM/Slot 覆盖、`MOONEP_KEEP_MC2` 和容量/EP 校验不变。旧值 1 不再强制 AllGather，原有 MC2 建链问题的手动规避入口不再提供，本次不新增自动回退；历史实验记录保留原样。

`MOONEP_EARLY_PREFETCH` 与专用的 `MOONEP_EARLY_PF_DEV` 已移除，固定采用原 `MOONEP_EARLY_PREFETCH=0` 的行为：不再用同一层上一次执行的 plan 在 attention 前投机预取，也不再安装对应 hook、保存预测行或进行预测修补。旧变量设为 1 不再启用该能力。同层真实 planning 后的副流预取由 `MOONEP_PREFETCH_SIDE_STREAM_OVERLAP` 控制；共享流/事件、MTE/UDMA 保护及缓存联动继续保留，Home GEMM 模式仍要求关闭持久缓存。`MOONEP_EARLY_PLAN` 独立诊断、MXFP8、融合 publish、有效行解包及 BF16 zero-copy 不变；历史实验脚本和结果保留原样。



### 持久专家缓存的层容量


迁移说明：`MOONEP_SLOT_CACHE` 已重命名为 `MOONEP_CACHED_REDUNDANT_EXPERT_COUNT`，只保留新名称的中央注册与读取，不提供旧名兼容别名。旧名单独设置时被忽略而使用新默认 0；新旧同时设置时仅采用新名。负数原先按关闭处理，现在明确报错，非整数继续拒绝。参数在起服/模块加载时固定，不增加逐层环境查询，也不支持运行中切换。旧缓存实验脚本需先迁移参数再重跑，否则旧值 8 等会被忽略；历史脚本与结果不改写。默认关闭、临时 Slot 数、按实际注册层数分配缓存及其他优化保持不变。

`MOONEP_CACHED_REDUNDANT_EXPERT_COUNT` 默认仍为 0，关闭时不分配持久缓存。设置为正整数 N 时，每个 EP rank 为每个实际注册的 MoE 层保留最多 N 个远端专家（权重及必要 scale），循环替换策略不变。取值要求为整数且 N>=0，不限制 N<=E/R，也不限制 N<=临时冗余 Slot 数；缓存可跨执行步骤保留不同专家，但实际容量受可用显存与底层分配约束。临时 Slot 仍由同形状层共享，持久缓存按层隔离。

`MOONEP_SLOT_CACHE_LAYERS` 已移除，不保留兼容别名，旧值不再控制容量。模型加载完成后，对非空注册层按对象身份去重，保持注册顺序，并按现有权重/payload 几何分组。每个池、每种 payload 一次性分配 `[该池实际层数, N, rows, cols]` 的连续 VMM 缓存；使用池内缓存索引，原全局层编号继续用于层标识与日志。不是读取模型配置的总层数，也不再固定预留 64 层。

各池逻辑缓存字节数为“该池实际层数 × N × 单专家所有 payload 字节数”；实际显存还受 VMM 对齐影响。实际层数小于旧预留值时可节省缓存显存，超过时会增加以覆盖所有层。开启缓存但显存不足时明确报错，不自动减少 N、跳过层或关闭缓存。所有缓存/槽位在 SHMEM 集体初始化之前完成分配，不在推理过程中扩容；重用池时若层容量改变则拒绝，层索引及缓存形状在初始化时校验。

缓存命中屏蔽、循环替换、权重及 scale 原始字节、预取副流及事件顺序保持不变。`dispatch|home_gemm1` 仍要求关闭持久缓存；此次只调整层容量管理，不修复此前缓存开启后的设备异常，不扩大动态模型或异构 EP 布局支持。历史实验脚本与结果保留原样，SDK 和算子无需因此重新编译。

### 同层预取副流重叠

`MOONEP_PREFETCH_SIDE_STREAM_OVERLAP` 在起服前设置，只接受以下精确取值：

| 取值 | 行为 | 对应旧值 |
|---|---|---|
| `none` | 主流串行执行 dispatch、必要的权重预取/缓存处理及完整 GEMM；不关闭 Slot 或必要的权重搬运 | `MOONEP_PF_STREAM=0` |
| `dispatch`（默认） | 真实 planning 后发起副流预取，与 dispatch 重叠；缓存回填或完整 GEMM 读取 Slot 前等待完成事件 | `MOONEP_PF_STREAM=2` |
| `dispatch\|home_gemm1` | 进一步先执行 Home GEMM1，再等待预取、执行 Slot GEMM1；GEMM2 仍合并 | `MOONEP_PF_STREAM=3` |

组合值在 shell 中必须加引号，例如：

```bash
export MOONEP_PREFETCH_SIDE_STREAM_OVERLAP='dispatch|home_gemm1'
```

这是启动配置，不支持服务运行中切换。关闭重叠时，SDK 使用当前流，MTE/UDMA 内核保留传输完成检查；UDMA 使用现有非重叠队列布局。两种重叠模式保留 planning、dispatch、预取、消费者及槽位复用之间的事件依赖，UDMA 仍要求独立 QP 及现有设备能力检查。Home GEMM1 模式仍要求 `MOONEP_CACHED_REDUNDANT_EXPERT_COUNT=0`，并满足现有 BF16/W4A8MXFP 后端、形状、激活及专用算子库限制；不新增模型支持或静默回退。持久缓存自身的既有问题不在本次重命名修复范围内。

迁移说明：`MOONEP_PF_STREAM` 已移除，不保留兼容别名；原事件管道诊断值 1 一并移除。默认值从原 0 改为 `dispatch`，与最近 GSM8K/LongBench-v2 优化实验显式使用的原值 2 一致。只设置旧名时（包括 0、3 或非法值）会被忽略而采用新默认；新旧同时设置时仅采用新名。历史脚本及结果不改写，重跑前需迁移参数，不能再依据旧值 0/3 判断当前模式。原生 Off 或 EP<=1 不安装副流层 hook，`KEEP_MC2` 正常选路保持不变；MXFP8、融合 publish、有效行解包和其他优化不变。

`MOONEP_NO_PIN` 已移除，固定采用原默认值 0 的行为：Ascend 平台继续声明支持 pinned host memory，原生 Off 与 MoonEP Slot 使用相同的平台能力查询；不改写调用处显式指定的 `pin_memory=False`。旧值 1 不再切换到 pageable memory，原有 `aclrtMallocHostWithCfg` 207001 分配失败的手动规避入口不再提供；本次不修复该分配问题，也不增加失败后自动回退。SHMEM/VMM、权重预取、副流、MXFP8、融合 publish、有效行解包及 BF16 zero-copy 不变，历史实验脚本和记录保留原样。

`MOONEP_ALLOW_CAPTURE` 已移除，固定采用原默认值 0 的捕获保护：MoonEP 的 token dispatch 和 Slot forward 在检测到图捕获时继续报错，旧值 1 不再绕过保护。使用 `--enforce-eager` 或关闭相应图模式；本次不自动更改图配置或增加静默回退，也不宣称 MoonEP 已支持图模式。Slot forward 在无执行上下文时的既有处理、AICPU 启动预热的捕获跳过逻辑、原生 Off 的图能力及其他优化保持不变，历史性能实验和捕获研究记录保留原样。

`MOONEP_SLOT_QUANT` 已移除，固定采用原默认值 1 的行为：Slot 始终接受现有代码支持并通过布局校验的量化权重 payload，旧值 0 不再禁止该路径。量化权重及必要 scale 仍按原字节布局镜像、预取并建立计算别名，不增加量化/反量化；BF16 模型继续使用 BF16 分支。缺失 scale、元素大小、shape/stride、逐专家连续性、存储边界和载体容量等校验保持不变，不扩大模型或量化格式支持范围。`slot_quant_enabled()` 保留为恒 True 的兼容能力查询，不表示当前模型一定量化，也不启用 MoonEP。dispatch dtype、其他优化和原生 Off 不变，量化权重继续保留原张量，历史实验脚本和结果保留原样。

`VLLM_ASCEND_MOONEP_FUSED_PUBLISH` 已移除，vLLM Slot 固定使用融合 publish：与旧默认值 0 不同，未设置或旧值为 0 时也采用此前显式值 1 的优化路径。将原始 BF16 专家输出与设备端有效行 counts 交给本地融合内核，一次完成有效行写入 SHMEM 和尾部清零，保留 NaN 尾部安全处理；FP32 路由权重乘加及最终 BF16 舍入不变。SDK 公共 `fused_publish/zero_copy` 参数和默认值不变，独立调用仍可选择旧路径；历史非融合实验脚本和结果保留原样，旧变量不能再用于新代码的非融合对照。

publish、MXFP8 wire 和 Home/Slot 范围 GEMM1 现已合入 A5 主设备库 `libascend_moonep_kernels.so`，各功能源码继续独立维护。Python 包装默认通过 ctypes 加载同一个主库，保留导出函数、wire/publish ABI检查和显式 library_path；缺库、缺符号、ABI不匹配或非法输入明确报错，不回退旧分库，不扩大设备/模型支持范围。`_C` 和 SHMEM/CANN依赖继续独立，合库不改变现有主通信 ABI。

于 ascend-moonep 仓根目录构建候选（不安装）：

```bash
bash scripts/build_kernels.sh --arch a5
```

候选产物为 `build/lib/libascend_moonep_kernels.so`，包含全部A5优化；原三个构建脚本保留为这个统一构建入口。验证候选并备份后，使用同目录临时文件加原子替换安装；旧进程保持已有映射，新服务重启后加载新版本。旧三个分库文件保留供既有进程和显式历史对照使用，新默认不再加载，SDK打包时排除陈旧分库。

本次只合并设备库产物，保留原内核源码、数据格式、有效行解包、NaN尾部安全处理、融合publish/FP32加权归并、launch次数和流事件顺序；MTE/UDMA、副流/Home GEMM1、缓存、通信选路及原生Off不变。算子微基准结果不代替模型端到端性能结论，历史脚本与结果不改写。

`ASCEND_MOONEP_NO_RECONCILE` 与 `ASCEND_MOONEP_PREFETCH_RELAXED` 已成对移除，固定采用原实验值 1/1，而非旧默认 0/0。SDK 的 `async_finish=True` 分支继续在当前流排队并返回完成事件，不再附加全设备 synchronize 和 dist.barrier；prefetch 的 C++ 入口始终不插入前后跨 rank barrier。planning/dispatch/combine 及初始化、权重发布的必要同步、预取内核内部完成机制和 UDMA 独立 QP 保护保留。

该约定要求 home 权重已发布且在使用期间不变，各 rank 的 collective 顺序一致；跨流消费和槽位复用由调用者建立事件依赖。它不提供权重热更新或任意跨流并发的自动同步。旧值 0 不再恢复保守同步，历史实验脚本和结果保留原样。vLLM 的普通非 async_finish 调用不会因此额外少掉“三次同步”；原生 Off 与其他优化不变。

这是 Python 与 C++ 配套变更：必须重编译并安装匹配的 `ascend_moonep._C` 后再启动新进程，不能只更新 Python。已有进程保留已加载版本，更新后须按原计划重启才能采用新行为；设备 kernel 库无需因这次主机侧清理而重编译。

`MOONEP_B` 已重命名为 `MOONEP_REDUNDANT_EXPERT_SLOT_COUNT`，只保留新名称，不提供旧名回退。未设置新名时仍按每层 E/EP 默认取值；旧名单独设置时被忽略，新旧同时存在时只采用新名。整数解析、范围校验及当前 planning 的 `B==E/EP` 限制不变，改名不表示支持关闭槽位或任意减少槽数。该值表示每个 EP rank 的临时冗余专家预取槽位数，不是全机总数，也不是 `MOONEP_CACHED_REDUNDANT_EXPERT_COUNT` 持久缓存数量；SDK 的 B 参数、计划字段、槽位分配和其他优化保持不变，历史实验脚本和结果保留原样。

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

> 路径说明：本文出现的 `/vllm-workspace/...`、`/home/tmp/...` 等是**编写机器**上的取值，
> 换机器请按实际路径替换（尤其 `ASCEND_CUSTOM_OPP_PATH`）。

```bash
source /usr/local/Ascend/cann/set_env.sh
unset MOONEP_SHMEM_IPPORT
export ASCEND_RT_VISIBLE_DEVICES=0,1             # 空闲的健康卡
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH   # ascend_moonep 由 ascend-moonep 仓的
                                                 # pip editable 提供（build.sh 以 pip install -e . 收尾）
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MOONEP=1 MOONEP_SHMEM_BOOTSTRAP_TYPE=ipport      # 引擎必须 ipport(config-store)引导,见限制#1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False  # URMA 必需,见限制#1②
export MOONEP_TRANSPORT_TYPE=mte         # mte（默认）或 udma；只接受精确小写值
```

## 使用

### 槽位使用统计

`MOONEP_SLOT_STATS=1` 会在进程退出时打印：dispatch 总次数、发生槽平衡的次数、槽内行数、预取的远程专家槽数。

### MoE 负载采集（打点）

打点只数本 rank 的量、不做任何额外集合通信；跨 rank 聚合与峰均比（PAR）报告由离线脚本完成
（`VLLM_ASCEND_MOE_PAR_*` 四个环境变量控制，采集模块 `moe_par_stats.py`）。

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
| `aclrtMemSetAccess(va=..., size=...) failed with 507899` | DP 收窄可见设备后**物理 vs 逻辑 device id 混用**（见限制 #7，库仓已修）；若仍复现，先确认 `import ascend_moonep` 解析到的是本机 editable 安装的 ascend-moonep 仓（`python3 -c "import ascend_moonep; print(ascend_moonep.__file__)"`） |
| `aclshmemx_init_attr_with_buffers failed with -100`，日志带 `Unsupported physical allocation properties ... locationId=...` | 同一个 id 空间问题的 **shmem 侧**：VMM 属性里是物理号、`aclrtGetDevice()` 是逻辑号，比较必然失败（修法与 dlsym 陷阱见限制 #7） |
| `ImportError ... '_build_info'` / 找不到 `vllm_ascend`、`ascend_moonep` | 仓库没装：vllm-ascend 与 ascend-moonep 两个 editable 包没都挂上（ascend-moonep 由其 `build.sh` 的 `pip install -e .` 安装） |
| OOM / `Failed to apply for memory`，但 `npu-smi` 看着空闲 | ① 别的容器占着卡（`npu-smi` 看不见，只有 ACL 级 `torch.npu.mem_get_info` 才反映）；② 上一次 `kill -9` 的残留进程仍持卡 |
| `aclrtMallocPhysical(16777216, device_id=N) failed with 207001`（池初始化阶段） | **容量池档数过多**：每档除 hidden 块外还要为各算子分配 workspace（~10 个 16 MiB 级 VMM 物理块），几十档就把驱动的物理块吃光。降低档数目标并核对实际档位列表（`MOONEP_POOL_MAX_TIER_COUNT`） |
| `cannot register a symmetric descriptor after buffer-backed shmem initialization` | 想在 shmem init 之后新建 `Buffer`（"懒建档"）——**库不支持**，对称堆在 init 时一次性定死 |
| 参数改过之后跨路线 token 不再逐条一致 | **预期**：分组长度/tiling 变 ⇒ 数值末位差异 ⇒ 贪心解码分叉。跨实现不该要求 bit-exact，判据用任务准确率 |
| `Free memory ... less than desired GPU memory utilization` | 同卡别的容器中途占回显存（`npu-smi` 之外也要看 ACL 级可用量）；自适应 GMU 会因此在运行中失效 |
| 崩溃日志是空的（kill 后无 traceback） | 别用 `… \| grep … \| tail -N` 跑长任务：`tail` 会缓冲到 EOF，进程被杀就什么都不落盘 |
| `HcclBroadcast ... port 16666` 等 HCCL 建链异常 | **未定根因**（个别机器只出现过一次、未复现）。遇到时先按 HCCL 错误码定位：`19 = HCCL_E_NETWORK`、`4 = 内部错误 / TLS 配置不一致`（曾见 0-3 卡 tls disabled、4-7 enabled 导致 8 卡建链失败） |

容量池预算迁移：`MOONEP_POOL_BUDGET_MB` 已重命名为 `MOONEP_POOL_MEMORY_BUDGET_MB`，只保留新名称的中央注册与读取，不提供旧名兼容别名。默认仍为 2048，后缀虽为 MB，换算仍是每单位 `2^20` 字节（MiB）；整数解析不变，正数参与估算，0 或负数取消预算约束。旧名单独设置时被忽略而使用新默认，新旧同时存在时仅采用新名。

该值是每个形状容量池在自动生成档位时使用的估算预算，不是整卡或全部容量池的总显存硬上限，也不包含全部工作区、元数据及 VMM 对齐开销。基础档位和最大容量仍保留，`MOONEP_POOL_MAX_TIER_COUNT` 及选档算法不变。显式指定 `MOONEP_POOL_TOKEN_CAPACITY_TIERS` 仍先返回手动档位，不读取或解析预算；预算只在生成档位时查询，不增加逐层推理开销。历史实验脚本与结果不改写，自定义旧预算的启动配置重跑前需迁移。

## 关键配置/环境开关

Planning 负载统计前的屏障由 SDK 参数 `MOONEP_PLANNING_LOAD_STATS_BARRIER_MODE` 控制，严格接受精确小写 `auto`/`pull`/`push`，默认 `auto`；SDK 在原 Planning 配置位置首次成功读取后缓存，各EP rank必须一致，不支持运行中切换。vLLM只增加一次性环境审计，不增加逐层解析或host同步。`auto`保留SHMEM自动选择（当前EP≤32用v3 pull，EP>32用v4 push），与此前8卡GSM8K/LongBench-v2默认路径一致；显式`pull`固定v3，`push`固定v4，不能关闭屏障，也不改变统计数据的gather方式。

旧名 `ASCEND_MOONEP_BARRIER_V4` 不再生效，不提供兼容别名；旧值1强制push的配置需迁移为新值`push`，旧名单独设置时采用新默认`auto`。必须配套部署重编译的SDK `_C`和`libascend_moonep_kernels.so`，保留56字节tiling布局但新增pull编码，不能混用旧设备库。保留写入→屏障→读取、内存可见性、代次计数、核间同步及跨rank调用顺序；原生Off、通信选择、源端去重、MTE/UDMA、副流/Home GEMM1、缓存、MXFP8、有效行解包和融合publish/加权combine均不变。不改写历史脚本与结果，也不宣称已重新验证多rank同步或性能。


MoonEP dispatch格式按每层实际 `MoEQuantParams` 自动选择，不再读取 `VLLM_ASCEND_MOONEP_DISPATCH_DTYPE`；删除中央注册与固定模式属性，不保留兼容别名，旧变量任意值均被忽略。`W4A8MXFP` 自动且必须走MXFP8，其余类型保留原先可用的BF16传输及MLP内必要量化，不新增INT8、普通FP8或MXFP4通信能力，不扩大模型支持范围。

MXFP8继续要求原始连续BF16输入、E4M3激活、group32 E8M0激活scale和现有wire几何（H在1024～16384且为1024倍数）；MXFP元数据缺失、格式或尺寸不兼容时明确报错。量化参数中未显式指定的激活/激活scale类型沿用标准E4M3/E8M0默认语义。选中后缺wire库、ABI/接口不匹配或执行失败也直接报错，不回退BF16，不把预量化FP8直接强转BF16当作反量化。

首次成功使用每种量化类型/H配置时记录自动选择及原因，不逐请求重复日志，不新增设备计数回读或host同步。保留量化一次、激活/scale打包、按有效行解包、输出缓冲复用、BF16 zero-copy、融合publish与FP32融合加权；MTE/UDMA、副流、Home GEMM1事件等待与scale复用、持久缓存限制及源端去重默认0/加权combine保护不变。现有模型布局、EP、图模式等校验继续生效；不将历史缓存设备异常视为已修复。

默认语义变化：W4A8MXFP从旧变量未设时的BF16改为自动MXFP8，与近期显式mxfp8的LongBench-v2/GSM8K优化实验一致；未量化BF16模型仍走BF16。历史脚本即使显式设置旧值bf16也不能再强制BF16对照，复现该对照需使用历史代码；历史脚本和结果不改写。原生Off、All-to-All、SDK公共接口与二进制均不变。


Planning 源端去重使用 SDK 参数 `MOONEP_PLANNING_SOURCE_DEDUPLICATION_ENABLED`，严格接受0/1，默认0不去重（等价历史优化实验的 `ASCEND_MOONEP_SKIP_DUP=1`），1启用去重；SDK扩展首次成功读取后缓存，所有EP rank须保持一致。旧名不保留兼容，旧名单独设置任何值均被忽略；新旧同时存在时仅采用新名。此处默认改变了旧SDK未设置时执行去重的行为，依赖旧去重默认的独立SDK调用需显式设置新值1。

当前Slot固定使用融合加权combine，因此新参数必须为0，MTE/UDMA主机校验继续拒绝1；现有去重前导会先无权合并专家输出，与逐路FP32加权不兼容，本次不新增两者共存能力。只迁移一次性环境审计项，不增加逐层环境查询；实际开关由SDK的匹配新版 `_C` 读取，不能继续使用旧二进制。内部tiling ABI、设备算法、dispatch dtype、副流、缓存、Home GEMM1、有效行解包、融合publish及原生Off均不变。历史脚本/结果不改写，旧的“开启去重”配置重跑前需迁移；默认关闭与已核对的优化版LongBench-v2/GSM8K配置一致。


内核并行参数清理：vLLM 不再读取或透传 `MOONEP_NUM_SMS`，不保留兼容入口；旧值（包括非法字符串）均被忽略。创建 Buffer 时省略 `num_sms`，继续采用 SDK 默认32。此前通过旧变量把大容量档 Planning 改成8～31的配置需要迁移至 `MOONEP_PLANNING_NUM_SMS`；旧变量未设或设为32、56时的 Planning 选择保持一致。

保留 SDK 的 `MOONEP_PLANNING_NUM_SMS`：未设置或为空时，按每个 Buffer 的容量 `S×K` 自动选核，≤8192使用8，其余在 SDK 默认值下使用32；该容量不是逐请求的实际有效 token 数。显式非空值沿用 int 解析与[8,56]校验，覆盖自动规则。最终 Planning block 数同时用于工作区大小计算和 PlanningOp 创建，不能只改变启动核数而沿用旧工作区。

此次不删除或更改 SDK 公共 `Buffer(num_sms=...)` 接口，独立 SDK 调用仍可传参；不修改 SDK 自动选核算法、源码或二进制。Dispatch、Combine、Prefetch 的底层启动策略、专家 GEMM、副流、MXFP8、有效行解包、融合 publish、BF16 zero-copy 和原生 Off 均不变；不涉及独立的 `MOONEP_NUM_SMS_DEDUP` 参数。清理 vLLM 中将旧变量描述为统一控制各阶段核数的过时注释，历史实验脚本与结果不改写。

传输类型迁移：`MOONEP_DISABLE_UDMA` 已删除，统一使用 `MOONEP_TRANSPORT_TYPE`，严格接受精确小写 `mte` 或 `udma`，不自动去空格或转换大小写。**默认改为 `mte`**：旧代码未设置/0时请求UDMA，旧值1选择MTE；新代码未设置时选择MTE。旧名不保留兼容，单独设置任何旧值（包括0）均被忽略，新旧同时存在时只采用新名；旧UDMA配置必须改为 `MOONEP_TRANSPORT_TYPE=udma`。

该参数在原初始化位置读取，统一控制vLLM MoonEP的dispatch、combine和权重prefetch，不表示激活或权重量化格式；planning保持SDK原生路径。Buffer与SHMEM使用一致的后端选择，初始化时继续核对实际后端，不匹配明确报错，不新增静默降级。原生Off与KEEP_MC2选路不变，不增加逐层环境查询或运行中切换。

UDMA副流仍要求Ascend950 direct-mode构建、独立QP初始化/ready校验和显式完成事件；MTE不启用UDMA上下文或专用QP，保留MTE副流重叠。缓存、Home GEMM1、MXFP8、有效行解包、融合publish和BF16 zero-copy的原限制与事件依赖均不变。本次不修复或重新验证UDMA/URMA与分配器的历史兼容问题。SDK独立布尔接口及内部逐阶段后端变量不改；SDK创建上下文时会写入这些内部变量，不能把它们当作vLLM统一后端选择的替代配置。通用 `service/start_service_mode.sh` 保持显式MTE；历史实验脚本和结果不改写，重跑前需迁移旧名。

SHMEM 引导端口迁移：`MOONEP_SHMEM_IP_PORT` 已重命名为 `MOONEP_SHMEM_BOOTSTRAP_PORT`，只保留新名称的中央注册与读取，不提供旧名兼容别名；默认仍为28765。值只表示整数端口，IP继续来自vLLM的DP master配置，并按原方式组装 `tcp://IP:端口`。旧名单独设置时被忽略而采用新默认，新旧同时存在时仅采用新名；历史脚本与结果不改写，自定义旧端口的配置重跑前需迁移。

该参数仍只在 `MOONEP_SHMEM_BOOTSTRAP_TYPE=ipport` 分支、原初始化位置读取，保留int解析和现有校验策略，不新增端口范围限制；非整数在该分支报错。`uid` 分支不读取或解析端口，即使端口值非法也不影响其选择。多机地址检查、AICPU预热/SHMEM初始化顺序、SDK的 `ip_port` 接口参数以及MTE/UDMA、副流QP和其他优化均保持不变，不增加逐层推理查询。

SHMEM 引导参数迁移：`MOONEP_SHMEM_BOOTSTRAP` 已重命名为 `MOONEP_SHMEM_BOOTSTRAP_TYPE`，只保留新名称的中央注册与读取，不提供旧名兼容别名。默认仍为 `ipport`；旧名单独设置（包括旧值 `uid`）时被忽略而采用新默认，新旧同时存在时仅采用新名。历史实验脚本与结果不改写，重跑旧 UID 配置前需迁移参数。

新参数严格接受精确小写 `ipport` 或 `uid`：不自动去空格或转换大小写，空字符串、拼写错误及其他值在初始化读取处明确报错，不再像旧代码那样进入 UID 分支。`ipport` 仍使用 vLLM 的 DP master 地址及 `MOONEP_SHMEM_BOOTSTRAP_PORT`（默认28765）建立 config store，并保留多机地址校验；`uid` 仍调用 SDK 的 UID/HCCL 广播初始化。AICPU 预热、引导读取、SHMEM 初始化和权重发布顺序不变，不增加逐层推理查询。两者只选择启动引导方式，不改变 MTE/UDMA 传输、副流 QP 或其他优化；本次不修复或重新验证历史 UID 初始化故障。

容量池档位参数迁移：`MOONEP_POOL_CAPS` 已重命名为 `MOONEP_POOL_TOKEN_CAPACITY_TIERS`，表示逗号分隔的 token 容量档位列表；`MOONEP_POOL_MAX_RUNGS` 已重命名为 `MOONEP_POOL_MAX_TIER_COUNT`，表示自动生成档位的数量目标，默认仍为 32。只保留新名称，不提供旧名兼容别名；仅设置旧名时采用新的默认行为（自动生成、档数目标32），新旧同时存在时仅采用新名。

手动档位未设置或为空字符串时仍自动生成；非空配置保持原整数解析、忽略空项、去重、升序、过滤超过最大容量的值及补上最大档的行为，不新增数值校验。手动模式仍跳过档数目标及内存预算的读取和解析；自动模式仍受原档数目标和估算预算共同控制，基础档与最大档继续保留。所有档位仍在 SHMEM 初始化前预分配，运行时按跨 DP 一致的 token 数选择满足需求的最小档。历史脚本及结果不改写，重跑自定义档位/档数实验前需迁移新名称。

| 开关 | 作用 |
|---|---|
| `MOONEP=1`（additional_config `enable_moonep`） | 启用 MoonEP comm（需 EP>1） |
| `MOONEP_SHMEM_BOOTSTRAP_TYPE=ipport` | SHMEM 引导类型，只接受 `ipport`/`uid`，默认 `ipport`；引擎沿用 config-store 路径，UID/HCCL 广播的历史故障见限制#1 |
| `MOONEP_SHMEM_BOOTSTRAP_PORT` | ipport 引导的 config-store 端口（默认 28765） |
| `MOONEP_TRANSPORT_TYPE=mte` | dispatch/combine/权重预取统一使用 mte（默认）或 udma；planning 保持原生路径，非法值明确报错 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` | **开 URMA 时必需**（torch_npu expandable segments × URMA 死锁，见限制#1②） |
| `ASCEND_MOONEP_DISPATCH_USE_UDMA` | SDK内部逐阶段后端状态；创建上下文时由SDK写入，不是vLLM统一传输类型的替代开关 |
| `MOONEP_REDUNDANT_EXPERT_SLOT_COUNT` | 每个 EP rank 的冗余专家预取槽位数，默认 E/EP；当前 planning 要求等于 E/EP，其它值仍会报错，与持久缓存数量不同 |
| `MOONEP_SLOT_STATS=1` | 槽位统计 |
| `VLLM_MOONEP_DEBUG=1` / `ASCEND_MOONEP_TRACE=1` | 调试打印（引擎/库内） |
| `MOONEP_POOL_TOKEN_CAPACITY_TIERS=128,1024` | 手动指定容量池档位（覆盖默认的"预算化档位表"） |
| `MOONEP_MIN_DISPATCH_TOKENS=16` | 每步选档的最小 token 下限，必须 ≥1；仅识别此变量，未设置时默认 128，实验脚本可显式设 16 |
| `MOONEP_POOL_MEMORY_BUDGET_MB=2048` | 档位表的估算预算，单位 MiB，影响档位密度；不是包含全部工作区的总显存硬限制 |
| `MOONEP_POOL_MAX_TIER_COUNT=32` | 自动生成的档数目标；基础档和最大档仍保留，不是严格硬上限。**受 VMM 物理块约束**（51 档会 `207001` 初始化失败，见排障） |
| `MOONEP_PAD_CUT=1` | 实验性：把补零行踢出 grouped GEMM。**实测错误，默认关**（见排障） |
| `MOONEP_TIME_STATS=1` | 分阶段计时，退出时打印 `[moonep-tstats]` |
| `ASCEND_MOONEP_TIME_STATS=1` | 库内分段计时（barrier/planning 核/dispatch 核/边界拷贝，`[moonep-lib-tstats]`） |
| `MOONEP_CAP_TRACE=1` | 打印每步选中的 dispatch 容量 |
| `DUMP_TOKENS=1` | 导出贪心 token ids（`/tmp/bench_tokids_<TAG>_rank*.json`），用于跨路线对拍 |

MoonEP dispatch 固定将有效行数保留在设备端，通过 count-mode `group_list` 限定 GEMM 工作量，并在 combine 中安全处理容量尾部，不再提供逐层 `.item()` 回读行数的开关。跨 rank 通信屏障、副流事件等待，以及独立开关控制的诊断/调试同步仍保留。

MoonEP dispatch 固定按 Buffer 复用 planning 输出表和通信输出缓冲；MXFP8 路径同样复用激活、scale 和权重缓冲。返回数据仅在该 Buffer 下一次 dispatch 前有效，消费方须按 `group_list` 使用有效行。可释放的输出 view、权重快照、尾部清理及副流等待继续保留。

普通 BF16 分发固定 `zero_copy=True`，直接读取 SHMEM 接收分片，省去分片拷出。MXFP8 仍共享 SHMEM 字节载体，按有效行解包到复用的 FP8/scale 缓冲；必要解包、融合 publish 和 combine 路径保持不变。SDK 公共 `zero_copy` 参数及默认值保持不变，独立调用仍可选择 copy 路径。

迁移说明：`MOONEP_DISPATCH_IN_SHARD` 已移除，旧脚本设为 0 不再关闭 BF16 zero-copy。历史实验脚本、结果和审计记录保留原样；旧 BF16-copy 草稿及仅检查环境变量的审计不能直接用于新代码的 copy 对照。历史 MXFP8 路径不依赖此开关，其有效行解包行为不受影响。

Slot 的 home 权重发布后，符合条件的 BF16 权重固定由 home 存储接管，保留 Parameter 对象身份、shape/dtype 校验及接管后的销毁保护。量化权重及 scale 继续保留原张量，不以 BF16 字节载体接管原始量化权重存储；DSV4 Flash 属于此跳过释放的路径。Slot 发布后仍执行 `empty_cache()` 和显存统计，Off 路径不变。该静态推理路径不支持初始化后的权重热更新、sleep/RL 场景。

MoonEP 固定启用自适应容量选档，使用跨 DP 一致的 step token 数选取足够大的最小档；上下文或必要元数据缺失时，仍安全回退到最大容量。`MOONEP_POOL_TOKEN_CAPACITY_TIERS`、档数目标默认 32、估算预算默认 2048 MiB 和最小选档下限默认 128 保持不变，历史实验显式设置的下限 16 继续生效。容量尾部不参与真实专家负载及 GEMM 工作量。

MoonEP 启用且 EP>1 时，首次运行时初始化固定在 SHMEM 集体初始化、建立跨设备映射前尝试预热 AICPU custom 调度器。保留 1-token metadata 调用及启动阶段同步，不进入逐层或逐请求路径；图捕获期间或目标算子未注册时跳过，算子调用/同步失败时告警并返回 False，延续现有失败处理策略。Off 和 EP≤1 的提前返回保持不变。

仿真路由清理：`VLLM_ASCEND_MOE_SIM_ROUTE`、`VLLM_ASCEND_MOE_SIM_ROUTE_SEED`、`VLLM_ASCEND_MOE_SIM_ROUTE_EVERY` 和 `VLLM_ASCEND_MOE_SIM_ROUTE_TABLE` 已移除，不提供兼容别名，旧环境变量不再注入合成专家分布。真实流量继续使用模型原生的融合/非融合/自定义 router 和原有逻辑到物理专家映射；启动 profile 的原生 force_load_balance 及 DP dummy 调用保持不变。

通用起服脚本已移除 `SIM_ROUTE`、`SIM_ROUTE_SEED`、`SIM_ROUTE_EVERY`、`SIM_ROUTE_TABLE` 透传入口及“仿真已开启”提示。历史合成倾斜度、二分阈值实验的脚本、路由档案与结果保留原样；这些实验需要历史代码才能按原注入方式重跑，不能把新代码忽略旧变量后的真实路由结果当作原合成实验。

真实负载统计 `VLLM_ASCEND_MOE_PAR_STATS`、目录/标签/落盘频率参数，以及 `MOONEP_PLAN_STATS`、`MOONEP_EARLY_PLAN` 等独立诊断均保留。MoonEP Slot、预取副流、缓存、MXFP8、有效行解包、融合 publish 和其他优化不变。源码曾记录仿真注入绕过特定 DSV4 单 token dummy 的原生 gating 故障；当前版本是否仍复现未重新验证，本次不新增绕过，也不认定该历史故障已修复。

## 限制与已知问题（重要）

1. **引擎内 UDMA 路径已可用（2026-09-09 实测定案）**，两个坑：
   - **① shmem 引导**：默认 `uid` bootstrap 在引擎的 HCCL 组上做 UID 广播会触发
     设备任务异常（`UB_INLINE_WRITE` / `[HDCommunicate][Put] Invalid length` +
     `Device oom 0x7110012`）→ 死锁。**必须 `MOONEP_SHMEM_BOOTSTRAP_TYPE=ipport`**
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
   `aclrtMemSetAccess ... 507899`（= ACL_ERROR_RT_DRV_INTERNAL_ERROR，驱动内部错误）。
   **早期「需设备 reset / 升级驱动」的判断已被实测推翻**：根因见第 7 条——
   `aclrtMemSetAccess` 的 `aclrtMemLocation.id` 要物理号，而 VMM 其余接口要逻辑号，
   `ASCEND_RT_VISIBLE_DEVICES` 是真子集时两者不等，故只在重映射场景出现。
3. **planning 要求 B == E/R**；GPU 原版“推理用小 B”在 Ascend 移植库上暂不支持。
4. **超容量/倾斜**：owner 模式在极端路由倾斜下可能丢行（见模块注释的容量包络）；
   计划中增加 `plan.remote_stats` 保护（越界自动回退原生 comm）。
   **历史 owner 合成倾斜对照中记录**：owner 模式下规划核的
   rebalancer 会把 rank 级负载抹平，dispatch 核再把超出该 rank 容量 `S×top_k` 的行**静默丢弃**——
   于是「路由倾斜 → rank 级计算」这条链在 owner 腿上是断的（rank 级计算反映的是再平衡结果，
   不是路由本身）。**要测路由倾斜对 rank 级计算的直接影响，用 off 腿或 slot 腿**：off 腿
   AlltoAll 直测接收行数；slot 腿行留本地，rank 负载与路由解耦。
   丢行本身**不打日志**（comm 路径静默，见上），只有账目行里的 `dropped=` 记着数——
   `BALANCE=1 MOE_PAR=1` 起 owner 腿，`grep -c 'dropped=' start_service_owner.log`：**0 = 没丢行**，
   ≥1 即越过包线（该项为零时根本不打印，所以 grep 不到就是没丢）。
   仿真路由的 `home-rank rows` 是预测的**分发前**分布，超没超包线要靠这个 `dropped=` 判。
5. 量化只支持 W4A8MXFP（DeepSeek-V4 布局）且 slot 模式要求 **EP≥8**（显存账见
   W4A8MXFP 需 EP≥8）；无 dynamic EPLB/mix、tp=1..N 均可
   （EP 跨 TP 组）；需 `--enforce-eager`（MoonEP 集体内核不能进 CUDA 图）。
6. **性能仍低于原生 MC2**（dp2ep2 下 MTE 约为 MC2 的 0.69，URMA 约 0.52；两者相对差距
   与 tp2 拓扑一致，说明瓶颈是每层 dispatch 的 host 段 + 无重叠，与拓扑无关）。已削掉的
   host 段见上文"性能优化点"；剩余大头是逐层 `.item()` 同步 + planning/dispatch 核的设备
   时间（`disp_post` ≈0.33ms）与库内 per-call host 开销（`disp_lib` ≈0.17ms）。下一步方向：
   库内输出复用/planning 输出表静态化、把 histogram 挪进 planning 核、planning 与 dispatch
   合核（省一次 launch + 一次全 rank 栅栏），以及分块双缓冲做真重叠。
7. **DP 拓扑下的 MoonEP 需要每个 rank 进程看到全部参与设备**（vLLM DP 默认会把
   `ASCEND_RT_VISIBLE_DEVICES` 收窄成单卡，MoonEP 的 VMM/shmem 初始化在收窄进程里会踩
   CANN 的 device id 语义问题：`aclrtMemSetAccess` 要物理号、`aclrtMallocPhysical` 要逻辑号）。
   库侧已修 `memSetAccess`（见 `ascend_moonep/buffer_c.py: acl_vmm_access_device_id`）；
   **另一半在 shmem 子模块**：其 `validate_user_buffer_property` 拿 VMM 句柄的
   `location.id`（物理号）与 `aclrtGetDevice()`（逻辑号）比较，收窄进程里必然不等 ⇒
   `aclshmemx_init_attr_with_buffers` 直接返回 `-100 ACLSHMEM_NOT_SUPPORTED`，日志伴随
   `Unsupported physical allocation properties ... locationId=...`——已按
   `shmem_init.cpp` 设备号补丁修复（改动随 3rdparty/shmem 子模块工作区携带）。
   若要修：把该比较改成物理号，**取物理号必须用
   `dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId")`，不要用
   `DlAclApi::AclrtGetPhyDevIdByUserDevId`** —— 后者的函数指针在 `DlAclApi::LoadLibrary()`
   之前是空指针，补丁会**静默失效**；而报错行号会随改动位移，很容易误判成"已生效"。
   实测 dp2ep2 需要保留全量可见 + 给 vllm-ascend 0.23.0 的 worker 补 DP 偏移（测试用
   shim：`/home/tmp/dpshim/sitecustomize.py`，`MOONEP_DP_FULL_VISIBLE=1`）。
   **本仓运行还依赖若干落在其它仓/子模块的修复**（catlass 编译、shmem init `-100`、本例、
   DeepSeek-V4-Flash 的 moonep 适配），背景/根因/解法与重置后如何重打见工作区的
   `skills/ascend-moonep-950-setup/`。

## 代码结构

- ascend-moonep Python 层**不再随本仓 vendored**（单一代码源：由 ascend-moonep 仓提供，
  `pip install -e` 后 `import ascend_moonep` 直接解析到该仓的包目录，编译产物 `_C*.so`/`lib/`
  也在其中，无需 `ASCEND_MOONEP_BUILD_DIR`）。
- `vllm_ascend/ops/fused_moe/moonep_comm.py` — MoonEP comm 方法（owner 模式、运行时池、选择逻辑）
- `vllm_ascend/ops/fused_moe/moonep_balance.py` — slot/平衡模式（权重镜像、预取、[epn+B] 别名 GEMM）
- 其余小改动：`ascend_config.py`（开关）、`ascend_forward_context.py`（MoECommType.SHMEM）、
  `worker/model_runner_v1.py`（load_model 末尾 wiring-time init）等。

## 脚本

调测/分析脚本（A/B、数值 harness、探针、MoE PAR 报告套件、dev 线工具集）与过程报告已按
「只保留功能代码」的清理决定移出本仓；现存的 `scripts/dsv4flash_service/` 是起服/占卡/
bench 调度与 sim-route 载荷档案的运行工具。MoE 负载打点仍在功能层：
`vllm_ascend/ops/fused_moe/moe_par_stats.py`（只记录本 rank、零集合通信，开关
`VLLM_ASCEND_MOE_PAR_STATS`）。负载口径：专家负载按逻辑专家记（副本算同一个专家）、
rank 负载按实际落到它所持专家上的对数记、pad 不计入；有冗余专家时 PAR 是上界。
