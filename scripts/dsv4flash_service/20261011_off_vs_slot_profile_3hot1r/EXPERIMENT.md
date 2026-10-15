# 实验：off vs slot(+优化组合) bench+profiler 同窗采集 · 2026-10-11

## 要回答的问题

在峰均比 ≈2.5（3hot1r，s=2.48）、max_num_batched_tokens=16384、输入 16384 token 的
条件下，对比 off 与 slot（+三件套优化）的 **profiling trace**（时间线层面两腿的算子/
通信/空转差异），bench 数据仅作参考（含 profiler 开销，见下）。

## 方法（run_profile_leg.sh，RUNS.txt 为两腿清单）

- 两腿各一台全新服务（`../service/start_service_mode_dp4.sh`，PROFILE=1 起服）：
  - off：无 MoonEP（标准 MC2/EP）
  - slot：三件套前缀 `ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1
    ASCEND_MOONEP_SKIP_DUP=1` + 默认优化（real-T 等），`[moonep-env-audit]` 佐证
- 条件：SIM_ROUTE=@../sim/sim_route_3hot_1rank.txt（s=2.48）· BATCHED_TOKENS=16384 ·
  MOONEP_MIN_DISPATCH_TOKENS=16；其余服务参数同 op2 实验（见其 EXPERIMENT.md §3）
- 采集窗口（用户指定）：**只跑一轮**——warmup（不采集）→ `POST /start_profile` →
  一轮 L16k bench（32 prompts/并发 16/输出 1，**全新 seed 21001，两腿相同 → 输入流量
  完全一致**）→ `POST /stop_profile` → 等 trace 导出（文件数稳定 30s）
- profiler：vllm torch profiler（`--profiler-config.profiler=torch`，
  `with_stack=false`——NPU 上该开关等价 with_modules、开销大）；4 个 rank 的 trace
  共用同一目录（文件名自带 rank 后缀）

## ⚠ 数据使用注意

该轮 bench 在 profiler 开启下测得，**性能数字含采集开销，不用于统计结论**；
性能结论以 `../20261011_opt2_off_vs_slot/` 的干净数据为准。本实验的 bench json
（`data/<leg>_L16k_profiled.json`）仅用于确认流量完整（completed=32）与两腿可比。

## 数据（data/ 与 profile/）

- `data/<leg>_L16k_profiled.json`：profiler 开启下的那轮 bench 结果
- `data/<leg>.log`：该腿完整服务日志（配置横幅、env 审计行、sim-route 步数）
- `profile/<leg>/`：4 rank 的 torch trace（chrome trace 格式，tensorboard 可看：
  `tensorboard --logdir profile/`）

## 结论（2026-10-11 23:50 跑完）

- 两腿均完整：bench 32/32 成功、4 rank trace + host trace 全部落盘
  （off 869M / slot 807M，ascend_pt 格式：FRAMEWORK/torch.op_mark+op_range +
  PROF_*/device_0+host，可用 msprof 导出为 op_summary/task_time 后分析，
  方法同 legacy_gsm8k_perf 报告的设备侧分析）。
- 同窗参考数字（**含 profiler 开销，不作统计用**）：off mean_ttft 19.2s / slot 10.1s；
  与 op2 干净数据（11.0s / 9.6s）对照可见 profiler 对 off 腿拖累更大（+74% vs +5%），
  侧面说明 off 的 host 侧事件更密集——trace 分析时可关注这点。
- 查看：`tensorboard --logdir profile/`，或 msprof 离线导出（rank0 目录为
  dp0_pp0_tp0_dcp0_ep0_rank0_*_ascend_pt）。

## 可查看分析数据（2026-10-12 已按官方指南生成）

对全部 8 个 `*_ascend_pt` 目录跑过 `torch_npu.profiler.profiler.analyse`
（即 vllm-ascend《服务性能分析指南》Ascend PyTorch Profiler 一节的 analyse 步骤），
每个 rank 的 `ASCEND_PROFILER_OUTPUT/` 现已含：

| 文件 | 用途 |
|---|---|
| `trace_view.json`（~550M/rank） | chrome trace 时间线：MindStudio Insight 或 chrome://tracing 打开 |
| `op_statistic.csv` / `operator_details.csv` | 算子级统计/明细（含 moonep_* 内核） |
| `kernel_details.csv` / `task_time.csv` / `step_trace_time.csv` | 内核明细、任务/步段时间 |
| `api_statistic.csv` | torch API 统计 |
| `communication.json` / `communication_matrix.json` | 通信矩阵 |
| `analysis.db` / `ascend_pytorch_profiler_0.db` | MindStudio Insight 数据库 |

（op_statistic 抽查：slot rank0 上 planning_kernel 占 17.8%、moonep_dispatch_kernel
占 11.1%，MooneP 内核在统计中可见。）



## 性能分析报告（2026-10-12）

`gen_profile_analysis.py` 读 analyse 产物现算生成 `report/profile_analysis_report.html`：
off 62% 设备时间在 HCCL 等热点 rank（40ms 级长尾）；slot 瓶颈是 MooneP 三件套占 34.4%
设备时间、其中 90% 是等对端（纯传输 dispatch p50 仅 9μs），折合每 rank 占窗口 21%——
优化首位是把集合点等待与计算重叠。

## 追加验证实验：切分场景 profiling（2026-10-13，待跑）

**动机**：用户质疑"预算8192+输入16k 被切成两个 8k，为何不等于两个 8k/8k"。逐步对账
（op2 数据，uniform 档，每步耗时=TTFT÷步数）：

| 场景 | slot 每步 | 结论 |
|---|---|---|
| 预算8192·输入8k（8 步×8192） | 593ms | 基准 |
| 预算16384·输入16k（8 步×16384） | 1194ms | =2×593，**严格线性**（推翻"起步价按步数翻倍"的旧解释） |
| 预算8192·输入16k（16 步×8192） | 783ms | **+32%，被切分的 8192 步本身变贵** |
| off 三种场景 | 513/1028/520ms | 全部线性，切分不变贵 |

待验证假设：a) chunk-2 读 16k KV 使 attention 变贵 → rank 到达 MooneP 集合点时刻差拉大
→ 等待变长（slot 通信内核 85-99% 是等对端，敏感；off 被热点等待淹没，不敏感）；
b) 切分边界的调度/波次对齐开销。

**方法**：`run_profile_leg_chunked.sh slot`（同 run_profile_leg.sh 但 BATCHED_TOKENS=8192，
其余全同：3hot1r、L16k、seed 21001、torch profiler 单轮同窗）。与既有 mbt16384 profile
逐内核对比（重点：KvQuantSparseAttn 每步均值、MooneP 三内核等待尾）定位 +190ms/步 的去向。

**状态**：两次尝试未成——① 首次起服挂 Vector core timeout(507034)（早期初始化，硬件级
偶发）；② 重试时卡 3 被其他容器占用。已挂监视器等卡 3 释放自动重跑。

**状态更新（2026-10-14 05:30）**：连续 4 次起服失败，停止重试。失败特征各异：
①Vector core timeout(507034, DP2 早期初始化)；②HCCL ranktable detect 失败(EI0015, DP1)；
③④EngineCore 启动失败（DP0 等），全部发生在权重加载之前。同期宿主机 dmesg 持续报
Ascend 驱动层错误（dms_module check_and_release_event_procs "Get container pid failed"，
约 1 条/秒），疑似另一容器任务（占卡 3 至 04:59）异常退出后驱动状态未清理。
**待人工决定**：a) 等驱动错误自愈后重试；b) 对卡 0-3 做 npu-smi 复位（需确认，影响这些卡
上所有任务）；c) 容器/主机重启；d) 错峰重试。实验脚本就绪（run_profile_leg_chunked.sh），
跑通后按"动机"一节的方法做逐内核对比即可。

**结论（2026-10-15 18:04，卡组 4-7，两腿同窗 profiling 对比定案）**

| 内核（4rank 合计，切分腿×2步归一后与基准比） | 基准总 | 切分总 | 切分:基准 |
|---|---:|---:|---:|
| **planning_kernel** | 9.9s | **31.3s** | **3.17×** |
| KvQuantSparseAttn（attention） | 9.7s | 9.7s | **1.00×** |
| moonep_dispatch / combine | 11.9 / 10.9s | 15.8 / 11.6s | 1.33× / 1.07× |
| GroupedMatmul / TensorMove / 胶水 | — | — | 1.1~1.4× |

- TTFT：基准 10019ms（8 步×16384）vs 切分 13219ms（16 步×8192）→ 每 token +32%（与 op2 完全一致）
- **超额全部落在 planning 内核的跨卡栅栏等待**：等待尾 469 次×16.3ms（基准）→ 1852 次×15.6ms（切分），
  单次等待时长不变、次数×4；planning 均值 1277μs→3369μs
- **排除"attention 变贵"假设**（1.00×：切分后 attention 总量不变——8k+8k 与 16k 的 KV 读总量相同）
- 机制：切分使调度步×2 → planning 的"每步一次跨卡汇合"付×2；且 chunk 边界使各 rank 步进不齐，
  先到者空转的层数翻倍（等待次数×4）→ 每步固定型等待成本翻倍，而收益侧（消除倾斜等待）不随步数增长
- 优化指向：planning 栅栏等待与计算重叠（profiling 报告优化方向 #1 的具体落点）；同一 prompt 的两个
  chunk 在路由分布相同时 plan 可复用（SIM_ROUTE 下两 chunk 的 plan 完全相同），省一半 planning 调用

**数据位置**：profile/baseline_mbt16384/ 与 profile/chunked_mbt8192/（各 4 rank；已删 550M×8 的
trace_view.json 瘦身至 3.8G，CSV 统计齐全；bench 均值在两份 leg 日志，基准腿 json 被切分腿同名
覆盖是 runner 归档 bug，已修复——RESULT_SUFFIX 参数化）

**根因细化（2026-10-16 补，bench 窗口内精确统计，剔除 dummy 步污染）**

以"大 attention 内核只出现在 prefill 步"为时间锚，只统计 bench 窗口（基准 22.0s / 切分 30.1s，
rank0）：

- planning 快速体（纯计算）切分腿仅 23ms/988 次——计算量可忽略，等待是全部
- 基准：每步 planning 等待恒定 ~200ms（54,198,199,…）；attention 每步恒定 ~300ms——
  各 rank 步形状一致，只剩小抖动
- 切分：每步等待 0~882ms 乱跳（均值 425ms）；attention 每步 55↔258ms 交替（chunk1 读 8k KV、
  chunk2 读 16k KV，差 2~4×）——步内容不均匀 + 各 rank 队列相位错位（先完成 prompt 的 rank 混入
  decode/对齐步）→ 同一栅栏处 rank 间"便宜步 vs 贵步"错位 → 等待=整段成本差
- 每 16k token 的 planning 等待：1456ms（基准）→ 6800ms（切分）＝步数×2 × 单次等待×2.3
- 与"单独 8k"的判别：op2 中单独 8k 每步 593ms，×2=1186≈基准 1194（线性）——单独 8k 无此惩罚；
  惩罚专属"chunk1/chunk2 交替 + 相位错位"的切分场景
- 修正：前表"planning 489/309ms/步"混入 dummy 步；窗口内真值 426/200ms/步

## 追加实验：8k/8k（S=8192 · 输入 8192）off vs slot · 3hot1r（2026-10-16）

**目的**：补齐 2×2 拼图——这是 slot 在 s≈2.5 下**获益**的场景（op2 干净数据 −4.5%），
与已采集的 16k 两腿（基准/切分）构成完整对照。方法同前：run_profile_leg.sh 参数化后
（BENCH_LEN/BATCHED_TOKENS/RESULT_SUFFIX 环境变量），卡组 4-7，单轮同窗 torch profiler，
seed 21001，SIM_ROUTE=3hot1r(s=2.48)，SIM 下两腿输入流量一致。

**bench（profiler 开启下，不作统计口径）**：off 32/32 mean_ttft 5,665ms；slot 32/32
mean_ttft 5,156ms（**slot −9.0%**；op2 干净数据为 −4.5%，方向一致，profiler 对 off
拖累更大的现象与 16k 实验一致）。

**profiling 摘要（4 rank 合计）**：off：HcclLaunchAicpuKernel 30.1s（窗口 28.8s/rank，
倾斜下 HCCL 等待 45% 设备时间）、Computing 41%/未重叠通信 12%/Free 47%。slot：
planning 8.8s + dispatch 6.3s + combine 5.8s（MooneP 三件套合计 20.9s）、Computing 55%/
未重叠通信 0%/Free 45%；attention 两腿相等（4.6s）。**即使 8k 步为同构步，off 的
HCCL 等待（等热点 rank）仍大于 slot 的 MooneP 集合开销——这就是 8k/8k 高倾斜下
slot 获益的 profiling 表达。**

**数据**：profile/{off,slot}_L8k_mbt8192/（各 4 rank，ASCEND_PROFILER_OUTPUT 已生成，
trace_view.json 已删）；bench json 与服务日志在 data/（带 _L8k_mbt8192 后缀）。
