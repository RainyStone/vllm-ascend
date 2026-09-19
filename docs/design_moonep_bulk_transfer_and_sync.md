# 设计：MoonEP dispatch/combine 的搬运大块化与栅栏收敛

> 依据来自 `docs/ladder_T_ttft_imbalance.html`（T 阶梯实测）与 `dev_results_2026-09-16.md`
> 的 profiling 附录。本文只描述**改造方案**，所有数字都标注了来源。

## 1. 背景：两条已实测的事实

**事实一 —— MoonEP 的耗时随 T 接近线性，all-gather 是次线性。**

| T（每卡每步 MoE token 数）| 69 | 276 | 1104 | 4416 |
|---|---|---|---|---|
| MoonEP slot TTFT (ms) | 75.7 | 269.2 | 490.8 | 1493.3 |
| all-gather TTFT (ms) | 69.5 | 185.3 | 251.0 | 603.5 |
| slot/ag | 1.09 | 1.45 | 1.96 | 2.47 |

T 涨 64× ⇒ all-gather 涨 **8.7×**，MoonEP 涨 **19.7×**。

**事实二 —— 两条路线的跨卡字节量相等。**

| 路线 | 进 MoE 前 | 出 MoE 后 | 每卡跨卡字节量（R=8, K=8）|
|---|---|---|---|
| all-gather | `all_gather` | `reduce_scatter` | 2(R−1)·T·H = **14 T·H** |
| MoonEP | dispatch（MTE put）| combine（MTE get）| 2·T·K·(R−1)/R·H = **14 T·H** |

⇒ **差异不在"搬多少"，在"怎么搬"。**

## 2. 问题定义

### P1 搬运粒度：每条 route 一次 put

`kernels/moonep_dispatch.cpp` 的外层遍历 route（`token_id = i/K`, `topk_id = i%K`），
每条 route 发一次：

```cpp
uint64_t hidden_off_bytes = loff * H * sizeof(HiddenT);
aclshmemx_mte_put_nbi(hidden_symmetric + hidden_off_bytes,  // 对端对称内存
                      hidden_sh + token_id * H,             // 本地一行
                      row_elements,                         // = H（Qwen3: 4 KB）
                      dest_rank, event_id);
```

| | 一次搬 | 每层每卡次数 | 每步（48 层）|
|---|---|---|---|
| MoonEP dispatch | **4 KB** | **T×K**（T=4416 → 35,328）| **~170 万次** |
| all-gather | 数 MB 连续块 | ~67 | ~3,200 |

⇒ 报文数差约 **500×**。这是"线性 vs 次线性"的机制。

### P2 同步粒度：每层 4 次全局栅栏

**来源**：一次 `prefetch_projections` 调用传了 2 个投影（`w13` / `w2`，形状不同），
`api.py:734` 的 `for full_weight in weight_prefetch_args` 循环**逐投影**调用
`ctx['kernels']['prefetch'].compute(...)`，而 `csrc/ops/prefetch.cpp:118/:134`
在每个 kernel 前后各加一个 `aclshmemx_barrier_all_on_stream`
⇒ **2 投影 × 2 = 4 次/层**（profiler 实测 `barrier_on_stream_kernel` 4 次/层 ✓）。

**性质**（库作者自己的注释，`prefetch.cpp:105-116`）：

- 源是 peer 的 **home chunk**（init 时写好、之后不再改）⇒ **pre-barrier 无物可排**；
- kernel 自己的 **MTE3_MTE2 event** 已保证每次 GET 完成 ⇒ **post-barrier 对本地可见性也不需要**；
- 每个 rank 必须设同一个 env，否则**栅栏代际错位**。

**为什么默认不能关**：`api.py:503` 说明 prefetch 的 host 栅栏与 planning/dispatch 的
device `aclshmemx_barrier_all_vec` **共用同一个 `sync_counter` 代际**，rank 间代际不一致
就**死锁**。实测全集群一起关（`ASCEND_MOONEP_PREFETCH_RELAXED=1`）反而**慢 2.9 ms** ——
说明它们在当前结构下承担 rank 对齐。

## 3. 改造 A：搬运大块化

### A1 源端 compact（发送方本地）

扫一遍本卡的 `dst[]`（planner 已经为每条 route 算出 `dst = dest_rank*NvS + loff`），
按 `dest_rank` 分组，把散落的 hidden 行**连续化**到一块 staging：

```
in : hidden_sh[T, H]（按 token 顺序）
     dst[N]（每条 route 的目标 rank + loff）
out: staging[(R-1), max_rows, H]  或  每 dest 一段连续区间
     dest_row_idx[dest][j] = 本地第几个 token 的行（用于填 loff 表）
```

- 这是一次**卡内** DMA（读 T×H + 写 T×H），本地带宽，不进网络。
- staging 大小：`T×H×2B`（Qwen3 T=4416 → **18 MB**）；也可按 dest 分块流式做，几 MB 即可。
- **复用现成件**：内核里已有 `DataCopyPad` / `DataCopyExtParams` 的用法，不引入新原语。

### A2 大块 put（网络）

每个 dest 一次 `aclshmemx_mte_put_nbi`，搬 `[n_rows × H]` 的连续块：

- **put 次数：T×K → R−1 = 7 次/层/卡** ✓
- 分片：UB 只有 ~188 KB，块大于 UB 时按 `rows_per_chunk` 流水（**内核里已有这个分片逻辑**，
  见 `prefetch.cpp` 的 ping-pong 双缓冲）。

### A3 接收端 permute（对端本地）

**冲突点**：目标端当前布局是**专家连续**（GEMM 的 `group_list` 要求），而大块 put 要求
**来源连续** ⇒ 接收端要补一次 permute 回到 expert-major。

**不是新发明**：这正是 all-gather 路线里 `aclnnMoeInitRoutingV3` 干的事（实测 265 ms），
也是 MC2 / DeepEP 的标准 dispatch 结构。做完之后**接收端数据形态与 all-gather 一致，
两条路线可共用同一套 MLP 接口**。

### A4 涉及的接口与文件

| 层 | 文件 | 改动 |
|---|---|---|
| planner | `kernels/planning.cpp` | `loff` 分配从"专家连续"改为"来源连续"（每 (source,dest) 一段）|
| dispatch | `kernels/moonep_dispatch.cpp` | 加 A1 的 compact 段；put 改为按 dest 整块；粒度参数从 `H` 改为 `row_bytes` |
| 接收 | 新增一次 permute（可复用 vLLM 的 `MoeInitRouting`，或写进 dispatch 的接收侧）| — |
| combine | `kernels/moonep_combine.cpp` | 对称改造：从每个 peer 一次 get 一段连续区间，再本地 K-sum |
| staging | 对称堆或本地 workspace | 新增一块 `[T, H]` 量级的 buffer（每层复用）|

### A5 正确性验证

1. **黄金对比**：同一 routing 下，改造前后 `hidden_nvsh` 逐字节一致（先在单元测试里做，
   不依赖引擎）；
2. **端到端**：GSM8K 准确率与改造前对齐（**不要用逐 token 相等** —— 引擎 run-to-run
   不可复现，见 `dev_results` 附录四）；
3. **报文数**：`msprof` 里 `moonep_dispatch_kernel` 的调用数不变（还是 1 次/层），
   但内核内的 put 次数要用 `GetSystemCycle` 埋点或 profiler 的 MTE 计数确认下降。

## 4. 改造 B1：合并两个投影的 prefetch（栅栏 4 → 2）

**做法**：把 `weight_prefetch_kernel` 从"一次一个投影"改成"一次收一个投影列表"，
内核对每个投影用各自的 `(H, Hp)` 循环处理（内核本来就有 `rows_per_chunk` 的分片逻辑，
加一层投影循环即可）。

| | 现状 | B1 之后 |
|---|---|---|
| kernel launch | 2 次/层 | **1 次/层** |
| 跨卡栅栏 | **4 次/层** | **2 次/层** |

**代价/风险**：**低** —— 不动 `sync_counter` 代际机制，只是把两次调用合一次。
`Buffer.prefetch_projections` 的**接口不用改**（它本来就是收一个列表，是内部循环发多次）。

**收益估算**：48 层 × (1 次 launch + 2 次栅栏)。按前面标定的 ~20 µs/launch 量级，
单是 launch 就 ~1 ms/步；栅栏的等待是否真能省掉要看它当前是否在关键路径上
（需要用 B1 的 A/B 实测确认）。

## 5. 改造 B2：prefetch 同步解耦（栅栏 2 → 0–1）

**目标**：让 prefetch 不再依赖共享的 `sync_counter` 代际，从而可以自由地决定要不要栅栏。

**做法（二选一）**：

1. **event 方案**（推荐，改动最小）：prefetch 的存在意义是"权重先于 GEMM 到位"，
   而这个顺序是**本地**的 —— 用 `torch_npu` 的 event / `SetFlag-WaitFlag` 就能表达，
   完全不需要跨 rank 栅栏。把 `csrc/ops/prefetch.cpp` 的两个
   `aclshmemx_barrier_all_on_stream` 去掉，改由**调用方**用 event 把 prefetch 与
   本地消费者排序（`ASCEND_MOONEP_PREFETCH_RELAXED=1` 已经是这个语义，只是要把它
   变成默认，并**同时**解决代际问题）；
2. **flag 方案**：像 dup 路径那样，发送方把完成标志 one-sided 写进对端，接收方设备侧
   轮询（库里有现成模式：`WaitForSourceDupInbox` / `PublishSourceDupPayloadDoneFlags`）。

**为什么必须先做这一步**：只要 prefetch 还在共享代际序列里，**所有 rank 就必须
保持一致地进出栅栏**，栅栏数就降不下来；大块化省下的报文收益会被每层的栅栏等待吃掉一半。

**风险**：中 —— 动同步设计。必须验证：
- 去掉 prefetch 栅栏后，**代际计数在所有 rank 上仍然一致**（这是死锁的直接原因，
  我们在这个坑里待过：4 rank 挂死 + 设备超时 `507034`）；
- 建议加一个"代际一致性"的自检（启动时打一次各 rank 的 `sync_counter`）。

## 6. 实施顺序

| 阶段 | 内容 | 风险 | 前置 |
|---|---|---|---|
| **B1** | 合并两个投影的 prefetch | **低** | 无 —— 可以先做 |
| **A** | dispatch/combine 大块化 | 中（planner + 内核 + 接收端 permute）| 无 |
| **B2** | prefetch 同步解耦 | 中高（跨 rank 同步）| B1（先证明合并可行）|

**建议先做 B1**：它最小、风险最低，而且能先验证"栅栏数确实能从 4 降到 2 并带来收益"，
为 B2 和 A 的价值判断提供依据。

## 7. 验收标准

| 指标 | 现状 | 目标 |
|---|---|---|
| TTFT（T=4416）的 slot/ag 比值 | **2.47** | < 1.5（把线性拉向次线性）|
| 每层跨卡栅栏数 | **4** | **1–2** |
| dispatch 每层每卡 put 次数 | **T×K = 35,328** | **R−1 = 7** |
| dispatch/combine 的耗时随 T 的指数 | 19.7×（64× 的 T）| 向 8.7× 收敛 |
| 正确性 | — | GSM8K 准确率不降（**不用逐 token 比对**）|

## 8. 明确不做的事

- **不追求"打平 all-gather"**：本设计只是把 MoonEP 的劣势从 2.5× 收窄，不改变
  "在均衡负载上 all-gather 更优"这个结论（不均衡实测只有 1%，MoonEP 没有可均衡的对象）；
- **不动 chunk 切分**：用切 chunk 放大不均衡会同时把每层的固定开销×N，净亏
  （见 `ladder_T_ttft_imbalance.html` §③）；
- **不引入新的通信原语**：全程仍是 `aclshmemx_mte_put/get_nbi` 的**单边 RMA**，
  对面 CPU 不参与。
