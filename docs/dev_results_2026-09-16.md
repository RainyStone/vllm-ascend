# dev 分支实测记录（2026-09-16）：容量档、pad 与均衡

本轮把 MoonEP 的"容量/pad/负载均衡"这条线做了一次系统测量与优化。
所有数字都来自本机 4 卡（npu 4-7）实跑，工具在 `scripts/moonep_dev/`。

## 1. 负载与几何

| 项 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B（E=128, K=8, H=2048, 48 层, bf16） |
| 并行 | DP4 × TP1 = EP4（每卡 32 个专家） |
| prompt | GSM8K 原始题面（`scripts/moonep_dev/make_prompts.py`，p50=69 token） |
| batch | N_SEQ=256（每卡 64 条） |
| MBT | 8192（每卡 prefill ≈ 4.7-6.4k token，1-2 步装下） |
| 输出 | 上限 1024 token，自然 EOS |

## 2. 跨卡负载不均（逐层 max/mean，四卡）

判据：**每一层**取"最忙卡行数 ÷ 四卡平均行数"，再看整批层的分布。

| 路线 | p50 | 均值 | p90 | 最大 |
|---|---|---|---|---|
| MC2 decode | 1.200 | 1.253 | 1.500 | 3.000 |
| MC2 prefill | 1.033 | 1.160 | 1.450 | 1.628 |
| **MoonEP slot（decode + prefill）** | **1.000** | **1.000** | **1.000** | **1.000** |

- MC2 的基线在**四次独立运行**里完全复现（p50 1.198/1.200/1.200/1.200）⇒ 可信 ✓
- 把整段 run 平均后各卡只差 12%，**"平均"会把瞬时 20–50% 的不均抹掉** ⇒ 必须用逐层数据 ✓
- slot 的 1.000 是**逐层精确**（连 max 都是 1.000），不是平均意义上的均衡 ✓

## 3. 卡内专家组不均（max 组 / 平均组）

| 路线 | decode | prefill |
|---|---|---|
| MC2 | 3.85×（p90 5.66×）| 3.30×（p90 7.25×）|
| slot（旧档位表）| 3.86× | **6.33×** ✗ |
| slot（新档位表）| 3.86× | **3.12×** ✓ |

prefill 的极端不均是 **pad 造成的**：补零行全部路由到"本卡最后一个专家"，
堆出一个 6–14 倍于平均的组。档位表做细后（见 §4）补零行降了 87% ⇒ 回到 MC2 水平 ✓。
**decode 的 3.86× 是模型路由本身的偏好，两条路线一样，不属于通信层。**

## 4. 容量档位表与 pad

旧实现是 pow2 阶梯（≥64 或 ≥128）；prefill 步 T≈6.4k 时被迫选 S=8192，
补零行 = (8192−6400)×8 ≈ 14,336 行/卡/层，全部白算且堆在单专家上。

新实现（`MOONEP_POOL_BUDGET_MB` / `MOONEP_POOL_MAX_RUNGS`）：

```
16,32,64,128,256,512          (2 的幂：decode 尾巴)
1024,1536,…,8192              (等步长 512：prefill)
```

| | 旧 | 新 |
|---|---|---|
| 档数 | 8 | **21**（池 2.2 GB）|
| prefill 补零行/卡/层 | 14,336 | **2,048（−86%）** |
| prefill 行/卡/层 | 65,536 | **53,248** |
| 卡内不均（prefill）| 6.33× | **3.12×** |

**为什么是等步长而不是几何档**：每档浪费 = (S−T)×K 行，**等步长把浪费钳在 step×K 以内**；
几何档的绝对浪费随 S 增长、落点还靠运气（实测 T=4676：比率 1.35 浪费 688 行，比率 1.22 反而 6,416 行）。

## 5. 端到端（同一 prompt 集、同 batch、同 MBT）

| 指标 | MC2 | slot 旧档 | slot 新档 |
|---|---|---|---|
| TPOT | 82.1 ms | 90.5 ms（+17%）| **90.4 ms（+10%）** |
| TTFT | 395 ms | 687 ms（+74%）| **491 ms（+24%）** |
| prefill 吞吐 | 18,097 tok/s | 8,797 | **12,892（+47%）** |
| 生成吞吐 | 760 tok/s | 673 | 682 |

（MC2 自身 run-to-run 有 ±5% 波动，比值看趋势 ✓。**均衡省下的 ~1% 抵不过每层 prefetch 的固定开销** ⇒ TPOT 仍高 10%。）

## 6. 释放 vLLM 原权重（`MOONEP_RELEASE_WEIGHTS`，默认开）

slot 模式把本卡专家权重复制进对称内存（home mirror，288 MB/层 × 48 = **13.8 GB/卡**，
MTE prefetch 需要"源 VA"、GEMM 需要 `[epn+B]` 零拷贝别名）。计算只用别名，
因此 `publish()` 后可以**原地改 `layer.w13_weight.data`** 指向 home chunk（必须原地改：
运行时用 `id(w13_weight)` 查状态），把原张量释放掉。

| 验证 | 结果 |
|---|---|
| A/B（`RELEASE=0` vs `1`，4 卡 × 64 prompt × 128 token）| **逐 token 100% 相同** ✓ ⇒ 对计算无副作用 |
| 开关有效性 | `RELEASE=0` ⇒ 0 行 release 日志；`=1` ⇒ 196 行（48 层 × 4 卡）✓ |
| 释放后可用显存 | 56.6–56.8 GB / 79.8 GB ✓ |
| **GPU KV cache size** | `=1`: 353,152 / 353,024；`=0`: 348,416 / 351,744 ⇒ **几乎不变** ✗ |

**KV 容量没变的原因**：引擎按 **profile 阶段的峰值显存** 算 KV 预算，而 swap 那一刻
两份权重同时存在 ⇒ 峰值没降。**所以这 13.8 GB 的价值是"常驻余量/抗 OOM"**，要兑现成 KV
得把 GMU 调高（每 +0.01 ≈ +0.8 GiB ≈ **+8.5k token KV**）。

## 7. 正确性判据的变化（重要）

档位表改动会改变 prefill 的补零行数 ⇒ grouped GEMM 分组长度/tiling 变 ⇒ 前向数值末位不同
⇒ 写进 KV 的值不同 ⇒ 从很早就与 MC2 分叉：

| gen | 首 token 一致率 | 全长一致率 |
|---|---|---|
| 32（旧档位表）| 100% | 100% |
| 1024（新档位表）| 86–97% | 0–3% |

⇒ **跨实现比 bit-exact 不是有效判据**（两个不同内核顺序必然分叉 ✓）。
改用任务准确率：

| 路线 | GSM8K zero-shot 精确匹配（256 条，CoT 末数字解析）|
|---|---|
| MC2 | 111/256 = 43.4% |
| slot（新档位表 + 释放权重）| 140/256 = **54.7%** |

两条路线都在合理区间、slot 不差 ⇒ 改动**没有损伤精度** ✓（解析规则粗糙但两腿同一规则 ✓）。

## 8. 本轮撞到的硬约束

1. **Buffer 档数受 VMM 物理块限制，不是字节数**：每档除了 hidden 块，还要为
   dispatch/combine/planning 等算子各分配一批 workspace（~10 个 16 MiB 级 VMM 块）。
   实测 **51 档 ⇒ 池初始化阶段 `aclrtMallocPhysical(...) failed with 207001`** ✗；
   31 档安全 ✓。默认已锁 `MOONEP_POOL_MAX_RUNGS=32`。
2. **懒建档不可行**：想在运行时按 `round_up(T,8)` 新建 Buffer，被库显式拒绝
   （`cannot register a symmetric descriptor after buffer-backed shmem initialization;
   it would not belong to the active heap`）⇒ 对称堆在 init 时一次性定死 ✓。
   `scripts/moonep_dev/probe_lazy_buffer.py` 是这条结论的复现脚本。
3. **pad 无法从 grouped GEMM 里摘掉**：`MOONEP_PAD_CUT` 的实现假设"补零行是专家组的尾部"，
   实测**错误**（规划器在组内按自己的 k 切片排行）⇒ smoke harness 相对误差 0.6–0.8% PASS
   变成 **74% FAIL** ✗。旋钮保留但默认关闭，代码里有实测记录。
4. **GMU 用自适应值有风险**：同租户中途占回显存会让引擎拒绝启动或运行中被 OOM 杀
   （曾出现 DP0 静默消失、其余三卡死等 8 分钟）。

## 9. 待办

1. **卡内组不均到底吃不吃时间**：埋点已就绪（`MOE_ROW_TIMING=1` 逐层同步计时 +
   `scripts/moonep_dev/analyze_skew_time.py` 在 decode 平台做相关性），**尚未跑**。
   若不吃时间（Ascend grouped matmul 会把大组内部再切给多核），就不该花力气治。
2. **TPOT +10% 的来源**：拆 prefetch 分段计时，判断是带宽受限还是被串在关键路径上。
3. **GMU 兑现**：在 `GMU=0.92` 下验证 KV 是否从 353k 涨到 ~400k。

---

# 附录（2026-09-17）：设备侧 profiler 分析 —— "为什么慢"的定论

## 方法

`msprof`（CANN 自带，不依赖 vLLM 的 profiler API）分别采集 **MC2** 与 **MoonEP slot** 两条路线，
同一负载（N_SEQ=64、每卡 16、PROMPT_LEN=512、GEN=8、DP4=EP4、MBT=2048）。
原始数据（8 个 rank 的完整 trace + `op_summary` CSV）：

```
/home/tmp/msprof/msprof_mc2/   2.1G
/home/tmp/msprof/msprof_slot/  3.3G
python3 scripts/moonep_dev/analyze_profile.py /home/tmp/msprof/msprof_mc2 /home/tmp/msprof/msprof_slot --top 12
```

## 设备时间总账

| | MC2 | slot |
|---|---|---|
| 总设备时间 | **4.86 s** | **13.68 s**（2.8× ✗）|
| 通信相关 | `MoeDistributeDispatchV3` 1.041s + `CombineV3` 0.432s = **1.473s（30.3%）** | `planning_kernel` 3.470s + `barrier_on_stream_kernel` 3.403s + `prefetch_weight_kernel` 1.716s + `combine` 1.161s + `dispatch` 0.604s |
| MoE GEMM | `GroupedMatmulV5` 0.841s | 0.987s（同量级 ✓）|
| attention 的 matmul | `MatMulV3` 0.639s | 0.664s（同量级 ✓）|

**逐层折算**：MC2 融合通信 ≈ **120 µs/层**（85+35）；MoonEP 的真搬运 ≈ dispatch **48 µs** + combine **93 µs** + prefetch **69 µs** ≈ **210 µs/层**。

## 两个方法论要点（都实测验证过）

1. **含跨 rank 同步的算子，其 `Duration` 是"等到齐"的时间，不是算力成本** ✗。
   `barrier_on_stream_kernel` 每层 4 次 × 68 µs = 272 µs，但用去栅栏补丁
   （`ASCEND_MOONEP_PREFETCH_RELAXED=1`）实测：**去掉它反而慢 2.9 ms** ✗ ⇒ 那是
   **slack**（等最慢 rank），不是可省的时间。`planning_kernel` 的 278 µs 同理含内部
   `aclshmemx_barrier_all_vec` 的等待。
2. **host 侧分段计时只能反映"入队"** ✗。我们此前所有 host 计时（`disp_lib` 0.12 ms、
   `cross_rank_barrier` 1 µs、prefetch 0.49 ms…）都是入队时间；设备上的真实耗时是它们的
   数倍 → 本轮多次误判都源于此。

## 结论：慢在哪

- **真成本**：MoonEP 的通信工作量约 210 µs/层（≈10 ms/步），MC2 的融合通信约 120 µs/层（≈5.8 ms/步）；
- **更大的差**：**MC2 的通信与 GEMM 融合、互相掩盖**；MoonEP 是 planning → dispatch → GEMM → combine **串行**、无重叠 ⇒ 通信全部暴露 ⇒ 叠加起来正好是实测的 **+12~13 ms/步**；
- **额外项**：`planning_kernel` 278 µs/层（跨 rank 直方图交换 + 逐行分配 + dup/槽位规划）在 MC2 路径里根本不存在（其调度在算子之外）。

## 端到端（N_SEQ=256、每卡 64、MBT=8192、DP4=EP4）

| 路线 | step_ms | TPOT | prefill tok/s |
|---|---|---|---|
| MC2 | **53.9** | 54.7 | **18,194** |
| MoonEP owner | 66–67（+23% ✗）| 67.5 | 11,559（−36% ✗）|
| MoonEP slot | 69.7（+29% ✗）| 70.0 | 12,886 |

## 已被实测排除的方向（不要再走）

| 方向 | 实测 |
|---|---|
| host 侧 Python 优化（预分配 scratch、env/几何缓存）| ≈1%（噪声级）✗ |
| 旁路流并行（投机预取）| 死锁（库栅栏）✗；打补丁后 +23 ms/步 ✗ |
| 持久化槽缓存 | 68% 命中、跳过 80% 调用，仍 **+9 ms/步** ✗（host 读比省下的 prefetch 贵 3 倍）✗ |
| 去掉跨 rank 栅栏（relaxed）| **+2.9 ms** ✗ |
| prefetch 本身 | 不在关键路径 ✗（砍掉 80% 调用 step 无变化）|

## 下一步（按性价比）

1. **`num_sms` / `MOONEP_NUM_SMS_DEDUP` 调参** —— 目标是 278 µs/层的 `planning_kernel`（默认 `num_sms=32`，对 Ascend950 偏少）；纯 env 改动，可能省 5–10 ms/步；
2. **prefetch 缓存要兑现收益，必须让槽内容在步间持久** ⇒ 按层独立槽 chunk（每层 ~19 MB、48 层 ~900 MB），命中时无需拷贝也无需 host 读；代价是每层一份 `[epn+B]` 别名（+96 VMM 映射）；
3. **重叠**（真正的大头）需库侧把 dispatch + GEMM + combine 融合，属大改。
