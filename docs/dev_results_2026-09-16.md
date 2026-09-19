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

1. ~~**`num_sms` 调参**~~ —— **已做完，结论是死路**（见 §附录二）；
2. **prefetch 缓存要兑现收益，必须让槽内容在步间持久** ⇒ 按层独立槽 chunk（每层 ~19 MB、48 层 ~900 MB），命中时无需拷贝也无需 host 读；代价是每层一份 `[epn+B]` 别名（+96 VMM 映射）；
3. **重叠**（真正的大头）需库侧把 dispatch + GEMM + combine 融合，属大改。

---

# 附录二（2026-09-17）：`num_sms` 扫描 —— planning_kernel 是延迟受限，不是算力受限

## 硬约束（先撞到的）

```
RuntimeError: planning core_num must be in [8, 56], got 64
WorkerProc initialization failed ... (启动即失败)
```

`num_sms=64 / 128` 两条腿**引擎起不来**（不是跑得慢，是 init 抛异常）。
`planning_workspace_sizes()` → `local_workspace_elements(..., core_num)` 里做的校验：
**planning 的核数上限就是 56**。库默认 `num_sms=32`（`api.py:211`），所以合法区间只有 `[8,56]`。

## 设备侧：确实加速了

同样 config（N_SEQ=64、每卡 16、PROMPT_LEN=512、GEN=8、MBT=2048、DP4=EP4），
只用 `MOONEP_NUM_SMS=56` 换个环境变量，`msprof` 逐 kernel 对比：

| kernel | num_sms=32 | num_sms=56 | Δ |
|---|---|---|---|
| `planning_kernel` | **278.0 µs/层** | **242.4 µs/层** | **−12.8%** ✓ |
| `moonep_dispatch_kernel` | 48.4 µs/层 | 38.7 µs/层 | −20.0% ✓ |
| `moonep_combine_kernel` | 93.0 µs/层 | 80.6 µs/层 | −13.3% ✓ |
| `barrier_on_stream_kernel` | 68.2 µs/次 | 62.6 µs/次 | −8.2% |
| `prefetch_weight_kernel` | 68.8 µs/次 | 68.7 µs/次 | 0 ✗ |
| `GroupedMatmulV5` | 0.987 s | 0.987 s | 0（不受影响，预期内）|
| **总设备时间** | **13.68 s** | **12.69 s** | **−7.2%** ✓ |

## 端到端：白干

| num_sms | step_mean (ms) | TPOT p50 | 相对默认 |
|---|---|---|---|
| 32（库默认）| 71.87 | 72.11 | — |
| 40 | 71.72 | 71.98 | −0.2% |
| 48 | 73.51 | 73.78 | **+2.3%** ✗ |
| 56 | 71.33 | 71.57 | −0.8% |

四条腿落在 **71.3–73.5 ms**，即今日噪声带宽 ~±1.5%（今晨 slot 基线是 69.7 ms，
同 config 两天差 3% ⇒ 环境漂移比信号大）⇒ **墙钟上没有可信收益** ✓。

## 为什么"设备省了 1 s、墙钟不动"

- 用 Amdahl 估：纯算力受限时 56 核相对 32 核最多 1.75×，实测只有 **1.13×**
  ⇒ planning 的 **~2/3 是固定延迟**（跨 rank 直方图交换 + 内部 `aclshmemx_barrier_all_vec`），
  不是并行度不够；
- 而它省下的那部分，**正好被栅栏的等待时间吃掉**：`barrier_on_stream_kernel`
  自己从 68.2 → 62.6 µs/次（每层 4 次）⇒ 最长 rank 早到，其余 rank 就多等一会儿；
- ⇒ 这个旋钮的天花板是 278 µs 里的 13% ≈ 36 µs/层 ≈ 1.7 ms/步，实测兑现 **0.5 ms**（其余是 slack）。

## 处置

- `MOONEP_NUM_SMS` 保留为 env 旋钮（默认不设 = 32，与库一致）：
  **设备时间 −7% 是实打实的**，在抢卡/多租户场景下能多留余量；墙钟中性；
- **不要再往这个方向投时间**：planning 的瓶颈是跨 rank 交换的延迟，要动就得动交换本身
  （减少交换的数据量/次数），不是加核。

---

# 附录三（2026-09-17）：planning 到底是什么受限 —— 同步，不是算力

## 先纠正一个数字

附录二里 `planning_kernel` 的 **278 µs/层**是 **profiler 口径**，不是真实成本。
拆开来看：

| 项 | 值 | 来源 |
|---|---|---|
| 引擎内 `planning_kernel` Duration | 278 µs/层（p50 257）| msprof |
| ↳ 其中"等其它 rank 到达" | **~167 µs** | 见下"栅栏判定" |
| ↳ 剩下（profiler 开销 + 入队）| 其余 | — |
| **隔离测量的真实设备成本** | **~58 µs/层**（32 块）| `bench_planning.py` 的 `seq` |

**`iso` vs `seq` 的区别很重要**：`iso` 是"发一次调用 + sync"，里面含着 host 入队延迟；
`seq` 是"连发 N 次只 sync 一次"，host 早跑到前面去了，量到的是纯设备时间。
同配置下 iso≈100 µs 而 seq≈58 µs ⇒ **~40 µs 是 host 入队，不是 kernel**。

## 栅栏判定：四个算子的 end skew 都是常数

取每次调用在 4 个 rank 上的 start/end 时刻：

| kernel | start skew p50 | end skew 均值 / 标准差 |
|---|---|---|
| `moonep_dispatch_kernel` | 184.5 | **179.7 / 5.4** |
| `moonep_combine_kernel` | 206.8 | **179.7 / 5.7** |
| `barrier_on_stream_kernel` | 201.0 | **179.7 / 5.4** |
| `planning_kernel` | 346.8 | **179.4 / 5.4** |

四个算子的 end skew **mean≈max≈p90≈180 µs、标准差只有 5.4 µs** ⇒ 这不是"等待"，
这是 **4 张卡的时钟基准差了一个常数 ~180 µs**。减掉它之后：

| kernel | 真实 start skew | 真实 end skew |
|---|---|---|
| `dispatch` | **5 µs** | ~0 |
| `barrier_on_stream` | 21 µs | ~0 |
| `combine` | 27 µs | ~0 |
| **`planning`** | **167 µs** | ~0 |

⇒ **所有这些都是"结束对齐"的栅栏算子**，谁最后到谁决定大家什么时候走。
planning 的 167 µs 就是**上一层尾部（attention + router）累积出来的 rank 间偏斜**，
在 planning 的栅栏处被吸收掉了 —— 它不是 planning 的成本，是**系统的偏斜**。
这也解释了为什么砍 planning 的 13% 设备时间对墙钟没影响。

## 那 58 µs 里是什么

**pipe 计数（msprof，`planning_kernel`）**：`aiv_time` 278 µs，但真正发射的只有
`vec 7.3 + scalar 15.6 + mte2 3.4 + mte3 2.8 ≈ 29 µs` ⇒ **90% 在停等**。
（对照 `prefetch_weight_kernel`：mte2 65.6 / aiv 68.0 = 96% 忙，说明这个口径可信。）

**核数扫描**（R=4、E=128、K=8，隔离，单位 µs，`seq`）：

| 块数 | N=128 | N=512 | N=2048 | N=8192 | N=32768 | N=65536 |
|---|---|---|---|---|---|---|
| **8** | **48.7** | **52.6** | **54.3** | **61.1** | 111.5 | 188.3 |
| 32（库默认）| 58.3 | 60.5 | 61.6 | 63.7 | **71.7** | **85.8** |
| 56 | 67.1 | 72.5 | 72.4 | 74.4 | 79.6 | 90.7 |

⇒ 固定开销**随块数线性涨**（≈0.4 µs/块），路由数在 128→8192 之间只涨 12 µs。

**E 扫描**（R=2、S=64、32 块，`seq`）：epn=32→50.1，64→51.5，128→55.6，256→54.7，512→74.6。
分配迁移里那个 `(R+epn)·epn` 的**标量循环**是 O(epn²)，epn 涨 16 倍只多了 5 µs
⇒ **它不是瓶颈**（若是，epn=512 该是 epn=32 的 256 倍代价量级）。

**结论**：planning ≈ **40-45 µs 纯栅栏/收敛开销 + 十几 µs 实际工作**，
对 S 和 E 都几乎不敏感，只对**块数**敏感。要再往下压，只能减少同步轮次
（现结构：1 次跨卡 `barrier_all_vec` + **14 次 `SyncAll<true>`** + 19 处 SIMT 调用），
属库侧重写，风险和收益（~1% 步时间）不成比例。

## 本轮的改动（已提交 → dev）

**给 planning 单独的（更窄的）块数**：`_planning_num_sms()` 按路由数选
**N ≤ 8192 用 8 块**，否则 `min(num_sms, 32)`；dispatch/combine/prefetch **保持 32**
（它们是带宽受限的，prefetch 实测 96% MTE2 忙）。
decode 几何（{R=4, E=128, K=8, S=64}）下 **60.5 → 52.6 µs（−13%）**。
`MOONEP_PLANNING_NUM_SMS` 可覆盖。

正确性：`test/mte/test_planning.py --ep-size 2 --num-tokens 64 --num-topk 8`
的 balanced/skewed/duplicate/random 四例在 8 块路径上**全部通过** CPU 黄金参考比对。

## 一个必须记住的坑

`sys.path[0]` 是 **cwd**，而 `/home/ascend-moonep/ascend_moonep/` 本身就是一个包目录
⇒ **在 `/home/ascend-moonep` 目录下跑测试，导入的是外部仓库的那份 Python 层，
不是 `third_party/ascend_moonep/` 里改了的那份**（两份 api.py 相差 90 行，inode 不同）。
引擎从 `/home/tmp` 起跑，导入的才是 third_party 那份。
**验证改动是否生效的第一步永远是打印 `ascend_moonep.api.__file__`。**
（`ascend_moonep` 还有 editable install 的 meta-path finder 指到 `/home/ascend-moonep`，
但它是 `append` 到 `sys.meta_path`，所以 cwd/PYTHONPATH 仍然优先。）

---

# 附录四（2026-09-17）：砍 host 算子条数，端到端 −3 ms/步

## 改了什么

| # | 改动 | 为什么安全 |
|---|---|---|
| 1 | `tokens_per_expert` 不再每层用 `torch.histc(topk.to(f32), ...).to(i32)` 重算（2 个 Cast + 1 个 Histc，注释自标价 ~40 µs/层），改成一个一次分配的零 scratch | 库**不消费**它：`PlanningOp` docstring 明说 *"retained for MoonEP API compatibility … the kernel computes local expert counts from topk_flat"*；`_collect_inputs` 只查 shape/dtype；**它根本不在 `launch_planning_op(...)` 的参数表里** ⇒ 内容到不了任何 kernel |
| 2 | 行数统计不再用 `counts[0] = cu64[0]` 单独拼一个标量；改用 `cu_pad`（nrows+1，`[0]` 恒 0），一个 `sub` 得到 `cu[i]−cu[i−1]`（含隐含 `cu[-1]=0`）| 对 200 组随机 `cu` 验证新式 == 旧式 == `diff(cat([0],cu))` |

## 一个必须先知道的方法论问题：**这个引擎不可复现**

本来打算用"逐 token 完全一致"做 A/B，**结果同一份代码连跑两次输出就不同**：

```
rank0: after vs after2 identical=False
   after [:9] = [220, 7281, 11, 10339, 279, 7354, 304, 7716, 624, ...]
   after2[:9] = [220, 7281, 11, 10339, 279, 7354, 304, 7716,  13, ...]
```

⇒ **run-to-run 非确定性**，token 相等在两个方向上都不是判据。正确性只能靠
"按构造论证 + 单测 + 准确率统计"，性能只能靠 **多轮交替测量**。

## 实测（2 卡 DP2×EP2，交替 3 轮）

配置：`N_SEQ=64`、`GEN=128`、`MBT=2048`；`after`/`before` 交替执行以抵消租户负载漂移
（同一台机器上另外 6 张卡被别的租户占着）。

| | step_ms 各轮 | 均值 | 极差 |
|---|---|---|---|
| **after**（改动）| 61.12, 60.91 | **61.02** | **0.21** |
| **before**（基线）| 63.22, 64.67, 64.32 | **64.07** | 1.45 |

**两个分布完全不重叠**（after ∈ [60.91, 61.12]，before ∈ [63.22, 64.67]）：

```
匹配对 r1        : after 61.12 vs before 63.22  ->  -2.10 ms  (-3.3%)
全部已完成轮次均值: 61.02 vs 64.07             ->  -3.06 ms  (-4.8%)
decode 吞吐      : 1024.7 vs 977.6 tok/s       ->  +4.8%
```

> 有一轮 `after` 因租户把卡打到 100% 而**没在 600 s 内跑完**（无结果，故不计入）；
> 它的配对 `before` 那一轮是三条 before 里最慢的（64.67），说明那段时间确实在抢卡。
> 剔除该轮不偏向任何一方——after 的两轮极差只有 0.21 ms。

## 交叉验证：算子条数确实降了

msprof，2 卡，97 步 × 48 层：

| op | before | after |
|---|---|---|
| `aclnnHistc_Histogram` | 1.00 / 层 | **0.00** ✓ |
| `aclnnInplaceCopy_Cast` | 5.08 / 层 | **3.08** ✓ |
| **kernel 总数 / 层 / 卡** | **40.41** | **37.41（−7.4%）** |

## 结论与下一步

- **3 个 launch/层 = −3 ms/步（−4.8%）**，收益比预期的大：说明这些空档里
  `histc` 那一条（带 `to(float32)`/`to(int32)` 两次全量 Cast）确实是大头；
- 换算到 4 卡（decode T=64/卡、步时间 ~72 ms）：**绝对量应该同样是 ~3 ms/步（≈4%）**，
  因为省下的是 host 侧每次 launch 的空档，与 token 数无关；
- 剩余候选（按性价比，都不大）：
  `own_counts` 的 copy 改 view（1 个 launch）、`T == capacity` 时跳过 3 次 pad copy
  （只在 T 正好等于档位值时成立，decode 里很少）、combine 的 `Mul`+`Cast` 融合
  （**会改数值**，需先定精度）；
- **大方向**：即使把 MoonEP 的算子全删掉也只能到 ~24 个/层，因为 vLLM 自己的
  attention+router+采样 就有 ~24 个 —— 真正的大头是 **eager 模式下的 vLLM 算子条数**，
  下一步该看能不能开 CUDA graph / 减少 vLLM 侧算子。

## 运维教训（自己踩的）

`nohup` 起的长跑腿被 `kill` 掉时，**它 spawn 出去的 worker 会变成孤儿继续占着卡**
（这次两个 `VLLM::Worker_DP*` 各占 35 GB 卡住了 2、3 号卡，导致后面 6 条腿全部
`aclrtMallocPhysical(...) failed with 207001`）。收尾必须按 **明确的 PID** 清掉孤儿
（`kill -9`，SIGTERM 对它们无效），`pkill -f <pattern>` 还会把自己的 shell 一起打死。

---

# 附录五（2026-09-18）：四个算子在 prefill / decode 的负载归属

数据源：`/home/tmp/msprof/msprof_slot`（4 卡 DP4×EP4、N_SEQ=64、PROMPT_LEN=512、GEN=8、MBT=2048）。
按**每层的 token 数 T** 分层——取自 `aclnnMoeGatingTopK` 的 Input Shape（`T,128`），
这是 prefill/decode 最干净的判据。

`Duration` 是 profiler 的算子墙钟（**含跨 rank 栅栏的等待**），
`issue` = `aiv_vec + aiv_scalar + aiv_mte2 + aiv_mte3`，即**真正发射的指令时间**。

## 逐层（µs，4 卡平均）

| | planning | dispatch | combine | prefetch | 四件套 | issue 占比 |
|---|---|---|---|---|---|---|
| **PREFILL**（T≥254，576 层）Duration | **436.7** | 230.3 | 395.2 | 96.6 | 1158.8 | |
| ↳ issue | 55.0 | **217.4** | **245.6** | 82.1 | 600.1 | **52%** |
| ↳ issue/Duration | 13% | **94%** | **62%** | 85% | | |
| **DECODE**（T=16，本 run 主区间，960 层）Duration | **441.3** | 52.3 | 55.9 | 106.2 | 655.6 | |
| ↳ issue | 35.6 | 12.4 | 8.7 | **106.4** | 163.1 | **25%** |
| ↳ issue/Duration | **8%** | 24% | 16% | **100%** | | |
| DECODE（T=1，10272 层）Duration | 254.8 | 38.1 | 82.1 | 61.7 | 436.7 | |
| ↳ issue | 25.6 | 7.3 | 7.5 | 59.2 | 99.6 | 23% |

## 折成每步（48 层）

| | planning | dispatch | combine | prefetch | MoE 合计 | 其中真活 |
|---|---|---|---|---|---|---|
| PREFILL（T=2048） | 21.0 | 11.1 | 19.0 | 4.6 | **55.6 ms** | 28.8 ms |
| DECODE（T=16） | 21.2 | 2.5 | 2.7 | 5.1 | **31.5 ms** | **7.8 ms** |

## 结论：bottleneck 分两层，答案不一样

**1）按"占用关键路径的时间"（Duration）——两边都是 `planning`**

decode 441 µs/层（占 MoE 四件套的 67%）、prefill 437 µs/层（38%）。
**但它 87–92% 是等待**：真实发射只有 35.6 µs（decode）/ 55.0 µs（prefill）。
那 400 µs 是**其它 rank 还没到**——见附录三：图层间的偏斜在 planning 的跨卡栅栏处被吸收。
⇒ 让 planning 更快解决不了它（已验证：砍 13% 设备时间、墙钟零变化）。

**2）按"真正把设备干满"（issue）——两条路线完全不同**

| | 真瓶颈 | 依据 |
|---|---|---|
| **PREFILL** | **dispatch + combine** | issue 217 / 246 µs/层，分别 94% / 62% 忙；这是真正的数据搬运 |
| **DECODE** | **prefetch** | issue **106.4 µs/层、100% 忙**——四个算子里唯一"没有任何余量"的；其余三个加起来才 56.7 µs |

**3）注意这跟"谁在关键路径上"是两回事**

prefetch 在 decode 里 100% 满负荷（106 µs/层），但此前实测**砍掉 80% 的调用步时间不变**
⇒ 它虽然满忙，却**可以被重叠掉**（它的等待是被别的等待盖住的）。
反过来 dispatch/combine 在 decode 里只忙 12/9 µs，却是"发射即阻塞"的串行环节。

**4）总量口径**

decode 时 MoE 四件套占 **31.5 ms/步**的 Duration，其中真活只有 **7.8 ms** ⇒
**23.7 ms/步（75%）是等待**（跨 rank 栅栏 + 等 host 发射）。
prefill 时 55.6 ms/步里真活 28.8 ms（52%）。

⇒ 想把 decode 的步时间打下来，**继续压 planning 的 Duration 是无效的**（它本来就是"别人的时间"）；
有效的是两条：(a) 消掉那 23.7 ms 的等待（栅栏结构 / host 发射），(b) 把 prefetch 那 5.1 ms 真活重叠掉。

## 口径提醒

- `Duration` 含跨 rank 栅栏等待，**不能当算力成本**（附录三已证）；
- 这是 profiled 数据，msprof 会放大 host 空档；
- T=1 那 10272 层是低批区间（214 步），不代表正常 decode，故单列；
- 4 卡、每卡 16 条序列，生产配置（N_SEQ=256 / 每卡 64）下各值都会更大。

---

# 附录六（2026-09-18）：全层负载归属 —— 含 attention 与 GEMM

## 口径

- 数据：`/home/tmp/msprof/msprof_slot`（4 卡 DP4×EP4、N_SEQ=64、PROMPT_LEN=512、GEN=8、MBT=2048）；
- 按**每层 token 数 T** 分层（取自 `MoeGatingTopK` 的 Input Shape）；只有 **T=16 那一档是正常 decode**
  （attention : MoE 层 = **1:1**，GroupedMatmul = 2/层，combine = 1/层 —— 结构自洽）。
  占样本多数的 T=1 档（attention:MoE = 0.14）是低批阶段，不代表正常 decode，故单列。
- **`issue`** = `aiv_vec + aiv_scalar + aiv_mte2 + aiv_mte3 + aic_mac + aic_scalar + aic_mte1/2/3 + aic_fixpipe`
  —— 各条流水线**真正发射指令**的时间之和（cube 算子只有 `aic_*`，混合算子两者都有）。
  `busy = max(aiv_time, aicore_time)`。**`issue% > 100` 表示多条流水线并行发射（流水做得好），不是超载**；
  `issue% ≈ 10%` 表示墙钟里 90% 的时间没有任何流水线在发射 ⇒ **纯等待**。

## 每层每卡（µs）

| | PREFILL (T≥254, 576 层) | | | DECODE (T=16, 960 层) | | |
|---|---|---|---|---|---|---|
| | Duration | issue | 占该层 | Duration | issue | 占该层 |
| **① MoE 四件套** | **1589.6** | 885.1 (56%) | **56%** | **1112.8** | 489.7 (44%) | **82%** |
| ↳ planning | 436.7 | 55.0 (**13%**) | 15% | 441.3 | 35.6 (**8%**) | **33%** |
| ↳ dispatch | 230.3 | 217.4 (94%) | 8% | 52.3 | 12.4 (24%) | 4% |
| ↳ combine | 395.2 | 245.6 (62%) | 14% | 55.9 | 8.7 (16%) | 4% |
| ↳ prefetch | 193.1 | 164.2 (85%) | 7% | 212.3 | **212.8 (100%)** | 16% |
| ↳ barrier ×4 | 334.2 | 202.8 (61%) | 12% | 351.1 | 220.2 (63%) | 26% |
| **② MoE GEMM** | 422.3 | 759.7 (180%) | 15% | 92.9 | 149.6 (161%) | 7% |
| **③ attention**（FIA 等）| 17.0 | 29.3 (172%) | 1% | 14.8 | 27.4 (185%) | 1% |
| **④ 投影/归一化/rope** | 240.9 | 426.8 (177%) | 8% | 77.8 | 105.2 (135%) | 6% |
| ⑤ router/采样 | 33.0 | 26.1 (79%) | 1% | 17.5 | 14.2 (81%) | 1% |
| ⑥ 拷贝/搬运 | 467.1 | 570.4 (122%) | 16% | 35.8 | 28.9 (81%) | 3% |
| ⑧ 其它 | 69.5 | 123.5 (178%) | 2% | 5.7 | 5.1 (89%) | 0% |
| **全部** | **2839.3** | 2820.8 (99%) | 100% | **1357.3** | 820.2 (60%) | 100% |

## 结论

**1）decode 时，MoonEP 的通信/规划路径吃掉整层的 82%。**

```
decode 每层 1357 µs  =  MoE四件套 1113 (82%)  +  MoE GEMM 93 (7%)  +  其余 151 (11%)
```
MoE GEMM 只有 93 µs、attention 的 FIA 只有 15 µs —— **通信路径是计算路径的 12 倍**。

**2）但这 82% 里大部分是"等"，不是"干"。**

- decode 全层 issue 只有 60%；MoE 四件套只有 **44%**；
- 三个最大项里，**planning（441 µs，8% 忙）和 barrier（351 µs，63% 忙）都是等待** ——
  等其它 rank 到达栅栏（barrier 的 63% 是标量自旋，也是空转）；
- **真正满负荷的只有 prefetch：212.8 µs、100% busy**。

**3）真瓶颈：prefill 是 dispatch+combine，decode 是 prefetch。**

| | 生产性工作量（issue）| 说明 |
|---|---|---|
| PREFILL | **dispatch 217 + combine 246** = 463 µs/层 | 94% / 62% 忙，真正的跨卡搬运 |
| DECODE | **prefetch 212.8** µs/层（100% 忙）| MoE 路径里唯一没有余量的；是 MoE GEMM（149.6）的 1.4 倍 |

**4）折成每步（×48）：**

| | MoE四件套 | 其中 planning | prefetch | barrier | 全层 |
|---|---|---|---|---|---|
| PREFILL | 76.3 ms | 21.0 | 9.3 | 16.0 | 136.3 ms |
| DECODE | 53.4 ms | 21.2 | 10.2 | 16.9 | 65.2 ms |

⇒ decode 每步 **16.9 ms 花在 4×48 次跨卡栅栏的自旋上**，21.2 ms 花在 planning 的等待上。
这两项合计 38 ms，占整层预算的 58%，而它们**都是"等最慢的 rank"**。

## 一个需要重新验证的旧结论

文档前面写过"prefetch 不在关键路径（砍掉 80% 调用 step 无变化）"。
但新数据显示 **prefetch 每层 212.8 µs、issue% = 100%**，即它**全程满负荷**，
在单流串行的前提下这就是 10.2 ms/步的实打实工作量。
两者矛盾，**旧的"不在关键路径"结论需要用当下的 slot 模式重测**（当时那两次实验分别是
slot-cache 与 early-prefetch 的上下文，可能不可比）。

## ⚠️ 附录六的重要修正：这个 run 的 context 只有 81 个 token

**attention 只占 1% 不是统计错误，是负载太短。**

benchmark 自己的账本（`/tmp/bench_dp2ep2_msprof_slot_g8_b256.json`）：
`prefill_tokens=4676`、`n_seq=64` ⇒ **每个 prompt 73 个 token**（GSM8K 题面本来就 ~73 token，
`PROMPT_LEN=512` 只是个上限，根本没截到）；再加 `GEN=8` ⇒ **decode 全程 context ≤ 81 token**。
FIA 的 shape 也对上：query `16,32,128`、KV 是分页 cache、写入 token 数 16 量级。

**KV 读取随 context 线性增长，而 MoE 路径不随 context 变**（它只跟 batch×K 有关）。
Qwen3-30B-A3B 每 token 每层 KV = 2×4(kv heads)×128×2B = **2048 B**：

| context | KV 读/层 | 推算 attention | vs MoE四件套 1113 µs |
|---|---|---|---|
| **81**（本 run）| 2.7 MB | **15 µs**（实测 14.8 ✓）| **~0%** |
| 512 | 16.8 MB | 95 µs | 3% |
| 2048 | 67.1 MB | 379 µs | 12% |
| 8192 | 268 MB | 1517 µs | 48% |
| 32768 | 1.07 GB | 6066 µs | 193% |

（按实测点 2.7 MB → 14.8 µs 定标，即 ~177 GB/s 有效带宽；线性外推是粗略的，
小尺寸时是开销受限、大尺寸时应好于线性，所以这里给的是上界量级。）

**⇒ 结论的适用范围必须限定**：

1. 附录六的"MoE 通信占 82%"**只对短 context（≤ 数百 token）成立**；
2. context 到 2K 时 attention 已经不可忽略，8K 时与整条 MoE 路径同量级，32K 时反过来碾压；
3. 因此 **"MoonEP 比 MC2 慢 29%"这个结论也要按 context 打折** —— MoE 路径的开销是
   与 context 无关的常数，context 越长它在总时间里的占比越小；
4. **本 run 的 `gen=8` 也偏小**：整个 profiling 只覆盖了 512 个 decode token（8×64），
   ITL 只有 ~100 ms 量级，样本很少。

**下一步该做**：用**真实长度的 context**（比如 2K / 8K）重跑一遍这个负载表，
才能得到生产形态下的瓶颈归属。当前这张表只能回答"短 context 下谁是大头"。

---

# 附录七（2026-09-18）：长 context 重测 —— 并修正附录六的口径错误

## 为什么要重测

附录六的 trace 里 attention 只占 1%，经查是**负载太短**：benchmark 的账本
`prefill_tokens=4676 / n_seq=64` ⇒ 每个 prompt 只有 **73 token**（`PROMPT_LEN=512` 只是上限，
GSM8K 题面根本没到），加 `GEN=8` ⇒ **decode 全程 context ≤ 81**。

**重测方法**：`PROMPT_FILE` 不设时 `bench_dp.py` 会合成**恰好 `PROMPT_LEN` token** 的 prompt，
于是只改 context、其它全部不变（DP2×EP2、`N_SEQ=32`=16 序列/卡、`GEN=32`）：

| leg | prompt | 实测 prefill_tokens/seq | decode context |
|---|---|---|---|
| ctx62（对照）| `PLEN=64` | 62.0 | 62 → 94 |
| ctx2046 | `PLEN=2048` | 2046.0 | 2046 → 2078 |

## ⚠️ 修正：附录六用的是**均值**，被少数层严重污染

同一份 trace 里 **均值 vs 中位数** 差一个量级 —— 因为**少数层**撞上很大的跨 rank 偏斜，
把均值拉飞了。**两个口径回答的是不同问题，都要看**：
- **均值**回答"这个算子占了整步多少时间"（步时间 = 48 层之和 ⇒ 要用均值）；
- **中位数**回答"典型的一层长什么样"。

| 项（decode, T=16）| 附录六（均值）| 中位数（ctx62 / ctx2046）| 说明 |
|---|---|---|---|
| planning | 441 µs，issue **8%** | **47.2 / 55.3 µs，issue 57% / 54%** | 441 是少数层的偏斜等待 |
| prefetch | 212.8 µs，**100% busy** | **6.3 / 6.4 µs** | 多数层几乎不搬，少数层很贵 |
| barrier（4 次/层）| 351 µs | **241.6 / 230.7 µs** | **两个口径都最大** |
| attention | 14.8 µs | 14.2 / **72.2** µs | 均值≈中位数，无异常层 |
| MoE 四件套 | 82% | 66% / 71% | |

**⇒ 附录六里"planning 87–92% 在等待""prefetch 是 decode 的真瓶颈"两条要撤回**：
那是均值口径的假象。中位数显示 planning 的典型耗时 **47–55 µs、一半以上在干活**
（与隔离测量的 58 µs 吻合 ✓），prefetch 的典型耗时只有 **6.3 µs**。

## 干净的 context 对比（中位数，每层每卡 µs）

| | ctx≈62 | ctx≈2046 | 倍数 | 占比62 | 占比2046 |
|---|---|---|---|---|---|
| **① MoE 四件套** | 642.1 | 673.8 | 1.0× | **71%** | **66%** |
| ↳ barrier ×4 | **241.6** | **230.7** | 1.0× | **27%** | **22%** |
| ↳ planning | 47.2 | 55.3 | 1.2× | 5% | 5% |
| ↳ dispatch | 18.7 | 20.9 | 1.1× | 2% | 2% |
| ↳ combine | 9.5 | 9.9 | 1.0× | 1% | 1% |
| ↳ prefetch | 6.3 | 6.4 | 1.0× | 1% | 1% |
| **② MoE GEMM** | 98.1 | 101.5 | 1.0× | 11% | 10% |
| **③ attention** | 14.2 | **72.2** | **5.1×** | 2% | **7%** |
| ④ 投影/归一化/rope | 68.8 | 70.8 | 1.0× | 8% | 7% |
| ⑥ 拷贝/搬运 | 30.7 | 33.7 | 1.1× | 3% | 3% |
| ⑤ router/采样 | 7.5 | 8.9 | 1.2× | 1% | 1% |
| **全部** | **898.9** | **1028.6** | 1.1× | 100% | 100% |

### 结论

1. **attention 确实是随 context 长的，但远慢于线性**：context 涨 33 倍（62→2046），
   单层 attention 只涨 **5.1 倍**（14.2 → 72.2 µs），占比 2% → 7%。
   附录六里那张"线性外推到 8K 就 48%"的表**过于悲观**，应以本次实测为准：
   小 context 时 attention 是开销受限，大 context 才逐步转向带宽受限。
2. **MoE 路径完全不随 context 变**（1.0×）✓ —— 它只跟 batch×K 有关。
3. **>60 岁的头号单项是 4 次跨 rank barrier（220–240 µs/层，22–27%）**，
   两个口径都最大，而且它由 prefetch 产生（2 次 prefetch × 2 个栅栏）。
   这与"去掉栅栏反而慢 +2.9 ms"并不矛盾：**栅栏就是系统吸收 rank 偏斜的地方**，
   230 µs 里大半是"等最慢的 rank"。
4. **prefill 侧（ctx2046，T≈8192/层）**：MoE 四件套 3745 µs/层（37%），其中
   **dispatch 1634 + combine 1755 = 3389 µs（33%）** 是绝对大头；MoE GEMM 2165（21%）；
   拷贝/搬运 2149（21%）；attention 650（6%）。**prefill 的瓶颈是实打实的跨卡搬运**。

---

# 附录八（2026-09-18）：context = 20460 —— attention ∝ √context，但 MoE 仍是 75%

## 配置与可行性

目标：把 context 从 2046 再拉大 10 倍到 **20460**，MC2 与 slot 各跑一遍，4 卡。

| leg | 结果 |
|---|---|
| **slot @ 20460** | ✅ 完成（4 卡 DP4、每卡 4 序列、`GEN=16`、`MBT=8192`、`MAXLEN=22528`）；`n_finished=16`、`n_preempted=0`、KV cache 366,720 token（**没有串行准入**）|
| **MC2 @ 20460** | ❌ **两次都挂死**：engine 初始化完成后 10 分钟无进展（worker 30 s 只走 1 个 cpu tick、日志冻结、每卡占 27 GB）。两次分别在 `N_SEQ=32`（8 序列/卡）和 `N_SEQ=16`（4 序列/卡）下发生 |
| slot @ 20460（第一次，`N_SEQ=32`）| ❌ HCCL 初始化失败：`hcclCommInitRootInfoConfig ... error code 4` / `RootInfoDetect failed` |

⇒ **本环境上 MC2 在 20460 context 跑不完**，所以这一档**给不出 MC2 vs slot 的对比**。
（2 卡试过：KV cache 只有 5.4 GiB，而 8 序列 × 20460 需要 15 GiB ⇒ vLLM 只准入 ~2 条序列，
测出来的是串行化后的假 batch，已弃用。）

> 顺带记一个 harness bug：`bench_dp.py` 写出的结果文件名**恒为 `bench_dp2ep2_*`**，
> 不反映实际 DP（本次 DP=4 也写成 dp2ep2）。判断配置要看 JSON 里的 `dp`/`ep` 字段。

## slot 每层负载（decode，context 20460，每卡 4 序列）

中位数，每层每卡 µs：

| | Duration | issue | issue% | 占比 |
|---|---|---|---|---|
| **① MoE 四件套** | **1272.2** | 321.1 | 25% | **75%** |
| ↳ planning | 630.2 | 31.5 | **5%** | **37%** |
| ↳ barrier ×4 | 327.3 | 116.8 | 36% | 19% |
| ↳ prefetch | 216.2 | **4.3** | **2%** | 13% |
| ↳ combine | 74.9 | 6.2 | 8% | 4% |
| ↳ dispatch | 47.6 | 7.7 | 16% | 3% |
| **② MoE GEMM** | 63.5 | 93.0 | 146% | 4% |
| **③ attention** | **239.4** | 645.2 | 270% | **14%** |
| ④ 投影/归一化/rope | 64.5 | 84.1 | 130% | 4% |
| **全部** | **1690.4** | 1192.2 | 71% | 100% |

## attention 的 context 曲线（每次调用，µs）

| context | 62 | 2046 | **20460** |
|---|---|---|---|
| FIA Duration p50 | 14.2 | 68.8 | **236.2** |
| 相对 ctx62 | 1× | 4.8× | **16.6×** |
| context 倍数 | 1× | 33× | 330× |
| 拟合指数 | — | 0.45 | **0.48** |

**⇒ attention ≈ O(√context)，远不是线性。** 分布极紧（p10 235.6 / p90 236.9，跨卡仅差 3.0 µs）
⇒ 是稳定的算力/带宽行为，不是抖动。
（附录六那张线性外推表因此彻底作废；附录七已经改过一次，这次是第三次修正，以实测为准。）

## 层内占比随 context 的变化（decode）

| | ctx62 | ctx2046 | ctx20460 |
|---|---|---|---|
| **MoE 四件套** | 71% | 66% | **75%** |
| **attention** | 2% | 7% | **14%** |
| MoE GEMM | 11% | 10% | 4% |
| **全部 µs/层** | 899 | 1029 | **1690** |

**⇒ context 拉大 330 倍，MoE 通信路径仍然是 75%。** attention 从 2% 涨到 14%，
按 √context 外推要到 ~64K context 才追平 MoE 路径 —— **在 20K 这一档，MoonEP 的通信路径仍是绝对大头**。

## 机制：MoE 路径随 context 增长的是"等待"，不是工作量

- planning 中位数 **630 µs，issue 只有 31.5（5%）**
- prefetch 中位数 **216 µs，issue 只有 4.3（2%）**
- barrier 327 µs，issue 117（36%）
- 三者 issue 合计 153 µs，只占它们 Duration（1173 µs）的 **13%**

跨卡分析（同一调用序号，end skew 的常数部分 = 卡间时钟基准差）：

| ctx20460 | start skew | end skew | 真实 start skew | Duration 跨卡差 |
|---|---|---|---|---|
| barrier | 297.8 | 296.5 | **1.2** | 95.2 |
| **planning** | 435.5 | **296.5** | **139.0** | **578.6** |
| combine | 300.0 | 296.8 | 3.2 | 91.5 |
| attention | 378.8 | — | — | **3.0** |

⇒ **4 卡在 planning 处重新对齐（end skew = 时钟常数），但开始时刻差 139 µs、Duration 差 579 µs**
—— planning 又是那个"吸收 rank 间偏斜"的位置。attention 的跨卡差只有 3 µs（完全确定性）。

**⇒ 长 context 下 MoE 路径变慢，是因为它吸收的偏斜变大了，不是它自己的工作量变大了。**
这跟附录三/七的结论一致：planning 与 prefetch 的 Duration 是"别人的时间"。

> 口径提醒：本 leg 每卡 4 序列（T=4），ctx2046 那轮是 8 序列（T=8），
> 所以 **MoE 部分不可直接比**（batch 不同）；attention 的 per-call 值可直接比。

---

# 附录九（2026-09-18）：PREFILL 的负载 —— 大头是 attention，不是 MoE

附录八只分析了 decode。prefill 完全是另一幅图。

## 方法：固定 chunk 大小，只改累积 context

两个 run 都开了 chunked prefill（`MBT=8192`），因此**每层处理的 token 数都是 8192**
（`MoeGatingTopK` 的 Input Shape 第一段 = 8192）。两个 run 的区别只在**已累积的 context**：

| run | prompt | prefill chunk | T=8192 的层样本 |
|---|---|---|---|
| ctx2046 | 2046 | 8192（多个序列打包）| 384（每卡 96）|
| ctx20460 | 20460 | 8192（单条长序列的第 1..3 段）| 2304（每卡 576）|

⇒ **同一 chunk 大小、不同 KV 长度**，正好隔离出"context 对 prefill 的影响"。

## 每层每卡（中位数，µs）

| | ctx2046 | ctx20460 | 倍数 | 占比46 | 占比20k |
|---|---|---|---|---|---|
| **① MoE 四件套** | 3817.1 | 3873.6 | **1.01×** | 34% | 26% |
| ↳ dispatch | 1652.4 | 1560.5 | 0.94× | 15% | 11% |
| ↳ combine | 1823.7 | 1773.4 | 0.97× | 16% | 12% |
| ↳ prefetch | 112.9 | 218.5 | 1.94× | 1% | 1% |
| ↳ planning | 96.7 | 120.4 | 1.24× | 1% | 1% |
| ↳ barrier ×4 | 123.4 | 115.1 | 0.93× | 1% | 1% |
| **② MoE GEMM** | 2199.6 | 2109.1 | 0.96× | 20% | 14% |
| **③ attention** | **740.8** | **4479.3** | **6.05×** | **7%** | **31%** |
| ④ 投影/归一化/rope | 1394.3 | 1395.3 | 1.00× | 13% | 10% |
| ⑥ 拷贝/搬运 | 2220.6 | 2207.1 | 0.99× | 20% | 15% |
| ⑧ 其它 | 692.7 | 700.6 | 1.01× | 6% | 5% |
| **全部** | **11091.2** | **14625.6** | **1.32×** | 100% | 100% |

## 结论

**1）prefill 的最大单项是 attention，不是 MoE 通信。**

```
ctx20460 prefill 每层 14626 µs
  = attention 4479 (31%)  +  MoE四件套 3874 (26%)  +  拷贝 2207 (15%)
  + MoE GEMM 2109 (14%)   +  投影/rope 1395 (10%)  +  其它 700 (5%)
```

⇒ **和 decode 正好相反**（decode 里 MoE 四件套 75%、attention 14%）。

**2）attention 在 prefill 里随 context 是 ~线性增长（6.05× / 10×），不是 decode 里的 √context。**
理由很直接：prefill 的 attention 计算量 ∝ `q_len × kv_len`，而 `kv_len` 就等于已累积的
context；decode 里 `q_len=1` 且受带宽/开销摊薄，所以呈 √。

**3）MoE 路径在 prefill 里完全不随 context 变（1.01×）** ✓ —— 它只跟 chunk 大小（token 数）有关。

**4）prefill 里 MoE 的主要成本是 dispatch + combine（合计 3334 µs = 23%），
两个都是真搬运**（此前测过 issue 94% / 62% 忙）；planning/barrier/prefetch 加起来只有 3%。

**5）拷贝/搬运类（Cast/TensorMove/Contiguous/…）占 15%**，是 prefill 里第三大的组。

## 折成每步（×48）与 TTFT 自洽性

| | ctx2046 | ctx20460 |
|---|---|---|
| 每步（8192 token/chunk）| 532 ms | **702 ms** |
| ↳ attention | 36 ms (7%) | **215 ms (31%)** |
| ↳ MoE 四件套 | 183 ms | 186 ms |

ctx20460 的 prompt 20460 token ÷ 8192 = 3 个 chunk，每卡 4 序列 ⇒ 约 10 个 prefill step
× 702 ms ≈ **7.0 s**；而实测 `ttft_ms` p50 = **4.64 s**（请求在不同 step 完成，中位数约 7 步）
⇒ **与逐层账自洽** ✓

---

# 附录十（2026-09-19）：MC2 在 8 卡上跑不通（EP8 链路协议无实现）

## 现象

8 卡（`DP8 × TP1 = EP8`）跑 **MC2** 路线（`enable_moonep=False`，走原生
`MoeDistributeDispatch`/`Combine`）：8 个 rank 全部正常加载模型（9.65 GB/卡），
**第一个 forward 即失败**，之后引擎挂住不再推进。

```
RuntimeError: npu_moe_distribute_dispatch_v2:.../MoeDistributeDispatchV2KernelOpApi.cpp:194
  NPU function error: call aclnnMoeDistributeDispatchV4 failed, error code is 561000
```

CANN plog（`/root/ascend/log/debug/plog/plog-<pid>_*.log`）给出真正的根因：

```
[mc2_context.cpp:160][GetHcclCommLink] OpName:[Mc2Context]
    No matching communication protocol found in HCCL links protocol is 6
[mc2_context.cpp:333][CreatMc2Context] Get HCCL communication resource failed
[mc2_context.cpp:536][GetMc2ContextTensor] errno[561000]
```

⇒ **MC2 在建立"通算融合上下文"时，8 卡组的 HCCL 链路协议（protocol 6）没有对应实现**。
失败发生在 CANN 算子内部的 context 创建阶段，**早于任何模型代码**。

## 对照实验（关键）

| 配置 | 结果 |
|---|---|
| MC2, **EP8**（卡 0-7）| ❌ 确定性失败，**复现 2 次**（`561000`）|
| MC2, **EP4**（卡 0-3）| ✅ **EXIT=0，跑通** |
| MC2, **EP4**（卡 4-7）| ✅ 早前也跑通（`/home/tmp/msprof/msprof_mc2` 那次）|

⇒ **变量是 rank 数（4 → 8），不是卡的集合**。

## 结论

- **MC2 在 EP8 上不可用，EP4 正常**；
- 这是**算子/框架侧的限制**，不是 MoonEP 适配引入的（MC2 路径不经过我们的代码）；
- 推测与 8 卡全互联使用 UB（protocol 6）而 MC2 context 只实现了部分协议有关。

## 下一步

1. **交叉验证**：用**原版 vllm-ascend**（非本分支）跑一次 8 卡 MC2。我们的 bench 虽然
   `enable_moonep=False` 走原生 dispatcher，但分支本身有改动，交叉后才能断言这是通用问题；
2. 试 `additional_config={"enable_mc2_hierarchy_comm": true}`（vllm-ascend 有此开关，
   默认 false，`ascend_config.py:304`），看是否是更大 EP 的路径；
3. 报给算子/框架侧：`mc2_context.cpp:GetHcclCommLink` 对 protocol 6 无实现。

## 运维教训（再次踩到）

**引擎在启动后失败/挂住时，`run_multi.sh` 退出但 spawn 出的 worker/EngineCore 会变成孤儿
继续占卡**（本次一度留下 16 个进程、每卡 16 GB，填满全部 8 张卡，还让我误判成"租户抢卡"）。
收尾必须按 PID 清（SIGTERM 无效，需 `kill -9`）；判断归属用 `/proc/<pid>/environ` 里的
`MOONEP_DP_FULL_VISIBLE` + `cwd=/home/tmp`。
