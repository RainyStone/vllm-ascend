# MoonEP 图模式（ACL Graph）可行性分析

**日期**：2026-09-23 · **配置**：Ascend 950，DP=2 / EP=2，Qwen3-30B-A3B，slot 模式

> **术语**：MoonEP 上游路线图里写的是 CUDA Graph（NVIDIA）。在本机（Ascend）
> 对应物是 **ACL Graph**，vllm-ascend 侧实现在 `vllm_ascend/compilation/acl_graph.py`
> 与 `vllm_ascend/worker/v2/aclgraph_utils.py`。

---

## 1. 为什么 MoonEP 这张牌值得打

单腿下 MoonEP host 路径 **0.35 ms/层**（双腿 1.27 ms/层），其中**没有任何单项超过
0.11 ms** —— 全是 10~50 μs 的小操作堆出来的。数一下每层的 kernel launch：

```
planning 1 + dispatch 1 + prefetch 2(+4 barriers) + combine 1 + glue 4 ≈ 13~17 次
→ 每步 650~800 次
```

按每次 launch 的 Python + pybind + C++ + ACL 开销 40~80 μs 算，
**光 launch 就是 26~64 ms/步**。而**删代码救不了它** —— 每一笔都对应一次真实的下发。
**唯一的结构性手段是把这些下发压成一次图重放。**

---

## 2. MoonEP 的静态性核查（逐项实测，非推断）

| 要素 | 是否静态 | 依据 |
|---|---|---|
| **dispatch 行数 `NvS = S*K`** | ✅ 与路由倾斜无关 | 均衡器保证每 rank 恒为 S×K 行（已实测：99:1 偏斜下两卡每层都是 131,072 行） |
| `dst[N]` / `cu_seqlens[E+B]` / `dup_*` | ✅ 形状只依赖 `S` | `allocate_planning_outputs` 用 `ctx['S']`/`ctx['N']` |
| 每层 torch 分配 | ✅ **零** | `_pads_for`→`_MONEP_RUNTIME._pads`、`_scratch_for`→`sc`、`_geometry`→`_GEO_CACHE`、`_select_buffer_for_step` 纯 dict 查找 |
| `create_sym_tensor_from_source` | ✅ 已缓存 | 今天新增（`(source_va, shape, dtype, device, allow_logical_extent)` 键 + `_destroyed` 存活检查） |
| plan 输出张量 | ✅ 已缓存 | `reuse_outputs=True`（`MOONEP_REUSE_OUT` 默认 1）走 `_reused_plan_outputs` |
| 每层 host 设备同步 | ✅ **零** | 逐个核过守卫：`_reconcile_async_ranks`（async 分支）、`_ENV_PACE_SYNC`、`MOONEP_PF_TIMING` 全默认关 |
| Buffer / capacity rung 选择 | ✅ 固定 `T` 下确定 | `_select_buffer_for_step(tokens)` → `next(c for c in caps if c >= need)`，无分配 |

**结论：在固定容量桶（固定 `tokens`）下，MoonEP 的数据面与 Python 路径都是确定性的 ——
满足"图重放要求输入形状一致"这个前提。**

---

## 3. 现状：两道显式禁止 + 一个启动参数

```python
# moonep_comm.py:911  TokenDispatcherWithMoonEP.token_dispatch
if _EXTRA_CTX.capturing:
    raise RuntimeError(
        "MoonEP dispatch cannot run inside captured CUDA graphs; "
        "run with --enforce-eager (or disable cudagraph) when enable_moonep is on.")

# moonep_balance.py:682  run_slot_forward
if capturing:
    raise RuntimeError("MoonEP slot mode cannot run inside captured graphs.")
```

**两道都是无条件 raise，且注释直接指向 `--enforce-eager`** —— 从措辞看是**保守禁止**
（"先别在图里跑"），不是记录了某个具体的数据依赖。**但这是推断，必须用实验证伪。**

`/home/tmp/serve_moonep.sh` 当前就带 `--enforce-eager`。

---

## 4. 障碍面盘点

### 4.1 已解决（今天的工作顺带清掉的）

| 曾经的障碍 | 现状 |
|---|---|
| sym tensor 每层新建 → 地址不稳定 | ✅ 已缓存，地址整轮不变 |
| plan 输出每层 8 次分配 | ✅ 已走 `_reused_plan_outputs` 缓存 |
| 每层全设备同步 | ✅ 本来就没有（之前的判断是错的） |

### 4.2 真障碍（明确，需改）

| 障碍 | 位置 | 改法 |
|---|---|---|
| 两道 `raise` | `moonep_comm.py:911`、`moonep_balance.py:682` | 换成开关控制，验证后再默认放开 |
| `--enforce-eager` | `serve_moonep.sh` | 去掉 |
| `_ENV_PACE_SYNC` / `MOONEP_PF_TIMING` 的 `npu.synchronize()` | 默认关，但**捕获期必须确保不被打开** | 已有默认关；加 assert |

### 4.3 未知（必须实验回答，是本次分析的核心不确定性）

| 未知 | 为什么关键 | 怎么验 |
|---|---|---|
| **shmem 内核的 `<<<>>>` 发射能否被 ACL Graph 捕获** | planning/dispatch/combine/prefetch 全是 AscendC 内核，走 Bisheng `<<<>>>` → `aclrtLaunchKernel` | 最小实验：捕获一个只含 `moonep_dispatch_kernel` 的片段 |
| **`aclshmemx_barrier_all_on_stream` 的 host barrier 能否捕获** | 它 `<<<1>>>` 起一个 kernel；ACL Graph 对 host 侧显式 launch 的兼容性未知 | 同上，或开 `PREFETCH_RELAXED` 绕开 |
| **跨 rank 集合操作能否在图上 lockstep 重放** | planning 内含跨卡栅栏；图重放必须所有 rank 同步进入 | vllm-ascend 的 TP/EP 图已经在做这件事，**大概率可用**，但要确认 EP group 是否被图模式覆盖 |
| **slot 别名（`[epn+B]` 权重视图）在图里是否稳定** | GEMM 读的是 `state.alias_w13`（`_packed_alias` 造的视图） | 捕获前后对比 `data_ptr()` |

---

## 5. 局部图模式的三个候选粒度（按风险从低到高）

用户已明确"局部的图模式都可以"，所以不必一次吞下整层。

### 候选 A：只捕获 vLLM 侧 glue（4 次 launch）

```
TensorMove / Cast / Sub / ConcatD   （cu_seqlens → group_list 的转换链）
```
- **收益**：省 4 次 launch/层 ≈ 6~10 ms/步（占 host 的 ~30%）
- **风险**：最低（不碰 shmem，不碰跨卡）
- **但**：这段在 `moonep_comm.py:1042-1049`，是 vLLM 侧代码，与 MoonEP 图模式路线图无关

### 候选 B：捕获"dispatch 之后 → GEMM 之前"这一段（推荐首选）

```
prefetch → (join) → GEMM(w13) → SwiGLU → GEMM(w2)
```
- **收益**：prefetch 2(+4 barrier) + GEMM 2 ≈ 6~8 次 launch/层
- **风险**：中。prefetch 有跨卡 RMA（读对端 home chunk），但**无跨卡栅栏**（开 `RELAXED` 后）；
  GEMM 是纯本地计算
- **前置**：`PREFETCH_RELAXED=1`（今天已验证 barrier 消失）

### 候选 C：捕获整层 MoE（planning → combine）

```
planning → dispatch → prefetch → GEMM → combine
```
- **收益**：13~17 次 launch 全消 ≈ **26~64 ms/步**（这才是路线图的目标）
- **风险**：最高。planning/dispatch/combine 都含**跨卡栅栏**，要求所有 rank 在图重放时 lockstep

---

## 6. 建议的验证路径（每步都是可证伪的小实验）

```
Step 1  消融：把两道 raise 改成受开关控制（默认仍 raise），跑一轮确认行为不变
        —— 排除"改了守卫本身有副作用"

Step 2  最小捕获：写一个独立脚本，捕获【一次 dispatch 内核】然后重放，
        比对输出 dst 与 eager 路径逐位相同
        —— 这一步直接回答 4.3 的前两个"未知"

Step 3  若 Step 2 通过 → 上候选 B（dispatch 之后到 GEMM）
        用 overlap_pct 那套口径 + 输出比对验证

Step 4  若 Step 3 通过 → 试候选 C（整层 MoE）
        重点验跨 rank lockstep：用 R=2 时两卡的 dst 逐位比对
```

**每步都要有的对照**：
- 输出 `torch.equal` 与 eager 路径逐位相同（不是"看起来对"）
- 用 `slot_bench.sh`（单腿、`--offset` 多采样）量 TTFT，单腿噪声 ≈ 33 ms

---

## 7. 收益上限（诚实推算）

| 粒度 | 省下的 launch | 每步 | 占 TTFT |
|---|---:|---:|---:|
| 候选 A（glue） | 4 | 6~10 ms | 0.4~0.6% |
| 候选 B（prefetch+GEMM） | 6~8 | 10~16 ms | 0.6~1.0% |
| 候选 C（整层 MoE） | 13~17 | **26~64 ms** | **1.7~4.1%** |

**注意**：这些是**按 host 路径推算的上限**。今天多次"设备侧优化端到端测不出来"
的教训说明 —— **要真正拿到收益，前提是 host 确实是当前瓶颈**。而实测：

```
单腿 host 0.35 ms/层 = 16.8 ms/步，TTFT 1535 ms 的 1.1%
```

**所以候选 C 的"26~64 ms"里，只有 16.8 ms 是真实存在的 host 成本**，
其余是"多条腿 / 争用下才会出现"的部分。**单腿下候选 C 的天花板是 16.8 ms/步（1.1%）。**

**⇒ 在单腿场景里，图模式不是大杠杆。它的价值在双腿（host 争用把 host 抬到 1.27 ms/层
= 61 ms/步）—— 那时候选 C 的 61 ms/步（4%）才有意义。**

---

## 8. 结论

| 问题 | 答案 |
|---|---|
| MoonEP 数据面适合图捕获吗？ | **适合** —— 逐项核查过，固定容量桶下无每层分配、无每层同步、形状全静态 |
| 当前为什么不能？ | **两道显式 `raise` + `--enforce-eager`，是保守禁止而非已知依赖** |
| 最大的未知 | **shmem 内核与 host barrier 能否被 ACL Graph 捕获**（4.3），必须先做 Step 2 最小实验 |
| 值得做吗？ | **单腿下不值得**（天花板 1.1%）；**双腿下有 4% 的空间** —— 先决定部署形态再投入 |
| 推荐的切入点 | **候选 B**（prefetch+GEMM），风险/收益比最好 |

> **一句话**：静态性这块 MoonEP 是合格的（甚至比路线图预期的还好，因为今天把
> 动态分配和同步都清掉了）；真正的未知在 **AscendC 内核的捕获兼容性**，
> 而那个用一个 20 行的最小实验就能回答。
