# 2026-09-23 — MoonEP host 侧优化 + 训练代码清理

**范围**：`/home/vllm-ascend-moonep`（集成侧）、`third_party/ascend_moonep`（库 Python）、
`/home/ascend-moonep`（内核 + csrc）。

**配置**：slot 模式，DP=2 / EP=2，Qwen3-30B-A3B，T=16384/rank，skew=0.95，
单腿（只跑 slot，不起 ag）。

---

## 1. 结论速览

| 项 | 结果 |
|---|---|
| 当前 slot-only 基线 | **TTFT_p50 ≈ 1535 ms**（4 样本：1544.8 / 1525.5 / 1562.1 / 1526.1） |
| 端到端相对今天起点 | **−2% 左右**，处于噪声边缘 |
| 可归因的 host 侧净收益 | **~8 ms/步**（两个缓存：3.3 + 4.8） |
| 训练代码移除 | **Python 323 行**，已验证不影响推理（TTFT 不变） |
| 真正的瓶颈 | **每层 13~17 次 kernel launch**（host 侧），以及 host 多进程争用 |

---

## 2. host 侧优化（已落地并验证）

### 2.1 单腿 host 路径的分段（优化后）

```
buffer.dispatch()      0.11 ms/层   其中 planning op 0.05 + dispatch op 0.02 + get_stream/bnd_copy 0.04
disp_post              0.05         w_buf.copy_() 一次 D2D 拷贝的发射
combine                0.10         comb_pre 0.05 + comb_buf 0.05
prefetch               0.07         pf7_opcall 0.02 + 其余
unquant_apply_mlp      0.05         GEMM 发射
──────────────────────────────────
合计                  ~0.38 ms/层  = 16.8 ms/步 ≈ TTFT 的 1.1%
```

**没有任何单项 > 0.11 ms。** 全是 10~50 μs 级的小操作，且每一笔都对应一次真实的
kernel launch 或 pybind 调用 —— **删不掉，除非不发射。**

### 2.2 修掉的两个"纯浪费"

| 改动 | 位置 | 效应 |
|---|---|---|
| `aclrtGetDeviceInfo(AICORE_CORE_NUM)` 改为 static 缓存 | `csrc/ops/prefetch.cpp` | 实测 **34.8 μs/次**，每层 2 次 → **3.3 ms/步** |
| `create_sym_tensor_from_source` 结果缓存 | `buffer_c.py` | `pf5_home` 0.10→0.00、`pf6_slots` 0.07→0.00（ms/次），**prefetch 总计 0.16→0.06 ms/层** → **4.8 ms/步** |

**两个都是"值整轮不变却每次重算"的典型。** `aclrtGetDeviceInfo` 是 ACL 运行时查询
（对 `aclrtGetDevice` 的 0.4 μs 而言，34.8 μs 高两个数量级）；sym tensor 原本每次
都要「线性扫 `_VMM_REGISTRY` + `acl_shmem_get_buffer_ptr`（C++）+ 构造 storage」。

**缓存的失效条件**：命中时仍做一次 `allocation._destroyed` 属性检查（廉价），
`reset_symmetric_descriptors()` 里一并 `clear()`。避免了悬垂 view。

### 2.3 打开过但**不建议默认开**的开关

| 开关 | 结论 | 证据强度 |
|---|---|---|
| `ASCEND_MOONEP_PREFETCH_RELAXED=1` | barrier 确实消失（**4/层 → 0/层**，两次跑都是 0） | **强**（二值，无噪声） |
| ↳ 它的墙钟影响 | 单样本下 profiler 开时 +29%；**无 profiler 下单样本 −1.6%** | **弱** ← 需重测 |
| `MOONEP_PF_STREAM=2` | prefetch 确实挪到 side stream 且 **overlap_pct 中位 0.97 / 92% 层 ≥0.5**（真并行） | **强** |
| ↳ 它的墙钟影响 | 战利品只有 rank1 的 241.5 μs/层 = **11.6 ms/步 = TTFT 的 0.75%**，小于噪声 | **强**（是上限推算） |
| `ASCEND_MOONEP_SKIP_DUP=1` | 设备侧长尾 1219.7→24.5 μs（dp0），**效果明确** | **强** |
| `ASCEND_MOONEP_NO_RECONCILE=1` | **对推理路径是 no-op**（见 §3） | — |

> **更正**：曾把"RELAXED 单独开慢 29%、必须配 NO_RECONCILE"写成结论。该说法有两个问题：
> ① 每档只有 **1 个样本**（当时噪声 ±150 ms）；
> ② 归因建立在"`_reconcile_async_ranks` 在热路径上"这个**已被推翻**的判断上。
> 见 §4.6。

---

## 3. 训练代码清理（本轮完成的部分）

`ascend_moonep` 最初为训练设计，推理从不调用其训练面。

### 已移除（Python 侧，323 行）

| 文件 | 移除内容 |
|---|---|
| `planning.py` | `MoonEPCommPlan.zero_fill_ranges` / `remote_stats` 两个死字段（含校验、clone、分配、构造点） |
| `ops.py` | `GradReduceOp` 整类 |
| `launchers.py` | `launch_grad_reduce` |
| `api.py` | `Buffer._run_reduce_grad_on_current_stream` / `reduce_grad` / `reduce_projection_grads` |
| `ops.py` `api.py` `__init__.py` | 7 处断掉的 import / kernels 注册 / udma 标志 / `__all__` 条目 |

**验证**：`import ascend_moonep` + `import ascend_moonep.api` 通过；三个方法确实不在
`Buffer` 上；端到端推理 `steps=2 gen=1`、TTFT 与移除前一致（~1535 vs ~1530）。

**备份**：`/home/tmp/backup_gradrm/`（6 个文件）。

### 未移除（待办）

| 项 | 量 | 备注 |
|---|---|---|
| `kernels/moonep_grad_reduce.cpp` | 409 行 | 要改 `CMakeLists.txt:94-95` + 重编 |
| `kernels/moonep_grad_reduce_udma.cpp` | 174 行 | 同上 |
| `csrc/ops/grad_reduce.cpp` + `.h` | ~150 行 | + `torch_classes.cpp:19,554` 的注册 |
| `test/*/test_*grad_reduce*.py`、`benchmarks/bench_grad_reduce.py` | — | 配套测试 |
| `scripts/run_audit.sh` 的相关条目 | — | |
| `_common.py` footprint 的 `grad_reduce` 项 | 1 行 | **只是估算，不真分配**（Buffer 构造只分配 hidden/weights/barrier/combine_sync/dispatch_sync） |

> ⚠️ 注意：`_common.py` 的 footprint 函数 docstring 明确写着「buffer-backed 路径下
> 对称堆大小由注册进 `init_with_buffer` 的 VMM descriptor 决定，**本函数仅作预算参考**」。
> 所以删 grad-reduce **不释放任何运行期内存**，只让估算诚实（当前多算 1~2 GB）。

### 推理路径上**没有**每层设备同步（已核实）

一度以为 `_reconcile_async_ranks`（每层 3 次 `npu.synchronize()` + `dist.barrier()`）
落在推理热路径上 —— **这个判断是错的**。它的 4 个调用点全部在 `if not async_finish:`
**块的外面**，即 `async_finish=True` 那条分支：

```python
def dispatch(..., async_finish: bool = False, ...):
    if not async_finish:                       # ← 我们走这里
        self._run_dispatch_on_current_stream(...)
        return hidden_nvsh, ..., plan          # ← 直接返回
    # ↓ 对我们不可达
    self._run_dispatch_on_current_stream(...)
    self._reconcile_async_ranks(ctx)
    return ..., done
```

三个调用点（`buffer.dispatch` / `prefetch_projections` / `combine`）都用默认
`async_finish=False`，所以**该函数在推理时从不执行**。

**推理路径上 host 侧设备同步的完整清单**（逐个核过守卫）：

| 位置 | 守卫 | 推理时执行 |
|---|---|---|
| `api.py:599` `_reconcile_async_ranks` | async 分支 | **否** |
| `api.py:622-623` | `destroy()`（退出时） | 否 |
| `moonep_comm.py:1152` | `_ENV_PACE_SYNC`（默认 False） | 否 |
| `moonep_balance.py:718/782/813/827` | `MOONEP_PF_TIMING`（默认 False） | 否 |

**⇒ 每层的设备同步数是 0。** 设备空转 51~60% 的原因是 **host 本身算得慢**
（单腿 0.35 ms/层、双腿 1.27 ms/层），不是被同步摁住的。

**唯一真正存在的每层同步是 `MOONEP_BALANCE_STATS` 那两个 `.item()`** —— 已移除。

> ⚠️ 我加的 `ASCEND_MOONEP_NO_RECONCILE=1` 对推理路径是 no-op（只影响 async API）。
> 之前"`RELAXED` 单独开慢 29%、配 `NO_RECONCILE` 修好"的结论**是单样本、且建立在
> 上述错误判断上，需重测**（见 §4.6）。

---

## 4. 测量方法论（今天踩过的坑，务必保留）

### 4.1 单腿 vs 双腿 —— 差 3.6 倍

```
MoonEP host 路径       单腿        双腿
dispatch              0.20       0.43  ms/层
combine               0.09       0.18
prefetch              0.06       0.66
──────────────────────────────────────
合计                  0.35       1.27  ms/层
```

**两条腿（4 个 rank 进程）在同一 host 上互相抢**，把 MoonEP host 路径抬高 3.6 倍，
`prefetch` 一项高达 11 倍。**任何 host 侧结论都必须在单腿下测。**

### 4.2 采样协议

- **`stop_profile` 会打死 engine** → 一个服务世代只能采一个 profile
- 测 TTFT **不需要 profiler**：一个进程里用 `--offset 0,2,4,...` 跑多次 probe 即可
- `--reps N` 是 **best-of-N**，只打一行 → 用 `--offset` 拿多样本
- 单腿下 6 样本的极差 ≈ **33 ms**（双腿是 ±150 ms）
- `make_long_prompts.py --seed` 可换 prompt 池

### 4.3 必须关掉的诊断开关

`MOONEP_BALANCE_STATS`（每层 2 次 `.item()` → 757.7 ms 设备空转）、
`MOONEP_SLOT_STATS`、`MOONEP_PLAN_STATS`、`MOONEP_PF_TIMING`、`MOONEP_MLP_SYNC`。
已在 `serve_moonep.sh` 里硬性 `unset`。

**`MOONEP_TIME_STATS` / `MOONEP_LIVE_STATS`** 不同步设备但会在推理路径上加计时器
（26 处，实测开启后 host 数值整体抬高 ~3×）→ **默认关**。
（库内探针的开关是 **`ASCEND_MOONEP_TIME_STATS`**，前缀不同，别搞混。）

### 4.4 `MAXLEN` 陷阱

`serve_moonep.sh` 里 `MAXLEN=${MAXLEN:-4096}`。**手动起服务时漏传 `MAXLEN=24576`，
16384-token 的 prompt 会被 API 以 400 拒掉**，probe 报出 `TTFT=66ms / steps=0 /
gen=0 / [warn: client 16384 vs server 0]` —— 看起来像服务坏了，其实是配置漏了。

### 4.5 判真假并行要看 `overlap_pct`，不是"有没有交集"

粗算「prefetch 与某次 dispatch 有交集」得到 85%，按规范口径（每层 2 个 prefetch
的并集窗口 vs 该层 dispatch，`overlap = 交集 / min(两侧窗口长)`）得到
**中位 0.97 / 92% 的层 ≥0.5** —— 结论一样但可信度完全不同。

### 4.6 「守卫归属」判错 —— 今天犯了三次的同一类错误

三个错误案例：

| 我当时的判断 | 真相 | 怎么错的 |
|---|---|---|
| `_VMM_REGISTRY` 线性扫描是 prefetch 0.66 ms 的主因 | 只值 9~37 μs（微基准确认） | 读了代码就推断，没量 |
| 按"腿"绑 NUMA 能消除 host 争用 | 两条腿一起变差（CPU 池砍半） | 没意识到"隔离"和"减半"是同一个动作 |
| `_reconcile_async_ranks` 每层 3 次同步，是热路径遗留 | **它只在 async 分支，推理不可达** | **守卫检测脚本往上找最近的 `if`，但没判断调用是在 `if` 块内还是块后** |

**共同点：靠缩进/上下文猜控制流，没读到底。**

**判一个调用是否真的执行，必须读完整的分支结构**，不能只看最近的 `if`。
本仓最容易踩的形态就是这个：

```python
if not async_finish:
    do_work()
    return X            # ← 同步路径到此为止
# ↓ 下面是 async 路径，与上面的 if 是并列的，不是它的 else
do_work()
reconcile()
return X, done
```

**`if X: ...; return`** 这种"提前返回"写法会让"块外"的代码看起来像"块内"。

**另一条纪律**：单样本不下结论。今天所有被推翻的"收益/退化"结论，
追溯回去都是 **1 个样本 + 噪声 ±150 ms**（双腿）或没做对照。

---

## 5. 下一步的优先级

| 优先 | 事项 | 量级 | 说明 |
|---|---|---|---|
| 1 | **图模式（capture）可行性分析** | **26~64 ms/步** | 每层 13~17 次 launch 是 host 成本的主体；但 `run_slot_forward` 开头明确 `raise`，要先查清障碍面 |
| 2 | **重测 `PREFETCH_RELAXED` 的墙钟影响** | 未知 | 旧结论（+29%）是单样本且归因错误。用 `slot_bench.sh` 单腿多样本重做；barrier 消失本身是确凿的 |
| 3 | 清内核侧 grad_reduce | 583 行 | 纯代码清晰度，需重编 + 删测试 |
| 4 | host 侧核绑（按 **rank** 分不相交 CPU 组） | 未知 | 之前按"腿"分（都钉 node1）导致两条腿一起变差；按 rank 分没试过 |
| 5 | `_common.py` footprint 的陈旧 `grad_reduce` 项 | 估算诚实 | 不释放真实内存 |

> **图模式的前置**：sym tensor 地址已稳定（今天的缓存）；但
> "全设备同步可关"这条**前提已不成立**（本来就没有每层同步）—— 所以
> 图模式的障碍面要重新盘，不能再按旧的假设走。
