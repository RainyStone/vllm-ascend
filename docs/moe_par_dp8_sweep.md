# DP8 三配置跑测（gsm8k）：mc2 / moonep-owner / moonep-slot

**目的**：在同一台 A5（Ascend950DT，8 卡）、同一份 DeepSeek-V4-Flash 上，用同一批 gsm8k
请求跑三遍，只改 MoE 通信方式，然后一次出报告对比负载（PAR）与性能（吞吐/TTFT/TPOT），
顺带取一份 GSM8K 准确率和一份 torch_npu trace。

| leg | `--moonep-mode` | 环境 | 含义 |
|---|---|---|---|
| `mc2` | `off` | 不设 `MOONEP` | 标准 all-to-all EP（对照组） |
| `moonep-owner` | `owner` | `MOONEP=1 MOONEP_SLOT_MODE=0` | MoonEP owner 模式 |
| `moonep-slot` | `slot` | `MOONEP=1 MOONEP_SLOT_MODE=1 MOONEP_SLOT_STATS=1` | MoonEP slot（镜像 expert 行） |

为什么是 DP 而不是 TP：DSA 稀疏注意力要求每 rank 64 或 128 个 q 头，而这份 checkpoint 一共
只有 64 个头，`TP=2` 直接 AICPU 22007。DP 是唯一能跑的形态，且必须多进程
（`LLM(data_parallel_size>1)` 在单进程里 raise），驱动脚本因此自己当父进程、每个 rank 起一个子进程。

## 跑法

一条命令（推荐）：

```bash
cd vllm_ascend_moonep
python scripts/moe_par/moe_par_sweep.py \
    --dp 8 --cards 0,1,2,3,4,5,6,7 \
    --model /mnt/weight/A5-weights/DeepSeek-V4-Flash \
    --dataset scripts/moe_par/gsm8k_test.jsonl \
    --num-prompts 0 --max-tokens 256 --batch-size 32 --max-num-batched-tokens 1024 \
    --gpu-mem-util 0.80
```

先看不跑（会打印每个 leg 的完整命令与 env，可直接复制手跑）：

```bash
python scripts/moe_par/moe_par_sweep.py ... --dry-run
```

只跑某几个 leg / 关掉额外的 pass：

```bash
python scripts/moe_par/moe_par_sweep.py ... --legs mc2,moonep-slot      # 只要两条腿
python scripts/moe_par/moe_par_sweep.py ... --skip-score --skip-profile  # 只要负载与报告
```

单条 leg（sweep 内部就是这么调的，手动调试用）：

```bash
python scripts/moe_par/moe_par_bench.py --dp 8 --cards 0,1,2,3,4,5,6,7 --tp 1 \
    --moonep-mode slot --model /mnt/weight/A5-weights/DeepSeek-V4-Flash \
    --dataset scripts/moe_par/gsm8k_test.jsonl --stats-dir scripts/moe_par_stats --tag my-slot
```

`--dp 1` 时退回单进程行为（原来的 TP 用法完全不变）。

## 负载测试与性能数据采集（怎么做）

三条腿怎么跑见上一节；这一节是操作口径——一次可信的负载测试要满足什么，三类性能数据分别怎么采。

### 负载测试（load pass）

口径只有一条：**同一批请求、同一组 batch 参数，只改 MoE 通信方式**。sweep 就是这么组三条腿的，
所以在 sweep 之外手改单腿参数再拿去对比，就不再是受控实验。

参数怎么选：

- `--num-prompts 0` = 整份数据集；只走通链路时用几十条即可，但三条腿必须相同。
- `--max-tokens` / `--batch-size`（映射到 `max_num_seqs`）/ `--max-num-batched-tokens` 共同决定
  **每个 step 的 token 数**，而 MoE 通信方式是逐 step 按这一步的 token 数选的
  （`vllm_ascend/ascend_forward_context.py` 的 `select_moe_comm_method`）——所以一份报告只对
  「这组参数」成立。要外推到别的负载形态，就把 `--batch-size` 扫一个区间重跑。
- 负载腿默认 `--ignore-eos` + `temperature=0`：每条请求都跑满 `--max-tokens`，长度一致、各 rank
  步数才对齐——这是 PAR 可比的前提。

什么算通过（sweep 自动判，失败进 `sweep_summary`，也体现在退出码上）：

| 判据 | 看哪里 | 不通过说明什么 |
|---|---|---|
| 对账 | dump 的行数/token 数 vs `bench.json` 的请求记录 | 有数据没落盘（fork 不吃 `atexit`），或混进了 dummy 批次 |
| `kv_check` | `bench.json["kv_check"]` | KV 池装不下一个整批 → 引擎靠抢占推进，负载数字描述的是抢占 |
| M7 | `[moonep-stats]` 的 `total_slot_rows` | slot 腿其实退化回了 owner 路径，这次对比无效 |
| 8 rank 到齐 | `state/` 下有 `done_rank*`、无 `fail_rank*` | 有 rank 中途死掉，dump 不完整 |

### 性能指标（吞吐 / TTFT / TPOT）

- **采集**：bench 自己记，不用额外开关。每个 rank 写 `bench_rank<r>.json`，父进程合并成
  `<dump>/bench.json` 的 `perf` 段。
- **口径**：`generate_s` 取**各 rank 的最大值**（最慢那个 rank 的窗口），不是父进程墙钟——模型加载
  占了大部分时间，用墙钟会把加载算进吞吐。
- **出表**：`analyze_moe_par.py` 的 `report.html` 性能面板 / `--json payload.json`；sweep 另外把每腿
  一行写进 `sweep_summary.md`。
- TTFT/TPOT 是逐请求统计，prompt 长度不齐时看 p50/p90，别只看均值。

### MoE 负载（PAR）

- **采集**：`VLLM_ASCEND_MOE_PAR_STATS=1`（bench/sweep 会替调用方设），逐 (step, layer) 写本 rank 的
  dump；落盘粒度由 `VLLM_ASCEND_MOE_PAR_FLUSH_EVERY`（默认 16 步）控制。记录器**不做任何跨 rank
  通信**，聚合放在离线。
- **分析**：`analyze_moe_par.py --run <名字>=<dump 目录>` → 三种 PAR + 自包含 HTML；报告先读
  peak/mean，再看峰值对应的 step。
- dump 目录里**只能放同一 tag 的一份 dump**：analyzer 把整个目录当一个 run 读。

### torch_npu trace（profiling）

唯一能回答「时间花在哪」的数据。两条路走的是同一份 worker 代码
（`vllm_ascend/profiler/torch_npu_profiler.py`），trace 名自带并行维度后缀，8 个 rank 共用一个目录安全。
机制与锁步约束见下面 §profiling。

- **离线**：bench `--profile-only`，固定窗口、8 rank 锁步，采到的是饱和稳态。
- **服务化**：`vllm serve ... --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<目录>`，
  然后 `curl -X POST <host>/start_profile` 与 `/stop_profile` —— 一次 HTTP 会广播到所有 DP engine core，
  不用逐 rank 操作。

用离线数据说服务化的事之前，先过一遍这张表：

| 维度 | 离线（本套脚本） | 服务化（`vllm serve` DP） |
|---|---|---|
| 请求来源 | 父进程静态切片，一次灌满 | 前端动态到达，coordinator 按队列长度分派 |
| coordinator / wave | 没有（offline 下 `run_coordinator` 恒为 False） | 有：wave 对齐各 rank，wave 之间引擎 pause |
| 批次形状 | 恒定满并发、长度一致 | 参差、prefill/decode 混批、有空转 |
| trace 窗口 | 固定条数 × 固定 token 数，8 rank 步数一致 | 真实流量，参差 |

结论：**内存账本、M7、准确率、EP/卡数相关的结论可以直接迁移**；**三条腿的相对比较也可以**（同一批
请求、只换通信方式）；**只有绝对吞吐与 PAR 分布只代表「满负载饱和点」**，不能直接当作服务化的性能
预测——MoE 通信方式逐 step 由 token 数决定，token 数分布不同时，腿之间的相对优劣都可能反转。

## 产物

```
scripts/moe_par_stats/<sweep-tag>/
├── mc2/ moonep-owner/ moonep-slot/     每 leg 的 dump 目录；里面 bench.json + moe_par_<tag>_rank<r>.log
│   ├── bench_rank<r>.json              每个 rank 自己的计时/请求记录
│   ├── log/rank<r>.log                 该 rank 的完整 stdout（内存账本、报错都在这里）
│   └── state/                          DP rendezvous 标记（done_rank*/go/fail_rank*）
├── mc2-score/ ...                      准确率 pass（--score，独立进程、--no-stats）
│   └── score.json                      准确率、invalid 率、逐题 (idx,pred,gold,ok,text)
├── <leg>-prof/                          profiling pass
│   └── profile.json
├── logs/<leg>.log                      该 leg 的完整控制台输出
├── traces/<leg>/                       torch_npu trace：tensorboard --logdir <这里>
├── report.html                         三 leg 对比报告（analyze_moe_par.py 产出）
├── payload.json                        报告的数据（PAR 等，供脚本消费）
└── sweep_summary.json / .md            汇总表（吞吐/TTFT/TPOT/PAR/内存/准确率/slot 计数）
```

`analyze_moe_par.py` 只认 `<dir>/bench.json`，所以合并后必须叫这个名字——`bench_rank<r>.json`
是中间产物，别把它当报告输入。

单项报告也可以自己出：

```bash
python scripts/moe_par/analyze_moe_par.py --run mc2=<dir> --run moonep-slot=<dir> -o report.html
```

## 环境（脚本自己会 pin，别在 shell 里乱改）

| 变量 | 值 | 为什么 |
|---|---|---|
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:False` | DP+EP 下不设会挂（见 cross_repo_fixes.md） |
| `MOONEP_DISABLE_UDMA` | `1` | UDMA 打开时 shmem 初始化返回 -3 |
| `HCCL_BUFFSIZE` | `1024` | 首条跨 engine 集合通信需要 |
| `PYTHONPATH` | 本仓 `third_party/ascend_moonep` 放**最前** | 上一层的 `ascend-moonep/` 是未打补丁的旧副本，解析到它会缺 `acl_vmm_access_device_id`，非整组卡时 slot 模式报 `507899` |
| `ASCEND_MOONEP_BUILD_DIR` | 指向编好的 `ascend_moonep`（含 `_C.cpython-*.so`、`lib/libshmem_utils.so`） | python 包在 vendored，`.so` 在 build 目录 |

启动时会做一次自检（`python -c "import ascend_moonep"`），解析到的路径不是 vendored 就**直接退出**——
这个错误不这样做的话会一直到上卡才以驱动错误的形式暴露。

三个 env 只在调用方没设时注入（`setdefault`），`MOONEP*` 则按 leg 权威设置/清除：上一次 leg 留在
shell 里的 `MOONEP_SLOT_MODE=1` 会静默让下一条腿跑错配置。

## 内存账本与 KV 校验

每个 rank 的日志里有两行是排查内存问题的唯一入口：

```
[worker.py:593] Available KV cache memory: X GiB
[worker.py:759] ... Actual usage: W GiB for weights, A GiB for peak activation,
                N GiB for non-torch memory, G GiB for NPU graph memory.
[EngineCore] [kv_cache_utils.py] GPU KV cache size: T tokens
```

`bench.json["memory"]` 逐 rank 存这些数，`bench.json["kv_check"]` 给出判断：最小的 KV 池装不下
一个整批（`max_num_seqs*(prompt+max_tokens)`）或最长的那条请求时 `ok=false`（`--max-model-len`
只上报、不参与判定：vLLM 装不下它本来就起不来）——
这时引擎靠抢占推进，负载/延迟数字描述的是抢占而不是 MoE 通信，sweep 会把它算作失败
（`--allow-tiny-kv` 可放行）。这两个数被写进报告而不是只在日志里，因为 non-torch 一项就能把
KV 池吃掉两个数量级：DP4/EP4 上只换 `--moonep-mode`，owner 是 2.77 GiB（KV 3.86M tokens）、
slot 是 37.82 GiB（KV 82k tokens）—— 之前的「同配置 2.87 vs 37.93 GiB」就是 owner 与 slot 之差，
不是随机。slot 多出来的那份是镜像的 expert 行，约 `epn*12.75MiB*layers`（EP8/epn=32 时 ≈17 GiB）。

**评分腿要的 `--score-batched-tokens`（默认 4096）比负载腿的 `--max-num-batched-tokens`
大得多**，5-shot 提示约 2k token，小批装不下。vLLM 用这个值跑建 KV 之前的 profile run，
所以它直接决定峰值激活：DP4/EP4 的 slot 腿（non-torch 37.8 GiB）在这里顶爆，引擎启动即
`No available memory for the cache blocks`，评分腿失败。EP8 权重与非 torch 都减半，余量充足；
真要压，降 `--score-batched-tokens` 到 2048 或提高 `--gpu-mem-util`。

## slot 模式的 M7 判据

`MOONEP_SLOT_STATS=1` 时每个 worker 退出前打印：

```
[moonep-stats] dispatch_calls=N calls_with_slot_rows=N total_slot_rows=N total_prefetched_expert_slots=N
```

判据是 `total_slot_rows > 0 且 calls_with_slot_rows > 0`：一条 slot leg 如果一行 slot 都没走到，
说明它其实退化回了 owner 路径，这一次的「slot 对比」无效。sweep 的汇总表里有一列 slot rows
（取各 rank 最小值），`sweep_summary.json` 里保留了完整四元组。

## profiling

单独一个进程跑（`--profile-only`），因为记录器在 import 时就锁定了开关，和负载 pass 同进程会把
profile 的步数混进同一份负载 dump。窗口是固定的：`--profile-prompts`（**每个 rank** 多少条，
默认 8；sweep 把它乘 `--dp` 后作为 `--num-prompts` 传给 bench，bench 再把 `--num-prompts`
均分给各 rank）× `--profile-max-tokens`（默认 32）、greedy、`ignore_eos`，保证每个 rank 的步数
完全一致——`stop_profile()` 在 worker 里
**同步导出**，导出期间该 rank 不执行任何 step，一旦各 rank 收尾不齐就会卡在集合通信上。

`--num-prompts` 因此必须是 `--dp` 的正整数倍（不整除直接拒绝启动）：不整除就有 rank 少跑甚至
空窗口，而**一个空窗口的 rank 会杀掉整条腿**——它退出时关掉自己的 gloo socket，此时正在
startup 集合通信（warmup 的 `_sync_metadata_across_dp`）里的 rank 立刻收到
`Connection closed by peer`，跟着 EngineCore 死掉，整条 leg 连锁失败。同理，窗口**只许切一次**：
bench 已经在 `run_worker` 里按 rank 切过 `questions[rank::dp]`，任何地方再来一次同样的切片
都会让除 rank0 外全空（rank r 取的是 `r::dp` 里的第 `r::dp` 个）。

trace 按 rank 命名，多 rank 共用一个目录安全：`tensorboard --logdir <traces/<leg>>`。服务化下怎么采
（一次 HTTP 广播到所有 DP engine core）见上面 §负载测试与性能数据采集；采集机制相同不等于窗口相同。

## 已知坑

- **DP 子进程不能提前退出**：DP engine 每 32 步做一次跨 rank all-reduce，谁先拆引擎，别人的下一次
  集合通信就会失败。所以 bench 的父子之间有 rendezvous：子进程跑完先写 `done_rank<r>` 并等
  `state/go`，父进程等所有 rank 到齐才放行。这个窗口也覆盖 profiler 的导出。
- **各 rank 工作量要齐**：`--dp-slice`（DP>1 默认开）让 rank r 跑 `questions[r::DP]`；不齐的话跑得快的
  rank 会去跑 dummy 批次，而 dummy 批次**会被记录器记进 dump**（记录器只挡 profile/dummy-at-startup），
  负载就脏了。
- **跑之前会挡两类情况**：卡上有别人的进程（`npu-smi` 自由 HBM < `--min-free-gib`，默认 60 GiB）
  直接拒绝启动；目标 dump 目录里已经有别的 tag 的 dump 也会拒绝（analyzer 把整个目录当一个 run 读）。
- **EngineCore 启动失败不会写 `fail_rank`**：引擎核心在自己的进程里建 KV cache（上面那条
  `No available memory` 就是在 `_initialize_kv_caches` 抛的），它一死，dp-rank 子进程会卡在
  vLLM mp client 的启动握手上——不抛异常、不写标记、不退，父进程于是白等整个 `--timeout`
  且卡一直被占。父进程因此另有一条判据：**某个 rank 的日志里有 traceback 且 60s 没有增长**
  （`_log_signature` 只读每个日志的尾部 64 KB）→ 立刻杀这个 rank 的进程组，其余 rank 照常走
  收尾窗口。健康 rank 一直在写日志，不会误伤；模型加载那 ~100s 的静默里也没有 traceback。
- **超时只杀自己**：父进程用 `os.killpg` 只杀自己起的子进程组，不会碰其他租户的进程。
