# MoE 负载采集与峰均比报告

本文说明本仓库新增的一套 MoE expert 负载采集 / 峰均比（PAR）分析工具：怎么用、产出哪些文件、
文件长什么样、报告里每张图在回答什么问题。

---

## 1. 这套东西回答什么问题

四个问题，一份报告全部回答：

| 问题 | 报告位置 |
|---|---|
| **负载口径是什么**——每个量数的是什么、数据从哪来、不含什么；「谁持有」与「谁真的算了」两套口径差在哪 | 报告最上的「口径与统计 · 先读这里」面板 + 「负载账本（分层）」 |
| **峰均比（PAR）** 有多大——最忙的 rank 比平均忙多少 | KPI 卡片、PAR 折线、箱线对比 |
| **token 在专家上的分布**——哪些专家是热点 | 层 × 逻辑专家热力图、专家负载剖面、单步专家剖面 |
| **token 在 rank 上的分布**——哪张卡扛得多 | 每 rank 负载柱状图、各 rank 负载随步折线 |
| **专家在 rank 上的分布**——哪个专家住在哪张卡 | rank × 逻辑专家热力图、专家放置矩阵 |

三条硬约束（这是整个设计的前提）：

1. **只打点本 rank 的负载**，不做任何额外的集合通信。PAR 本身是跨 rank 的量，但跨 rank 的聚合放在
   **离线脚本**里做：每个 rank 只写自己的那一份，分析脚本再把它们拼起来。热路径上只有
   「每 (步, 层) 一次设备内拷贝 + 每 N 步一次 `.cpu()`」。MoonEP 走的发送侧直方图同样如此：
   它是设备侧**已有**的一张向量，交给打点器不新增任何通信。
2. **不依赖 EPLB**。打点复用的就是 EPLB 也用的那份「每专家的 token-专家对数」，但开关独立，
   `dynamic_eplb` 保持关闭也能跑。
3. **模型可更换**。层数、专家数、topk、rank 数一律从 dump 的元数据里读，脚本不硬编码任何维度。

---

## 2. 组成与本次新增的文件

```
引擎内（热路径）                      离线（随时可跑，可反复跑）
┌──────────────────────────┐        ┌───────────────────────────────┐
│ moe_par_stats.record()   │        │ analyze_moe_par.py            │
│  · 每个 (步, 层) 记一行   │        │  · 读各 rank 的 dump           │
│  · 设备侧环形缓冲         │        │  · 跨 rank 拼成 R×T×L          │
│  · 攒够 N 步 .cpu() 落盘  │        │  · 算三种 PAR、出中文 HTML     │
└──────────┬───────────────┘        └───────────────┬───────────────┘
           │ moe_par_{tag}_rank{N}.log              │ bench.json（可选，性能面板）
           ▼                                        ▼
      dump 文本  ─────────────────────────────►  单个自包含 HTML
```

| 文件 | 状态 | 作用 |
|---|---|---|
| `vllm_ascend/ops/fused_moe/moe_par_stats.py` | **新增** | 打点核心：把每 (步, 层) 的各本地专家计数写成本 rank 的 dump。零集合通信 |
| `vllm_ascend/envs.py` | 修改（+4 个变量） | 集中定义 `VLLM_ASCEND_MOE_PAR_*` 开关（AGENTS.md 要求所有环境变量定义在此） |
| `vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py` | 修改（+调用点） | vllm 0.23.0 实际生效的 `AscendFusedMoE`，在 EPLB 计数点旁挂上打点 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 修改（+调用点） | 其它 vllm 版本的兼容分支，同样挂上打点 |
| `vllm_ascend/ops/fused_moe/moonep_comm.py` | 修改（+调用点） | MoonEP 调度器把 plan 之前的发送侧直方图 `tpe` 连同 pad 对数/落点交给打点（`note_route_counts`），plan 定稿后再把本 rank 真算的 `local_counts` 交给打点（`note_compute_load`，写出第二套口径）；引擎行为不变 |
| `scripts/moe_par/moe_par_bench.py` | **新增** | 驱动：跑数据集 → 写 dump + `bench.json`（含逐请求 TTFT/TPOT 与吞吐） |
| `scripts/moe_par/analyze_moe_par.py` | **新增** | 分析器：多份 dump → 三种 PAR + 中文自包含 HTML 报告 |
| `tests/ut/ops/test_moe_par_stats.py` | **新增** | CPU 可跑的 UT，同时覆盖打点写入与解析读出两侧 |
| `scripts/moe_par/gsm8k_test.jsonl` | **新增** | bench 的默认数据集（GSM8K 问题，一行一个 JSON 对象） |

打点接入点选在**通信方式已经算完 `expert_tokens` 之后**，与 EPLB 累加 `layer.moe_load` 的位置同级：

```python
if moe_par_stats.enabled():
    moe_par_stats.record(
        self,
        fused_experts_results.expert_tokens,     # 每本地专家的 token-专家对数
        fused_experts_results.group_list_type,   # 0 = 累加态，1 = 计数态
        type(_EXTRA_CTX.moe_comm_method).__name__,  # 本步用的是哪种通信方式
    )
```

五种通信实现（`AllGatherCommImpl` / `MC2CommImpl` / `All2AllCommImpl` / `FusedMC2CommImpl` /
`MoonEPCommImpl`）都通过同一个字段报数，所以**一套打点覆盖所有通信方式**，换通信方式不用改代码。
报告里会显示该步实际用的是哪一种（`method=` 列）。

唯一的例外是 MoonEP：它上报的 `expert_tokens` 是**接收侧**的量（含 pad、且可能被配额搬动、
还可能被静默丢行），不能当作路由账。所以它在 dispatch 之前把**发送侧**直方图也交给打点器，
打点器优先用后者（`counts_scope=global`）。见 §6「MoonEP 的容量 pad」。

---

## 3. 快速开始

前置条件：跑测试的 shell 里已经具备运行环境（`ASCEND_RT_VISIBLE_DEVICES` 选好卡、
`PYTHONPATH`、`ASCEND_MOONEP_BUILD_DIR`、`LD_LIBRARY_PATH` 等）。**入口就是本仓 `scripts/` 下的
驱动脚本**（`moe_par_bench.py` / `bench_long.py` / `moe_ab_test.py`，内部是离线 `vllm.LLM(...)`）；
仓库外 `../scripts/` 下的服务化脚本（`start_prefill.sh`）**当前跑不起来，不作为入口**。
最小环境（少一项就起不来，见 [跨仓修复记录](cross_repo_fixes.md)）：

```bash
cd /home/x00893896/vllm_ascend_moonep_integration/vllm_ascend_moonep
REPO=$PWD
WS=/home/x00893896/vllm_ascend_moonep_integration
export PYTHONPATH=$REPO/third_party/ascend_moonep:$REPO:$PYTHONPATH   # 必须 :$PYTHONPATH 追加
export ASCEND_MOONEP_BUILD_DIR=$WS/ascend-moonep/ascend_moonep

# ① 第 1 次运行：MC2（EP 打开、不开 moonep）
python scripts/moe_par/moe_par_bench.py \
    --tp 2 --batch-size 16 --max-num-batched-tokens 512 --max-tokens 256 \
    --num-prompts 100 --tag tp2-ep-b16-bt512-o256
#    结束时打印 dump 目录、每份 dump 的行数/步数，以及可直接复制的出报告命令

# ② 第 2 次运行：同样的负载参数，只换通信方式（MoonEP / SHMEM）
ASCEND_RT_VISIBLE_DEVICES=6,7 python scripts/moe_par/moe_par_bench.py \
    --tp 2 --batch-size 16 --max-num-batched-tokens 512 --max-tokens 256 \
    --num-prompts 100 --moonep --tag tp2-ep-b16-bt512-o256-moonep

# ③ 出报告（多份 dump 对比；报告落在当前目录）
python scripts/moe_par/analyze_moe_par.py \
    --run mc2=<dump-dir-1> --run moonep=<dump-dir-2> \
    -o /tmp/moe_par_report.html
```

报告是**单个自包含 HTML**（CSS/JS/数据全部内联，无 CDN、无外部字体），直接双击或用浏览器打开即可，
可以整份发给别人。仓库根目录下运行 `analyze_moe_par.py` 时报告会落在仓库里，所以 `.gitignore`
已经忽略 `/*_report.html`。

---

## 4. `moe_par_bench.py`：驱动脚本

用途：把一份数据集灌进离线 `LLM`，跑完一遍，产出「每 rank 的负载 dump」+「本次运行的性能与配置 `bench.json`」。

> DP 场景（8 卡 DP+MoonEP、DP8 三配置对比、负载测试与 profiling 的采集口径）见
> [`moe_par_dp8_sweep.md`](moe_par_dp8_sweep.md)。

```bash
python scripts/moe_par/moe_par_bench.py --help
```

### 参数（默认值来自脚本内的 argparse）

**workload**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `<scripts/moe_par>/gsm8k_test.jsonl` | 一行一个 JSON 对象的 jsonl |
| `--prompt-field` | `question` | 取对象里哪个字段当 prompt |
| `--num-samples` / `--num-prompts` | `0` | 跑多少条；`0` = 整份文件 |
| `--max-tokens` | `256` | 每条请求生成多少 token |
| `--ignore-eos` | 开 | 生成满 `--max-tokens` 个 token，保证每次运行的负载一致 |
| `--no-ignore-eos` | — | 遇 EOS 就停（更真实，但输出长度逐次浮动，两次运行的负载不可比） |
| `--temperature` | `0.0` | 贪心解码，可复现 |

**batching**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch-size` / `--max-num-seqs` | `32` | decode 并发数，即 batch size；**这是控制纯 decode 步 token 数的手段** |
| `--max-num-batched-tokens` | `1024` | 每步 token 预算；**通信方式是按这个数逐 step 选的** |

**parallelism**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `/mnt/data/weight/Qwen3-Coder-30B-A3B-Instruct` | 权重路径 |
| `--tp` | `4` | tensor parallel；EP 组就是 TP 组 |
| `--no-ep` | — | 关掉 EP（`ep_size=1`，退化成 all-gather MoE）。**EP 默认是开的**，只有显式传它才关 |
| `--moonep` | 关 | 开 `enable_moonep`（EP 走 SHMEM，而不是默认的 all-to-all） |
| `--max-model-len` | `40960` | |
| `--gpu-mem-util` | `0.80` | |
| `--no-enforce-eager` | — | 允许图捕获；**默认 enforce-eager 开着** |
| `--additional-config` | `""` | 额外 JSON，合并进 `additional_config`，如 `'{"foo": 1}'` |

**stats**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stats-dir` | `<scripts>/moe_par_stats` | dump 的根目录，实际落在 `<stats-dir>/<tag>/` |
| `--dump-dir` | `""` | 直接指定本次 dump 目录（原样使用，不再拼 `<tag>/`），给出时覆盖 `--stats-dir` |
| `--tag` | `""` | 本次运行的标签，用于目录名、文件名和报告里的运行名；留空则按配置自动生成 |
| `--force` | 关 | 目录里已经有**别的** tag 的 dump 时，仍然往里写 |
| `--flush-every` | `16` | 攒多少步做一次 `.cpu()` 落盘 |
| `--no-stats` | — | 关掉采集（做「开/关打点」的开销对照用） |
| `--out` | `""` | `bench.json` 的路径，默认 `<dump-dir>/bench.json` |
| `--drain-timeout` | `30.0` | 等待各 worker 把 dump 刷完的秒数 |

**misc**：`--seed`（默认 `0`）。

### 默认标签（tag）

留空时由配置拼出：`tp{tp}-{ep|noep}[-moonep]-b{batch_size}-bt{max_num_batched_tokens}-o{max_tokens}`，
例如 `tp2-ep-b16-bt512-o256`。这样从名字就能看出两次运行差在哪。

### 输出布局

```
<stats-dir>/<tag>/                         默认 <repo>/scripts/moe_par_stats/<tag>/
├── moe_par_<tag>_rank0.log                 rank 0 的负载 dump（每 rank 一份）
├── moe_par_<tag>_rank1.log
└── bench.json                              本次运行的配置、统计与性能数据
```

### 护栏与提示

- `--no-ep --moonep` 目前是**静默无效组合**：EP 关掉时通信方式退化为 all-gather，但 tag 里仍会带
  `moonep`。属于已知坑，不是特性；要对比就两个参数配套使用。
- `moe_par_bench.py` 会先把开关写进环境变量，**再** `import vllm`——因为打点模块在 import 时读这些
  变量（见下节）。
- 采集目录里若已有**同一个** tag 的 dump，只警告并覆盖；若已有**别的** tag 的 dump，直接报错退出，
  因为分析脚本是把「一个目录里的所有 dump」当成**同一次运行**来读的，混在一起会张冠李戴。
- 一个进程**只能建一次 `LLM`**（见第 12 节）。每次运行都要单独起进程。
- `MOONEP_SLOT_MODE`、`MOONEP_DISABLE_UDMA` 这类环境开关**不参与 tag 生成**：两次都开 `--moonep`
  但 slot 模式不同的运行，tag 会一样，默认会落到同一个 dump 目录里互相覆盖。要分开就显式 `--tag`
  或 `--dump-dir`。
- `scripts/moe_par/moe_par_bench.py` 会把 `VLLM_WORKER_MULTIPROC_METHOD` 默认设成 `spawn`（`setdefault`，
  调用方环境里写死的值优先）。原因见 §6「收尾与边界情况」：fork 出来的 worker 走 `os._exit()`，
  不执行 `atexit`，dump 的收尾汇总行和最后不足 `--flush-every` 步的数据都会丢。若环境里是别的值，
  脚本会在 stderr 打一条警告并说明后果。

---

## 5. 打点模块与开关

四个开关都在 `vllm_ascend/envs.py` 里集中定义：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_ASCEND_MOE_PAR_STATS` | `0` | 总开关，`1` 才采集 |
| `VLLM_ASCEND_MOE_PAR_STATS_DIR` | `/tmp/moe_par_stats` | 输出目录 |
| `VLLM_ASCEND_MOE_PAR_STATS_TAG` | `run` | 运行标签，写进文件名 |
| `VLLM_ASCEND_MOE_PAR_FLUSH_EVERY` | `16` | 攒多少步做一次 `.cpu()`，至少 1 |

**这些变量在模块 import 时读取一次**，之后不再查环境。所以：

- 用 `moe_par_bench.py` 时不用担心，它在 `import vllm` 之前就把变量设好了；
- 自己写脚本时，`os.environ[...] = ...` 必须放在 `import vllm` **之前**；
- 运行时改环境变量**不会**生效。

打点本身的行为细节：

- **自动过滤 profile / dummy 步**：读 `_EXTRA_CTX.in_profile_run`，启动期的 profile forward 不会污染 dump。
- **层编号按首次出现顺序**（`layer_ord`），不假设层数、不假设层顺序，换模型自适应。
- **步边界**：同一步内每层只跑一次，因此「某一层再次出现」即意味着上一步已结束。
- **开销**：每 (步, 层) 一次设备内拷贝进环形缓冲；每 `FLUSH_EVERY` 步一次 `.cpu()` 落盘；
  退出时 `atexit` 兜底刷一次。**全程无集合通信**。
  注意 `atexit` 在 fork 出来的 worker 里**不会执行**（`os._exit` 绕过钩子），所以收尾行改为「每次落盘
  都重写」，不依赖退出钩子——见 §6「收尾与边界情况」。
- 不同层的 `local_num_experts` 不一致时，会告警一次并跳过该层（不写进 dump）。
- **MoonEP 走另一条数据来源**：`moonep_comm.py` 在 plan 之前把发送侧直方图
  `tpe = torch.histc(topk_c, bins=全局专家数)` 交给 `moe_par_stats.note_route_counts(...)`，
  由它替代方法返回的接收侧 `expert_tokens`，并按全局专家 id 扣掉 pad。这一路径同样**不上通信**：
  `tpe` 是设备侧已有的一张向量，交接只是把已有张量的引用传过去（记录时才 `.cpu()`）。
- **MoonEP 一次派发写两行**：同一步同一层，除了上面那行「路由」（发送侧、扣过 pad），
  `moonep_comm.py` 还会把 **plan 定稿之后的 `local_counts`** 交给
  `note_compute_load(...)`，写成同一 (步, 层) 的另一行（`scope=compute`）——那是**本 rank 真要算的行**，
  按全局专家 id 记、**含 pad**。两行是同一批 token 的两套归属口径（见 §6「两套 rank 口径」）。
  它同样是设备侧已有张量的快照（`.detach().clone()`，几十个整数），**不上通信**；
  只对 MoonEP 这一步多一次拷贝，非 MoonEP 的 method 不写这一行。

---

## 6. dump 文件格式

每个 rank 一个文本文件，`moe_par_{tag}_rank{rank}.log`，三类行：

### `# meta`：自描述头（第一行）

```
# meta model=... tag=tp2-ep-b16-bt512-o256 rank=0 world_size=2 ep_rank=0 ep_size=2 \
  moe_layers=48 topk=8 counts_scope=global global_experts=128 local_experts=64 redundant_experts=0 \
  flush_every=16
```

| 字段 | 含义 |
|---|---|
| `model` | 被服务的模型名（从 layer 的 `vllm_config` 读；读不到时回落到 `$MODEL`，再回落到 `unknown`） |
| `tag` | 本次运行标签 |
| `rank` / `world_size` | 本进程的全局 rank 与世界大小 |
| `ep_rank` / `ep_size` | 本 rank 在 EP 组内的序号与 EP 组大小 |
| `moe_layers` | MoE 层数（本模型 48） |
| `topk` | 每个 token 每层选几个专家（本模型 8） |
| `counts_scope` | **数列的口径**：`global` = 发送侧、按逻辑专家 id、宽 = `global_experts`；`local` = 接收侧、按本地槽位、宽 = `local_experts`；`global+compute` = 同时带「路由」与「实际计算」两套行（见 §6「两套 rank 口径」）。分析器据此分流 |
| `global_experts` | 全局（= 逻辑）专家数（本模型 128） |
| `local_experts` | **本 rank 上**的专家槽位数 = `global_experts ÷ ep_size`（本模型 64） |
| `redundant_experts` | 冗余专家数；本工程不开 EPLB，恒为 0 |
| `flush_every` | 本次运行的 `VLLM_ASCEND_MOE_PAR_FLUSH_EVERY`（默认 16）。分析器用它算出「进程没跑到退出钩子时，这份 dump 最多可能缺多少步」，见下文 |

分析脚本只用这一行就能确定所有维度，因此换模型不用改脚本。

### `# layer`：层 → 全局专家的映射（每层一行，静态）

```
# layer layer_ord=0 instance_id=0 log2phy=None expert_map=[0,1,...,63,-1,-1,...]
```

- `layer_ord`：本 dump 内的层序号（0 起，按首次出现顺序）。
- `instance_id`：层对象的 `moe_instance_id`（构造顺序的全局唯一计数）。
- `log2phy`：EPLB 的逻辑→物理专家映射；不开 EPLB 时为 `None`。
- `expert_map`：长度 = `global_experts` 的数组，`expert_map[逻辑专家 id] = 本 rank 上的槽位号`，`-1` 表示该专家
  不在本 rank 上。两个方向都用得上：`local` 口径靠它把槽位上的计数还原成逻辑专家号；
  `global` 口径靠它反过来回答「谁持有这个逻辑专家」，把全局路由归属到各 rank。
  **「token 在专家上的分布」和「专家在 rank 上的分布」都靠它。**

### 数据行：一个 (步, 层) 一行

```
step=0 layer=0 method=MC2CommImpl counts=[12,0,45,...]
```

| 字段 | 含义 |
|---|---|
| `step` | 服务过程中的第几步（0 起，连续编号；一步 = 引擎的一次 forward） |
| `layer` | 层序号，对应 `# layer` 里的 `layer_ord` |
| `method` | 本步实际使用的通信实现类名（见下表） |
| `scope` | **可选**，只在 dump 里同时有两种口径时才写：`global` = 路由行、`compute` = 实际计算行、`pad` = pad 行。单口径 dump 不写这个字段（头部已经说清每行是什么），形状与旧版完全一致 |
| `counts` | 整数数组，含义由头部的 `counts_scope`（必要时再叠加行上的 `scope=`）决定（见上表）：`global` = 本 rank 在 dispatch **之前**按逻辑专家 id 数出的路由直方图，长度 = `global_experts`；`local` = 本 rank 各本地槽位收到的 token-专家对数，长度 = `local_experts`；`compute` = 本 rank 这一步**实际要算的行**（`[epn 本地专家 \| B 个预取槽]` 折叠成全局专家 id、**含 pad**），长度 = `global_experts` |

**`scope=pad` 行是稀疏的**：只有真的注入了 pad 的 (步, 层) 才写，一行就是「这个 rank 在这一层
往哪个专家塞了多少对 pad」——宽 = `global_experts`，除了那一格全是 0。它把「从路由行里扣掉的那一块」
作为账目留在 dump 里，而不是只留一个总数。所以同一个 (步, 层) 里，一个 rank 最多三行：
`global`（路由）/ `compute`（实际计算）/ `pad`（注入的 pad）。

`method` 的取值与报告里的显示名：

| dump 里的值 | 报告里显示 | 何时出现 |
|---|---|---|
| `AllGatherCommImpl` | all-gather | 关闭 EP 时 |
| `MC2CommImpl` | MC2 | EP 打开、token 数在阈值内、未开 moonep |
| `All2AllCommImpl` | all-to-all | 同上，另一种实现 |
| `FusedMC2CommImpl` | fused-MC2 | 融合路径 |
| `MoonEPCommImpl` | MoonEP (SHMEM) | 开了 `enable_moonep` 且本步 token 数不超过 SHMEM 容量 |

**通信方式是逐 step 动态选的**（按该步的 token 数），所以一次运行里 `method=` 可能不止一种；
同一步内所有层共用一个方式。只要开了 `--moonep` 就认为「这次跑的是 MoonEP」是不对的，
要看 dump 里的 `method=` 或报告里的「通信方式」列。

#### 这个 `method=` 是怎么选出来的

一句话：**每个 forward step 开始时决定一次，判据是该步的 token 数加一组静态配置**；同一步内所有
MoE 层共用同一种方式，与层号无关（所以按 `(step, layer)` 打标签是成立的）。

选型链路（`ascend_forward_context.py:99-104`，建立 forward context 时）：

| 步骤 | 代码 | 说明 |
|---|---|---|
| ① 取本步 token 数 | `max_num_tokens = num_tokens_across_dp.max()`（`:99`） | 取 **DP 组内各 rank 的最大值**，好让整个 EP 组选出同一种方式（否则集合通信会错配）；没有 DP 时就用本 rank 的 `num_tokens` |
| ② 选类型 | `select_moe_comm_method(max_num_tokens, vllm_config, is_draft_model)`（`:100`，定义在 `:335`） | 返回一个 `MoECommType`（或 `None`） |
| ③ 取实现 | `get_moe_comm_method(moe_comm_type)`（`:101`，定义在 `moe_comm_method.py:51`） | 从注册表取**单例**；注册发生在 `setup_moe_comm_method`（`moe_comm_method.py:55`），在建 MoE 层时调用一次（`AscendMoERunner.__init__`，`fused_moe.py:440`），逐 step 只是选哪个 |
| ④ 记进 dump | `moe_par_stats.record(..., type(_EXTRA_CTX.moe_comm_method).__name__)`（`fused_moe.py:716`） | 记的就是**这一步实际用的那个**，与 ② 的选择一一对应 |

注册表里有什么（`moe_comm_method.py:55-68`）：`ep_size > 1` 时注册 all-to-all / all-gather / MC2 / fused-MC2，
开了 `enable_moonep` 再注册 MoonEP (SHMEM)；`ep_size == 1` 时**只注册 all-gather**。

判定顺序（`select_moe_comm_method`，`ascend_forward_context.py:335-411`）：

```text
select_moe_comm_method(num_tokens)
│
├─ 1. 不是 MoE 模型 ───────────────────────────────→ None
│
├─ 2. MoonEP 覆盖（插在最前，优先级最高）：
│     enable_moonep 且 enable_expert_parallel
│     且 ep_group.world_size > 1
│     且 SHMEM 容量 S 已建好
│     且 num_tokens <= S ─────────────────────────→ SHMEM（MoonEP）
│
├─ 3. 没开 EP，或 ep_group.world_size == 1 ───────→ ALLGATHER
│
└─ 4. 按芯片分派（soc_version）：
      ├─ A2：每卡专家数 <= 24 且 ep >= 16
      │      且 num_tokens <= mc2容量 ────────────→ MC2
      │      否则 ─────────────────────────────────→ ALLGATHER
      │
      ├─ A3：num_tokens <= mc2容量
      │       ├─ enable_fused_mc2==1 且 ep <= 32 ─→ FUSED_MC2
      │       ├─ enable_fused_mc2==2 且草稿模型走
      │       │   dispatch_gmm_combine_decode、
      │       │   且量化是 w8a8_dynamic ───────────→ FUSED_MC2
      │       └─ 否则 ────────────────────────────→ MC2
      │      否则（大 batch / prefill）
      │       ├─ enable_fused_mc2==1 且 ep <= 32 ─→ FUSED_MC2
      │       └─ 否则 ────────────────────────────→ ALLTOALL
      │
      ├─ A5：num_tokens <= mc2容量 且 world_size > 1 → MC2
      │      否则 world_size <= num_experts_per_tok → ALLGATHER
      │      否则 ──────────────────────────────────→ ALLTOALL
      │
      └─ 310P ──────────────────────────────────────→ ALLGATHER
```

MoonEP 是**插在最前面的覆盖**：只要条件成立就选它，连 A3 上的 fused-MC2 也排在它后面。

两个容量是两码事，分界就靠它们：

| 容量 | 谁设 | 值 | 作用 |
|---|---|---|---|
| **MoonEP 容量 `S`** | `set_moonep_tokens_capacity`（`:232`），在 model_runner 初始化时无条件调用（`worker/model_runner_v1.py:502`，v2 在 `worker/v2/model_runner.py:130`） | `max_num_batched_tokens`，有 cudagraph capture 时取其上限的较大者 | 对称内存的每 rank 静态容量。**调度器一步永远不会超过 `max_num_batched_tokens`，所以只要第 2 步其余条件成立，整轮每一步都走 MoonEP**（dump 里每一行都是 `MoonEPCommImpl`，不会 MC2、MoonEP 交替出现） |
| **MC2 容量** | `set_mc2_tokens_capacity`（`:204`），同样在 model_runner 初始化时（`model_runner_v1.py:501`） | 开 `enable_prefill_mc2` 时取 `max_num_batched_tokens`，否则取 cudagraph capture 上限，再否则 `max_num_reqs × uniform_decode_query_len`；按 TP 上取整后**截到 512/rank**，最后 `× tp_size` | 判「这一步的 token 数放不放得进 MC2 融合算子」 |

**本机是 A5**（Ascend950PR；`get_ascend_device_type()` 按 `soc_version == 260` 判定，`utils.py:857`），
所以本机不开 moonep 时走的是上表的 A5 分支；A2 / A3 / 310P 的规则不同，换机器别照搬。

对读报告的意义：

- **「这次跑的是不是 MoonEP」只能看 dump 的 `method=` 或报告的「通信方式」列，不能看命令行**：
  开了 `--moonep` 但没开 EP、或 EP 只有 1 个 rank 时，第 2 步不成立，会**静默退回**芯片默认规则，
  dump 里一行 `MoonEPCommImpl` 都不会有；
- 一次运行里出现多种 `method=` 是正常的（不开 moonep 的一次实跑里同时出现过
  `AllGatherCommImpl` 与 `MC2CommImpl`，逐 step 交替）；报告按 step 列出通信方式，三种 PAR 口径也按 step 分组；
- **负载口径跟着 `method=` 走**：只有 MoonEP 的计数含容量 pad、只有 MoonEP 的 dump 是
  `counts_scope=global` 或 `global+compute`（后者多一套「实际计算」行），其余都是接收侧本地槽位口径
  （见下一节）。

### 负载口径：数的是什么

**先把口径定死**，后面所有数字都按它读（报告最上方的「口径与统计」面板会把这三条再讲一遍）：

| 量 | 定义 | 不含 |
|---|---|---|
| **专家负载** | **逻辑专家**收到的 token-专家对数。一个 token 在每个 MoE 层被路由 `topk` 次，一对 = 一次专家计算；**副本算同一个专家** | pad 行 |
| **rank 负载** | **实际落到该 rank 所持专家上**的对数——token 来自所有 rank，落到谁的专家就记谁。**不是「这个 rank 发出了多少」** | pad 行 |
| **pad 对数** | MoonEP 为对齐池化容量注入的零行 `(capacity − T) × topk`，由打点侧在产生它的 rank 上扣除 | —— |

为什么 rank 负载不能按「本 rank 发出的 token」算：EP 下每个 rank 只持有本步的一部分 token
（`Σ_r T_r` = 该步 token 数），但 token 会被投递到**别的** rank 的专家上。实测只记「本地口径的
发送侧切片」会少算约一半、与 bench 对账差 ~2×。

由此得到的恒等式（`global` 口径）：

- `Σ(所有 rank, 所有步, 所有层) counts` = `Σ步 token 数 × topk × moe_layers`，**不含 pad**。
  本模型 `topk × moe_layers = 8 × 48 = 384`。
- 逐层对数 = 该步 token 数 × `topk`，**与层无关**——这是「层间平坦」自查的依据。
- `Σ_r rank负载[r,t,l]` 也与层无关；各 rank 之和 = 全部真实路由（每个对恰好归属一个持有者）。

### 两套 rank 口径：路由归属 vs 实际计算

上一节那张表的 rank 负载是**路由归属**口径：它按「谁**持有**这个逻辑专家」记账。
这在只有整行搬迁、没有冗余副本时是准的，但**一旦 plan 把某些行搬到别的 rank 上去算**
（MoonEP 的预取槽 / 专家副本），它就不再等于「谁真的忙」：一个 token 分给专家 A、专家 A 的属主是
rank 0，可它是**在 rank 1 上算的**——按持有关系记在 rank 0 头上，rank 1 白干。

所以 MoonEP 的 dump 里同时记两套 rank 口径（头部写 `counts_scope=global+compute`），
**报告并列给出、以「实际计算」为主、以「路由归属」为参照**：

| 口径 | rank 负载怎么记 | 回答什么问题 | 边界 |
|---|---|---|---|
| **路由归属**（发送侧，`scope=global`） | 按**属主**：`(token, 专家)` 对记在**持有该专家**的 rank 上 | 「gate 把负载分给了谁」——决定 EPLB 该搬谁的量；也是 plan 的**输入** | **上界**：行被搬到别处算时，它仍记在原属主头上，所以真实的 rank 间差异只会比它**更平** |
| **实际计算**（接收侧，`scope=compute`） | 按**写者**：哪个 rank 算的，就记哪个 rank | 「谁真的在忙」——直接对应同步屏障前各 rank 的等待 | 这里**才是实测值**；含 pad（见下），且依赖 plan 把「搬到哪去了」如实交出来 |

**搬迁不影响专家维度**：行被搬到哪个 rank 上算，它算的都是**逻辑专家 A**，所以两套口径在专家轴上是
同一批流量、同一个热点形状；两者的专家负载只差 pad 那一块（计算的含 pad，路由的不含），
差额就是报告专家表里的 `计算量 = 真实 + pad − 丢弃`。**口径差异只发生在 rank 轴上**。

#### 分层账：发送量 = 真实路由 + pad，计算量 = 发送量 − 丢弃

pad 的零行**同样进 matmul、同样占算力**，所以「实际计算量」里有它们的一份。报告不把 pad 混进
真实负载，而是把总量分层摆开（`renderLedger`，只在两套口径的 dump 上出现）：

| 量 | 怎么来的 | 精确性 |
|---|---|---|
| **真实路由** | gate 选出的 `(token, 专家)` 对 = 发送侧直方图（已扣 pad） | 精确 |
| **pad 计算量** | MoonEP 为对齐容量注入的零行，由 `scope=pad` 行给出 | 精确（专家维度也精确） |
| **发送量** | 真实路由 + pad | 精确 |
| **丢弃** | `发送量 − 计算量`：容量包络在派发途中丢掉的对，**只能作为残差知道** | 精确（前提是两套打点都对得上） |
| **实际计算量** | 各 rank 写下的 `scope=compute` 行之和 | 精确（专家维度）；**rank 维度的 pad/真实拆分是估计** |

两个恒等式，报告里逐 rank 给出：

- `实际计算[r] = 归属真实[r] + 归属pad[r] + 净搬入[r] − 被丢弃[r]`，
  等价的读法是报告「净差」列：`实际计算 − 归属合计 = 净搬入 − 被丢弃`。
  净差为负 = 搬迁把行从忙的 rank 挪给了闲的（这正是 slot 模式想要的效果）。
- `Σ_e 计算量[e] = Σ_e 真实[e] + Σ_e pad[e] − 丢弃`，逐专家精确。

**唯一估的一处**：某一行是 pad 还是真实 token，**写下来的时候分不出来**（pad 行和真实行同槽，
plan 还可能把它搬到别的 rank）。所以 rank 维度的「这里面有多少 pad」只能按行数比例摊：
`pad_est[r] = Σ_e 该 r 在专家 e 上的计算行数 × (pad[e] / 发送[e])`——即「专家 e 的 pad 均匀摊在
它被发送的每一行上」。把各 rank 加回去只得到**活下来的 pad**（≤ `pad[e]`），
所以它只会少算、不会凭空造出 pad；`真实_est[r] = 计算量[r] − pad_est[r]`。
**总量、专家维度、净差都是精确的，只有 rank 维度的这个拆分是估计**——报告里这一列的列名
就写着「（估）」。想让它更准就把 pad 比例压低（pad 一多，估计的方差就大）。

### MoonEP 的容量 pad：为什么 moonep 运行的 counts 会虚高，以及怎么扣干净

这是本工具踩过的最大一个坑，**用 moonep 采集时必须知道**。

MoonEP（`MoonEPCommImpl`，即 `enable_moonep` 且本步 token 数在 SHMEM 容量内）的调度器
（`moonep_comm.py` 的 `TokenDispatcherWithMoonEP`）**每步先把 MoE 的输入 pad 到池化容量**再通信：

1. 容量取 `MOONEP_MIN_DISPATCH_TOKENS = 128` 起的 2 的幂阶梯（`MOONEP_POOL=1`，默认开），
   所以 `capacity ≥ 本 rank 本步的 token 数 T`；
2. pad 出来的行 `hidden` 置零、`topk` 全部写成**本 rank 的第一个本地专家**
   `pad_expert = rank × local_experts`（`MOONEP_PAD_EXPERT0=1` 时改写成全局 0 号专家）；
3. pad 行数 = `(capacity − T) × topk`。

**`expert_tokens` / `group_list` 这个量本身就含 pad**——EPLB 累加 `moe_load` 拿到的也是同一个量。
后果（在没有扣除的 dump 里）：

| 现象 | 原因 |
|---|---|
| 每格 (步, rank, 层) 的 `counts` 之和 ≈ `capacity × topk`（本模型 128×8 = 1024），而不是 `T × topk` | pad 把每步的输入撑到容量 |
| 第一个本地专家那一格常年几百上千，48 层无一例外 | pad 全部落在它头上；**专家分布图上那根突出的柱子是 pad，不是热点** |
| 「与 bench.json 对账」差出几十倍（实测 878,101 vs 64,877） | 对账用的是 `Σ counts ÷ (topk × 层数)`，pad 计入后推出的是 `Σ capacity` 而不是真实 token 数 |
| 「逐层自查」仍有几个百分点的层间偏差 | pad 与层无关、各 rank 也一样多，各层之和只剩真实路由的差别；**偏差小不代表数据干净** |

#### 扣在哪里：必须在**发送侧**、按**逻辑专家**扣

**在接收侧扣是错的路子，已被实测证伪。** 之前试过「本 rank 在本地 0 号槽扣掉
`(capacity − T) × topk`」，结果 45% 的格子算出负数、总量差 −64,634 对、逐层 join 偏差中位
+80~93%。原因有三，缺一不可：

1. `cu` 差分的 `local_counts` 是**接收侧**的量，pad 行和真实行混在一起，本地分不出谁是谁；
2. 那份计数已经被 plan 按配额搬过（每 rank 的总量被压成 `capacity × topk`），
   本地槽位上的数与「这个 rank 本步发出/收到多少 pad」没有固定的换算关系；
3. 接收侧还存在**静默丢行**：派发 kernel 在目标 chunk 越界时会丢掉该行
   （`moonep_dispatch.cpp` 的 `if (dst_loff >= NvS) continue;`），且没有任何丢行计数
   （本 build 不写 `plan.remote_stats`）。丢多少离线不可知、也不可还原。

**现在扣在发送侧。** `moonep_comm.py` 里、plan 生成**之前**的那份直方图

```python
E, K = self.num_experts, self.top_k      # E = 全局（= 逻辑）专家数
tpe = torch.histc(topk_c, bins=E, min=0, max=E-1)   # 按逻辑专家 id 统计，精确
```

是**每个 rank 自己发出的真实路由**，它满足 `Σ tpe = T × topk + (capacity − T) × topk`，
且 pad 全部落在它自己那个 `pad_expert`（全局 id）上。于是：

- 调度器把这张直方图交给打点模块：`note_route_counts(tpe, pad_pairs=(capacity − T) * topk, pad_index=pad_expert)`；
- 打点模块用它**替代**方法返回的 `expert_tokens`（那是接收侧、含 pad），并在**全局专家空间**里
  `counts[pad_index] -= pad_pairs`（复制后再减，不改原张量）；实测不会出现负数；
- 于是每一行数列的含义变成「本 rank 在这一层 dispatch 之前、按逻辑专家 id 数出来的路由」，
  宽度 = **全部逻辑专家**（本模型 128，是原先本地口径 64 的两倍）。

**引擎侧语义完全不变**：`tpe` 本来就是 plan **不消费**的量（Ascend 路径上 plan 自己从 `topk_idx`
重建直方图，见 `kernels/planning.cpp`；`ops.py` 也注明 "validated but not consumed"），
交给打点器不影响任何计算；pad 照旧写、plan 照旧算、EPLB 看到的负载照旧含 pad——只有 dump 变干净。
这条路径**没有集合通信、没有跨 rank 同步**，网格不变。

#### 分析器怎么用它

各 rank 的发送侧直方图按 `(step, layer)` 相加 = 这一步的**真实全局路由** `c[t,l,e]`，然后：

- 专家负载 = `c` 按逻辑专家求和（副本算同一个专家）；
- rank 负载（**路由归属**）= 把 `c` 按 `expert_map` 归属到「谁持有这个逻辑专家」
  （`einsum("tle,le->tl", c, mask_r)`）。

如果 dump 里还有 `scope=compute` 行（MoonEP 的两套口径），再算一遍第二套：

- 计算行已经按**全局专家 id** 摊好（`[epn 本地专家 | B 个预取槽]` 折叠过），
  所以按 (步, 层) 把各 rank 的**写者**加起来就是 `compute[t,l,e]`；
- rank 负载（**实际计算**）= 直接取写者的行和——谁写的记谁，不再查 `expert_map`；
- 专家负载（计算口径）= `compute` 按逻辑专家求和，含 pad；
- 逐专家核对 `计算 = 真实 + pad − 丢弃`，逐 rank 核对 `净差 = 净搬入 − 被丢弃`（见上一节）。

每份 dump 的头部会写明数列口径：`counts_scope=global`（发送侧，MoonEP 新 dump）、
`counts_scope=global+compute`（发送侧 + 实际计算，MoonEP 的两套口径）、
或 `counts_scope=local`（接收侧，其它 method 与旧 dump）。分析器按这个字段分流，
旧 dump 仍按本地槽位解读、并用下面的指纹识别告警。

收尾行会带上扣除记录：`# meta partial_steps=0 pad_pairs_subtracted=NNN pad_expert_global=I`
（每个 rank 只记自己扣掉的那部分；报告 T6 表按 rank 列出，并在报告最上方的「口径与统计」面板里
给出总量与占比；两套口径的 dump 另有三项对账计数，见上一节末尾）。
这个总数**只累计已经落盘的那些行**（和 `pad_rows_total` 一样在落盘时结算），所以它和
「所有 `scope=pad` 行的和」按定义相等，分析器的对账不会因为「note 比落盘早一层」而报假警。

两条边界：

1. **`MOONEP_PAD_EXPERT0=1` 时**pad 全部发给全局 0 号专家，只有 rank 0 收得到；其它 rank 的直方图里
   没有这一桶，扣除会落到不存在的格子上，打点侧记 `pad_notes_misplaced` 并在报告里告警。
   要精确计数就用默认的 pad 落点（别设这个变量）。
2. **冗余专家/预取槽**：真有副本时，副本各自处理多少在**路由归属**口径里不可见（发送侧只知道
   「这个专家被路由了多少」，不知道副本各摊了多少），报告因此把负载记在一个持有者头上、
   不去猜分摊，所以**路由归属口径的 PAR 是上界**；而**实际计算口径记的就是各 rank 真算的量**，
   不受这个限制——这正是它存在的理由。只带一套口径的旧 dump 只能用上界那一套，
   报告会在「口径与统计」面板和每处 PAR 结论标注这一点。

分析器还会对**没有扣除记录的旧 dump**做指纹识别：全部 step 都走 MoonEP、没有任何
`pad_pairs_subtracted`、且「每行最忙的本地专家占比中位数 > 0.5」同时成立时，报告会明确写出
「成因是 MoonEP 的容量 pad，不是各 rank 的步没对齐」，并提示用带修正的打点重跑。
这样的旧 dump 里 pad 占了绝对负载的 90% 上下，真实信号只剩几个百分点，**离线无法挽救**。

### 收尾与边界情况

- 正常退出时，**在飞行中的那一步若层不全，整条丢弃**（不写进 dump）。
- 若某一步确实比已知层数跑得少（例如变长执行路径），该步会以「缺失层 = 0」写入，并在文件末尾补一行
  `# meta partial_steps=N`，说明有 N 步属于这种情况——这些步的负载是**被低估**的。
  分析脚本读到这一行会在报告里给出提示。
- 同一条收尾行里还会带上 MoonEP 的 pad 扣除记录（`pad_pairs_subtracted` / `pad_expert_global`，
  必要时还有 `pad_notes_misplaced` / `negative_cells_clamped`），见上一小节。
- 两套口径的 dump 还会多三项：`pad_rows_total`（写了几行 `scope=pad`，只在有 pad 行时写）、
  `compute_rows_total` / `compute_pairs_total`（写了几行 `scope=compute`、共多少对）。
  **它们存在的意义是与数据行对账**：分析器把它们和真正读到的行数/对数比对，
  不一致就告警（「打点侧说写了 N 行、dump 里只读到 M 行」）——同一 (步, 层) 的第二行会覆盖第一行、
  窗口没落盘会整批少行，这两种损坏只有行数能看出来，只看总量发现不了。
  `compute_pairs_total` 可以和 `[moe-par-ledger] … compute=…`（记录器自己打的，按定义一致）
  以及派发侧的 `[moonep-balance] … rows_local=…` 对上，用来确认打点看到的计算量就是派发真的算了的量。
  和派发侧那行对账前要先对齐**覆盖范围**：两边都逐层累计，行里的 `layers=` 和 dump 的
  `compute_rows_total`（÷ 每步层数）各自说明覆盖到哪一步；服务化腿里 dump 常常比它少**正好一步**
  ——那一步在停服时还在飞行中（见下），差一步以内是收尾差而不是打点错。旧 dump 没有这三项，
  缺失不等于有问题。
- 旧 dump 里若出现 `model=unknown`，说明是「从 layer 读模型名」这个修复之前的产物，**不影响分析**
  （报告不使用该字段）。

#### 收尾行是「每批落盘都重写」的，不是退出时才写的

这一步是为了让 dump 在**进程没有走退出钩子**时也是完整的：`flush_every` 步落一次盘，每次落盘前后
把文件末尾那行汇总**先删掉、写完数据行、再以最新累计值补回**（`out.seek/_truncate` 到上一行的偏移，
文本模式下 `tell()` 返回的就是字节偏移，已实测）。所以任何时刻去看 dump，最后一行都是当前的最新账；
退出钩子只额外补上「还在飞行中的那批」。

**为什么非要这样：`atexit` 在 fork 出来的 worker 里不会执行。** `multiprocessing` 的 fork 子进程
结束时走 `os._exit()`，绕过所有 atexit 钩子（spawn 子进程会正常执行，已用最小样例实测对照）。
后果是 dump 会缺两样东西：末行汇总，以及**最后不足 `flush_every` 步的整批数据**——因为那批还没落盘。
`scripts/moe_par/moe_par_bench.py` **默认替调用方设成 `spawn`**（用 `setdefault`，环境里显式的选择优先）；
如果环境里是别的值，脚本会在 stderr 打一条警告，
说明这份 dump 会缺最后不足 `flush_every` 步、对账自查会报出来。

分析器对一个没有收尾行的 dump 不会把「扣了多少 pad」「有几步不完整」当成 0，而是明确写
**「不可知」**：T2 的 ③⑤ 两格、以及 T6 表在整个运行上的判定都会变成「不可知（dump 里没有退出汇总行）」，
`pad` 总数也不显示。对账那一格则给出解释：dump 结束在一个 flush 边界上，缺失的是「未落盘的整批」，
需要 spawn（或 `--flush-every 1`）重跑。

**服务化的停服路径还有一层：连 spawn 也救不回来。** `vllm serve` 的 API server 用
`shutdown_timeout=0` 收 EngineCore（`Process manager: force killing remaining processes`），
worker 是它们的子进程，一起被 SIGKILL 掉——不是 fork 的 `os._exit()` 问题，而是根本没走到解释器退出，
所以 worker 里所有 `atexit` 打点（`[moonep-balance]`、`[moonep-stats]`、`[moonep-tstats]`）在服务化
这条腿上**一行都不会出现**。两个对策：

- dump 侧：`--flush-every 1`（工作区脚本 `start_service_mode.sh` 的负载腿已默认设成 1）。
  每步落盘后，SIGKILL 丢掉的就从「不足一整批」缩到「**最后那一步**」——它是唯一还没收尾的一步，
  换谁来杀都留不下。
- 账目侧：`MOONEP_BALANCE_STATS_EVERY=<层数>` 让账目行不再只在退出时打，而是**每 N 层打一行累计账**
  （0 = 只在退出时打）。设为模型的 MoE 层数（DSV4-Flash 是 43）就是「一步一行」。这条变量下有
  两本互补的账：
  - `[moe-par-ledger r0 tag=…] steps=… rows=… pads=… compute=…`：**记录器自己**打的（两条腿都有），
    数字全部来自落盘行在主机侧的累计，所以它和 dump 的 `# meta` 行**按定义一致**，没有覆盖差。
    `rows` 的含义跟着口径走：MoonEP 腿是扣掉 pad 的真实路由，local 口径的腿就是本 rank 算的行。
    `rows + pads − dropped = compute` 一眼可见。
  - `[moonep-balance r0] layers=… tokens=… rows_local=… rows_all=… pad_rows=…`：**派发侧**打的
    （只有 MoonEP 腿），覆盖它真正 dispatch 过的每一层，因此天然比 dump 多出「还在飞行中、没落盘」
    的那一步 —— `layers=` 与 dump 的 `compute_rows_total` 一比就知道少了哪一步。它**和 dump 一样跳过
    profile/dummy 前向**（否则会正好差一个满容量批），这是两本账能对上的前提。
  两本账的 `rows_local` 与 `compute` 是同一个量，直接可比。

---

## 7. `analyze_moe_par.py`：分析器

```bash
python scripts/moe_par/analyze_moe_par.py --run <名字>=<目录> [--run <名字>=<目录> ...] \
    [-o 报告.html] [--json payload.json]
```

| 参数 | 说明 |
|---|---|
| `--run [NAME=]DIR` | **必填，可重复**。一个 dump 目录（同一目录里的所有 `moe_par_*.log` 视为同一运行的不同 rank）。不写 `NAME=` 时用目录名 |
| `-o` / `--out` | 报告路径，默认 `moe_par_report.html`（**相对当前目录**） |
| `--json PATH` | 额外把算出来的数据写成 JSON（调试或二次加工用） |

运行时往 stderr 打一行摘要 + 每条告警，退出码 0（下图是示例，跑完 `--tag mc2` 那一条即可看到自己的版本）：

```
[moe-par] mc2: 2 ranks x 1800 steps x 48 layers | counts local (local slots)
          | PAR per-layer mean 1.140 max 1.282
          | PAR per-step (whole step) mean 1.026 | PAR whole run 1.007
          | cross-rank join flat (spread 0.00e+00)
[moe-par] wrote moe_par_report.html (2914 KiB)
```

两套口径的 dump（`counts_scope=global+compute`）不再打一行，而是把同一份数据的两套 PAR 并排打两条
（下面的数字只示意格式，行在终端里会自然折行）：

```
[moe-par] serve_slot: 8 ranks x 64 steps x 43 layers | counts global+compute (logical experts)
          | cross-rank join flat (spread 0.00e+00) | MoonEP pads subtracted 14843460 (global ids)
[moe-par] serve_slot (实际计算·主): PAR per-layer mean 1.062 max 1.165 | PAR per-step (whole step)
          mean 1.016 | PAR whole run 1.004 | 计算量 15920000 = 真实 14435660 + pad 1484340 - 丢弃 12000
          (pad 占发送量 9.3%)
[moe-par] serve_slot (路由归属·参照): PAR per-layer mean 1.079 max 1.203 | PAR per-step (whole step)
          mean 1.024 | PAR whole run 1.010  [被搬到别处算的行仍记属主，故 rank 维度上是上界]
```

- `counts global (logical experts)` / `counts global+compute` / `counts local (local slots)`：
  这份 dump 的数列口径，取自头部 `counts_scope`。第一种是 MoonEP 的发送侧直方图（按逻辑专家 id），
  第二种在它之外还带一套「实际计算」行，第三种是各 method 上报的接收侧本地计数。
- 两套口径时的第二、三行：`(实际计算·主)` 与 `(路由归属·参照)` 各给一组 PAR，
  后者行尾明写它是上界。`计算量 = 真实 + pad − 丢弃` 那一段就是分层账的入口
  （`pad 占发送量` = pad ÷ (真实 + pad)，不是占真实路由）。
- 还有一条只在旧 dump 上出现的告警：**汇总行说扣了 pad、但 dump 里没有 pad 行**（旧版打点只记总数），
  这时「真实路由」是准的，而 `发送量 = 真实路由 + pad` 里的 pad **不可拆分**——不要当成 0。
- `cross-rank join flat / NOT FLAT`：检查每个 (步, 层) 在各 rank 上是否**收齐**了（`flat` = 收齐；
  `NOT FLAT` 说明有 rank 的 dump 缺行，通常是采集没跑完或层宽不一致被跳过）。
- 结尾的 `| MoonEP pads subtracted N (global ids)` 表示这份 moonep dump 已扣掉 N 个容量 pad
  （位置是全局/逻辑专家 id）；若显示 `| MoonEP PADS PRESENT (top expert 94% of every row, not subtracted)`，
  说明这是**修正前**的 dump，报告各项绝对值不可用（见 §6「MoonEP 的容量 pad」）。
- 若 dump 里**没有**收尾汇总行（末行不是 `# meta partial_steps=…`），摘要会打
  `| no exit-summary line: pads subtracted and partial steps UNKNOWN`，并在 stderr 给一条警告。
  此时「扣了多少 pad」和「有几步不完整」**不可知**，报告不会拿 0 冒充：T2 的 ③⑤、T6 表判定都写
  「不可知」，对账那一格解释成因是「进程没走退出钩子、最后未落盘的整批丢了」，
  并提示用 `spawn`（或 `--flush-every 1`）重跑（见 §6「收尾与边界情况」）。
- `PAR` 至少需要两个 rank；只有一个 rank 时显示 `n/a`。
- 没有 `--check` 之类的开关，校验是自动做的，结论直接打进报告。
- 报告本身也自查一次：写盘前把 HTML 里那一段 JS 交给 `node --check` 解析（PATH 上有 node 时），
  解析不过就打一条 `warning: 报告的 JS 第 N 块解析不过（浏览器里会白屏）`。模板里多一个花括号
  就是这样——HTML 本身完全合法、分析器照样退出 0，只有浏览器里是一片空白，所以必须单独查一次；
  机器上没有 node 时这一项静默跳过（报告本身不依赖 node）。

---

## 8. 报告怎么读

### 8.1 三种 PAR 口径

| 口径 | 公式 | 含义 | 报告位置 |
|---|---|---|---|
| **逐层 PAR** | 对每个 (步, 层)：`max_rank(load) / mean_rank(load)`，再对该步内各层取平均 | **EPLB 自己优化的那个比值**（`EplbWorker._compute_imbalance` 同定义）。衡量单层内部有没有热点 | KPI 卡片、逐层 PAR 折线、箱线左图 |
| **整步 PAR** | 先把一步内所有层对每个 rank 求和，再 `max_r / mean_r` | 这一步里最慢的 rank 要在同步屏障前等多久 | KPI 卡片、整步 PAR 折线、箱线右图 |
| **全程 PAR** | 把每个 rank 在**全部步、全部层**上求和，再 `max_r / mean_r` | 整个服务过程一个数：总账上谁做得多。用于评估放置与容量规划，**预测不了任何一次等待** | KPI 卡片、「整个服务过程」面板 |

三者都在 `[1, rank 数]` 之间：1 = 完全均衡。**口径越大，比值越小**，因为层与层之间、步与步之间的
倾斜方向不一致，加总时会互相抵消。报告里 KPI 卡片、三口径表旁边都给出「均衡度 = 1 ÷ PAR」，
可读作最忙那张卡的利用率上限。

三者的分子分母都按 §6「负载口径」数：**专家按逻辑专家，rank 视口径而定**（见下）。

注意：PAR 是**路由（gate）的性质**，不是通信方式的性质——同一批 token 无论由哪个 all-to-all 承载，
都会去同一批专家。两次运行之间的 PAR 差异主要来自 batch 组成（每步多少 token），所以报告专门画了
「PAR 与每步 token 数的关系」。

### 8.2 两套口径怎么并列读

MoonEP 的两套口径 dump（`counts_scope=global+compute`）里，上表每一个 PAR 都有**两个值**，
报告并列给出（KPI 卡片、各口径对照表、逐 rank 表都分两段）：

| | 实际计算（**主**） | 路由归属（参照） |
|---|---|---|
| rank 负载 | 谁算的记谁（含 pad），是**实测** | 谁持有专家记谁（不含 pad），是**上界** |
| 回答什么 | 谁在同步屏障前等；搬迁（slot）到底把负载搬平了没有 | gate 把负载分给了谁；EPLB 该搬谁 |
| 报告里的位置 | 主行、主柱、主结论 | 参照列/参照行，**差 = 主 − 参照** 一列 |

读法：**先看「实际计算」的 PAR**（那是要解释的现象），再看「路由归属」的 PAR（那是 gate 给出的、
EPLB 会去优化的形状），两者之差就是**搬迁**的贡献。`实际计算 < 路由归属` 说明搬迁把行从忙的 rank
挪给了闲的（逐 rank 的「净差」列会给出具体是谁挪给谁）。两者相等则说明这一步没有搬迁、
或者搬了但净效果为零。

**但要注意 pad 会把这个比值钉在 1.000**：池化容量对所有 rank 是对称的，pad 把每个 rank 补齐到同一个数，
所以只要 pad 占计算量大头，「实际计算」的逐层/整步/全程 PAR 就都精确等于 1.000——
那不是「负载均衡」，而是容量对齐的**构造结果**，真实路由的差额全被 pad 抹平了。
这种情况下负载差异只在「路由归属」那一套、以及分层账 T2 的「净差」与 T3 的专家行里可见。
DP8 小 batch 实测（`--max-num-batched-tokens 16384`、每步约 1 个 token、43 层）：每 rank 每层恰好
算 768 对 = 128（池化容量下限）× 6（topk），pad 占计算量 **97.0%**，于是实际计算 PAR = 1.000，
而路由归属逐层 PAR 均值 2.287、全程 1.073——**两套一比，pad 与搬迁各改写掉多少就一目了然**。

分层账要在「负载账本（分层）」面板里读：**总发送量 = 真实路由 + pad**，
**总计算量 = 总发送量 − 丢弃**；专家维度三个数都精确，rank 维度只有总量精确、
其中 pad 的拆分是估计（列名带「（估）」）。**pad 算计算量**——它是零行，但零行同样进 matmul、
同样占算力，所以要单独列出来看清楚它占了多少，而不是把它藏进「实际计算」里。

### 8.3 图表清单

| 图 | 回答什么 |
|---|---|
| **口径与统计 · 先读这里** | 报告最上的面板：T0 两套口径说明（路由归属 vs 实际计算，含算例）、T1 负载口径表（每个量数的是什么、数据从哪来、不含什么）、运行级口径徽标、冗余边界、T2 统计与自查表、T6 pad 记账表 |
| KPI 卡片 | 结论速览：逐层/整步/全程 PAR、均衡度、每步 token 数、rank × 专家规模；两套口径时分两段给（实际计算·主 / 路由归属·参照），另有一张「计算量分层」卡片（真实 + pad − 丢弃） |
| **负载账本（分层）** | 只在两套口径的 dump 上出现：T1 总账（真实路由 / pad / 发送量 / 丢弃 / 实际计算量）、T2 按 rank（两套归属 + 计算量的 pad/真实拆分，净差 = 净搬入 − 被丢弃）、T3 按专家（计算量 = 真实 + pad − 丢弃，逐专家精确） |
| 怎么读这份报告 · 术语表 | 报告里出现的**每一个词**的说明（含「路由归属口径」「实际计算口径」「发送量/计算量」「pad 计算量」「丢弃」「预取槽」），外加「报告地图」——想看什么该看哪张图 |
| 每个 rank 的负载（跨步均值） | 各 rank 平均扛了多少；虚线是均值，峰值与均值之比就是整步 PAR。表里另有「所持逻辑专家」与「相对最忙」两列；两套口径时以**实际计算**为主图，并加「路由归属」「净差」两列 |
| 各 rank 负载随步变化 | 每条线一个 rank；最高线与最低线之差 = 同步屏障要为此付的代价 |
| 逐层 PAR / 整步 PAR 折线 | PAR 随步的波动；两条线之间的差距就是层间倾斜被抵消掉的部分 |
| 整个服务过程（全程累计） | 全程总账柱状图 + 三口径并排对照表 |
| token 去了哪里：rank × 逻辑专家 | **核心图**：一行一个 rank、一列一个逻辑专家，格子 = token-专家对数。行和 = 该 rank 的负载，列的深浅 = 专家是否热点 |
| 层 × 逻辑专家 | 同样的流量按层拆开：某层的深色带说明它主导了逐层 PAR |
| 专家负载剖面：单层（可选层） | 单层内按负载降序排列的专家柱状图，颜色按「高于/低于该层均值」发散着色，表里给出持有 rank、占该层比例与「热点/正常/偏冷」判定 |
| 专家负载剖面：单步（可选步） | 单步内的同类视图，标题里给出该步 token 数与通信方式 |
| 每个专家住在哪个 rank | 专家放置矩阵：每个（层, 专家）归属哪个 rank。无副本时这是一个划分，正是 EPLB 会去搬动的形状；有副本时报告会标注**路由归属口径**的 PAR 是上界 |
| 各运行的峰均比 | 多份 dump 的箱线对比（逐层 / 整步）+ 汇总表，以及两套口径的逐项对照（怎么算 / 实际计算 / 路由归属 / 差 / 均衡度）；结论：PAR 相同时说明差异来自 batch 组成而非通信方式 |
| 性能：TTFT / TPOT / 吞吐 | 见下节 |
| 数据 | 逐 step 表（含每步通信方式）、每 rank 表、逐层表——图里画的东西全都能查到原始数字 |

报告右上角有「深色 / 浅色」切换；所有图和表都支持悬停看精确值。

---

## 9. 性能面板：TTFT / TPOT / 吞吐

负载回答「谁在等」，性能回答「等了多久」。两张表放在一起看，才能判断一次负载倾斜值不值得修。

数据全部来自 `bench.json`，**不是**从 dump 推的（dump 里没有任何时间信息）：

| 指标 | 定义 | 来源 |
|---|---|---|
| **TTFT** | 从请求**到达**到第一个 token 产出；**含排队时间** | vLLM `RequestStateStats.first_token_latency` |
| **TPOT** | decode 段耗时 ÷ (输出 token 数 − 1)，即不含 prefill 产出的那个 token | `mean_time_per_output_token` |
| **端到端时延** | TTFT + decode 段（**不是** `末 token 时间 − 到达时间`） | 由上面两项相加 |
| **排队 / prefill 段 / decode 段** | 入队→调度、调度→首 token、首 token→末 token | 引擎内部时间戳 |
| **吞吐** | 输入 / 输出 / 总 tok/s、req/s，按整个 `generate()` 的墙钟时间算 | bench 自己计时 |

两个必须知道的坑：

1. **端到端不能用两个时钟相减**。`arrival_time` 是前端墙钟，而 `queued/scheduled/first_token/last_token`
   是引擎进程内的单调时钟；只有同为墙钟或同为单调时钟的量才能相减，所以端到端由 TTFT + decode 相加得到。
2. **TTFT 含排队**。离线 `LLM` 会把所有 prompt 一次性塞进去，靠后的请求要等前面的跑完，因此
   TTFT 的 p99 通常明显大于 p50。`queue_ms` 与 `prefill_ms` 分开给出，便于区分这两段。

还有一处是脚本内部已经处理掉的，自己改代码时要留意：离线 `LLM` 默认 `disable_log_stats=True`
（`vllm/entrypoints/llm.py`），此时引擎不分配 `RequestStateStats`，`RequestOutput.metrics` 是 `None`、
时延会读成 0。`moe_par_bench.py` 显式传了 `disable_log_stats=False`。

报告对数据缺失有三种降级，不会假装有数据：

| 情况 | 报告表现 |
|---|---|
| `bench.json` 齐全 | 完整性能面板：卡片 + 多运行对比表 + TTFT/TPOT 逐请求分布箱线 + 逐请求折线/表 |
| 有 `timing` 但无逐请求时延（旧版 bench 产出，或起服务时关了 metrics） | 只有吞吐，并给出一条说明 |
| 目录里没有 `bench.json`（例如直接用 `bench_long.py` 采的 dump） | 整块性能面板跳过 |

---

## 10. 生成文件一览

| 文件 | 落点 | 格式 | 量级（本例） | 作用 |
|---|---|---|---|---|
| `moe_par_<tag>_rank<N>.log` | `<dump-dir>/` | 文本，见第 6 节 | 15 MB / rank（`global` 口径约 ×2） | 该 rank 每 (步, 层) 的计数（含义见 `counts_scope`）。**分析的原始输入** |
| `bench.json` | `<dump-dir>/`（可用 `--out` 改） | JSON | 14 KB | 本次运行的配置、dump 清单、计时、逐请求时延、性能汇总、出报告命令 |
| 报告 HTML | `-o` 指定，默认当前目录 | 自包含 HTML | 2–3 MB | 结论与图表；可单文件传阅 |
| payload JSON | `--json` 指定 | JSON | 2 MB | 报告背后的数据，供二次加工 |

`bench.json` 的主要字段：

| 字段 | 内容 |
|---|---|
| `tag` / `model` / `dataset` | 运行标签、模型路径、数据集路径 |
| `config` | tp / ep / moonep / batch_size / max_num_batched_tokens / max_tokens / ignore_eos / temperature / max_model_len / enforce_eager / additional_config / num_prompts / rows_in_dataset / seed |
| `devices` | 本次运行时的 `ASCEND_RT_VISIBLE_DEVICES` |
| `dump_dir` / `dumps[]` | dump 目录；每份 dump 的路径、字节数、行数、步数、`# meta` 字段 |
| `timing` | `generate_s`、输入/输出 token 数、`output_tok_s`、因 EOS 提前停止的请求数 |
| `requests[]` | 逐请求：序号、输入/输出 token 数、结束原因、逐请求时延明细 |
| `perf` | 上面各项的分位数汇总 + 吞吐（报告的性能面板就读这一块） |
| `analyze_cmd` | **可直接复制执行的出报告命令**（`-o` 用当前目录） |

分析脚本对账：`Σ counts` 应等于 `Σ(输入 token + 输出 token − 1)`。减 1 是因为**最后一个采样出来的
token 不会再喂回模型**，不会产生任何专家路由。实测两次数据精确相等（32223 = 32223）；这条检查在报告
的「与 bench.json 对账」一节里给出。

---

## 11. 验证与自测

```bash
# 单元测试：打点写入 + 解析读出，CPU 即可跑
pytest -sv tests/ut/ops/test_moe_par_stats.py
```

UT 的设计意图：打点与分析两侧**一起测**——一侧改了格式，另一侧立刻挂，而不是几周后在报告里发现异常。
MoonEP 的发送侧直方图路径（`note_route_counts` → 全局专家空间扣 pad → 头部 `counts_scope=global`
→ 收尾行的 `pad_pairs_subtracted`）也在同一份 UT 里覆盖，包括「note 只用一次」「直方图赢过接收侧计数」
「note 为 0 时仍然替换计数」「pad 索引越界只记账不扣减」「扣出负数时夹到 0 并计数」几种边界。
另有一组 4-rank 的合成往返用例：把真实路由拆成各 rank 的发送侧直方图 + pad 后写出，
再由分析器拼回来，断言 `layer_expert` / `rank_expert` / `pad_subtracted` 全部精确复原。
同一组用例还覆盖**两套口径**：写出带搬迁（部分行落到别的 rank）与丢弃的接收侧计数，
断言「路由归属」的 PAR 是上界、「实际计算」的 PAR 更小且等于逐 rank 真算的和、
逐专家 `计算 = 真实 + pad − 丢弃` 成立、`Σ pad_est ≤ pad 总量`（估计只会少算、不会造出 pad）、
以及收尾行的 `compute_rows_total` / `compute_pairs_total` 与真正读到的行数对数一致。
还有一组专门盯「没有退出钩子」的用例：只靠一次 flush 就能写出收尾行、重复 flush 是**改写**而非追加、
改写不会吃掉已写的数据行；分析器一侧则有用例断言「没有收尾行」时报告给的是「不可知」而不是 0。
最后还有一组盯**报告模板本身**的用例（PATH 上有 node 时才跑）：模板里的 JS 必须能过 `node --check`，
并且故意往模板里插一个多余的花括号时必须被抓住——HTML 合法但 JS 少一个花括号，浏览器里是一片空白，
分析器却会照样退出 0，所以这条只能靠用例守（见 §7 末尾）。

实跑一次之后，可以从这几处确认采集是干净的：

1. 每份 dump 的行数 ≈ `步数 × moe_layers`（同一步内每层一行）；
2. `# meta` 里的 `moe_layers` / `global_experts` / `topk` / `counts_scope` 与模型配置、通信方式吻合；
3. 存在 `# meta partial_steps=` 行且值为 0：既说明没有「层不全的步」，也说明进程走到了收尾逻辑
   （这一行缺失意味着 dump 不完整，见 §6「收尾与边界情况」）；
4. 报告里没有 `NOT FLAT` 告警，说明各 rank 的步/层收齐了；
5. 报告的「与 bench.json 对账」一节两边数字相等；
6. 走 moonep 的 dump 应是 `counts_scope=global` 或 `global+compute`，收尾行里出现
   `pad_pairs_subtracted=`（数值 ≥ 0），且**不应**出现 `pad_notes_misplaced=`；报告最上方的
   「口径与统计」面板里，T2 的自查 ①②③④ 全绿、T6 表给出每个 rank 的 pad 占比。
   若报告反而出现「成因是 MoonEP 的容量 pad」那条告警，说明这份 dump 是**修正前**的产物，
   绝对负载与 PAR 不可用，必须重跑。
7. 两套口径的 dump 另有三项对账：收尾行 `pad_rows_total` / `compute_rows_total` /
   `compute_pairs_total` 应与报告「负载账本」里的行数、对数一致（不一致分析器会告警）；
   逐专家的 `计算量 = 真实 + pad − 丢弃` 应逐行成立；rank 维度的 `净差` 列应等于
   `净搬入 − 被丢弃`（slot 模式下它是负的还是 0，就是搬迁有没有把负载搬平的判据）。
   `compute_pairs_total` 还可以和 `[moe-par-ledger] … compute=…`（应与收尾行严格相等）以及
   派发侧打印的 `[moonep-balance] … rows_local=…` 对上（和派发侧对账前先按行里的 `layers=`
   对齐覆盖到第几步，见 §6）。

---

## 12. 已知边界与坑

1. **一个进程只能建一次 `LLM`**。打点模块的状态（层映射、步计数、打开的文件句柄）是模块级的，
   不会重置；同进程里建第二个 `LLM` 会按旧的层映射往同一份 dump 里追加。要跑多个配置就多起进程。
2. **不同层的 `local_num_experts` 不一致**时，宽度对不上的层会被跳过并告警一次（本模型所有层一致）。
3. **`MOONEP_SLOT_MODE` / `MOONEP_DISABLE_UDMA` 不进 tag**：两次都开 `--moonep`、只差 slot 模式的运行
   会得到同样的 tag，从而写进同一个 dump 目录。要么显式 `--tag`，要么给 `--dump-dir`。
4. **报告体量**：2–3 MB 的内联数据，浏览器打开正常，但不要放进聊天工具里逐条 diff。
5. **本工具不开启、也不修改 EPLB**。`dynamic_eplb` 保持关闭。冗余专家（副本）会影响解读：
   专家的负载仍按逻辑专家记（副本算同一个专家），但副本**各自**处理了多少在发送侧不可见，
   报告不去猜分摊、把负载记在一个持有者头上，所以 **「路由归属」口径的 PAR 是上界**；
   **「实际计算」口径记的是各 rank 真算的量，不受这个限制**（这正是两套口径并存的意义）。
   只带一套口径的 dump（旧 dump、非 MoonEP 的 method）只能用上界那一套，报告会在口径面板和
   相关结论处标注。
6. **pad 的扣除只发生在 dump 侧**：走 moonep 时 EPLB 看到的 `moe_load` 仍然含容量 pad（
   见 §6「MoonEP 的容量 pad」）。也就是说报告的绝对负载是真实路由，而 EPLB 优化的是含 pad 的那个量；
   两边不是同一个数，比较时不要混用。另外打点侧只**读**直方图与计数、不改 `group_list`、
   不改 plan 的输入（`tpe` 本来就不被 plan 消费），所以 MLP 与调度的计算结果与不开打点完全一致。
7. **接收侧不适合做负载账**：派发 kernel 在目标 chunk 越界时会静默丢行
   （`moonep_dispatch.cpp` 的 `if (dst_loff >= NvS) continue;`），本 build 不写 `plan.remote_stats`，
   丢了多少本地不可知。所以 `local` 口径只用于「实收」的近似观察，`global`（发送侧）才是精确的
   路由账。本 build 的 plan 也不做跨 rank 搬行（`CAP = NvS`，`balance ≤ 0` 直接 `break`），
   预取槽 `experts_to_copy ≡ -1`，`cu[E:]` 全 0。
   **在这种 build 上，「实际计算」只比「路由归属」多两件事**：pad 算进来、丢弃扣出去——
   逐 rank 的净差只会 ≤ 0（净搬入恒为 0，差额全部由丢弃解释），量级通常很小。
   也就是说 rank 轴上两套口径的差别**只有 pad 与丢弃**，报告会如实显示成这样。
   只有 slot 模式（`MOONEP_SLOT_MODE=1`，`experts_to_copy != -1`）才会出现由搬迁带来的情形，
   那时 compute 行才真正带来「谁被搬平了」的新信息。
8. 采集期间**不要同时改 `--flush-every` 之外的打点参数**（tag、目录）；它们只在进程启动时生效。
9. **fork 出来的 worker 不执行 `atexit`**：`multiprocessing` 的 fork 子进程以 `os._exit()` 结束，
   绕过所有 atexit 钩子，于是「退出时刷最后一批」不会发生——dump 会缺收尾汇总行和最后不足
   `flush_every` 步的数据（对账自查会先报出来）。收尾行已改为每次落盘重写，但**未落盘的那批无法找回**，
   所以要 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（`scripts/moe_par/moe_par_bench.py` 已默认）或用
   `--flush-every 1` 把窗口压到最小。这个坑的表现是「dump 看起来正常、数值也自洽，只是总量比 bench 少
   百分之几」，不查对账不容易发现。
10. **开了 `--moonep` 不等于这次走的就是 MoonEP**：通信方式是逐 step 选的（见 §6「这个 `method=`
    是怎么选出来的」），条件不满足时会静默退回芯片默认规则（本机 A5 是 MC2 / all-to-all / all-gather）。
    判断依据只有 dump 的 `method=` 或报告的「通信方式」列；命令行开关和 `bench.json` 里的配置项
    （`config.moonep` 等）只说明**要求了什么**，不说明**实际走了什么**。

---

## 相关文档

- [跨仓修复记录](cross_repo_fixes.md)：moonep / catlass / shmem 等**本仓之外**的修复，以及它们与
  本工具的相互影响（例如选卡是可见设备真子集时，必须把哪一份 Python 层放在 `PYTHONPATH` 最前）。
- 仓库根 `README.md`：环境搭建、引擎内 A/B、MoonEP 的性能优化点与已知问题。
