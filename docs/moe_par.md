# MoE 负载采集与峰均比报告

本文说明本仓库新增的一套 MoE expert 负载采集 / 峰均比（PAR）分析工具：怎么用、产出哪些文件、
文件长什么样、报告里每张图在回答什么问题。

---

## 1. 这套东西回答什么问题

四个问题，一份报告全部回答：

| 问题 | 报告位置 |
|---|---|
| **峰均比（PAR）** 有多大——最忙的 rank 比平均忙多少 | KPI 卡片、PAR 折线、箱线对比 |
| **token 在专家上的分布**——哪些专家是热点 | 层 × 专家热力图、专家负载剖面、单步专家剖面 |
| **token 在 rank 上的分布**——哪张卡扛得多 | 每 rank 负载柱状图、各 rank 负载随步折线 |
| **专家在 rank 上的分布**——哪个专家住在哪张卡 | rank × 专家热力图、专家放置矩阵 |

三条硬约束（这是整个设计的前提）：

1. **只打点本 rank 的负载**，不做任何额外的集合通信。PAR 本身是跨 rank 的量，但跨 rank 的聚合放在
   **离线脚本**里做：每个 rank 只写自己的那一份，分析脚本再把它们拼起来。热路径上只有
   「每 (步, 层) 一次设备内拷贝 + 每 N 步一次 `.cpu()`」。
2. **不依赖 EPLB**。打点复用的是 EPLB 也用的那份「每本地专家的 token-专家对数」，但开关独立，
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
| `scripts/moe_par_bench.py` | **新增** | 驱动：跑数据集 → 写 dump + `bench.json`（含逐请求 TTFT/TPOT 与吞吐） |
| `scripts/analyze_moe_par.py` | **新增** | 分析器：多份 dump → 三种 PAR + 中文自包含 HTML 报告 |
| `tests/ut/ops/test_moe_par_stats.py` | **新增** | CPU 可跑的 UT，同时覆盖打点写入与解析读出两侧 |
| `scripts/gsm8k_test.jsonl` | **新增** | bench 的默认数据集（GSM8K 问题，一行一个 JSON 对象） |

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

---

## 3. 快速开始

前置条件：跑测试的 shell 里已经具备运行环境（`ASCEND_RT_VISIBLE_DEVICES` 选好卡、
`PYTHONPATH`、`ASCEND_MOONEP_BUILD_DIR`、`LD_LIBRARY_PATH` 等）。本机这些变量写在**仓库外**的
`../scripts/init_env.sh`（即 `/home/x00893896/vllm_ascend_moonep_integration/scripts/init_env.sh`），
先 `source` 它。如果选卡是可见设备的真子集，注意 [跨仓修复记录](cross_repo_fixes.md) 里
「一份 Python 层要放在 `PYTHONPATH` 最前」那条。

```bash
cd /home/x00893896/vllm_ascend_moonep_integration/vllm_ascend_moonep
source ../scripts/init_env.sh

# ① 第 1 次运行：MC2（EP 打开、不开 moonep）
python scripts/moe_par_bench.py \
    --tp 2 --batch-size 16 --max-num-batched-tokens 512 --max-tokens 256 \
    --num-prompts 100 --tag tp2-ep-b16-bt512-o256
#    结束时打印 dump 目录、每份 dump 的行数/步数，以及可直接复制的出报告命令

# ② 第 2 次运行：同样的负载参数，只换通信方式（MoonEP / SHMEM）
ASCEND_RT_VISIBLE_DEVICES=6,7 python scripts/moe_par_bench.py \
    --tp 2 --batch-size 16 --max-num-batched-tokens 512 --max-tokens 256 \
    --num-prompts 100 --moonep --tag tp2-ep-b16-bt512-o256-moonep

# ③ 出报告（多份 dump 对比；报告落在当前目录）
python scripts/analyze_moe_par.py \
    --run mc2=<dump-dir-1> --run moonep=<dump-dir-2> \
    -o /tmp/moe_par_report.html
```

报告是**单个自包含 HTML**（CSS/JS/数据全部内联，无 CDN、无外部字体），直接双击或用浏览器打开即可，
可以整份发给别人。仓库根目录下运行 `analyze_moe_par.py` 时报告会落在仓库里，所以 `.gitignore`
已经忽略 `/*_report.html`。

---

## 4. `moe_par_bench.py`：驱动脚本

用途：把一份数据集灌进离线 `LLM`，跑完一遍，产出「每 rank 的负载 dump」+「本次运行的性能与配置 `bench.json`」。

```bash
python scripts/moe_par_bench.py --help
```

### 参数（默认值来自脚本内的 argparse）

**workload**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `<scripts>/gsm8k_test.jsonl` | 一行一个 JSON 对象的 jsonl |
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
- 不同层的 `local_num_experts` 不一致时，会告警一次并跳过该层（不写进 dump）。

---

## 6. dump 文件格式

每个 rank 一个文本文件，`moe_par_{tag}_rank{rank}.log`，三类行：

### `# meta`：自描述头（第一行）

```
# meta model=... tag=tp2-ep-b16-bt512-o256 rank=0 world_size=2 ep_rank=0 ep_size=2 \
  moe_layers=48 topk=8 global_experts=128 local_experts=64 redundant_experts=0
```

| 字段 | 含义 |
|---|---|
| `model` | 被服务的模型名（从 layer 的 `vllm_config` 读；读不到时回落到 `$MODEL`，再回落到 `unknown`） |
| `tag` | 本次运行标签 |
| `rank` / `world_size` | 本进程的全局 rank 与世界大小 |
| `ep_rank` / `ep_size` | 本 rank 在 EP 组内的序号与 EP 组大小 |
| `moe_layers` | MoE 层数（本模型 48） |
| `topk` | 每个 token 每层选几个专家（本模型 8） |
| `global_experts` | 全局专家数（本模型 128） |
| `local_experts` | **本 rank 上**的专家数 = `global_experts ÷ ep_size`（本模型 64） |
| `redundant_experts` | 冗余专家数；本工程不开 EPLB，恒为 0 |

分析脚本只用这一行就能确定所有维度，因此换模型不用改脚本。

### `# layer`：层 → 全局专家的映射（每层一行，静态）

```
# layer layer_ord=0 instance_id=0 log2phy=None expert_map=[0,1,...,63,-1,-1,...]
```

- `layer_ord`：本 dump 内的层序号（0 起，按首次出现顺序）。
- `instance_id`：层对象的 `moe_instance_id`（构造顺序的全局唯一计数）。
- `log2phy`：EPLB 的逻辑→物理专家映射；不开 EPLB 时为 `None`。
- `expert_map`：长度 = `global_experts` 的数组，`expert_map[本地槽位] = 全局专家 id`，`-1` 表示该槽位
  在本 rank 上不存在。**「token 在专家上的分布」和「专家在 rank 上的分布」就是靠它把本 rank 的槽位归到全局专家的。**

### 数据行：一个 (步, 层) 一行

```
step=0 layer=0 method=MC2CommImpl counts=[12,0,45,...]
```

| 字段 | 含义 |
|---|---|
| `step` | 服务过程中的第几步（0 起，连续编号；一步 = 引擎的一次 forward） |
| `layer` | 层序号，对应 `# layer` 里的 `layer_ord` |
| `method` | 本步实际使用的通信实现类名（见下表） |
| `counts` | 长度 = `local_experts` 的整数数组：**本 rank 各本地专家收到的 token-专家对数** |

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

### 两个统计口径

- `counts` 记的是**真实路由对数**，与 EPLB 累加的 `moe_load` 同口径，**不含 pad**。
- 恒等式：`Σ(所有 rank, 所有步, 所有层) counts` = `Σ步 token 数 × topk × moe_layers`。
  本模型 `topk × moe_layers = 8 × 48 = 384`。

### 收尾与边界情况

- 正常退出时，**在飞行中的那一步若层不全，整条丢弃**（不写进 dump）。
- 若某一步确实比已知层数跑得少（例如变长执行路径），该步会以「缺失层 = 0」写入，并在文件末尾补一行
  `# meta partial_steps=N`，说明有 N 步属于这种情况——这些步的负载是**被低估**的。
  分析脚本读到这一行会在报告里给出提示。
- 旧 dump 里若出现 `model=unknown`，说明是「从 layer 读模型名」这个修复之前的产物，**不影响分析**
  （报告不使用该字段）。

---

## 7. `analyze_moe_par.py`：分析器

```bash
python scripts/analyze_moe_par.py --run <名字>=<目录> [--run <名字>=<目录> ...] \
    [-o 报告.html] [--json payload.json]
```

| 参数 | 说明 |
|---|---|
| `--run [NAME=]DIR` | **必填，可重复**。一个 dump 目录（同一目录里的所有 `moe_par_*.log` 视为同一运行的不同 rank）。不写 `NAME=` 时用目录名 |
| `-o` / `--out` | 报告路径，默认 `moe_par_report.html`（**相对当前目录**） |
| `--json PATH` | 额外把算出来的数据写成 JSON（调试或二次加工用） |

运行时往 stderr 打一行摘要 + 每条告警，退出码 0（下图是示例，跑完 `--tag mc2` 那一条即可看到自己的版本）：

```
[moe-par] mc2: 2 ranks x 1800 steps x 48 layers | PAR per-layer mean 1.140 max 1.282
          | PAR per-step (whole step) mean 1.026 | PAR whole run 1.007
          | cross-rank join flat (spread 0.00e+00)
[moe-par] wrote moe_par_report.html (2914 KiB)
```

- `cross-rank join flat / NOT FLAT`：检查每个 (步, 层) 在各 rank 上是否**收齐**了（`flat` = 收齐；
  `NOT FLAT` 说明有 rank 的 dump 缺行，通常是采集没跑完或层宽不一致被跳过）。
- `PAR` 至少需要两个 rank；只有一个 rank 时显示 `n/a`。
- 没有 `--check` 之类的开关，校验是自动做的，结论直接打进报告。

---

## 8. 报告怎么读

### 8.1 三种 PAR 口径

| 口径 | 公式 | 含义 | 报告位置 |
|---|---|---|---|
| **逐层 PAR** | 对每个 (步, 层)：`max_rank(load) / mean_rank(load)`，再对该步内各层取平均 | **EPLB 自己优化的那个比值**（`EplbWorker._compute_imbalance` 同定义）。衡量单层内部有没有热点 | KPI 卡片、逐层 PAR 折线、箱线左图 |
| **整步 PAR** | 先把一步内所有层对每个 rank 求和，再 `max_r / mean_r` | 这一步里最慢的 rank 要在同步屏障前等多久 | KPI 卡片、整步 PAR 折线、箱线右图 |
| **全程 PAR** | 把每个 rank 在**全部步、全部层**上求和，再 `max_r / mean_r` | 整个服务过程一个数：总账上谁做得多。用于评估放置与容量规划，**预测不了任何一次等待** | KPI 卡片、「整个服务过程」面板 |

三者都在 `[1, rank 数]` 之间：1 = 完全均衡。**口径越大，比值越小**，因为层与层之间、步与步之间的
倾斜方向不一致，加总时会互相抵消。报告里 KPI 卡片旁边还给出「均衡度 = 1 ÷ PAR」，可读作最忙那张卡
的利用率上限。

注意：PAR 是**路由（gate）的性质**，不是通信方式的性质——同一批 token 无论由哪个 all-to-all 承载，
都会去同一批专家。两次运行之间的 PAR 差异主要来自 batch 组成（每步多少 token），所以报告专门画了
「PAR 与每步 token 数的关系」。

### 8.2 图表清单

| 图 | 回答什么 |
|---|---|
| KPI 卡片 | 结论速览：逐层/整步/全程 PAR、均衡度、每步 token 数、rank × 专家规模 |
| 怎么读这份报告 · 术语表 | 报告里出现的**每一个词**的说明，外加「报告地图」——想看什么该看哪张图 |
| 每个 rank 的负载（跨步均值） | 各 rank 平均扛了多少；虚线是均值，峰值与均值之比就是整步 PAR |
| 各 rank 负载随步变化 | 每条线一个 rank；最高线与最低线之差 = 同步屏障要为此付的代价 |
| 逐层 PAR / 整步 PAR 折线 | PAR 随步的波动；两条线之间的差距就是层间倾斜被抵消掉的部分 |
| 整个服务过程（全程累计） | 全程总账柱状图 + 三口径并排对照表 |
| token 去了哪里：rank × 全局专家 | **核心图**：一行一个 rank、一列一个全局专家，格子 = token-专家对数。行和 = 该 rank 的负载，列的深浅 = 专家是否热点 |
| 层 × 全局专家 | 同样的流量按层拆开：某层的深色带说明它主导了逐层 PAR |
| 专家负载剖面：单层（可选层） | 单层内按负载降序排列的专家柱状图，颜色按「高于/低于该层均值」发散着色，附带该专家相对均值的倍数 |
| 专家负载剖面：单步（可选步） | 单步内的同类视图，标题里给出该步 token 数与通信方式 |
| 每个专家住在哪个 rank | 专家放置矩阵：按负载推断每个（层, 专家）归属哪个 rank。无冗余时这是一个划分，正是 EPLB 会去搬动的形状 |
| 各运行的峰均比 | 多份 dump 的箱线对比（逐层 / 整步）+ 汇总表；结论：PAR 相同时说明差异来自 batch 组成而非通信方式 |
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
| `moe_par_<tag>_rank<N>.log` | `<dump-dir>/` | 文本，见第 6 节 | 15 MB / rank | 该 rank 每 (步, 层) 的各本地专家计数。**分析的原始输入** |
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

实跑一次之后，可以从这几处确认采集是干净的：

1. 每份 dump 的行数 ≈ `步数 × moe_layers`（同一步内每层一行）；
2. `# meta` 里的 `moe_layers` / `global_experts` / `topk` 与模型配置吻合；
3. 没有 `# meta partial_steps=` 行（或该值为 0），说明没有「层不全的步」；
4. 报告里没有 `NOT FLAT` 告警，说明各 rank 的步/层收齐了；
5. 报告的「与 bench.json 对账」一节两边数字相等。

---

## 12. 已知边界与坑

1. **一个进程只能建一次 `LLM`**。打点模块的状态（层映射、步计数、打开的文件句柄）是模块级的，
   不会重置；同进程里建第二个 `LLM` 会按旧的层映射往同一份 dump 里追加。要跑多个配置就多起进程。
2. **不同层的 `local_num_experts` 不一致**时，宽度对不上的层会被跳过并告警一次（本模型所有层一致）。
3. **`MOONEP_SLOT_MODE` / `MOONEP_DISABLE_UDMA` 不进 tag**：两次都开 `--moonep`、只差 slot 模式的运行
   会得到同样的 tag，从而写进同一个 dump 目录。要么显式 `--tag`，要么给 `--dump-dir`。
4. **报告体量**：2–3 MB 的内联数据，浏览器打开正常，但不要放进聊天工具里逐条 diff。
5. **本工具不开启、也不修改 EPLB**。`dynamic_eplb` 保持关闭；如果哪天在 EPLB 打开的情况下采集，
   打点记录的仍是「本 rank 实际处理的量」，但冗余专家会改变负载的形状，报告的解读方式要相应调整。
6. 采集期间**不要同时改 `--flush-every` 之外的打点参数**（tag、目录）；它们只在进程启动时生效。

---

## 相关文档

- [跨仓修复记录](cross_repo_fixes.md)：moonep / catlass / shmem 等**本仓之外**的修复，以及它们与
  本工具的相互影响（例如选卡是可见设备真子集时，必须把哪一份 Python 层放在 `PYTHONPATH` 最前）。
- 仓库根 `README.md`：环境搭建、引擎内 A/B、MoonEP 的性能优化点与已知问题。
