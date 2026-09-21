# MoE 负载采集与峰均比报告

本文说明本仓库新增的一套 MoE expert 负载采集 / 峰均比（PAR）分析工具：怎么用、产出哪些文件、
文件长什么样、报告里每张图在回答什么问题。

---

## 1. 这套东西回答什么问题

四个问题，一份报告全部回答：

| 问题 | 报告位置 |
|---|---|
| **负载口径是什么**——每个量数的是什么、数据从哪来、含不含 pad；**真实行 / pad 行 / 合计行**三本账各回答什么 | 报告最上的「口径与统计 · 先读这里」面板 + 「分层账本」 |
| **实际计算量的峰均比（PAR）** 有多大——最忙的 rank 比平均忙多少；其中多少来自真实 token、多少来自 padding | KPI 卡片、PAR 折线、箱线对比 |
| **token 在逻辑专家上的热度差异**——这批输入把哪些专家喂饱了、有多少冷专家 | 逻辑专家热度面板、层 × 逻辑专家热力图、专家负载剖面、单步专家剖面 |
| **token 在 rank 上的分布**——哪张卡扛得多（**谁算谁背**，不按属主记账） | 每 rank 负载柱状图、各 rank 负载随步折线 |
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
- **开销**：每 (步, 层) 一次设备内拷贝进环形缓冲，**设备侧只此一个算子**——pad 的扣减与
  MoonEP 计算行的折叠都在落盘时于**主机**上做（见下两条），不占层内关键路径；
  每 `FLUSH_EVERY` 步一次 `.cpu()` 落盘（MoonEP 多一个计算行窗口，紧跟在第一次拷贝之后，
  此时流已经同步过，第二次只是传输、不再等待）；退出时 `atexit` 兜底刷一次。**全程无集合通信**。
  注意 `atexit` 在 fork 出来的 worker 里**不会执行**（`os._exit` 绕过钩子），所以收尾行改为「每次落盘
  都重写」，不依赖退出钩子——见 §6「收尾与边界情况」。
- 不同层的 `local_num_experts` 不一致时，会告警一次并跳过该层（不写进 dump）。
- **MoonEP 走另一条数据来源**：`moonep_comm.py` 在 plan 之前把发送侧直方图
  `tpe = torch.histc(topk_c, bins=全局专家数)` 交给 `moe_par_stats.note_route_counts(...)`，
  由它替代方法返回的接收侧 `expert_tokens`；扣 pad 在**落盘时**按全局专家 id 在主机上做——
  同一份直方图、同一个专家下标，只是把这次设备减法从层内挪到了已经同步过的落盘点。
  这一路径同样**不上通信**：`tpe` 是设备侧已有的一张向量，交接只是把已有张量的引用传过去
  （很快写进环形缓冲的那一步就是快照，`.cpu()` 要到落盘时才发生）。
- **MoonEP 一次派发写两行**：同一步同一层，除了上面那行「路由」（发送侧、扣过 pad），
  `moonep_comm.py` 还会把 **plan 定稿之后的 `local_counts`** 交给
  `note_compute_load(...)`，写成同一 (步, 层) 的另一行（`scope=compute`）——那是**本 rank 真要算的行**，
  按全局专家 id 记、**含 pad**。两行数的是两个不同的量：路由行是「这批 token 路由到谁」（发送侧、已扣 pad，
  按逻辑专家 id），计算行是「这个 rank 实际算了哪些行」（接收侧、含 pad）——分析器按 §6「一个负载、
  三本账」的规则把后者拆成真实行 / pad 行。
  它同样是设备侧已有张量的快照（`.detach().clone()`，几十个整数），**不上通信**；
  只对 MoonEP 这一步多一次拷贝，非 MoonEP 的 method 不写这一行。
  它**原样入库、落盘时才折叠**：记录器把 `[home | slots | slot 映射]` 三段一次写进计算行缓冲
  （两次设备内拷贝），落盘时在主机上折回全局专家坐标。折叠必须依赖散射（slot 装的是 plan 搬来的
  那个专家，得 `index_add`），放在设备侧就是每层 ~10 个小算子——那是**只有 MoonEP 腿付、且正好付在
  被对比的那条腿上**的税，换到主机只是几十个整数的 numpy 加法，且不占层内关键路径。
  dump 的字节与旧版**完全一致**（折叠结果落在同一行、同一宽度），所以分析器、报告与既有 dump 都不受影响。

---

## 6. dump 文件格式

每个 rank 一个文本文件，`moe_par_{tag}_rank{rank}.log`，三类行：

### `# meta`：自描述头（第一行）

```
# meta model=... tag=serve_slot rank=0 world_size=8 ep_rank=0 ep_size=8 \
  moe_layers=43 topk=6 flush_every=1 counts_scope=global+compute global_experts=256 local_experts=32 \
  redundant_experts=0
```

| 字段 | 含义 |
|---|---|
| `model` | 被服务的模型名（从 layer 的 `vllm_config` 读；读不到时回落到 `$MODEL`，再回落到 `unknown`） |
| `tag` | 本次运行标签 |
| `rank` / `world_size` | 本进程的全局 rank 与世界大小 |
| `ep_rank` / `ep_size` | 本 rank 在 EP 组内的序号与 EP 组大小 |
| `moe_layers` | MoE 层数（本次运行 43） |
| `topk` | 每个 token 每层选几个专家（本次运行 6） |
| `counts_scope` | **数列的口径**：`global` = 发送侧、按逻辑专家 id、宽 = `global_experts`；`local` = 接收侧、按本地槽位、宽 = `local_experts`；`global+compute` = 上面这套「路由」行之外，另带一套「实际计算」行（见 §6「一个负载、三本账」）。分析器据此分流 |
| `global_experts` | 全局（= 逻辑）专家数（本次运行 256） |
| `local_experts` | **本 rank 上**的专家槽位数 = `global_experts ÷ ep_size`（本次运行 32） |
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
  另一个方向靠它回答「谁持有这个逻辑专家」——报告的专家放置矩阵、「所持逻辑专家」列、
  以及 pad 专家上的真实行归给谁，都用它。
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

| 量 | 定义 | 口径 |
|---|---|---|
| **专家负载** | **逻辑专家**收到的 token-专家对数。一个 token 在每个 MoE 层被路由 `topk` 次，一对 = 一次专家计算；**副本算同一个专家** | 真实路由（不含 pad），精确 |
| **rank 负载·真实行** | 这批 token 真正路由出来的对里，**被该 rank 算掉的**那部分 | 精确到下面那条例外约定 |
| **rank 负载·pad 行** | 该 rank 算掉的 **pad 零行**（MoonEP 为对齐池化容量注入，同样进 grouped matmul） | 精确 |
| **rank 负载·合计行** | 真实行 + pad 行 = 该 rank 的全部计算量 | 精确（MoonEP 腿上 = `capacity × topk`） |
| **发送量** | 该 rank 自己的 token 路由出来的对（**中性参照**，不进任何一本账） | 精确 |

为什么 rank 负载不能按「本 rank 发出的 token」算：EP 下每个 rank 只持有本步的一部分 token
（`Σ_r T_r` = 该步 token 数），但 token 会被投递到**别的** rank 的专家上，plan 还可能把行搬走。
实测只记「本地口径的发送侧切片」会少算约一半、与 bench 对账差 ~2×。发送量在报告里单独列，
是「这批输入本来有多不均匀」的对照——MoonEP 把每个 rank 的派发都顶到容量，
所以它满足 `发送量 + 注入的 pad ≡ 容量`，正因为如此它不能和任何一本账相加。

由此得到的恒等式（分析器逐 (step, layer) 自查，报告「口径与统计」面板给出）：

- `Σ_r 真实行[r]` = 路由总量 = `Σ步 token 数 × topk`，**不含 pad**。本次运行 `topk = 6`。
- `Σ_r pad 行[r]` = 打点侧记录的注入 pad（各 rank `pad_pairs_subtracted` 之和）。
- `Σ_r 合计行[r]` = `capacity × topk`；off 腿没有 pad，合计 ≡ 真实行。
- 逐层对数 = 该步 token 数 × `topk`，**与层无关**——这是「层间平坦」自查的依据。

### 一个负载、三本账：真实行 / pad 行 / 合计行

**三本账走的是同一套算术**（`_par_pack` 作用在三个 `[R,T,L]` 数组上），所以三组数字之间的差别
只可能来自数据，不来自算法：

| 账本 | 是什么 | 主看什么 |
|---|---|---|
| **真实行** | 这批 token 真正路由出来的对，谁算的谁背 | **哪张卡真的忙、忙多少**（搬迁与丢弃的净效果都在里面） |
| **pad 行** | 为对齐池化容量注入的零行 | pad 落点是否均匀：它不携带 token，但占的算力是真的 |
| **合计行** | 真实行 + pad 行 | **恒为 1.000**，是构造结果不是结论，见下 |

**合计行在 MoonEP 腿上恒为 1.000**：每个 rank 每 (步, 层) 恰好算 `capacity × K` 行（capacity 是这一步
的池化量、各 rank 相同）⇒ 逐 rank 完全相等。它在报告里留着，是为了让「pad 把差异抹平了多少」可读；
真信息全在真实行 / pad 行的拆分里。报告也不会让读者误读：这句话写在 KPI 卡片、口径面板和页脚。

**逐 cell 的拆分规则**（分析器，逐 (步, 层)）：

1. `compute[r,e]`：本 rank 实际算的行（`scope=compute` 行）；off 腿没有 compute 视图，
   `local` 行本身就是「本 rank 算的行」，此时 pad 恒 0；
2. `tot[e]`：路由行按专家求和（`scope=global`；`local` 腿经 `expert_map` 折到全局专家 id）；
3. pad 专家的集合与各自的 pad 量：从 `scope=pad` 行的非零格子读出（**不写死 id**——
   pad 默认落在 `rank × local_experts`，开了 `MOONEP_PAD_EXPERT0` 时是 0 号专家，两种都覆盖）；
4. 对每个 pad 专家 `e`（持有者 `o`）：`真实行[o,e] = tot[e]`、`真实行[r≠o,e] = 0`、
   `pad[r,e] = compute[r,e] − 真实行[r,e]`（负值计数并告警）；
5. 其余专家：`真实行[r,e] = compute[r,e]`、`pad[r,e] = 0`——所以**被 plan 搬走/接手的真实行
   天然精确**，分析器不需要知道 plan 的搬运行情。

**唯一要约定的地方**：落在 **pad 专家**上的真实行该算给谁（那格里的真实行与 pad 行同槽，
打点时分不出来）。做法是把该专家收到的真实路由整个记给它的持有者 `o`。这条约定只在「真实行被搬到
别的 rank 上算」时才偏，而本 build 的 plan 不搬行——本次实测反推出的负值只有 **43 对**（3 个格子，
占真实行总量的 0.02%），两种候选算法（归持有者 / 按持有量摊）的全程真实行 PAR 只差 **0.1%**
（1.0830 → 1.0841）。采用前者，因为它在无重定位时严格成立。精确性分级：

- **专家维度与总量精确**：路由总量、注入的 pad 都是打点侧记下的整数，逐 cell 对得上；
- **rank 维度**：除上面那条例外约定外精确；**没有按比例摊的估计量**（旧版按行数摊 pad 的
  `pad_est` 已随两口径一起删除）。

**发送量只作参照**：它既不是真实行也不是 pad 行——MoonEP 把每个 rank 的派发顶到容量，
`发送量 + 注入的 pad ≡ 容量`，把它加进 rank 负载等于数「派发过程的形状」而不是「谁在算」。

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
| 每格 (步, rank, 层) 的 `counts` 之和 ≈ `capacity × topk`（本次 slot 腿每格 768 = 容量 128 token × topk 6），而不是 `T × topk` | pad 把每步的输入撑到容量 |
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
  宽度 = **全部逻辑专家**（本次运行 256，本地口径只有 32）。

**引擎侧语义完全不变**：`tpe` 本来就是 plan **不消费**的量（Ascend 路径上 plan 自己从 `topk_idx`
重建直方图，见 `kernels/planning.cpp`；`ops.py` 也注明 "validated but not consumed"），
交给打点器不影响任何计算；pad 照旧写、plan 照旧算、EPLB 看到的负载照旧含 pad——只有 dump 变干净。
这条路径**没有集合通信、没有跨 rank 同步**，网格不变。

#### 分析器怎么用它

各 rank 的发送侧直方图按 `(step, layer)` 相加 = 这一步的**真实全局路由** `c[t,l,e]`，然后：

- **专家负载（热度）** = `c` 按逻辑专家（全局 id）求和——副本算同一个专家，pad 已被打点侧扣掉，
  所以这张表就是「这批输入下 token 在各逻辑专家之间的热度」，逐专家精确；
- **rank 负载** = 见上一节的拆分规则：`scope=compute` 行给出「谁算了哪个专家多少行」（含 pad），
  其中落在 pad 专家上的那一块按规则还原成真实行，其余行原样计入真实行；
- 逐 cell 自查三条恒等式（真实行总量 = 路由总量、pad 行总量 = 注入的 pad、
  合计行 = `capacity × topk`）与「逐层平坦」。

off 腿（其它 method、以及带 `scope=local` 的旧 dump）没有 compute 视图：`local` 行本身就是
「本 rank 算的行」，pad 恒 0，三本账退化成两本（真实行 ≡ 合计行，pad 列显示 `–`）。

每份 dump 的头部会写明数列口径：`counts_scope=global`（发送侧，MoonEP 新 dump）、
`counts_scope=global+compute`（发送侧 + 实际计算行，MoonEP 的三本账）、
或 `counts_scope=local`（接收侧，其它 method 与旧 dump）。分析器按这个字段分流，
旧 dump 仍按本地槽位解读、并用下面的指纹识别告警。

收尾行会带上扣除记录：`# meta partial_steps=0 pad_pairs_subtracted=NNN pad_expert_global=I`
（每个 rank 只记自己扣掉的那部分；报告 T6 表按 rank 列出，并在报告最上方的「口径与统计」面板里
给出总量与占比；带 compute 行的 dump 另有三项对账计数，见上一节末尾）。
这个总数**只累计已经落盘的那些行**（和 `pad_rows_total` 一样在落盘时结算），所以它和
「所有 `scope=pad` 行的和」按定义相等，分析器的对账不会因为「note 比落盘早一层」而报假警。

两条边界：

1. **`MOONEP_PAD_EXPERT0=1` 时**pad 全部发给全局 0 号专家，只有 rank 0 收得到；其它 rank 的直方图里
   没有这一桶，扣除会落到不存在的格子上，打点侧记 `pad_notes_misplaced` 并在报告里告警。
   要精确计数就用默认的 pad 落点（别设这个变量）。
2. **冗余专家/预取槽**：有副本时一个逻辑专家的路由由多个持有者分摊，发送侧只知道「这个专家被
   路由了多少」，不知道副本各摊了多少。现在的口径不需要知道：**负载记的是各 rank 实际算的行**
   （`scope=compute`），谁算的谁背，副本分摊自动精确；唯一需要约定的只有 pad 专家上那 2.5%
   （见上一节）。旧 dump 没有 compute 视图时，`local` 行本身就是本 rank 收到的行，同样不受影响。

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
- 带 compute 行的 dump 还会多三项：`pad_rows_total`（写了几行 `scope=pad`，只在有 pad 行时写）、
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
- **只有发送侧路由行的 MoonEP dump（`counts_scope=global`，没有 `scope=compute` 行）没有 rank 维度**：
  它照样能回答专家维度的问题（含 pad 落点——`scope=pad` 行在），但「各 rank 实际算了多少」读不出来。
  分析器**不会**按专家属主把路由分摊回 rank（那是另一次实测否定过的口径），而是把整条 rank 维度标成
  **不适用**：`Σ_r` 那类自查（逐层平坦、T2 的①、T6 的三条恒等式）显示「不适用（这份 dump 没有
  rank 维度）」而不是 0 或通过；分层账本的「计算量 = 真实行 + pad 行」一格显示 `–`；逐 rank 的柱状图
  不画、只留一句说明；发送量（各 rank 自己发出的真实路由）照常精确。要让 rank 维度可用，就用现在
  的打点重跑一次（`moonep_comm.py` 已经写 `scope=compute` 行）。

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

- dump 侧：`--flush-every 1`（本仓脚本 `scripts/dsv4flash_service/start_service_mode.sh` 的负载腿已默认设成 1）。
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

有 compute 行的 dump（`counts_scope=global+compute`）不再打一行，而是把**三本账**各打一行，
后面再跟一行账本汇总（下面是 slot 腿的实跑输出，`serve_off` / `serve_slot` 两条腿一起分析）：

```
[moe-par] off: 8 ranks x 31 steps x 43 layers | counts local (local slots) | cross-rank join flat (spread 0.00e+00)
[moe-par] off (真实行): PAR per-layer mean 2.363 max 2.679 | PAR per-step (whole step) mean 1.193 | PAR whole run 1.059 | 总量 248196 对
[moe-par] off (pad 行): PAR per-layer mean n/a max n/a | PAR per-step (whole step) mean n/a | PAR whole run n/a | 总量 0 对  [这份 dump 没有 pad 行]
[moe-par] off (合计行): PAR per-layer mean 2.363 max 2.679 | PAR per-step (whole step) mean 1.193 | PAR whole run 1.059 | 总量 248196 对
[moe-par] off 账本: 逐 rank 送出真实路由 248196 对（送出 PAR 1.059） → 实际计算 248196 对 = 真实行 248196 + pad 行 0 | pad 占计算量 0.0%
[moe-par] slot: 8 ranks x 31 steps x 43 layers | counts global+compute (logical experts) | cross-rank join flat (spread 0.00e+00) | MoonEP pads subtracted 7942788 (global ids)
[moe-par] slot (真实行): PAR per-layer mean 2.276 max 2.705 | PAR per-step (whole step) mean 1.158 | PAR whole run 1.083 | 总量 247164 对
[moe-par] slot (pad 行): PAR per-layer mean 1.023 max 1.318 | PAR per-step (whole step) mean 1.005 | PAR whole run 1.002 | 总量 7942788 对
[moe-par] slot (合计行): PAR per-layer mean 1.000 max 1.000 | PAR per-step (whole step) mean 1.000 | PAR whole run 1.000 | 总量 8189952 对  [容量对称的构造结果，不是均衡证据]
[moe-par] slot 账本: 逐 rank 送出真实路由 247164 对（送出 PAR 2.313） → 实际计算 8189952 对 = 真实行 247164 + pad 行 7942788 | pad 占计算量 97.0%
[moe-par] wrote ../scripts/report.html (963 KiB)
```

- `counts global (logical experts)` / `counts global+compute` / `counts local (local slots)`：
  这份 dump 的数列口径，取自头部 `counts_scope`。第一种是 MoonEP 的发送侧直方图（按逻辑专家 id），
  第二种在它之外还带一套「实际计算」行，第三种是各 method 上报的接收侧本地计数。
- 三行 `(真实行) / (pad 行) / (合计行)` 是**同一套算术作用于三个数组**（见 §6），所以三组数字的
  差别只来自数据。`合计行` 后面那句明写它是构造结果：MoonEP 每个 rank 每 (步, 层) 都算
  `capacity × K` 行，所以它的 PAR 必然是 1.000——**真信息在真实行与 pad 行的拆分里**。
- 上面 off 腿那份是 `counts_scope=local`：它只有**一本账**，pad 行显示 `n/a` 并注明「这份 dump
  没有 pad 行」，真实行 ≡ 合计行。`账本` 那一行的 `pad 占计算量` = pad ÷ (真实 + pad)。
  注意 slot 腿那一行：**送出真实路由 247164 对**（送出 PAR 2.313）与**实际计算 8189952 对**
  （真实行 PAR 1.083）是两个完全不同的问题——前者是 gate 给的形状，后者是谁真的在忙。
- 若 MoonEP 的 dump **只有发送侧路由行**（`counts_scope=global`，没有 `scope=compute` 行），三行 PAR 会
  全打 `n/a`、总量打「n/a（没有 rank 维度）」，`账本` 行改打「这份 dump 没有计算行，实际计算量无从得知；
  打点侧扣掉的 pad 是 N 对（落在哪些专家上照常可知）」，并在 stderr 给一条警告：专家维度照常可用，
  rank 维度无从得知——本工具只按「谁算的」记账，不按专家属主分摊（见 §6「收尾与边界情况」）。
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

三者的分子分母都按 §6「负载口径」数：**专家按逻辑专家；rank 只有一套口径——谁算谁背**，
只是同一套算术分别作用在**真实行 / pad 行 / 合计行**上（见下）。

注意：PAR 是**路由（gate）的性质**，不是通信方式的性质——同一批 token 无论由哪个 all-to-all 承载，
都会去同一批专家。两次运行之间的 PAR 差异主要来自 batch 组成（每步多少 token），所以报告专门画了
「PAR 与每步 token 数的关系」。

### 8.2 三本账怎么读

报告右上角（以及 KPI 卡片上方）有 **真实行 / pad 行 / 合计行** 三个按钮，点哪个，**整页**（KPI、
箱线、折线、逐 rank 表、逐 step 表）就切到哪本账——三本账的数据结构完全相同，所以切换时形状是可比的，
差别只来自数据。页面上每处标题都会带上当前账本名，不会让人读错账。

| 账本 | rank 负载数的是 | 回答什么 | 报告里的位置 |
|---|---|---|---|
| **真实行**（默认） | 这批 token 真正路由出来的对里，被该 rank 算掉的 | **谁真的在忙**——同步屏障前谁在等；搬迁（slot）到底把负载搬平了没有 | 主行、主柱、主结论 |
| **pad 行** | 该 rank 算掉的 pad 零行 | pad 落点是否均匀。它不携带 token，但占的算力是真的，而且**可以被摊得更平** | 同一套图表的第二本账 |
| **合计行** | 真实 + pad | 该 rank 的总工作量。**MoonEP 腿上恒为 1.000**，是构造结果 | 用来读「pad 把差异抹平了多少」 |

读法：

1. **先读真实行的 PAR**（默认那本）——那是要解释的现象，也是搬迁与丢弃的净效果所在。
2. **再看 pad 行**：如果 pad 行的 PAR ≈ 1.000 而真实行很高（本次 slot 腿：真实行逐层 2.276、
   pad 行逐层 1.023），说明**pad 落点是均匀的、不均匀的是真实流量**——问题在专家热度/放置上，
   不在 pad 的摊法上。
3. **合计行的 1.000 不是好消息**：池化容量对所有 rank 对称，pad 把每个 rank 补齐到同一个数，
   所以只要 pad 占大头，逐层/整步/全程的合计 PAR 就都精确等于 1.000——那是容量对齐的**构造结果**，
   真实路由的差额全被 pad 抹平了。**看到 1.000 要去看真实行**，而不是下结论说负载均衡。
   DP8 小 batch 实测（`--max-num-batched-tokens 16384`、每步约 1 个 token、43 层）：每 rank 每层恰好
   算 768 对 = 128（池化容量下限）× 6（topk），pad 占计算量 **97.0%**，合计 PAR = 1.000，
   而真实行逐层 PAR 均值 2.276、全程 1.083。**同一批 token，送出 PAR 是 2.313**——
   这三个数放在一起才说明问题：不均匀的是送出与真实行，pad 与容量把它们压平成 1.000。

**逐 rank 表怎么读**（「每个 rank 的负载」面板）：`送出真实路由` 与 `注入 pad` 两列是**中性参照**，
`真实行 / pad 行 / 合计行` 三列才是该 rank 的负载。`送出` 与 `真实行` 讲的是完全不同的问题：
送出是各 rank 自己的 token 路由出来的对（slot 腿全程 `71466 … 17544`，极不均匀），
真实行是这批 token 派发之后各 rank 实际算掉的（`28667 … 33461`，平得多）。
**两者的差不是「搬迁搬掉了多少」**——派发把不均匀变成了另一种不均匀，中间还有丢弃。
off 腿可以拿来做对照：没有搬迁、没有 pad 时 `送出 ≡ 真实行 ≡ 合计行`
（本次两条腿的逐 rank 数字逐位相等），这时三本账才真的是同一件事。

分层账要在「分层账本」面板里读：**发送量 = 真实路由 + pad**，**计算量 = 发送量 − 丢弃**；
T3 的专家维度三个数都精确，rank 维度给出三本账（真实行 / pad 行 / 合计）也都精确，
唯一的约定在 pad 专家上（见 §6）。**pad 算计算量**——它是零行，但零行同样进 matmul、
同样占算力，所以要单独列出来看清楚它占了多少，而不是把它藏进「真实行」里。

### 8.3 图表清单

| 图 | 回答什么 |
|---|---|
| **口径与统计 · 先读这里** | 报告最上的面板：T0 两个维度的说明（rank 维度「谁算谁背」、专家维度「逻辑专家热度」）、T1 负载口径表（每个量数的是什么、数据从哪来、含不含 pad）、运行级口径徽标、冗余边界、T2 统计与自查表（含三条恒等式）、T6 pad 记账表 |
| **三个 PAR 口径 · 结论先看这里** | 报告的结论区，也是**三本账的切换按钮**（真实行 / pad 行 / 合计行）：切换后整页跟着变。逐层/整步/全程三个 PAR 并排 + 逐 rank 堆叠条 |
| KPI 卡片 | 结论速览：三本账各自的逐层/整步/全程 PAR、均衡度、每步 token 数、rank × 专家规模、热度 top-N 占比 |
| **分层账本** | T1 总账（真实路由 / pad / 发送量 / 丢弃 / 计算量 = 真实行 + pad 行）、T2 按 rank（送出 / 注入 / 三本账 / 真实占比 / 合计−送出）、T3 按专家（真实 + pad 两列） |
| 怎么读这份报告 · 术语表 | 报告里出现的**每一个词**的说明（含「真实行 / pad 行 / 合计行」「发送量/计算量」「pad 计算量」「丢弃」「预取槽」），外加「报告地图」——想看什么该看哪张图 |
| 每个 rank 的负载（跨步均值 · 账本） | 各 rank 平均扛了多少；虚线是均值，峰值与均值之比就是整步 PAR。表里另有「所持逻辑专家」「送出真实路由」「注入 pad」「真实占比」「相对最忙」几列；**表随账本切换** |
| 各 rank 负载随步变化 | 每条线一个 rank；最高线与最低线之差 = 同步屏障要为此付的代价 |
| PAR 与每步 token 数的关系 | PAR 随步的波动 × 该步 token 数：PAR 是路由的性质，看它跟 batch 组成怎么一起变 |
| 整个服务过程（全程累计） | 全程总账柱状图；合计账本下各柱等高，正是容量补齐的构造结果 |
| **逻辑专家热度**（每个运行一块） | 这批输入把哪些**逻辑专家**喂饱了：4 张 KPI（Gini / 归一化熵 / top 1% 占比 / 冷专家数）、top-40 柱状图、累计曲线 vs 均匀对角线、最冷专家表。**这是「token 在逻辑专家间的热度差异」的正面回答** |
| 谁在算哪个专家：rank × 逻辑专家 | 一行一个 rank、一列一个逻辑专家，格子 = 该 rank **实际算过**的对数（含 pad）。行和 = 该 rank 的负载，列的深浅 = 哪些专家热 |
| 层 × 逻辑专家热度 | 同样的流量按层拆开：某层的深色带说明它主导了逐层 PAR |
| 专家负载剖面：单层（可选层） | 单层内按负载降序排列的专家柱状图，颜色按「高于/低于该层均值」发散着色，表里给出持有 rank、占该层比例与「热点/正常/偏冷」判定 |
| 专家负载剖面：单步（可选步） | 单步内的同类视图，标题里给出该步 token 数与通信方式 |
| 每个专家住在哪个 rank（由负载推断） | 专家放置矩阵：每个（层, 专家）归属哪个 rank，静态放置与「谁真的算了它」并排。无副本时这是一个划分，正是 EPLB 会去搬动的形状 |
| 各运行的峰均比 | 多份 dump 的箱线对比（逐层 / 整步）+ 汇总表，以及三本账的逐项对照（怎么算 / 真实行 / pad 行 / 合计行 / 均衡度）；结论：PAR 相同时说明差异来自 batch 组成而非通信方式 |
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
再由分析器拼回来，断言 `layer_expert` / `pad_subtracted` 与各 rank 的计算行全部精确复原。
同一组用例还覆盖**三本账**：写出带搬迁（部分行落到别的 rank）与 pad 的接收侧计数，
断言三条恒等式成立（`Σ_r 真实行 == 路由总量`、`Σ_r pad 行 == 注入的 pad`、
`Σ_r 合计行 == 容量 × topk`）、真实行 == 各 rank 真算的行去掉 pad、
pad 落在 pad 专家上且**归该专家的持有者**、`has_compute=False`（off 腿）时 pad 恒 0 且真实行 ≡ 合计行、
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
   「口径与统计」面板里，T2 的自查全绿（含三条恒等式）、T6 表给出每个 rank 的 pad 占比。
   若报告反而出现「成因是 MoonEP 的容量 pad」那条告警，说明这份 dump 是**修正前**的产物，
   绝对负载与 PAR 不可用，必须重跑。
7. 带 compute 行的 dump 另有三项对账：收尾行 `pad_rows_total` / `compute_rows_total` /
   `compute_pairs_total` 应与报告「分层账本」里的行数、对数一致（不一致分析器会告警）；
   **三条恒等式**（`Σ_r 真实行 == 路由总量`、`Σ_r pad 行 == 注入的 pad`、
   `Σ_r 合计行 == 容量 × topk`）逐 (步, 层) 严格成立——本次两条腿实测误差都是 `0.000e+00`；
   合计行的 `容量` 读的是**每个 (rank, 步, 层) 格子**：本次 slot 腿每格恒为 768 对 =
   池化容量 128 token × topk 6，各格完全相等，即 `capacity_pinned = true`（报告就写在 T6 上；
   旧版是「按 (步, 层) 把各 rank 加总再看」，那个和与负载倾斜无关，任何 dump 都恒等，
   所以它既不会不通过、也不能说明任何事）。`compute_pairs_total` 还可以和
   `[moe-par-ledger] … compute=…`（应与收尾行严格相等）以及派发侧打印的
   `[moonep-balance] … rows_local=…` 对上（和派发侧对账前先按行里的 `layers=` 对齐覆盖到第几步，见 §6）。

顺带一提，**报告里合计行恒为 1.000 不是 bug**：它对每个 (步, 层) 断言「各 rank 算的行数完全相同」，
是容量补齐的定义。要判断负载平不平，看真实行（以及 pad 行落点是否均匀）。

---

## 12. 已知边界与坑

1. **一个进程只能建一次 `LLM`**。打点模块的状态（层映射、步计数、打开的文件句柄）是模块级的，
   不会重置；同进程里建第二个 `LLM` 会按旧的层映射往同一份 dump 里追加。要跑多个配置就多起进程。
2. **不同层的 `local_num_experts` 不一致**时，宽度对不上的层会被跳过并告警一次（本模型所有层一致）。
3. **`MOONEP_SLOT_MODE` / `MOONEP_DISABLE_UDMA` 不进 tag**：两次都开 `--moonep`、只差 slot 模式的运行
   会得到同样的 tag，从而写进同一个 dump 目录。要么显式 `--tag`，要么给 `--dump-dir`。
4. **报告体量**：2–3 MB 的内联数据，浏览器打开正常，但不要放进聊天工具里逐条 diff。
5. **本工具不开启、也不修改 EPLB**。`dynamic_eplb` 保持关闭。冗余专家（副本）不影响 rank 口径：
   专家的负载按逻辑专家记（副本算同一个专家），rank 负载记的是**各 rank 实际算的行**
   （`scope=compute`），副本分摊自动精确、不需要猜。唯一需要约定的只有落在 pad 专家上的那 2.5%
   （见 §6）。旧 dump（非 MoonEP、无 compute 视图）用 `local` 行，那本身就是本 rank 真算的行，
   同样不受影响；此时 pad 恒 0、报告会标注。
6. **pad 的扣除只发生在 dump 侧**：走 moonep 时 EPLB 看到的 `moe_load` 仍然含容量 pad（
   见 §6「MoonEP 的容量 pad」）。也就是说报告的绝对负载是真实路由，而 EPLB 优化的是含 pad 的那个量；
   两边不是同一个数，比较时不要混用。另外打点侧只**读**直方图与计数、不改 `group_list`、
   不改 plan 的输入（`tpe` 本来就不被 plan 消费），所以 MLP 与调度的计算结果与不开打点完全一致。
7. **接收侧不适合做专家路由账**：派发 kernel 在目标 chunk 越界时会静默丢行
   （`moonep_dispatch.cpp` 的 `if (dst_loff >= NvS) continue;`），本 build 不写 `plan.remote_stats`，
   丢了多少本地不可知。所以**路由量**只认 `global`（发送侧）那本账，`local` 口径只用于「实收」的
   近似观察。丢弃只能作为残差知道（`发送量 − 计算量`），报告在分层账本 T1 里单独列出来。
   本 build 的 plan 也不做跨 rank 搬行（`CAP = NvS`，`balance ≤ 0` 直接 `break`），
   预取槽 `experts_to_copy ≡ -1`，`cu[E:]` 全 0——这时真实行与**送出**在 rank 轴上几乎一样
   （差别只剩丢弃）；只有 slot 模式（`MOONEP_SLOT_MODE=1`，`experts_to_copy != -1`）
   才会出现真实的搬迁，`scope=compute` 行那时才带来「谁被搬平了」的新信息。
   off 腿的实测就是前一种：`送出 ≡ 真实行 ≡ 合计行`，逐 rank 逐位相等。
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
