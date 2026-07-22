# DyCP_Memory 跨环境 Reload 指南

本目录是 agent 记忆的**仓库副本**。agent 的记忆系统不读仓库，它读一个
**按当前工作目录哈希生成的本地 memory 目录**，该目录绝对路径随环境/工作
目录变化（不能写死），换环境或重新克隆后通常是空的。要在新环境恢复记忆，
需把本目录的文件拷回**当前环境的 memory 目录**并登记索引。

## 步骤

### 1. 定位当前环境的 memory 目录

memory 目录在 `~/.codefuse/engine/cc/projects/<某哈希>/memory/` 下，其中
`<某哈希>` 由当前工作目录决定，无法预知。用以下任一方法找到它：

```bash
# 方法 A：在 agent 已运行的会话里，系统提示中会给出 memory 目录绝对路径
#          （形如 /.../projects/<哈希>/memory/），直接用即可。

# 方法 B：列出所有 project 哈希目录，找含 memory/ 且对应当前工作目录的那个
ls -d ~/.codefuse/engine/cc/projects/*/memory 2>/dev/null
# 再用 MEMORY.md 的内容/最近修改时间判断哪个是当前工作目录对应的
```

> 注意：同一台机器上不同工作目录会对应不同哈希目录，别拷错。当前会话系统提示
> 里给出的 memory 路径是最可靠的来源。

记下找到的目录，下文记作 `$MEM`：

```bash
MEM=<把上面找到的 memory 目录绝对路径填这里>
```

### 2. 定位仓库副本目录

```bash
REPO_MEM=<vllm-ascend 仓库根的绝对路径>/DyCP_Memory
```

### 3. 拷贝记忆文件

```bash
cp -n "$REPO_MEM"/*.md "$MEM"/
# -n 不覆盖已存在的同名文件；如确定要覆盖用 cp -f
# 注意：不要拷 RELOAD.md 本身，它只是给人在仓库里看的操作指南，不是记忆
```

本目录下作为**记忆**的文件是：
- `dycp-design-principles.md` — DyCP 方案原理
- `git-commit-conventions.md` — git 提交规则
- `memory-sync-to-repo.md` — 记忆与仓库同步规则
- `vllm-debug-evidence-based.md` — 问题分析基于代码+日志取证、不瞎猜
- `pcp-dcp-kv-sharding.md` — 纯PCP/纯DCP方案原理与KV cache切分链路
- `mooncake-connector-pd-transfer-principles.md` — mooncake_connector P/D KV传输原理与PCP/DCP两级CP切分

各记忆作用说明（便于判断是否需要 reload 全部）：

- **dycp-design-principles**：DyCP（Dynamic Context Parallel）方案原理。
  含长短分流（按 prefill token 数与 `long_request_threshold` 比较决定走纯 DP
  还是 CP 子组）、CP 子组拓扑/路由（每 `dycp_size` 个相邻 DP 引擎构成子组、
  owner=cp_rank0）、CPAwareScheduler 子组共识（逐拍 all_reduce MIN 三态共识
  SCHEDULED/NOT_SCHEDULED/PREEMPTED）、DP 全组 wave 状态机（sync_dp_state、
  step_counter %32 优化）。理解 DyCP 代码与排查调度/通信问题时必读。
- **git-commit-conventions**：在 vllm-ascend 仓库提交的规则。① commit message
  用中文；② 正文记录解决的问题（根因/现象/修复要点）；③ 不加 Co-Author；
  ④ 沿用 `fix(DyCP): ...`/`docs(...): ...` 风格、分支以运行时为准；⑤ 提交前
  先与用户确认要提交哪些文件，不自行决定提交范围。
- **memory-sync-to-repo**：记忆维护规则。更新/新增 agent 记忆时，凡属已纳入
  `DyCP_Memory/` 范畴的，需同步改仓库副本并提交，保持两边一致；记忆中不写死
  路径；仅纳入已明确要求的记忆。
- **vllm-debug-evidence-based**：分析 vLLM 服务问题的工作方式。必须基于代码+
  日志取证、不瞎猜，区分"已证实"与"推断"；日志不足时不要继续猜，先与用户确认
  加日志、由用户跑实验提供数据再分析；修复要能从日志验证。
- **pcp-dcp-kv-sharding**：纯 PCP/纯 DCP 方案原理与 KV cache 切分完整链路。含
  Q/KV 切分（PCP 切 Q head-tail + KV 分片存算时 all-gather 聚齐；DCP 切 KV 省
  显存、Q 不切仅 head 维 ag + all-to-all 对账）、interleave slot_mapping 底座、
  scheduler 块膨胀（`single_type_kv_cache_manager.py:59-63`）与 worker virtual
  block 寻址的精确调用链、CP 世界数两套来源（scheduler 不含 dycp / worker 含
  dycp）、PCP decode 执行模型（序列并行非权重并行、Q replica/KV 分片、采样对齐
  无 broadcast 靠 logits 相同 + 确定性采样）。理解 PCP/DCP 代码、排查 KV 切分/
  块记账问题时必读。
- **mooncake-connector-pd-transfer-principles**：mooncake.connector 跨节点 P/D
  KV cache 拉取机制。双信道（ZMQ 握手传元信息 + mooncake TransferEngine RDMA
  传 KV）、逐层地址模型（base_addr/block_len/block_size_scale 三表）、scheduler
  构 ReqMeta 到 worker 的 `start_load_kv` 链路、PCP/DCP 两级 CP 切分
  （`_get_kv_split_metadata`：CP 组拓扑→D↔P port 配对→block 按 CP rank 切分取子集
  →TP 冗余拉取）、拉完仅 `is_group_transfer_end` 的 group 做 reformat（transpose
  还原 [block,split,token,head,dim]→[block,token,split,head,dim]）。扁平
  `dycp_size` 适配后，`_get_kv_split_metadata` 及其后方法尚未按按 cp_ranks 分组的
  block_ids 新结构改造，为后期适配主战场。理解 connector 传输/排查 P/D KV 拉取、
  扁平 dycp 适配时必读。

（`RELOAD.md` 是操作指南，不要放进 memory 目录。）

### 4. 登记索引（关键，否则不会被召回）

agent 只会加载 `$MEM/MEMORY.md` 里列了指针的文件。把以下几行追加进去
（若已存在同名行则跳过，避免重复）：

```
- [DyCP 方案原理](dycp-design-principles.md) — 长短分流、CP子组拓扑/路由、CPAwareScheduler子组共识、DP全组wave状态机
- [Git 提交规则](git-commit-conventions.md) — 中文 message、记录解决的问题、不加 Co-Author、只提交相关文件
- [记忆与仓库同步](memory-sync-to-repo.md) — 更新 agent 记忆时同步更新仓库 DyCP_Memory/，保持一致
- [vLLM 问题分析工作方式](vllm-debug-evidence-based.md) — 基于代码+日志取证不瞎猜，日志不足先确认加日志、由用户跑实验提供数据
- [纯PCP/纯DCP与KV分片](pcp-dcp-kv-sharding.md) — PCP切Q+KV分片算时聚齐；DCP切KV省显存Q不切仅head维ag+all-to-all；scheduler块膨胀与worker virtual block同因子；含PCP decode执行模型与采样对齐
- [mooncake_connector P/D传输原理](mooncake-connector-pd-transfer-principles.md) — 双信道(ZMQ握手+mooncake RDMA)、逐层地址模型、start_load_kv链路、PCP/DCP两级CP切分(split_metadata)与reformat；扁平dycp适配后CP切分未迁为待适配主战场
```

检查并去重：

```bash
sort -u "$MEM/MEMORY.md" -o "$MEM/MEMORY.md"   # 仅当每行一条且无顺序依赖时可用；否则手动核对
```

### 5. 验证

新的记忆要在**下一次会话**才会被加载（当前会话已读入的索引不会热更新）。
重启会话后让 agent 回忆 DyCP 相关内容，确认能召回。

## 反向：更新记忆后同步回仓库

当在某个环境**修改或新增**了已被纳入本目录的记忆，按 `memory-sync-to-repo.md`
的规则，把改动同步回本目录对应文件并提交，保持仓库副本与实际记忆一致：

```bash
MEM=<当前环境 memory 目录>
REPO_MEM=<仓库根>/DyCP_Memory
cp "$MEM/<被改的记忆文件>.md" "$REPO_MEM/"
# 若是新增记忆且决定纳入仓库，同样拷过来
cd <仓库根> && git add DyCP_Memory && git commit   # commit 规则见 git-commit-conventions.md
```

## 注意事项

- 路径都不要写死：`$MEM`、`$REPO_MEM` 按当前环境实际值填入。
- 拷回 memory 目录的文件 frontmatter 中 `node_type`、`originSessionId` 等
  字段由 memory 系统自动注入，缺失不影响加载，不必手动补。
- 仅纳入已明确要求的记忆，不要把整个 memory 目录都倒进仓库。