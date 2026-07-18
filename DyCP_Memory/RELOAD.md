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

（`RELOAD.md` 是操作指南，不要放进 memory 目录。）

### 4. 登记索引（关键，否则不会被召回）

agent 只会加载 `$MEM/MEMORY.md` 里列了指针的文件。把以下几行追加进去
（若已存在同名行则跳过，避免重复）：

```
- [DyCP 方案原理](dycp-design-principles.md) — 长短分流、CP子组拓扑/路由、CPAwareScheduler子组共识、DP全组wave状态机
- [Git 提交规则](git-commit-conventions.md) — 中文 message、记录解决的问题、不加 Co-Author、只提交相关文件
- [记忆与仓库同步](memory-sync-to-repo.md) — 更新 agent 记忆时同步更新仓库 DyCP_Memory/，保持一致
- [vLLM 问题分析工作方式](vllm-debug-evidence-based.md) — 基于代码+日志取证不瞎猜，日志不足先确认加日志、由用户跑实验提供数据
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