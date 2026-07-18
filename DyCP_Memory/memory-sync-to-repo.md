---
name: memory-sync-to-repo
description: 更新 agent 记忆时需同步更新仓库 DyCP_Memory/ 目录，保持仓库记忆与实际记忆一致
metadata:
  type: feedback
---

agent 记忆系统存储在 /root/.codefuse/engine/cc/projects/-sfs-turbo-hw-xiazhixiang/memory/（按工作目录哈希生成，不在 vllm-ascend 仓库内），换环境/重新克隆即丢失。为便于跨环境 reload，关键记忆已同步到 vllm-ascend 仓库的 `DyCP_Memory/` 目录（如 `dycp-design-principles`、`git-commit-conventions`、本条）。

**Why:** memory 目录非仓库托管，换环境即丢失；只能靠仓库副本 reload。若两边不同步，仓库记忆会与实际 agent 记忆脱节，reload 出来的内容是过时的。

**How to apply:**
- 每次更新或新增 agent 记忆时，若该记忆属于已纳入 `DyCP_Memory/` 的范畴，需同步修改仓库 `DyCP_Memory/` 下对应文件并随相关改动一起提交。
- 新增可纳入仓库的记忆时，复制到 `DyCP_Memory/` 并提交。
- 仅纳入已明确要求的记忆，不擅自把所有 memory 都塞进仓库。
- reload：把 `DyCP_Memory/` 下各文件拷回本地 `memory/` 目录，并更新 `MEMORY.md` 索引。

相关：[[dycp-design-principles]] [[git-commit-conventions]]