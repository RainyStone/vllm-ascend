---
name: memory-sync-to-repo
description: 更新 agent 记忆时需同步更新仓库 DyCP_Memory/ 目录，保持仓库记忆与实际记忆一致
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cbe0cf31-4bdd-4917-a180-762ad0763612
---

agent 记忆系统存储在一个**按当前工作目录哈希生成**的本地目录（不在 vllm-ascend 仓库内，路径随环境/工作目录变化，不要写死），换环境/重新克隆后该目录通常是空的，记忆即"丢失"。为便于跨环境 reload，关键记忆已同步到 vllm-ascend 仓库的 `DyCP_Memory/` 目录（如 `dycp-design-principles`、`git-commit-conventions`、本条）。具体如何定位当前环境的 memory 目录、如何拷贝与登记索引，见仓库 `DyCP_Memory/RELOAD.md`。

**Why:** memory 目录非仓库托管，且其绝对路径随环境变化（按工作目录哈希生成），换环境即丢失且无法预知路径；只能靠仓库副本 reload。若两边不同步，仓库记忆会与实际 agent 记忆脱节，reload 出来的内容是过时的。

**How to apply:**
- 记忆中**不要写死 memory 目录绝对路径或仓库绝对路径**，用"当前 memory 目录""vllm-ascend 仓库根"等相对指代；定位方法见 `DyCP_Memory/RELOAD.md`。
- 每次更新或新增 agent 记忆时，若该记忆属于已纳入 `DyCP_Memory/` 的范畴，需同步修改仓库 `DyCP_Memory/` 下对应文件并随相关改动一起提交，保持两边一致。
- 新增可纳入仓库的记忆时，复制到 `DyCP_Memory/` 并提交。
- 仅纳入已明确要求的记忆，不擅自把所有 memory 都塞进仓库。
- reload：按 `DyCP_Memory/RELOAD.md` 的步骤，把该目录下各文件拷回当前环境的 memory 目录，并把每条登记到 `MEMORY.md` 索引。

相关：[[dycp-design-principles]] [[git-commit-conventions]]