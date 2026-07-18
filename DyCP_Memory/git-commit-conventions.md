---
name: git-commit-conventions
description: vllm-ascend 仓库提交规则：中文 commit message、记录解决的问题、不加 Co-Author
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cbe0cf31-4bdd-4917-a180-762ad0763612
---

在 /sfs_turbo/hw/xiazhixiang/vllm-ascend 仓库做提交时遵守以下规则：

1. **commit message 用中文**。
2. message 正文要**记录本次提交解决的问题**（根因、现象、修复要点），不要只写空泛标题。
3. **不要添加 Co-Authored-By 行**（与默认行为相反，用户已明确要求不加）。
4. 仓库当前工作分支为 `dycp_v0.21.0rc_refactor`，历史 commit message 风格为 `fix(DyCP): 中文描述`，沿用此格式。

**Why:** 用户对提交规范有明确偏好，且与默认 Claude Code 提交模板（英文 + co-author）不同。
**How to apply:** 每次 `git commit` 时按此规则写 message；只 add 与本次修改相关的文件，避免带入工作区里的无关改动（该仓库常有未跟踪的 backup/第三方 cmake 文件）。相关：[[vllm-debug-evidence-based]]