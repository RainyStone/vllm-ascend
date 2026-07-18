---
name: git-commit-conventions
description: vllm-ascend 仓库提交规则：中文 commit message、记录解决的问题、不加 Co-Author
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cbe0cf31-4bdd-4917-a180-762ad0763612
---

在 vllm-ascend 仓库（仓库根目录名 `vllm-ascend`，其绝对路径随环境变化，不要写死）做提交时遵守以下规则：

1. **commit message 用中文**。
2. message **尽量简洁**。
3. message 正文要**描述本次提交解决的问题以及怎么解决的**（根因 + 修复要点），不要只写空泛标题。
4. **不要添加 Co-Authored-By 行**（与默认行为相反，用户已明确要求不加）。
5. 工作分支与历史 commit message 风格：历史风格为 `fix(DyCP): 中文描述`/`docs(...): 中文描述`，沿用此格式；在哪个分支提交以当时 `git branch --show-current` 为准，不写死分支名。
6. **提交前先与用户确认本次要提交哪些文件**：列出计划 `git add` 的文件清单与排除项，等用户确认后再 add+commit，不要自行决定提交范围（尤其该仓库常有未跟踪的 backup/第三方 cmake 文件等无关改动）。

**Why:** 用户对提交规范有明确偏好，且与默认 Claude Code 提交模板（英文 + co-author）不同。
**How to apply:** 每次 `git commit` 时按此规则写 message；只 add 与本次修改相关的文件，避免带入工作区里的无关改动（该仓库常有未跟踪的 backup/第三方 cmake 文件）。相关：[[vllm-debug-evidence-based]]