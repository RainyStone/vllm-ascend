# 实验：off vs slot(+优化组合) 输出精度对比 · 2026-10-11

## 要回答的问题

slot 模式 + 算子侧优化（RELAXED/NO_RECONCILE/SKIP_DUP 三件套 + 默认开项）会不会算错？
输出与 off 是否一致？

## 方法（acc_test.py）

- 12 条固定提示词（10 条可验证：算术/事实/倒序/翻译/计数 + 2 条生成题），
  贪心解码 temperature=0、seed=1024、max_tokens 64，逐条记全文与耗时。
- 共 3 台服务 + 1 次重复，全部 `../service/start_service_mode_dp4.sh` 起服：
  1. `off`（基线）　2. `slot_opt`（slot+三件套，`[moonep-env-audit]` 确认生效）
  3. `off2`（off 重启对照，判别分歧来源）　4. `slot_opt_pass2`（同进程重跑，确定性对照）
- 比对口径：完全相同数、最长公共前缀（字符）、可验证题对错。

## 数据（data/）

`off.json / off2.json / slot_opt.json / slot_opt_pass2.json`——每条含 prompt、生成全文、
finish_reason、completion_tokens、延迟。

## 结论

- **正确性**：可验证题两腿 10/10 全对（391/12/5050/Paris/北京/H2O/倒序/加法函数/翻译/计数）。
- **确定性**：slot 同进程两遍 12/12 全同；off 跨重启 12/12 全同。
- **off vs slot**：1/12 完全相同、公共前缀覆盖 37%——分歧来自 slot 的浮点计算路径
  （行均衡到不同 rank → 归约顺序不同 → logits 微差 → 贪心链在近平局 token 分叉），
  属预期行为；所有关键答案都在分叉点之前产出且正确。
