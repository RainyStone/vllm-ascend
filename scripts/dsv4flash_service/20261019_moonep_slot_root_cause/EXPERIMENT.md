# MoonEP slot / off 性能根因汇总报告（2026-10-19）

本目录汇总本会话已有实验，没有启动新的 NPU 服务或占卡任务。
遵循 dsv4flash_service 的“一个实验一个目录”约定。
报告入口：report/moonep_slot_vs_off_root_cause.html（离线 HTML）。

范围：16K 输入、8K token 预算下 slot 回退；高倾斜度下 8K/8K、8K/16K、16K/16K 的收益。
包含分词、请求就绪、DP 准入、KV 回收与 dummy 相位的因果链，关键路径分账，最小修复与重复验证。
最终性能结论基于 2026-10-17 首次占用告警前的三组 warm 配对，不将不同卡组、调度状态或采样模式拼接。
跨模式数值等价未被认证：仿真路由随 shape 改变，同一 token 的专家选择不一致，详见正文。

## 数据

- data/historical_matrix.json：从旧矩阵的 64 个 c2/c3 JSON 重算，保留源文件 SHA256。
- data/alignment_fix_validation_20261017：对齐修复、重复验证、正确性与原始元数据日志附件。
- data/guarded_validation_20261017：轻量 profile、自然对齐与关键路径边界。
- data/initial_scheduler_validation：首次调度链实机验证。
- manifest.json：本包各文件大小与 SHA256，单文件小于 1MB。

原始完整实验目录：
/home/x00893896/vllm_ascend_moonep_integration/scripts/20261017_scheduler_alignment

## 重建（不使用 NPU）

在本目录运行：python3 gen_report.py

生成器只用 Python 标准库和本地 data；报告不依赖外部脚本、字体、图片或 CDN。
extract_history.py 是历史指标读取脚本，需从集成工程根目录执行；只打印 JSON，不修改旧数据。
实验复现命令见 data/alignment_fix_validation_20261017/reproduce.txt。
重建报告不会运行这些实验命令。

本目录不覆盖旧报告、不修改服务启动脚本、不提交或推送仓库。

## 2026-10-19 图示与补丁使用说明更新

报告新增三张可缩放 SVG 图：切分后相位错位、KV 回收空轮来源、20ms 修复前后对照。
report_enhancements.py 提供图形和补丁使用章节；gen_report.py 重建时同步生成 report/assets/。
补丁用法包含状态检查、git apply、20ms 配置与导入路径检查、单测、配置与源码回退。
当前容器已应用补丁，只读反向检查通过；应用/反向回退命令另在隔离目录验证，不修改运行源码。

## 补丁目录与 slot 自身性能对照

补丁及独立使用说明统一位于直接子目录 patch/，data 中原 .patch 文件已迁出。
slot_comparison.py 从原始 bench JSON 重算第 9.2 节（补充）slot 的 2ms / 20ms 前后均值、逐组结果及同 seed 补充检查。
前 3 组与后 4 组为同服务窗口切换验证，seed 不同，不表述为同 seed 配对；env 补丁重启验证另列。

## 修复后 off / slot 主对照

第 9.1 节以双方 20ms 且对齐后的三组同 seed 配对作为主性能对照。
slot_comparison.py 从六个原始 bench JSON 重新计算绝对 TTFT、输入吞吐、整组耗时与稳态周期，并核对相位、token 总量及占用质量。
data/off_slot_after_fix.json 保存重算结果、统计口径和源文件 SHA256；slot 自身前后变化降为补充。
三组平均 TTFT：off 9.9917s / slot 9.4755s（降低 5.17%）；输入吞吐：off 21343.56 / slot 22585.28 token/s（提高约 5.82%）。
本次仅调整报告和重算归档数据，没有新跑实验；补丁及使用说明仍位于 patch/。
