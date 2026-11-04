# LongBench-v2 长输入上限评估（待确认）

日期：2026-11-04（Asia/Shanghai）。本阶段只执行 CPU tokenizer 统计和静态源码分析，未拉起 vLLM、未发送压测请求。

## 已确认

- 模型：`/mnt/share/weight/DeepSeek-V4-Flash`，DeepSeek-V4-Flash，模型上下文配置上限 1,048,576 token（1024K，1K=1024）。
- 目标：aisbench 流式性能测试，比较 off / MoonEP Slot；服务 max-num-seqs=16，客户端 batch_size=16，max_out_len=1，chunked prefill，max-num-batched-tokens=8192。
- 服务起服与压测入口均使用独立 sh 脚本，参数和环境变量在脚本内明确设置。脚本及 aisbench 配置全文需交用户确认；本阶段不执行实验。
- 暂沿用 8 卡 DP8/TP1/EP8 进行可行性分析；最终脚本内容仍待用户确认。

## 数据长度

数据源：`/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/LongBench-v2/data.json`。
SHA256：15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2。
503 条；使用当前模型 tokenizer.json、aisbench 原始 LongBench-v2 模板完整格式化后的 prompt、plain completions 接口口径，包含 tokenizer 特殊 token 处理；最终运行时应通过服务端 tokenize/usage 再核对。

最短 10,285，P50=100,414，P90=484,608，最长 4,643,843 token。

| 输入 token 上限 | max_model_len（预留1输出token） | 可原样保留样本 | 超限样本 | 8K chunk 数上界 |
|---|---:|---:|---:|---:|
| 32K（32,768） | 32,769 | 116 / 503（23.1%） | 387 | 4 |
| 64K（65,536） | 65,537 | 190 / 503（37.8%） | 313 | 8 |
| 128K（131,072） | 131,073 | 316 / 503（62.8%） | 187 | 16 |
| 256K（262,144） | 262,145 | 407 / 503（80.9%） | 96 | 32 |
| 512K（524,288） | 524,289 | 455 / 503（90.5%） | 48 | 64 |
| 1024K（输入最多1,048,575） | 1,048,576 | 473 / 503（94.0%） | 30 | 128 |

“chunk数上界”仅为单请求输入长度除8192向上取整；实际调度会被其它并发请求共享预算，并不保证每步只处理一条请求。

## 显存评估与限制

DSV4 Flash 使用滑窗与压缩 KV。不能用旧日志“4 GiB / 等效10405 token”直接线性推算长上下文：旧日志的等效容量依赖当时 max_model_len 和混合缓存规划。

历史同配置 Slot 的 npu_ready 快照：每卡总 HBM 98,304 MiB（96 GiB），占用约60,956～61,009 MiB（约59.5～59.6 GiB），其中固定KV预算4 GiB。按其它占用暂时不变的粗略账面估算：KV改为24 GiB，余量约16.4 GiB；改为32 GiB，余量约8.4 GiB。这些余量不是实测峰值，也未扣除长上下文新增工作区。

初步建议：优先128K或256K，off / slot 两组统一24 GiB/卡 KV（25,769,803,776 bytes），保持8K chunk预算。该配置需要后续获准执行时验证服务启动、最长输入和并发16请求，不作为可运行保证。

512K与1024K可作为后续探索：初始可考虑32 GiB/卡，但较小的运行时余量使OOM/抢占风险更高，暂不直接定为正式实验上限。

DP8/TP1：单条请求的 KV 不会被8个DP engine平均切分。客户端全局并发16时，通常需要每个engine容纳约2条请求；服务 max-num-seqs=16 是每个engine的调度上限。

`evaluate_context_limits.py` 从当前源码抽取纯数据结构与规划函数，以重建布局估算增长趋势，未加载权重或创建NPU tensor。该重建尚未复现旧4GiB/8K启动现象，context_limit_estimates.json中的数字不是测量结果，不能直接用作正式KV预算依据。

## 待确认的数据策略

A. 仅选完整prompt不超过所选上限的原始样本：保持原文，无截断；报告中必须注明样本子集和长度分布。
B. 保留503条，对超长context做头尾截断，保留问题和选项并重新核对完整prompt长度：这是LongBench-v2派生性能集，不能称为原始全文测试，也不做准确率结论。

两组必须共享同一冻结数据、同一顺序和prompt模板。原始数据不修改。当前仅统计，未生成截断或筛选后的数据文件。

## 其它设计项（仍待确认）

- 建议原生专家路由，彻底清除模拟路由环境变量；沿用Slot三项优化与REAL_T，关闭持久专家缓存、提前预测预取与独立流预取。
- 建议两对独立服务，顺序off→slot、slot→off；每次预热后测3轮，首轮保留但不计均值，取后两轮。用户尚未回答此重复次数选择。
- 两组都关闭prefix cache，避免重复输入和预热改变prefill计算量。
- aisbench：stream=True、batch_size=16、max_out_len=1、request_rate=0、retry=0、temperature=0、ignore_eos=True；`--mode perf --debug`，通过tee保留终端原始日志。
- 当前aisbench的性能原始明细由源码写入`performances/<model_abbr>/*_details.jsonl`，汇总与逐请求指标写入性能结果目录，完整保留整个work_dir。
- 输出1token主要测prefill，但TTFT仍包含排队、通信、首token计算和网络开销，不能当成纯prefill kernel计时。

## 下一步

用户先确认输入上限与超长处理方式。随后按选定参数写出两份独立服务sh、两份压测sh和隔离的aisbench配置，交用户逐项审阅。未经用户通知，不启动服务或执行压测。
