# LongBench-v2 原始全文 prefill 对照实验

> 当前进展（2026-11-04）：融合加权归并修复版完成 1 台 Off、2 台 Slot，共 9 轮、2844 条正式请求；最后一台 Off 的 3 轮按用户要求暂停。轻量结果见 `reports/run_20261104_202555_weighted_combine_fixed/`。
> 下文完整四服务、12轮、3792条请求指修复前历史基线，不代表修复版已完成完整两对。原始全文、debug日志和SQLite继续保存在共享盘 `data/`，不纳入Git。

2026-11-04：修复已获用户确认，完整四服务、12轮、3792条正式请求均已完成并校验通过。当前执行目录：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/data/executions/run_20261104_122031_literal

## 已确认配置
- DSV4 Flash：/mnt/share/weight/DeepSeek-V4-Flash。
- 128K以内316条原始全文，不截断，固定顺序。完整prompt长度10285～130285 token，max_model_len=131073。
- 8卡DP8/TP1/EP8；服务max-num-seqs=16，客户端全局并发16；chunked prefill，max-num-batched-tokens=8192。
- off / Slot统一24GiB KV/卡；enforce-eager；关闭prefix cache；原生专家路由。
- Slot开REAL_T和三项优化，关闭持久专家缓存、提前预取与独立流预取。
- 流式/v1/completions，max_out_len=1，temperature=0，ignore_eos=True。当前aisbench的retry=1表示总共只尝试一次，不自动重试。
- off→slot、slot→off两对独立起服，每台预热16条、测3轮，取后2轮均值。

## 已批准修复与正确性条件
用户已确认采用仅本实验内生效的LiteralLongBenchPrompt，单次插入字段，保留context中的原文字面量花括号。全部316条在CPU逐字验证通过，原始数据和筛选数据均未修改。
每轮发送请求前校验实际服务命令、8个worker环境、MoonEP库映射和Slot初始化日志；每轮结束校验316条成功、输出1token、输入token数和完整prompt SHA256。
服务启动仍由两份独立sh执行；两份aisbench sh均有--debug并用tee完整保存终端输出。参数均写在脚本或对应配置文件内。

## 进度和原始数据
- CURRENT_EXECUTION.json：当前执行目录。
- 执行目录remaining_state.json / remaining_events.jsonl：四服务的实时阶段。
- 执行目录cases/<pair>_<mode>/case_result.json：完成服务的结果与原始数据指针。
- data/services：独立起服日志、实际环境、PID和模式核验。
- data/aisbench：每服务三轮的debug日志、完整JSONL、关联SQLite以及汇总CSV/JSON/HTML，全部保留。
- 脚本会在错误时停下并只清理自己启动的进程，不自动降低实验参数。

## 历史无效尝试
run_20261104_114303的已完成首轮中，aisbench默认safe_format将3条代码context中的{idx}替换成样本编号，因此整轮排除，不能用于最终性能对照。失败和原始日志保留。该缺陷已按用户确认修复后重开当前执行。

## 结果
已在当前执行目录生成RESULT.md、results.json与rounds.csv；八卡和8444端口释放检查见CLEANUP_CHECK.json。

首token文本并非完全一致，同模式重复也有差异；本轮仅提供性能对照，不证明生成精度等价。


## 补充诊断完成

用户已批准并完成off/Slot各316条负载补采和48次固定请求，诊断计时未并入正式性能。报告：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/diagnostics/data/executions/diagnosis_20261104_140807/REPORT.md。
