# s≈2.4822 实验执行记录（2026-11-03）

当前没有可报告的off/slot有效性能配对。

1. data/20261103_141216_015297：首次off启动失败。VLLM_PLUGINS仅包含ascend，屏蔽Ascend模型注册，走到了上游NVIDIA实现，触发FlashMLA FP8 KV断言。已修正插件允许列表并验证AscendDeepseekV4ForCausalLM注册。
2. data/20261103_141611_350344：off的L8192/B8192完成预热和3轮测量，三轮均32请求成功/0失败；slot在9GiB KV cache分配阶段NPU OOM（额外申请348MiB时仅剩约100MiB）。矩阵已停止，无其它形状性能数据。
3. data/startup_20261103_142732_375900_B8192：按用户确认改用5.5GiB KV cache、max-model-len=16385，仅验证slot启动，未发送benchmark请求。容量校验报ValueError：服务16385长度至少需要6.81GiB KV，5.5GiB估算最大长度13234，未就绪。

已完成off基线（仅第二次session、原9GiB KV配置，取第2/3轮）：
- 平均TTFT：3691.693676 ms
- 平均输入吞吐：27915.699720 token/s
- 平均整轮耗时：9.392892865 s
- 每轮输入token总数262144、输出32、请求32、失败0。

这些数字不与后续更改KV/context参数的数据混合。缺失的slot结果未填0，不计算收益。
目前待用户决定：先缩到8K上下文验证，或分析MoonEP额外显存以争取保留16K输入。

## 仅8K的后续尝试

data/20261103_143223_920608：slot-first，8K输入/8K预算，5.5GiB KV、max-model-len=8193。
slot成功就绪，预热2请求成功；真实8K第一轮32个请求全部失败，worker在deepseek_v4.py:989的hidden_states.clone()申请256MiB时OOM（剩余150.76MiB）。本轮不可作为性能数据；off未执行。
下一步已向用户确认是否两组统一降到4GiB KV，负载及并发保持不变。暂未调整或重新启动。

## 4GiB KV尝试

data/20261103_143635_222181：8K输入/8K预算、4GiB KV、并发16；slot起服及2请求预热成功，首轮真实8K请求在npu_rms_norm申请512MiB时OOM（剩余293.91MiB），32请求全部失败。off未执行，无有效配对。
待用户确认下一步：保持8K输入与并发16，将token预算降为4096进行chunked prefill，或先分析额外显存。

## 最新状态：自然路由短请求检查通过，性能实验暂缓

最新用户指令已替代继续调整性能矩阵：先验证slot能否处理正常请求及是否实际生效，性能实验等其它卡空闲。
检查目录data/slot_requests_20261103_144311_229722：两个短请求均HTTP200、各16输出token；自然路由，未注入SIM_ROUTE；四个rank的dispatch/combine/prefetch调用均增长。证据和解析器修正说明见该目录RESULT.md与activity_verification.json。
check_slot_requests.py已按DP rank而非逻辑npu:0解析日志，避免不同worker都显示npu:0导致的误报。原始自动检查失败记录未改，离线复核通过。
服务已停止，已核实8444端口释放、后四卡无NPU进程。8K大负载OOM尚未解决，无有效off/slot性能配对；没有继续4096 token预算实验，也未开启后续性能矩阵。

## 全8卡后续测试已完成

用户随后确认恢复全8卡s≈2.48、8K off/slot性能对比。新脚本及数据在../20261103_dp8_test，正式session为data/20261103_151752_989513，结果见RESULT.md。
DP8/TP1/EP8、KV每卡4GiB，slot相对off TTFT降低9.93%、输入吞吐提高10.57%；两组各3轮成功，统计后2轮，无OOM。最终8卡资源与8444端口已释放。
