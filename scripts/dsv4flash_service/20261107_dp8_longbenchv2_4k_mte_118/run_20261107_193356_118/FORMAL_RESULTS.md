# 118 DP8 TP1 LongBench-v2，4K chunk，Off 与 MTE Slot

正式轮各独立起服，最长样本预检后各预热 16 条。每轮 316 条原始全文，并发 16，输出 1 token。预检和预热排除。

| 模式 | TTFT均值 ms | P50 ms | P95 ms | 输入 token/s | 请求/s | 正式秒数 |
|---|---:|---:|---:|---:|---:|---:|
| off | 15273.9 | 15263.5 | 29571.3 | 59536.1 | 1.016 | 311.05 |
| slot | 14832.7 | 14756.0 | 27779.4 | 60383.4 | 1.030 | 306.69 |

4K Slot 相对 Off，TTFT 均值 -2.89%，P50 -3.32%，P95 -6.06%，输入吞吐 +1.42%，请求吞吐 +1.43%。

历史 8K 单对的 Slot 相对 Off，TTFT 均值 −0.55%，输入吞吐 −0.13%。本轮 4K 同样只有一对独立起服。

## 正式轮 Debug 日志

- off /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/formal/data/formal/aisbench/pair01-off-20261107_193610-786163/round_1/aisbench_debug.log
- slot /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/formal/data/formal/aisbench/pair01-slot-20261107_194512-797329/round_1/aisbench_debug.log
