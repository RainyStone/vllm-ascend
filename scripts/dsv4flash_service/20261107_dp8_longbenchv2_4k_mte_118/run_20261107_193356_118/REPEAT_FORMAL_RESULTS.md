# 118 DP8 TP1 LongBench-v2，4K 正式对照重复结果

每对 Off → Slot 独立起服，各自校验最长样本并预热 16 条。正式轮 316 条全文、并发 16、输出 1 token。

| 轮次 | 模式 | TTFT 均值 ms | P50 ms | P95 ms | 输入 token/s | 请求/s |
|---|---|---:|---:|---:|---:|---:|
| pair01 (141.61.45.118) | off | 15273.9 | 15263.5 | 29571.3 | 59536.1 | 1.016 |
| pair01 (141.61.45.118) | slot | 14832.7 | 14756.0 | 27779.4 | 60383.4 | 1.030 |
| pair01 | Slot 相对 Off | -2.89% | -3.32% | -6.06% | +1.42% | +1.43% |
| pair02 (141.61.45.118) | off | 15279.8 | 15079.7 | 29264.7 | 59080.8 | 1.008 |
| pair02 (141.61.45.118) | slot | 14789.5 | 14349.8 | 29072.8 | 61054.4 | 1.042 |
| pair02 | Slot 相对 Off | -3.21% | -4.84% | -0.66% | +3.34% | +3.34% |
| pair03 (141.61.45.124) | off | 15225.5 | 15143.0 | 29114.3 | 58955.7 | 1.006 |
| pair03 (141.61.45.124) | slot | 14849.5 | 14855.9 | 28338.8 | 60815.4 | 1.038 |
| pair03 | Slot 相对 Off | -2.47% | -1.90% | -2.66% | +3.15% | +3.15% |
| 新增两对均值 | off | 15252.6 | 15111.4 | 29189.5 | 59018.2 | 1.007 |
| 新增两对均值 | slot | 14819.5 | 14602.8 | 28705.8 | 60934.9 | 1.040 |
| 新增两对均值 | Slot 相对 Off | -2.84% | -3.37% | -1.66% | +3.25% | +3.25% |
| 新增两对均值，各对百分比再平均 | Slot 相对 Off | -2.84% | -3.37% | -1.66% | +3.25% | +3.25% |
| 全部三对均值 | off | 15259.7 | 15162.1 | 29316.8 | 59190.9 | 1.010 |
| 全部三对均值 | slot | 14823.9 | 14653.9 | 28397.0 | 60751.0 | 1.037 |
| 全部三对均值 | Slot 相对 Off | -2.86% | -3.35% | -3.14% | +2.64% | +2.64% |
| 全部三对均值，各对百分比再平均 | Slot 相对 Off | -2.86% | -3.35% | -3.13% | +2.64% | +2.64% |

各指标先按模式对各轮做算术平均，再算 Slot 相对 Off 的变化。TTFT 负值表示降低。历史 8K 单对的 Slot 相对 Off 为平均 TTFT −0.55%、输入吞吐 −0.13%。

## AISBench Debug 原始日志

- pair01 off /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/formal/data/formal/aisbench/pair01-off-20261107_193610-786163/round_1/aisbench_debug.log
- pair01 slot /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/formal/data/formal/aisbench/pair01-slot-20261107_194512-797329/round_1/aisbench_debug.log
- pair02 off /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/repeat_formal/pair02/formal/data/formal/aisbench/pair02-off-20261107_201129-830121/round_1/aisbench_debug.log
- pair02 slot /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/repeat_formal/pair02/formal/data/formal/aisbench/pair02-slot-20261107_202034-840971/round_1/aisbench_debug.log
- pair03 off /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/repeat_formal/pair03_124/formal/data/formal/aisbench/pair03-off-20261107_201117-1236288/round_1/aisbench_debug.log
- pair03 slot /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/repeat_formal/pair03_124/formal/data/formal/aisbench/pair03-slot-20261107_202010-1242438/round_1/aisbench_debug.log
