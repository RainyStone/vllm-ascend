# 124 单机 8 卡 LongBench-v2 4K

| 组别 | TTFT 均值 ms | P50 ms | P95 ms | 输入 token/s | 请求/s |
|---|---:|---:|---:|---:|---:|
| off | 15213.5 | 15187.1 | 29199.9 | 59203.3 | 1.010 |
| mxfp8_slot | 14785.9 | 14648.0 | 27546.5 | 60832.7 | 1.038 |
| 历史 124 Off | 15225.5 | 15143.0 | 29114.3 | 58955.7 | 1.006 |
| 历史 124 MXFP8 Slot | 14849.5 | 14855.9 | 28338.8 | 60815.4 | 1.038 |

| 对比 | TTFT 均值 | P50 | P95 | 输入吞吐 | 请求吞吐 |
|---|---:|---:|---:|---:|---:|
| mxfp8_slot 相对 off | -2.81% | -3.55% | -5.66% | +2.75% | +2.75% |
| 历史 124 Slot 相对历史 Off | -2.47% | -1.90% | -2.66% | +3.15% | +3.15% |
| 本轮 Off 相对历史 124 Off | -0.08% | +0.29% | +0.29% | +0.42% | +0.42% |
| 本轮 MXFP8 Slot 相对历史 124 Slot | -0.43% | -1.40% | -2.80% | +0.03% | +0.03% |

以各自 Off 为基线，Slot 的相对变化比历史同机对照多 -0.34 个百分点 TTFT，-0.40 个百分点输入吞吐。两轮为独立端到端对照。

## 正式轮 Debug 日志

- off /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_longbenchv2_4k_dispatch_124/mxfp8/formal/data/formal/aisbench/mxfp8-off-20261110_183754-2039427/round_1/aisbench_debug.log
- mxfp8_slot /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_longbenchv2_4k_dispatch_124/mxfp8/formal/data/formal/aisbench/mxfp8-slot-20261110_184935-2051025/round_1/aisbench_debug.log
