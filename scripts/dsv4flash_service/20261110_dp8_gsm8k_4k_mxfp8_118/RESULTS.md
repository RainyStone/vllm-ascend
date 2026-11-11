# 118 单机八卡 GSM8K 4K 性能与负载对照

DeepSeek-V4-Flash，DP8 TP1 EP，单 DP 4096 token，并发 16，输出 1 token。每组校验最长输入、预热 16 条后，正式测 1,319 条。两组均成功，输入均为 77,756 token。

| 模式 | 平均 TTFT ms | P50 ms | P95 ms | 输入 token/s | 请求/s |
|---|---:|---:|---:|---:|---:|
| Off | 369.28 | 369.40 | 393.61 | 2531.98 | 42.95 |
| MXFP8 Slot | 329.50 | 331.00 | 354.98 | 2831.44 | 48.03 |
| Slot 相对 Off | -10.77% | -10.40% | -9.81% | +11.83% | +11.83% |

## 独立负载回放

每个对齐的 step×MoE 层按八卡最大 token–expert 行数除以八卡平均行数。加权均值以该窗口总行数为权重。累计峰均比先对每个 rank 跨步跨层求和，再算八卡最大除以平均。

| 负载口径 | 窗口数 | 逐窗口均值 | 加权均值 | 窗口 P95 | 累计峰均比 |
|---|---:|---:|---:|---:|---:|
| Off 实际执行 | 9,503 | 1.9452 | 1.8099 | 2.8333 | 1.0894 |
| Slot 原属 rank | 9,460 | 1.9394 | 1.8070 | 2.6800 | 1.0891 |
| Slot 实际执行 | 9,460 | 1.0040 | 1.0049 | 1.0194 | 1.0002 |

| EP rank | Off 实际执行 | Slot 原属 rank | Slot 实际执行 |
|---|---:|---:|---:|
| 0 | 2,580,536 | 2,581,436 | 2,541,864 |
| 1 | 2,351,208 | 2,352,410 | 2,541,762 |
| 2 | 2,574,894 | 2,576,189 | 2,541,855 |
| 3 | 2,497,692 | 2,499,041 | 2,541,702 |
| 4 | 2,648,459 | 2,649,494 | 2,541,746 |
| 5 | 2,768,091 | 2,768,387 | 2,542,289 |
| 6 | 2,403,519 | 2,404,780 | 2,541,772 |
| 7 | 2,502,905 | 2,503,307 | 2,542,054 |

Off 的对齐窗口包含 3,483 个 MC2 和 6,020 个 All-to-All。Slot 9,460 个窗口均使用 MoonEP。
统计覆盖回放前后边界内的已落盘步骤。计数来自服务的专家行，可能包含运行时对齐和空 rank 的占位工作。

## 原始数据

- [Off AISBench debug](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/formal/data/formal/aisbench/mxfp8-off-20261110_210419-3805688/round_1/aisbench_debug.log) · [Off 八 rank 原始负载](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/diagnostic/data/diagnostic/rank_raw/off/20261110_210932-3815470)
- [Slot AISBench debug](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/formal/data/formal/aisbench/mxfp8-slot-20261110_210742-3813914/round_1/aisbench_debug.log) · [Slot 八 rank 原始负载](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/diagnostic/data/diagnostic/rank_raw/slot/20261110_211252-3823020)
- [八 rank MXFP8 按有效行解包审计](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/formal/data/formal/aisbench/mxfp8-slot-20261110_210742-3813914/round_1/OPTIMIZATION_AUDIT.json)
