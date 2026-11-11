# 118 GSM8K 4K 独立起服重复对照

已完成 3 对。每对 Off → MXFP8 Slot 独立起服，每个服务先预检、预热 16 条，再测完整 1,319 条。先按模式求各轮算术平均，再用 Slot 均值除以 Off 均值计算变化。P50 和 P95 为各轮分位数的平均。

| 组别 | 平均 TTFT（ms） | P50（ms） | P95（ms） | 输入 token/s | 请求/s |
|---|---:|---:|---:|---:|---:|
| off 均值 | 362.71 | 363.00 | 389.74 | 2,581.43 | 43.79 |
| slot 均值 | 333.32 | 334.10 | 358.89 | 2,800.91 | 47.51 |
| Slot 相对 Off | -8.10% | -7.96% | -7.92% | +8.50% | +8.50% |

| 各对明细 | 平均 TTFT（ms） | P50（ms） | P95（ms） | 输入 token/s | 请求/s |
|---|---:|---:|---:|---:|---:|
| pair01 off | 369.28 | 369.40 | 393.61 | 2,531.98 | 42.95 |
| pair01 slot | 329.50 | 331.00 | 354.98 | 2,831.44 | 48.03 |
| pair01 Slot 相对 Off | -10.77% | -10.40% | -9.81% | +11.83% | +11.83% |
| pair02 off | 357.31 | 358.30 | 381.12 | 2,621.77 | 44.47 |
| pair02 slot | 343.41 | 343.90 | 373.86 | 2,720.16 | 46.14 |
| pair02 Slot 相对 Off | -3.89% | -4.02% | -1.91% | +3.75% | +3.75% |
| pair03 off | 361.53 | 361.30 | 394.50 | 2,590.55 | 43.94 |
| pair03 slot | 327.07 | 327.40 | 347.82 | 2,851.12 | 48.36 |
| pair03 Slot 相对 Off | -9.53% | -9.38% | -11.83% | +10.06% | +10.06% |

## 正式轮 AISBench debug

- [pair01 off](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/formal/data/formal/aisbench/mxfp8-off-20261110_210419-3805688/round_1/aisbench_debug.log) · [pair01 slot](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/formal/data/formal/aisbench/mxfp8-slot-20261110_210742-3813914/round_1/aisbench_debug.log)
- [pair02 off](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/repeat_formal/pair02/formal/data/formal/aisbench/pair02-off-20261110_212112-3839055/round_1/aisbench_debug.log) · [pair02 slot](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/repeat_formal/pair02/formal/data/formal/aisbench/pair02-slot-20261110_212424-3846517/round_1/aisbench_debug.log)
- [pair03 off](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/repeat_formal/pair03/formal/data/formal/aisbench/pair03-off-20261110_212730-3853934/round_1/aisbench_debug.log) · [pair03 slot](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/repeat_formal/pair03/formal/data/formal/aisbench/pair03-slot-20261110_213047-3861447/round_1/aisbench_debug.log)

负载诊断见[首对的独立回放报告](/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261110_dp8_gsm8k_4k_mxfp8_118/RESULTS.md)。
