# MoonEP off / slot performance comparison

Home-rank peak/mean describes the injected pre-balancing route distribution.
It does not measure post-MoonEP compute imbalance or numerical correctness.
Retained rounds: 2..3; all values are arithmetic means. Positive percentages mean MoonEP improves.
Per-request percentiles remain in raw JSON; table reports mean TTFT.

| Route | Peak/mean | Input / budget | Status | TTFT off/on (ms) | TTFT reduction | Input tok/s off/on | Input gain | Duration off/on (s) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 3hot1r | — | 8192 / 8192 | incomplete | — | — | — | — | — |

## Incomplete cases

- 3hot1r_L8192_B8192_slot: ValueError: Case metadata is incomplete or does not match the plan: ValueError: Failed/incomplete requests in /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261103_dp4_imbalance/data/20261103_143635_222181/3hot1r_L8192_B8192_slot/round_1.json
- 3hot1r_L8192_B8192_off: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261103_dp4_imbalance/data/20261103_143635_222181/3hot1r_L8192_B8192_off/metadata.json'

No aggregate speedup is claimed for unmatched/failed pairs.
These scripts do not automatically certify external resource isolation or output equivalence.
Raw service logs, NPU snapshots, commands, seeds and Git versions accompany the results.
