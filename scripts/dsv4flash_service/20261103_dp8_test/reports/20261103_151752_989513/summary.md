# MoonEP off / slot performance comparison

Home-rank peak/mean describes the injected pre-balancing route distribution.
It does not measure post-MoonEP compute imbalance or numerical correctness.
Retained rounds: 2..3; all values are arithmetic means. Positive percentages mean MoonEP improves.
Per-request percentiles remain in raw JSON; table reports mean TTFT.

| Route | Peak/mean | Input / budget | Status | TTFT off/on (ms) | TTFT reduction | Input tok/s off/on | Input gain | Duration off/on (s) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| rank_skew_2p48_dp8 | 2.4822 | 8192 / 8192 | complete | 2331.42/2099.89 | +9.93% | 49652.73/54901.31 | +10.57% | 5.280/4.779 |

No aggregate speedup is claimed for unmatched/failed pairs.
These scripts do not automatically certify external resource isolation or output equivalence.
Raw service logs, NPU snapshots, commands, seeds and Git versions accompany the results.
