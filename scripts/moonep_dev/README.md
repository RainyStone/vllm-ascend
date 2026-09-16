# moonep_dev — load-balance / capacity measurements (dev branch)

Tooling behind `docs/dev_results_2026-09-16.md`. Everything here is
harness-only: nothing in this directory ships with the model path.

## What each file is

| file | purpose |
|---|---|
| `bench_dp.py` | N-card DP×TP benchmark: prompts (plain or `PROMPT_FILE`), prefill/decode split from the engine's own step stats, TTFT/TPOT/ITL, per-step trace, greedy token dump |
| `gmu_now.py` | prints a safe `gpu_memory_utilization` from the *current* free memory of cards 4-7 (a co-tenant can take memory back mid-run — see the OOM note below) |
| `run_multi.sh` | generic N-card runner (`VISIBLE/DP/TP/N_SEQ/PROMPT_LEN/MBT/MAXLEN/GEN/GMU/TAG/URONEP*`) |
| `dpshim/sitecustomize.py` | `MOONEP_DP_FULL_VISIBLE=1` shim: keeps the full visible device list (CANN logical==physical id, see README limitation #7) + the `MOONEP_FORCE_GLOO` path |
| `make_prompts.py` | builds the GSM8K prompt file (seeded shuffle, one question per line) |
| `moonep_plan_dump.py` | standalone 4-rank plan probe: dumps `cu_seqlens`, `experts_to_copy`, per-expert demand vs owner counts (this is what proved the slot mechanism *moves* rows) |
| `probe_lazy_buffer.py` | checks whether a `Buffer` can be created after `ShmemRuntime.init_with_buffer` (it cannot: `cannot register a symmetric descriptor after buffer-backed shmem initialization`) |
| `analyze_skew_time.py` | per-layer group skew vs synchronised MLP time correlation (decode plateau only) |
| `cmp_tokids.py` | compares `DUMP_TOKENS` output of two legs (identical / first divergence) |
| `run_*.sh` | example invocations: plan dump, lazy-buffer probe, GSM8K MC2-vs-slot, release A/B, skew-time |

## How to run

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # full list + MOONEP_DEV_BASE
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export MOONEP_DP_FULL_VISIBLE=1 MOONEP_SHMEM_BOOTSTRAP=ipport
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

# GSM8K MC2 vs MoonEP slot, 64 sequences per card
python3 make_prompts.py
VISIBLE=0,1,2,3,4,5,6,7 MOONEP_DEV_BASE=4 bash run_gsm8k_imbalance.sh
```

## Measurement instrumentation (in the model path, env-gated)

| env | what it records |
|---|---|
| `MOE_ROW_STATS=1` | per-layer local-expert `group_list`: row sum, max group, non-zero group count → `/tmp/moe_rows_<TAG>_rank<r>.json`. Device-side accumulation, no per-layer sync. |
| `MOE_ROW_TIMING=1` | additionally records each layer's *synchronised* MLP wall time (measurement-only: it serialises the step) → `/tmp/moe_rowtimes_<TAG>.json` |
| `DUMP_TOKENS=1` | greedy token ids per prompt → `/tmp/bench_tokids_<TAG>_rank<r>.json` |

## Traps this tooling already ran into (don't repeat them)

1. **`… | grep … | tail -N` eats crash tracebacks.** `tail` buffers until EOF, so a
   killed run leaves an empty log. Redirect raw output to a file instead.
2. **`torch.distributed.get_rank()` is -1 inside vLLM workers** (vllm-ascend makes
   its own HCCL groups) — all ranks then write the same dump file and corrupt it.
   Derive the rank from `torch.npu.current_device() - MOONEP_DEV_BASE`.
3. **Killing by pattern is unsafe on a shared box**: `pkill -f` matches your own
   shell, and other tenants run `VLLM::*` processes. Kill explicit PIDs.
4. **`gpu_memory_utilization` from an adaptive probe can be stale** — a co-tenant
   taking ~10 GiB mid-run is what killed one leg (`Free memory on device
   (70.03/79.81 GiB) … less than desired … 0.9`). Re-read free memory right before
   launch and stay a few GiB below it.
