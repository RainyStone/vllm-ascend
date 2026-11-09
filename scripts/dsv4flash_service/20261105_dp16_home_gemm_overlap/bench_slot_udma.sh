#!/usr/bin/env bash
# One approved cold 503-request performance round; raw debug retained.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=1
PAIR_LABEL='pair01'  # Repeat index per mode: pair01 for its first service, pair02 for its second; see PLAN order.
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='slot_udma'
HOST='141.61.45.118'
PORT=8445
SERVED_MODEL='dsv4flash_dp16_longbench_slot_udma'
CONFIG="$EXPERIMENT_DIR/aisbench_slot_udma.py"
ROUNDS=1
FIRST_ROUND_WARMUPS=0
# ===== End of editable parameters; API batch_size/max_out_len are in CONFIG =====

if (( $# != 0 )); then
    echo 'No positional arguments: edit the values inside this script.' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft: script/config contents are awaiting user review. No requests sent.' >&2
    exit 2
fi
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export PYTHONPATH="$EXPERIMENT_DIR:/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Keep this experiment cache/temp on the shared filesystem; container overlay is full.
export HF_HOME="$EXPERIMENT_DIR/data/client_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR='/dev/shm/home_gmm_dp16_20261106'
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_all503_midcut128k.json"
test -r "$EXPERIMENT_DIR/data/full503_manifest.json"

# Prevent concurrent clients from accidentally exceeding total concurrency 16.
exec 8>"/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another aisbench client from this experiment is active.' >&2
    exit 1
fi
"$PYTHON" - "$HOST" "$PORT" "$SERVED_MODEL" <<'CHECK_SERVICE'
import json, sys, urllib.request
host, port, expected = sys.argv[1:]
with urllib.request.urlopen(f'http://{host}:{port}/v1/models', timeout=15) as response:
    names = [entry['id'] for entry in json.load(response)['data']]
if expected not in names:
    raise SystemExit(f'Wrong service: expected {expected}, found {names}')
print('Verified service:', expected)
CHECK_SERVICE

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/home_gemm3-$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/data/full503_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf '%s\n' "$PAIR_LABEL" > "$RUN_DIR/pair_label.txt"
printf 'One measured round per service, zero warmups. Quick order: off -> MTE -> UDMA; one run per mode.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
    "$PYTHON" "$EXPERIMENT_DIR/verify_cluster_audit.py" "$MODE" "$PAIR_LABEL" "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/mode_audit.log"
    npu-smi info > "$ROUND_DIR/npu_before.txt"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/started_at.txt"
    # --debug prints to the terminal; tee retains stdout + stderr in full.
    # No --num-prompts subset, no --reuse, no --pressure, no implicit retry.
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
        --max-num-workers 1 --num-warmups "$warmups" \
        --work-dir "$ROUND_DIR/aisbench" \
        2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
    "$PYTHON" "$EXPERIMENT_DIR/validate_round_dp16.py" \
        "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/full503_manifest.json" \
        2>&1 | tee "$ROUND_DIR/validation.log"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/ended_at.txt"
    npu-smi info > "$ROUND_DIR/npu_after.txt"
done
printf 'All raw results and debug logs retained in %s\n' "$RUN_DIR"
