#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=1
PHASE='formal'  # 与同组起服脚本保持一致
PAIR_LABEL='pair01'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_longbenchv2_4k_mte_118/run_20261107_193356_118/formal'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='slot'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_longbench_dp8_4k_slot'
CONFIG="$EXPERIMENT_DIR/aisbench_slot.py"
ROUNDS=1
FIRST_ROUND_WARMUPS=16
# ===== End of editable parameters; API batch_size/max_out_len are in CONFIG =====

if (( $# != 0 )); then
    echo 'No positional arguments: edit the values inside this script.' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft: script/config contents are awaiting user review. No requests sent.' >&2
    exit 2
fi
if [[ "$PHASE" != formal && "$PHASE" != diagnostic ]]; then
    echo "PHASE must be formal or diagnostic." >&2
    exit 2
fi
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export TMPDIR="/dev/shm/moonep_longbench_dp8_4k_run_20261107_193356_118_formal"
mkdir -p "$TMPDIR"
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
mkdir -p "$HF_DATASETS_CACHE"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_full_le128k.json"
test -r "$EXPERIMENT_DIR/data/selection_manifest.json"

# Prevent concurrent clients from accidentally exceeding total concurrency 16.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
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

RUN_DIR="$EXPERIMENT_DIR/data/$PHASE/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/data/selection_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf '%s\n' "$PAIR_LABEL" > "$RUN_DIR/pair_label.txt"
if [[ "$PHASE" == formal ]]; then
    printf 'One formal round of 316 requests; preflight and warmups excluded.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
else
    FIRST_ROUND_WARMUPS=0
    printf 'One diagnostic replay of 316 requests; timings excluded from formal comparison.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
fi
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
    "$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" "$PHASE" "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/mode_audit.log"
    if (( round == 1 )); then
        "$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$MODE" "$RUN_DIR" \
            2>&1 | tee "$RUN_DIR/preflight.log"
        if [[ "$MODE" == slot ]]; then
            service_dir=$("$PYTHON" - "$ROUND_DIR" <<'FIND_SERVICE'
import json, sys
from pathlib import Path
audit, = Path(sys.argv[1]).glob("*_mode_audit.json")
print(json.loads(audit.read_text())["service_dir"])
FIND_SERVICE
)
            "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$service_dir" "$ROUND_DIR" \
                2>&1 | tee "$ROUND_DIR/optimization_audit.log"
        fi
    fi
    if [[ "$PHASE" == diagnostic ]]; then
        "$PYTHON" "$EXPERIMENT_DIR/snapshot_rank_logs.py" "$ROUND_DIR" before
    fi
    npu-smi info > "$ROUND_DIR/npu_before.txt"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/started_at.txt"
    # --debug prints to the terminal; tee retains stdout + stderr in full.
    # No --num-prompts subset, no --reuse, no --pressure, no implicit retry.
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
        --max-num-workers 1 --num-warmups "$warmups" \
        --work-dir "$ROUND_DIR/aisbench" \
        2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
    if [[ "$PHASE" == diagnostic ]]; then
        "$PYTHON" "$EXPERIMENT_DIR/snapshot_rank_logs.py" "$ROUND_DIR" after
    fi
    "$PYTHON" "$EXPERIMENT_DIR/validate_round.py" \
        "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/selection_manifest.json" \
        2>&1 | tee "$ROUND_DIR/validation.log"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/ended_at.txt"
    npu-smi info > "$ROUND_DIR/npu_after.txt"
done
printf 'All raw results and debug logs retained in %s\n' "$RUN_DIR"
