#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_gsm8k_mxfp8_publish'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='off'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_gsm8k_off'
CONFIG="$EXPERIMENT_DIR/aisbench_off.py"
WARMUPS=16
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
export TMPDIR="/dev/shm/moonep_gsm8k_20261107"
mkdir -p "$TMPDIR"
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
mkdir -p "$HF_DATASETS_CACHE"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"

# Keep the pair sequential and retain a fresh work directory for each service.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another client from this experiment is active.' >&2
    exit 1
fi

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
ROUND_DIR="$RUN_DIR/round_1"
MANIFEST="$EXPERIMENT_DIR/data/selection_manifest.json"
mkdir -p "$ROUND_DIR/precheck" "$ROUND_DIR/postcheck"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$MANIFEST" "$RUN_DIR/selection_manifest.json"

"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" auto "$ROUND_DIR/precheck" \
    2>&1 | tee "$ROUND_DIR/mode_audit.log"
SERVICE_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["service_dir"])' \
    "$ROUND_DIR/precheck/mode_audit.json")

"$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$HOST" "$PORT" "$SERVED_MODEL" "$MANIFEST" "$ROUND_DIR" \
    2>&1 | tee "$ROUND_DIR/preflight.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/precheck" \
        2>&1 | tee "$ROUND_DIR/optimization_audit.log"
fi
npu-smi info > "$ROUND_DIR/npu_before.txt"
cd "$BENCHMARK_DIR"
"$AIS_BENCH" "$CONFIG" --mode perf --debug \
    --max-num-workers 1 --num-warmups "$WARMUPS" \
    --work-dir "$ROUND_DIR/aisbench" \
    2>&1 | tee "$ROUND_DIR/aisbench_debug.log"

# Recheck the same service while its workers are still alive.
"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
    2>&1 | tee "$ROUND_DIR/mode_postcheck.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
        2>&1 | tee "$ROUND_DIR/optimization_postcheck.log"
fi
"$PYTHON" - "$ROUND_DIR" <<'CHECK_AUDIT'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "precheck/mode_audit.json").read_text())
after = json.loads((root / "postcheck/mode_audit.json").read_text())
for key in ("service_pid", "source_sha256"):
    if before[key] != after[key]:
        raise SystemExit(f"Service or audited source changed during measurement: {key}")
workers_before = {worker["rank"]: worker for worker in before["workers"]}
workers_after = {worker["rank"]: worker for worker in after["workers"]}
for rank, worker in workers_before.items():
    current = workers_after[rank]
    if worker["pid"] != current["pid"]:
        raise SystemExit(f"Worker changed during measurement: rank {rank}")
    for library, digest in worker["library_sha256"].items():
        if current["library_sha256"].get(library) != digest:
            raise SystemExit(f"Library changed during measurement: {library}")
CHECK_AUDIT
"$PYTHON" "$EXPERIMENT_DIR/validate_round.py" "$ROUND_DIR/aisbench" "$MANIFEST" \
    2>&1 | tee "$ROUND_DIR/validation.log"
npu-smi info > "$ROUND_DIR/npu_after.txt"
printf 'Raw logs and validated metrics: %s\n' "$ROUND_DIR"
