#!/usr/bin/env bash
# Awaiting review. Select the already running service with off or slot.
set -Eeuo pipefail

SCRIPT_REVIEWED=1
HOST='127.0.0.1'
PORT=8444
WINDOW_SECONDS=30
CONCURRENCY=16
POLL_SECONDS=1
SATURATED_SAMPLES=3
GATE_TIMEOUT_SECONDS=180
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'

if [[ $# != 1 || ( "$1" != off && "$1" != slot ) ]]; then
    echo 'Usage: bash bench_profile.sh off|slot' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft awaiting review. No requests sent.' >&2
    exit 2
fi
MODE="$1"
CONFIG="$EXPERIMENT_DIR/aisbench_$MODE.py"
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export TMPDIR='/dev/shm/moonep_longbench_dp8_4k_30s_profile_118'
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_full_le128k.json"
test -r "$EXPERIMENT_DIR/data/selection_manifest.json"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$EXPERIMENT_DIR/data/formal/aisbench"

exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
flock -n 8 || { echo 'An AISBench client from this experiment is already active.' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/data/formal/aisbench/$MODE-$(date +%Y%m%d_%H%M%S)-XXXXXXXX")"
ROUND_DIR="$RUN_DIR/round_1"
mkdir "$ROUND_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/profile_window.py" "$RUN_DIR/profile_window.py"
cp -- "$EXPERIMENT_DIR/data/selection_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf 'Output directory %s\n' "$RUN_DIR"

cd "$BENCHMARK_DIR"
"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" formal "$ROUND_DIR" \
    2>&1 | tee "$ROUND_DIR/mode_audit.log"
"$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$MODE" "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/preflight.log"
if [[ "$MODE" == slot ]]; then
    service_dir=$("$PYTHON" - "$ROUND_DIR" <<'FIND_SERVICE'
import json
import sys
from pathlib import Path

audit, = Path(sys.argv[1]).glob("*_mode_audit.json")
print(json.loads(audit.read_text())["service_dir"])
FIND_SERVICE
)
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$service_dir" "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/optimization_audit.log"
fi
"$PYTHON" "$EXPERIMENT_DIR/profile_window.py" "$MODE" "$ROUND_DIR" \
    "http://$HOST:$PORT" "$WINDOW_SECONDS" "$CONCURRENCY" "$POLL_SECONDS" \
    "$SATURATED_SAMPLES" "$GATE_TIMEOUT_SECONDS" -- \
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
    --max-num-workers 1 --num-warmups 0 --work-dir "$ROUND_DIR/aisbench" \
    2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
"$PYTHON" "$EXPERIMENT_DIR/validate_round.py" \
    "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/selection_manifest.json" \
    2>&1 | tee "$ROUND_DIR/validation.log"
"$PYTHON" - "$ROUND_DIR/CAPTURE_STATUS.json" <<'CHECK_CAPTURE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = json.loads(path.read_text())
if not status["valid"]:
    raise SystemExit(f"Replay validated, capture invalid: {status['reason']}. See {path}")
print(f"Valid {status['window_seconds']}s window, eight original rank traces: {status['profile_dir']}")
CHECK_CAPTURE
