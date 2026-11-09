#!/usr/bin/env bash
# Separate, bounded diagnostic trace. Every parameter is edited here.
set -Eeuo pipefail
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
MODE='slot_udma'
OUTPUT='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/data/diagnostics/home_profiling_v2_20261106/slot_udma/profile_requests.json'
PYTHON='/usr/local/python3.11.10/bin/python'
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export PYTHONPATH="$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec 8>'/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/.aisbench_client.lock'
flock -n 8
"$PYTHON" "$EXPERIMENT_DIR/diagnostics/profiling/verify_cluster_audit.py" "$MODE" pair01 "$(dirname "$OUTPUT")/pre_audit"
exec "$PYTHON" "$EXPERIMENT_DIR/profile_requests.py" --mode "$MODE" --output "$OUTPUT"
