#!/usr/bin/env bash
set -Ee -o pipefail
PYTHON='/usr/local/python3.11.10/bin/python'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR='/dev/shm/home_profile_cpu_parse'
mkdir -p "$TMPDIR"
for MODE in off slot_mte slot_udma; do
  "$PYTHON" "$EXPERIMENT_DIR/offline_analyse_profile.py" "$EXPERIMENT_DIR/diagnostics/profiling/data/profiling_v2/$MODE" 2
done
