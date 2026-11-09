#!/usr/bin/env bash
set -Ee -o pipefail
PYTHON='/usr/local/python3.11.10/bin/python'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
PROFILE_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/diagnostics/profiling/data/profiling_v2/off/dp0_pp0_tp0_dcp0_ep0_rank0_3168476_20261106061435833_ascend_pt'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR='/dev/shm/home_profile_cpu_parse'
mkdir -p "$TMPDIR"
exec "$PYTHON" "$EXPERIMENT_DIR/offline_analyse_profile.py" "$PROFILE_DIR"
