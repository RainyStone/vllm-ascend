#!/usr/bin/env bash
set -Ee -o pipefail
PYTHON='/usr/local/python3.11.10/bin/python'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest'
PROFILE_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest/diagnostics/profiling/data/profiling/slot_mte'
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
unset LD_PRELOAD
source /usr/local/Ascend/cann/set_env.sh
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR='/dev/shm/optimized_profile_cpu_parse'
mkdir -p "$TMPDIR"
exec "$PYTHON" "$EXPERIMENT_DIR/offline_analyse_profile.py" "$PROFILE_DIR" 2
