#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_mxfp8_transport'
PYTHON='/usr/local/python3.11.10/bin/python'
CARDS='4'
RUN_ID='single_118_20261106_1206'
"$PYTHON" "$EXPERIMENT_DIR/guard_cards.py" --cards "$CARDS" --output "$EXPERIMENT_DIR/data/$RUN_ID/idle.json"
while IFS= read -r name; do
    case "$name" in MOONEP*|ASCEND_MOONEP*|VLLM*|SHMEM*) unset "$name";; esac
done < <(compgen -e)
unset ASCEND_VISIBLE_DEVICES PYTHONPATH LD_PRELOAD http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$WORKSPACE/vllm-ascend:$WORKSPACE/ascend-moonep:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export OMP_NUM_THREADS=4 OMP_PROC_BIND=false
export TMPDIR='/dev/shm/mxfp8_test_20261106'
mkdir -p "$TMPDIR"
exec "$PYTHON" "$EXPERIMENT_DIR/test_single.py" --library "$WORKSPACE/ascend-moonep/build/mxfp8_transport_20261106/lib/libascend_moonep_mxfp8_wire.so" --output "$EXPERIMENT_DIR/data/$RUN_ID/result.json"
