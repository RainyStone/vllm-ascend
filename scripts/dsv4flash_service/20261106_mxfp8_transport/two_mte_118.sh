#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_mxfp8_transport'
PYTHON='/usr/local/python3.11.10/bin/python'
CARDS='4,6'
RUN_ID='two_mte_118_20261106_1214'
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
export HCCL_IF_IP=141.61.45.118 GLOO_SOCKET_IFNAME=enp34s0f1 HCCL_SOCKET_IFNAME=enp34s0f1 TP_SOCKET_IFNAME=enp34s0f1
export MOONEP_SLOT_MODE=1 MOONEP_SLOT_QUANT=1 MOONEP_SLOT_CACHE=0 MOONEP_EARLY_PREFETCH=0 MOONEP_PF_STREAM=3 MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_SKIP_DUP=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_PREFETCH_RELAXED=1
export SHMEM_LOG_LEVEL=ERROR MOONEP_BALANCE_STATS=0 MOONEP_PLAN_STATS=0 MOONEP_TIME_STATS=0 ASCEND_MOONEP_TRACE=0 ASCEND_MOONEP_TIME_STATS=0
export MOONEP_DISABLE_UDMA=1 ASCEND_MOONEP_PLANNING_USE_UDMA=0 ASCEND_MOONEP_DISPATCH_USE_UDMA=0 ASCEND_MOONEP_COMBINE_USE_UDMA=0 ASCEND_MOONEP_PREFETCH_USE_UDMA=0
VENDOR="$WORKSPACE/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
exec "$PYTHON" "$EXPERIMENT_DIR/two_card_runner.py" --backend mte --output "$EXPERIMENT_DIR/data/$RUN_ID" --library "$WORKSPACE/ascend-moonep/build/mxfp8_transport_20261106/lib/libascend_moonep_mxfp8_wire.so" --master-port 29816 --shmem-port 29916 --master-addr 141.61.45.118
