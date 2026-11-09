#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest'
TEST_APPROVED=1
NODE=0
NODE_IP='141.61.45.118'
CARD='0,1,2,3,4,5,6,7'
RUN_ID='cross16_udma_20261106T1315'
MASTER_PORT=29713
SHMEM_PORT=29813
PYTHON='/usr/local/python3.11.10/bin/python'
[[ $# == 0 && "$TEST_APPROVED" == 1 ]] || { echo 'Test gate closed'; exit 78; }
"$PYTHON" "$EXPERIMENT_DIR/guard_cards.py" --cards "$CARD" --output "$EXPERIMENT_DIR/data/$RUN_ID/idle_node$NODE.json"
while IFS= read -r name; do
  case "$name" in MOONEP*|ASCEND_MOONEP*|VLLM*|SHMEM*) unset "$name";; esac
done < <(compgen -e)
unset ASCEND_VISIBLE_DEVICES PYTHONPATH LD_PRELOAD http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$WORKSPACE/vllm-ascend:$WORKSPACE/ascend-moonep"
export ASCEND_RT_VISIBLE_DEVICES="$CARD" OMP_NUM_THREADS=2 OMP_PROC_BIND=false
export TMPDIR='/dev/shm/udma_cross_sync_20261106'
mkdir -p "$TMPDIR"
export HCCL_IF_IP="$NODE_IP" GLOO_SOCKET_IFNAME=enp34s0f1 HCCL_SOCKET_IFNAME=enp34s0f1 TP_SOCKET_IFNAME=enp34s0f1
export NO_PROXY=127.0.0.1,localhost,141.61.45.118,141.61.45.124
export no_proxy="$NO_PROXY"
export MOONEP_SLOT_MODE=1 MOONEP_SLOT_QUANT=1 MOONEP_SLOT_CACHE=0 MOONEP_EARLY_PREFETCH=0 MOONEP_PF_STREAM=2 MOONEP_KEEP_MC2=0
export MOONEP_REAL_T=1
export ASCEND_MOONEP_SKIP_DUP=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_PREFETCH_RELAXED=1
export MOONEP_BALANCE_STATS=0 MOONEP_PLAN_STATS=0 MOONEP_TIME_STATS=0 ASCEND_MOONEP_TRACE=0 ASCEND_MOONEP_TIME_STATS=0
export MOONEP_DISABLE_UDMA=0 ASCEND_MOONEP_PLANNING_USE_UDMA=0 ASCEND_MOONEP_DISPATCH_USE_UDMA=1 ASCEND_MOONEP_COMBINE_USE_UDMA=1 ASCEND_MOONEP_PREFETCH_USE_UDMA=1
OUT="$EXPERIMENT_DIR/data/$RUN_ID"
mkdir -p "$OUT/shmem_node$NODE"
export SHMEM_LOG_LEVEL=ERROR SHMEM_LOG_PATH="$OUT/shmem_node$NODE"
VENDOR="$WORKSPACE/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep/build/dp16_optimized_20261106/lib:$LD_LIBRARY_PATH"
exec "$PYTHON" "$EXPERIMENT_DIR/operator_node_runner.py" --node "$NODE" --local-size 8 --backend udma --output "$OUT" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" --library "/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep/build/dp16_optimized_20261106/lib/libascend_moonep_mxfp8_wire.so"
