#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_fused_combine_publish'
TEST_APPROVED=0
NODE=1
NODE_IP='141.61.45.124'
CARD=1
RUN_ID='cross_udma_sync_audited_20261106T115850Z'
MASTER_PORT=29883
SHMEM_PORT=29983
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
export MOONEP_SLOT_MODE=1 MOONEP_SLOT_QUANT=1 MOONEP_SLOT_CACHE=0 MOONEP_EARLY_PREFETCH=0 MOONEP_PF_STREAM=3 MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_SKIP_DUP=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_PREFETCH_RELAXED=1
export MOONEP_BALANCE_STATS=0 MOONEP_PLAN_STATS=0 MOONEP_TIME_STATS=0 ASCEND_MOONEP_TRACE=0 ASCEND_MOONEP_TIME_STATS=0
export MOONEP_DISABLE_UDMA=0 ASCEND_MOONEP_PLANNING_USE_UDMA=0 ASCEND_MOONEP_DISPATCH_USE_UDMA=1 ASCEND_MOONEP_COMBINE_USE_UDMA=1 ASCEND_MOONEP_PREFETCH_USE_UDMA=1
OUT="$EXPERIMENT_DIR/data/$RUN_ID"
mkdir -p "$OUT/shmem_node$NODE"
export SHMEM_LOG_LEVEL=INFO SHMEM_LOG_PATH="$OUT/shmem_node$NODE"
VENDOR="$WORKSPACE/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
export MOONEP_UDMA_TEST_LIBRARY="$WORKSPACE/ascend-moonep/build/udma_weighted_final_20261106/lib/libascend_moonep_udma_weight_scalar_fix.so"
export LD_PRELOAD="$MOONEP_UDMA_TEST_LIBRARY"
exec "$PYTHON" "$EXPERIMENT_DIR/cross_node_runner.py" --node "$NODE" --output "$OUT" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" --library "$WORKSPACE/ascend-moonep/build/combine_publish_20261106/lib/libascend_moonep_publish.so"
