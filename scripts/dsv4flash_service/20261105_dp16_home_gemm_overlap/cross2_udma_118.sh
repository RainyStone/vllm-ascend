#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
PYTHON='/usr/local/python3.11.10/bin/python'
CARDS='1'
RUN_ID='home_cross2_udma_20261106_002617'
"$PYTHON" "$EXPERIMENT_DIR/resource_gate.py" --cards "$CARDS" \
    --output "$EXPERIMENT_DIR/data/$RUN_ID/idle_118.json"
while IFS= read -r name; do
    case "$name" in MOONEP*|ASCEND_MOONEP*|VLLM*|SHMEM*) unset "$name";; esac
done < <(compgen -e)
unset ASCEND_VISIBLE_DEVICES PYTHONPATH LD_PRELOAD http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$WORKSPACE/vllm-ascend:$WORKSPACE/ascend-moonep:$PYTHONPATH"
VENDOR="$WORKSPACE/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export HCCL_IF_IP='141.61.45.118'
export GLOO_SOCKET_IFNAME='enp34s0f1' TP_SOCKET_IFNAME='enp34s0f1' HCCL_SOCKET_IFNAME='enp34s0f1'
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false SHMEM_LOG_LEVEL=ERROR
export TMPDIR='/dev/shm/home_gmm_test_20261106'
mkdir -p "$TMPDIR"
export MOONEP_SLOT_MODE=1 MOONEP_SLOT_QUANT=1 MOONEP_SLOT_CACHE=0
export MOONEP_EARLY_PREFETCH=0 MOONEP_PF_STREAM=3 MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_SKIP_DUP=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_PREFETCH_RELAXED=1
export MOONEP_BALANCE_STATS=0 MOONEP_PLAN_STATS=0 MOONEP_TIME_STATS=0
export ASCEND_MOONEP_TRACE=0 ASCEND_MOONEP_TIME_STATS=0
LIBRARY="$WORKSPACE/ascend-moonep/build/home_gmm_20261105/lib/libascend_moonep_home_gmm.so"
export MOONEP_DISABLE_UDMA=0
export ASCEND_MOONEP_PLANNING_USE_UDMA=0
export ASCEND_MOONEP_DISPATCH_USE_UDMA=1
export ASCEND_MOONEP_COMBINE_USE_UDMA=1
export ASCEND_MOONEP_PREFETCH_USE_UDMA=1
exec "$PYTHON" "$EXPERIMENT_DIR/rank_group_runner.py" \
    --world 2 --node 0 --backend udma \
    --output "$EXPERIMENT_DIR/data/$RUN_ID" --library "$LIBRARY" \
    --master-port 29672 --shmem-port 29772
