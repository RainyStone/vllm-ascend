#!/usr/bin/env bash
# Model-free diagnostic, same CANN/HCCL environment as the approved off service.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PAIR_LABEL='home_gemm_pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='off'
SERVED_MODEL='dsv4flash_dp16_longbench_off'
CARDS='0,1,2,3,4,5,6,7'
HOST='141.61.45.118'
PORT=8445
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB/card; DP16 requires later startup validation.
SERVER_SEED=1024
# ===== End of editable experiment parameters =====

if (( $# != 0 )); then
    echo 'No positional arguments: edit the values inside this script.' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft: script/config contents are awaiting user review. No service started.' >&2
    exit 2
fi
# Time authorization is checked before any accelerator query.
"$PYTHON" "$EXPERIMENT_DIR/resource_gate.py" --cards "$CARDS" \
    --output "$EXPERIMENT_DIR/data/idle_off_118.json"

# Remove inherited experimental settings, including every synthetic-route option.
while IFS= read -r name; do
    case "$name" in
        MOONEP*|ASCEND_MOONEP*|VLLM*) unset "$name" ;;
    esac
done < <(compgen -e)
unset ASCEND_VISIBLE_DEVICES http_proxy https_proxy ftp_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY PYTHONPATH LD_PRELOAD
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
# CANN's vendor environment script may reference unset variables.
source "$CANN_ENV"
CANN_PYTHONPATH="$PYTHONPATH"
set -u

PLUGIN="$WORKSPACE/vllm-ascend"
MOONEP_PACKAGE="$WORKSPACE/ascend-moonep/ascend_moonep"
VENDOR="$PLUGIN/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export SHMEM_LOG_LEVEL=ERROR
unset SHMEM_LOG_PATH
export PYTHONPATH="$WORKSPACE/vllm:$PLUGIN:$WORKSPACE/ascend-moonep:$CANN_PYTHONPATH"
export VLLM_PLUGINS='ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model'
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export HCCL_IF_IP='141.61.45.118'
export GLOO_SOCKET_IFNAME='enp34s0f1'
export TP_SOCKET_IFNAME='enp34s0f1'
export HCCL_SOCKET_IFNAME='enp34s0f1'
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:False'
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export NO_PROXY='127.0.0.1,localhost,141.61.45.118,141.61.45.124'
export no_proxy="$NO_PROXY"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
# MoonEP disabled: no MoonEP or synthetic routing environment is exported.


RUN_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/data/native_hccl16_20261106_040000'
mkdir -p "$RUN_DIR"
exec 9>"/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/.service_node118_8445.lock"
flock -n 9
"$PYTHON" "$EXPERIMENT_DIR/resource_gate.py" --cards "$CARDS" --output "$RUN_DIR/node118_idle.json"
env | LC_ALL=C sort | grep -E '^(HCCL|GLOO|ASCEND|LD_LIBRARY_PATH|PYTHONPATH|OMP)' > "$RUN_DIR/node118_environment.txt"
exec "$PYTHON" -m torch.distributed.run --nnodes 2 --node_rank 0 --nproc_per_node 8 --master_addr 141.61.45.118 --master_port 29785 "$EXPERIMENT_DIR/tests/test_native_hccl_metadata.py" "$RUN_DIR/results"
