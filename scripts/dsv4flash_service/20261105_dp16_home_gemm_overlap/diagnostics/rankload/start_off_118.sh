#!/usr/bin/env bash
# Approved Home GEMM1 experiment. Edit all tuning values inside this file.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PAIR_LABEL='rankload_pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/diagnostics/rankload'
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

export VLLM_ASCEND_MOE_PAR_STATS=1
export VLLM_ASCEND_MOE_PAR_STATS_DIR="$EXPERIMENT_DIR/data/rank_load/off"
export VLLM_ASCEND_MOE_PAR_STATS_TAG='home3_off'
export VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1

# Local resource protection. Do not stop another experiment.
"$PYTHON" - <<'CHECK_IDLE'
import re,subprocess
text=subprocess.check_output(['npu-smi','info'],text=True)
if set(re.findall(r'No running processes found in NPU (\d+)',text))!=set(map(str,range(8))):
    raise SystemExit('Eight idle local cards are required; current single-node experiment is protected.')
CHECK_IDLE

mkdir -p "$EXPERIMENT_DIR/data/services"
exec 9>"/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/.service_node118_8445.lock"
if ! flock -n 9; then
    echo 'Another service script from this experiment still holds node 118 service lock.' >&2
    exit 1
fi
# Bind probe does not interrupt an existing listener.
"$PYTHON" - "$HOST" "$PORT" <<'CHECK_PORT'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((sys.argv[1], int(sys.argv[2])))
CHECK_PORT

RUN_DIR="$EXPERIMENT_DIR/data/services/$PAIR_LABEL-$MODE-node118-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/start_script.sh"
printf '%s\n' "$$" > "$RUN_DIR/service.pid"
ps -o pgid= -p "$$" > "$RUN_DIR/service.pgid"
env | LC_ALL=C sort | grep -E '^(MOONEP|ASCEND_MOONEP|SHMEM_|VLLM|ASCEND_RT_VISIBLE_DEVICES|ASCEND_CUSTOM_OPP_PATH|HCCL|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME|OMP|PYTORCH_NPU_ALLOC_CONF|PYTHONPATH|LD_LIBRARY_PATH|NO_PROXY|no_proxy)' > "$RUN_DIR/environment.txt"
npu-smi info > "$RUN_DIR/npu_before.txt"
exec > >(tee -a "$RUN_DIR/service.log") 2>&1
printf 'MODE=%s PAIR=%s LOG=%s\n' "$MODE" "$PAIR_LABEL" "$RUN_DIR/service.log"
cd "$PLUGIN"
# Foreground exec keeps this shell's PID; stop this service before starting the other.
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --api-server-count 1 \
    --data-parallel-size 16 --data-parallel-size-local 8 \
    --data-parallel-address 141.61.45.118 --data-parallel-rpc-port 29560 \
    --data-parallel-backend mp --tensor-parallel-size 1 --enable-expert-parallel \
    --served-model-name "$SERVED_MODEL" --seed "$SERVER_SEED" \
    --trust-remote-code --enforce-eager \
    --disable-log-stats --disable-uvicorn-access-log \
    --enable-chunked-prefill --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-memory-bytes "$KV_CACHE_BYTES" --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --additional-config '{"enable_moonep":false,"dp_allreduce_on_npu":true}'
