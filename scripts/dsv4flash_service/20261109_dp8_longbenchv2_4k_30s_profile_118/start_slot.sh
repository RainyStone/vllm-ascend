#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PHASE='formal'
PAIR_LABEL='pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='slot'
SERVED_MODEL='dsv4flash_longbench_dp8_4k_slot'
CARDS='0,1,2,3,4,5,6,7'
HOST='127.0.0.1'
PORT=8444
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=4096  # 每 DP engine 的单步 token 预算
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB/卡
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
export TMPDIR="/dev/shm/moonep_longbench_dp8_4k_30s_profile_118"
mkdir -p "$TMPDIR"
# CANN's vendor environment script may reference unset variables.
source "$CANN_ENV"
CANN_PYTHONPATH="${PYTHONPATH:-}"
set -u

PLUGIN="$WORKSPACE/vllm-ascend"
MOONEP_PACKAGE="$WORKSPACE/ascend-moonep/ascend_moonep"
VENDOR="$PLUGIN/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
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
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
# MoonEP Slot: MTE, persistent cache disabled, prefetch side stream enabled.
export MOONEP=1
export MOONEP_SLOT_MODE=1
export MOONEP_REAL_T=1
export MOONEP_MIN_DISPATCH_TOKENS=16
export MOONEP_SHMEM_BOOTSTRAP='ipport'
export MOONEP_DISABLE_UDMA=1
export ASCEND_MOONEP_BUILD_DIR="$MOONEP_PACKAGE"
export MOONEP_SLOT_QUANT=1
export MOONEP_SLOT_CACHE=0
export MOONEP_EARLY_PREFETCH=0
export MOONEP_PF_STREAM=2
export VLLM_ASCEND_MOONEP_DISPATCH_DTYPE=mxfp8
export VLLM_ASCEND_MOONEP_FUSED_PUBLISH=1
export MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_PREFETCH_RELAXED=1
export ASCEND_MOONEP_NO_RECONCILE=1
export ASCEND_MOONEP_SKIP_DUP=1
export LD_LIBRARY_PATH="$MOONEP_PACKAGE/lib:$LD_LIBRARY_PATH"
# No VLLM_ASCEND_MOE_SIM_ROUTE, balance stats, plan stats, or timing instrumentation.


mkdir -p "$EXPERIMENT_DIR/data/$PHASE/services"
exec 9>"$EXPERIMENT_DIR/.service_port_8444.lock"
if ! flock -n 9; then
    echo 'Another service script from this experiment still holds port 8444 lock.' >&2
    exit 1
fi
# Bind probe does not interrupt an existing listener.
"$PYTHON" - "$HOST" "$PORT" <<'CHECK_PORT'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((sys.argv[1], int(sys.argv[2])))
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((sys.argv[1], 28765))
CHECK_PORT

RUN_DIR="$EXPERIMENT_DIR/data/$PHASE/services/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR/profiling"
PROFILER_CONFIG=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","delay_iterations":0,"max_iterations":0,"ignore_frontend":true,"torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false}' "$RUN_DIR/profiling")
printf '%s\n' "$PROFILER_CONFIG" > "$RUN_DIR/profiler_config.json"
cp -- "$0" "$RUN_DIR/start_script.sh"
printf '%s\n' "$$" > "$RUN_DIR/service.pid"
ps -o pgid= -p "$$" > "$RUN_DIR/service.pgid"
env | LC_ALL=C sort | grep -E '^(MOONEP|ASCEND_MOONEP|VLLM|ASCEND_RT_VISIBLE_DEVICES|ASCEND_CUSTOM_OPP_PATH|HCCL|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME|OMP|PYTORCH_NPU_ALLOC_CONF|PYTHONPATH|LD_LIBRARY_PATH|NO_PROXY|no_proxy)' > "$RUN_DIR/environment.txt"
# 只读取卡状态。若任一卡已有进程，直接退出。
npu-smi info > "$RUN_DIR/npu_before.txt"
"$PYTHON" - "$RUN_DIR/npu_before.txt" <<'CHECK_CARDS'
import re, sys
from pathlib import Path
status = Path(sys.argv[1]).read_text()
free = set(re.findall(r"No running processes found in NPU (\d+)", status))
if free != set(map(str, range(8))):
    raise SystemExit(f"Expected all eight NPUs free; idle NPUs reported: {sorted(free)}")
CHECK_CARDS
exec > >(tee -a "$RUN_DIR/service.log") 2>&1
printf 'PHASE=%s MODE=%s PAIR=%s LOG=%s\n' "$PHASE" "$MODE" "$PAIR_LABEL" "$RUN_DIR/service.log"
cd "$PLUGIN"
# Foreground exec keeps this shell's PID; stop this service before starting the other.
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --api-server-count 1 \
    --data-parallel-size 8 --tensor-parallel-size 1 --enable-expert-parallel \
    --served-model-name "$SERVED_MODEL" --seed "$SERVER_SEED" \
    --trust-remote-code --enforce-eager \
    --enable-chunked-prefill --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-memory-bytes "$KV_CACHE_BYTES" --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --additional-config '{"enable_moonep":true}' \
    --profiler-config "$PROFILER_CONFIG"
