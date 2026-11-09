#!/usr/bin/env bash
# Approved Home GEMM1 experiment. Edit all tuning values inside this file.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PAIR_LABEL='profiling_pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/diagnostics/profiling'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='slot_udma'
SERVED_MODEL='dsv4flash_dp16_longbench_slot_udma'
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
    --output "$EXPERIMENT_DIR/data/idle_slot_udma_118.json"

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
# MoonEP Slot: native router, real-token bounds, shared working slots, three optimizations.
export MOONEP=1
export MOONEP_SLOT_MODE=1
export MOONEP_REAL_T=1
export MOONEP_MIN_DISPATCH_TOKENS=16
export MOONEP_SHMEM_BOOTSTRAP='ipport'
export MOONEP_DISABLE_UDMA=0
export ASCEND_MOONEP_PLANNING_USE_UDMA=0
export ASCEND_MOONEP_DISPATCH_USE_UDMA=1
export ASCEND_MOONEP_COMBINE_USE_UDMA=1
export ASCEND_MOONEP_PREFETCH_USE_UDMA=1
export ASCEND_MOONEP_BUILD_DIR="$MOONEP_PACKAGE"
export MOONEP_SHMEM_IP_PORT=29765
export MOONEP_SLOT_QUANT=1
export MOONEP_SLOT_CACHE=0
export MOONEP_EARLY_PREFETCH=0
export MOONEP_PF_STREAM=3
export MOONEP_KEEP_MC2=0
export MOONEP_BALANCE_STATS=0
export MOONEP_PLAN_STATS=0
export MOONEP_TIME_STATS=0
export MOONEP_PF_TIMING=0
export MOONEP_SLOT_STATS=0
export MOONEP_CAP_TRACE=0
export ASCEND_MOONEP_TRACE=0
export ASCEND_MOONEP_TIME_STATS=0
export ASCEND_MOONEP_PREFETCH_RELAXED=1
export ASCEND_MOONEP_NO_RECONCILE=1
export ASCEND_MOONEP_SKIP_DUP=1
export LD_LIBRARY_PATH="$MOONEP_PACKAGE/lib:$LD_LIBRARY_PATH"
# No VLLM_ASCEND_MOE_SIM_ROUTE, balance stats, plan stats, or timing instrumentation.


# Local resource protection. Do not stop another experiment.
"$PYTHON" - <<'CHECK_IDLE'
import re,subprocess
text=subprocess.check_output(['npu-smi','info'],text=True)
if set(re.findall(r'No running processes found in NPU (\d+)',text))!=set(map(str,range(8))):
    raise SystemExit('Eight idle local cards are required; current single-node experiment is protected.')
CHECK_IDLE

# Read-only gate: no SDK modification or legacy-combine fallback is performed.
# Require the real backend-specific Home-GEMM certificate; no boolean override.
"$PYTHON" - "$PLUGIN" "$EXPERIMENT_DIR" <<'CHECK_SLOT'
from pathlib import Path
import json,sys
source=Path(sys.argv[1])/'vllm_ascend/ops/fused_moe/moonep_comm.py'
if 'ip_port=f"tcp://127.0.0.1:' in source.read_text():
    raise SystemExit('Current Slot bootstrap is loopback-only; no two-node service is started.')
path=Path(sys.argv[2])/'data/preflight/slot_udma_dp16_validation.json'
d=json.loads(path.read_text())
assert d['world_size']==16 and d['valid_ranks']==16 and d['fused_weighted_combine']
assert d['backend']=='udma' and d['use_udma'] is True
assert d['cross_node_validated'] and d['expert_prefetch_validated']
assert d['planning_dispatch_validated'] and d['precision_passed']
assert d['planning_backend']=='mte'
assert d['prefetch_stream_mode']==3 and d['prefetch_side_stream_validated']
assert d['home_gemm1_validated'] and d['gemm2_unsplit']
import hashlib
for binary,sha in d['runtime_binary_sha256'].items():
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==sha
CHECK_SLOT

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
    --additional-config '{"enable_moonep":true,"dp_allreduce_on_npu":true}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/diagnostics/profiling/data/profiling_v2/slot_udma","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":0,"max_iterations":3}'
