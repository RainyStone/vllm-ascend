# LongBench-v2 补充诊断草案

原四服务性能矩阵继续按已批准脚本完成；以下独立诊断仅在矩阵结束后执行，不并入正式性能数据。

计划：两次额外起服off/slot，模型与原性能参数保持一致。开启现成MoE负载记录，每步落盘；316条同数据、并发16、输出1，补采一次负载。之后少量固定prompt按串行及固定并发重复，并请求top-logprobs，区分重复不稳定、跨模式变化与低margin翻转。无需修改共享aisbench预设或模型代码。

补采报告：每step/layer各rank实际负载、原生专家归属负载、rank最大/平均和专家最大/平均，p50/p90/p99、每rank总量、Slot重分配前后对比；并检查路由与实际计算条目守恒。不把补采不均衡比冒称为原性能轮逐步记录。

首token不一致仅为现象，不能提前认定普通浮点误差，也不能由单token实验判断完整LongBench精度。若仍无法定位，将列出证据和未解决环节。

新增服务环境（其余原参数不变）：VLLM_ASCEND_MOE_PAR_STATS=1，专用输出目录，专用TAG，VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1。两份独立sh均在文件内部设置，不接收外部环境参数。

## start_off_diagnostic.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PAIR_LABEL='diagnostic'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='off'
SERVED_MODEL='dsv4flash_longbench_off'
CARDS='0,1,2,3,4,5,6,7'
HOST='127.0.0.1'
PORT=8444
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB per card; candidate, startup not yet validated.
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
# CANN's vendor environment script may reference unset variables.
source "$CANN_ENV"
CANN_PYTHONPATH="$PYTHONPATH"
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
# MoonEP disabled: no MoonEP or synthetic routing environment is exported.


# Diagnostic-only instrumentation; do not use this service for timing comparisons.
export VLLM_ASCEND_MOE_PAR_STATS=1
export VLLM_ASCEND_MOE_PAR_STATS_DIR="$EXPERIMENT_DIR/diagnostics/data/off/routing"
export VLLM_ASCEND_MOE_PAR_STATS_TAG='longbench_diagnostic_off'
export VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1

mkdir -p "$EXPERIMENT_DIR/data/services"
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
CHECK_PORT

RUN_DIR="$EXPERIMENT_DIR/data/services/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/start_script.sh"
printf '%s\n' "$$" > "$RUN_DIR/service.pid"
ps -o pgid= -p "$$" > "$RUN_DIR/service.pgid"
env | LC_ALL=C sort | grep -E '^(MOONEP|ASCEND_MOONEP|VLLM|ASCEND_RT_VISIBLE_DEVICES|ASCEND_CUSTOM_OPP_PATH|HCCL|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME|OMP|PYTORCH_NPU_ALLOC_CONF|PYTHONPATH|LD_LIBRARY_PATH|NO_PROXY|no_proxy)' > "$RUN_DIR/environment.txt"
npu-smi info > "$RUN_DIR/npu_before.txt"
exec > >(tee -a "$RUN_DIR/service.log") 2>&1
printf 'MODE=%s PAIR=%s LOG=%s\n' "$MODE" "$PAIR_LABEL" "$RUN_DIR/service.log"
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
    --additional-config '{"enable_moonep":false}'
```

## start_slot_diagnostic.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=1
PAIR_LABEL='diagnostic'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='slot'
SERVED_MODEL='dsv4flash_longbench_slot'
CARDS='0,1,2,3,4,5,6,7'
HOST='127.0.0.1'
PORT=8444
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB per card; candidate, startup not yet validated.
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
# CANN's vendor environment script may reference unset variables.
source "$CANN_ENV"
CANN_PYTHONPATH="$PYTHONPATH"
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
# MoonEP Slot: native router, real-token bounds, shared working slots, three optimizations.
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
export MOONEP_PF_STREAM=0
export MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_PREFETCH_RELAXED=1
export ASCEND_MOONEP_NO_RECONCILE=1
export ASCEND_MOONEP_SKIP_DUP=1
export LD_LIBRARY_PATH="$MOONEP_PACKAGE/lib:$LD_LIBRARY_PATH"
# No VLLM_ASCEND_MOE_SIM_ROUTE, balance stats, plan stats, or timing instrumentation.



# Diagnostic-only instrumentation; do not use this service for timing comparisons.
export VLLM_ASCEND_MOE_PAR_STATS=1
export VLLM_ASCEND_MOE_PAR_STATS_DIR="$EXPERIMENT_DIR/diagnostics/data/slot/routing"
export VLLM_ASCEND_MOE_PAR_STATS_TAG='longbench_diagnostic_slot'
export VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1

mkdir -p "$EXPERIMENT_DIR/data/services"
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
CHECK_PORT

RUN_DIR="$EXPERIMENT_DIR/data/services/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/start_script.sh"
printf '%s\n' "$$" > "$RUN_DIR/service.pid"
ps -o pgid= -p "$$" > "$RUN_DIR/service.pgid"
env | LC_ALL=C sort | grep -E '^(MOONEP|ASCEND_MOONEP|VLLM|ASCEND_RT_VISIBLE_DEVICES|ASCEND_CUSTOM_OPP_PATH|HCCL|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME|OMP|PYTORCH_NPU_ALLOC_CONF|PYTHONPATH|LD_LIBRARY_PATH|NO_PROXY|no_proxy)' > "$RUN_DIR/environment.txt"
npu-smi info > "$RUN_DIR/npu_before.txt"
exec > >(tee -a "$RUN_DIR/service.log") 2>&1
printf 'MODE=%s PAIR=%s LOG=%s\n' "$MODE" "$PAIR_LABEL" "$RUN_DIR/service.log"
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
    --additional-config '{"enable_moonep":true}'
```

## bench_off_diagnostic.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=1
PAIR_LABEL='diagnostic'  # Set pair02 for the reversed independent-service pair.
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='off'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_longbench_off'
CONFIG="$EXPERIMENT_DIR/aisbench_off.py"
ROUNDS=1
FIRST_ROUND_WARMUPS=0
# ===== End of editable parameters; API batch_size/max_out_len are in CONFIG =====

if (( $# != 0 )); then
    echo 'No positional arguments: edit the values inside this script.' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft: script/config contents are awaiting user review. No requests sent.' >&2
    exit 2
fi
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_full_le128k.json"
test -r "$EXPERIMENT_DIR/data/selection_manifest.json"

# Prevent concurrent clients from accidentally exceeding total concurrency 16.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another aisbench client from this experiment is active.' >&2
    exit 1
fi
"$PYTHON" - "$HOST" "$PORT" "$SERVED_MODEL" <<'CHECK_SERVICE'
import json, sys, urllib.request
host, port, expected = sys.argv[1:]
with urllib.request.urlopen(f'http://{host}:{port}/v1/models', timeout=15) as response:
    names = [entry['id'] for entry in json.load(response)['data']]
if expected not in names:
    raise SystemExit(f'Wrong service: expected {expected}, found {names}')
print('Verified service:', expected)
CHECK_SERVICE

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/data/selection_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf '%s\n' "$PAIR_LABEL" > "$RUN_DIR/pair_label.txt"
printf 'Diagnostic routing census only. Instrumented timings excluded from formal comparisons.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
    "$PYTHON" "$EXPERIMENT_DIR/diagnostics/audit_diagnostic_mode.py" "$MODE" auto "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/mode_audit.log"
    npu-smi info > "$ROUND_DIR/npu_before.txt"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/started_at.txt"
    # --debug prints to the terminal; tee retains stdout + stderr in full.
    # No --num-prompts subset, no --reuse, no --pressure, no implicit retry.
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
        --max-num-workers 1 --num-warmups "$warmups" \
        --work-dir "$ROUND_DIR/aisbench" \
        2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
    "$PYTHON" "$EXPERIMENT_DIR/validate_round.py" \
        "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/selection_manifest.json" \
        2>&1 | tee "$ROUND_DIR/validation.log"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/ended_at.txt"
    npu-smi info > "$ROUND_DIR/npu_after.txt"
done
printf 'All raw results and debug logs retained in %s\n' "$RUN_DIR"
```

## bench_slot_diagnostic.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=1
PAIR_LABEL='diagnostic'  # Set pair02 for the reversed independent-service pair.
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='slot'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_longbench_slot'
CONFIG="$EXPERIMENT_DIR/aisbench_slot.py"
ROUNDS=1
FIRST_ROUND_WARMUPS=0
# ===== End of editable parameters; API batch_size/max_out_len are in CONFIG =====

if (( $# != 0 )); then
    echo 'No positional arguments: edit the values inside this script.' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft: script/config contents are awaiting user review. No requests sent.' >&2
    exit 2
fi
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_full_le128k.json"
test -r "$EXPERIMENT_DIR/data/selection_manifest.json"

# Prevent concurrent clients from accidentally exceeding total concurrency 16.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another aisbench client from this experiment is active.' >&2
    exit 1
fi
"$PYTHON" - "$HOST" "$PORT" "$SERVED_MODEL" <<'CHECK_SERVICE'
import json, sys, urllib.request
host, port, expected = sys.argv[1:]
with urllib.request.urlopen(f'http://{host}:{port}/v1/models', timeout=15) as response:
    names = [entry['id'] for entry in json.load(response)['data']]
if expected not in names:
    raise SystemExit(f'Wrong service: expected {expected}, found {names}')
print('Verified service:', expected)
CHECK_SERVICE

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/data/selection_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf '%s\n' "$PAIR_LABEL" > "$RUN_DIR/pair_label.txt"
printf 'Diagnostic routing census only. Instrumented timings excluded from formal comparisons.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
    "$PYTHON" "$EXPERIMENT_DIR/diagnostics/audit_diagnostic_mode.py" "$MODE" auto "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/mode_audit.log"
    npu-smi info > "$ROUND_DIR/npu_before.txt"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/started_at.txt"
    # --debug prints to the terminal; tee retains stdout + stderr in full.
    # No --num-prompts subset, no --reuse, no --pressure, no implicit retry.
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
        --max-num-workers 1 --num-warmups "$warmups" \
        --work-dir "$ROUND_DIR/aisbench" \
        2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
    "$PYTHON" "$EXPERIMENT_DIR/validate_round.py" \
        "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/selection_manifest.json" \
        2>&1 | tee "$ROUND_DIR/validation.log"
    printf '%s\n' "$(date -Is)" > "$ROUND_DIR/ended_at.txt"
    npu-smi info > "$ROUND_DIR/npu_after.txt"
done
printf 'All raw results and debug logs retained in %s\n' "$RUN_DIR"
```

## audit_diagnostic_mode.py

```python
#!/usr/bin/env python3
"""Audit the running service's mode using actual /proc state and worker logs."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parents[3]
REQUIRED_SLOT={
    "MOONEP":"1","MOONEP_SLOT_MODE":"1","MOONEP_REAL_T":"1",
    "MOONEP_SLOT_CACHE":"0","MOONEP_EARLY_PREFETCH":"0","MOONEP_PF_STREAM":"0",
    "MOONEP_KEEP_MC2":"0","ASCEND_MOONEP_PREFETCH_RELAXED":"1",
    "ASCEND_MOONEP_NO_RECONCILE":"1","ASCEND_MOONEP_SKIP_DUP":"1",
}

def main():
    mode,service_dir,session=sys.argv[1:]
    assert mode in ("off","slot")
    if service_dir=="auto":
        candidates=[]
        for candidate in (ROOT/"data/services").glob(f"*-{mode}-*"):
            pid_file=candidate/"service.pid"
            if pid_file.is_file() and (Path("/proc")/pid_file.read_text().strip()/"cmdline").exists():
                candidates.append(candidate)
        assert candidates, "No live service metadata for mode "+mode
        service_dir=str(max(candidates,key=lambda x:x.stat().st_mtime))
    directory=Path(service_dir);log=(directory/"service.log").read_text(errors="replace")
    parent=int((directory/"service.pid").read_text())
    cmd=(Path("/proc")/str(parent)/"cmdline").read_bytes().decode().rstrip("\0").split("\0")
    additional=json.loads(cmd[cmd.index("--additional-config")+1])
    assert additional["enable_moonep"] is (mode=="slot"),additional
    assert additional == {"enable_moonep": mode=="slot"}, additional
    for key, value in {"--max-num-seqs":"16","--max-num-batched-tokens":"8192",
                       "--data-parallel-size":"8","--tensor-parallel-size":"1",
                       "--served-model-name":"dsv4flash_longbench_"+mode}.items():
        assert cmd[cmd.index(key)+1] == value, (key,cmd)
    for flag in ["--enable-expert-parallel","--enable-chunked-prefill","--enforce-eager"]:
        assert flag in cmd
    assert "--no-enable-prefix-caching" in cmd
    assert cmd[cmd.index("--max-model-len")+1]=="131073"
    assert cmd[cmd.index("--kv-cache-memory-bytes")+1]=="25769803776"
    ranks={}
    for dp,ep,pid in re.findall(r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)",log):
        assert dp==ep,(dp,ep,pid)
        ranks[int(dp)]=int(pid)
    assert set(ranks)==set(range(8)),ranks
    runtime_audits={}
    pattern=r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)\s*\[moonep-env-audit\]\s+([^\n]+)"
    for dp,ep,pid,message in re.findall(pattern,log):
        if dp==ep:
            runtime_audits[int(pid)]=dict(re.findall(r"([A-Z0-9_]+)=([^\s(]+)",message))
    parent_environment={}
    for line in (directory/"environment.txt").read_text().splitlines():
        if "=" in line:
            k,v=line.split("=",1);parent_environment[k]=v
    workers=[]
    for rank,pid in sorted(ranks.items()):
        proc=Path("/proc")/str(pid)
        environment=dict(pair.split("=",1) for pair in proc.joinpath("environ").read_bytes().decode().split("\0") if "=" in pair)
        relevant={k:v for k,v in environment.items() if k.startswith(("MOONEP","ASCEND_MOONEP","VLLM_ASCEND_MOE_"))}
        synthetic=[k for k in relevant if k.startswith("VLLM_ASCEND_MOE_SIM_ROUTE")]
        assert not synthetic,(rank,synthetic)
        maps=proc.joinpath("maps").read_text()
        libraries=sorted(set(line.split()[-1] for line in maps.splitlines() if "/ascend-moonep/" in line or "moonep" in Path(line.split()[-1]).name.lower()))
        if mode=="off":
            allowed_stats={
                'VLLM_ASCEND_MOE_PAR_STATS':'1',
                'VLLM_ASCEND_MOE_PAR_STATS_DIR':str(ROOT/'diagnostics/data/off/routing'),
                'VLLM_ASCEND_MOE_PAR_STATS_TAG':'longbench_diagnostic_off',
                'VLLM_ASCEND_MOE_PAR_FLUSH_EVERY':'1',
            }
            assert all(k in allowed_stats and allowed_stats[k]==v for k,v in relevant.items()),(rank,relevant)
            assert not libraries,(rank,libraries)
        else:
            # /proc/environ is the exec-time snapshot, not Python's current os.environ.
            # vLLM reinjects worker variables after exec; use the runtime audit emitted
            # inside ensure_runtime_initialized, plus the actual Slot-only publish path.
            assert all(parent_environment.get(k)==v for k,v in REQUIRED_SLOT.items()),parent_environment
            for key,value in REQUIRED_SLOT.items():
                if key in environment:
                    assert environment[key]==value,(rank,key,environment[key])
            runtime=runtime_audits.get(pid,{})
            for key in ["MOONEP_SLOT_MODE","MOONEP_REAL_T","MOONEP_SLOT_QUANT",
                        "ASCEND_MOONEP_PREFETCH_RELAXED","ASCEND_MOONEP_NO_RECONCILE",
                        "ASCEND_MOONEP_SKIP_DUP"]:
                assert runtime.get(key)=="1",(rank,key,runtime)
            assert runtime.get("MOONEP_MIN_DISPATCH_TOKENS")=="16",(rank,runtime)
            assert libraries,("MoonEP library not mapped in worker",rank)
            assert f"[moonep-init r{rank}] stage=shmem-init-done" in log,rank
            assert f"[moonep-release r{rank}] home chunks now back the layers" in log,rank
        workers.append(dict(rank=rank,pid=pid,initial_exec_environment=relevant,
                            runtime_initialization_environment=runtime_audits.get(pid),
                            mapped_moonep_libraries=libraries))
    if mode=="off":
        assert "[moonep-env-audit]" not in log and "[moonep-init r" not in log
    sources=[WS/"vllm-ascend/vllm_ascend/ascend_forward_context.py",
             WS/"vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py",
             WS/"vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py"]
    report=dict(mode=mode,valid=True,service_pid=parent,service_dir=str(directory),
                actual_command=cmd,actual_additional_config=additional,workers=workers,
                native_router=True,prefix_cache_enabled=False,
                source_sha256={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                execution_contract=("enable_moonep=false: SHMEM method not registered; SoC communication selection applies"
                   if mode=="off" else "enable_moonep=true + KEEP_MC2=0: SHMEM override, no silent fallback; SLOT_MODE=1 selects run_slot_forward"),
                evidence_limit="This is actual process/configuration/library/initialization verification plus source execution contract; no profiler trace is claimed.")
    (directory/"MODE_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    output=Path(session)/(directory.name+"_mode_audit.json")
    output.write_text(json.dumps(report,indent=2)+"\n")
    print("MODE_AUDIT_OK",mode,"8/8 actual worker processes; native router; expected MoonEP flags and libraries.")
    print("AUDIT_FILE",output)

if __name__=="__main__": main()
```
