# 118 单机八卡 4K LongBench-v2 · 30 秒 profiling 审阅稿

本稿供确认。三个 Shell 入口都设有 `SCRIPT_REVIEWED=0`，审核后才启用。

- 顺序为 Off → MTE Slot。一次只起一套服务。两组各自检查模型最长样本，再预热 16 条，预热归零后启动 316 条正式回放。
- 两组共用 DSV4 Flash、DP8 TP1 EP8、单 DP 4096 token、并发 16、输出 1 token、最大序列 131073、24 GiB/卡 KV、eager、关闭 prefix cache，使用模型原生路由和上一轮原文数据清单。
- Off 检查实际进程和八个 worker 的 MoonEP 状态。Slot 启用 MTE、prefetch 副流、MXFP8 dispatch、融合 publish，专家持久缓存关闭；沿用加权归并路径。压测前检查八个 worker 的模式、已映射库和 Slot 优化路径。
- 按 `model_name` 和 `engine` 读取八个 DP 的运行数、等待数和输入 token 累计值。默认每秒采样，连续 3 次在服请求数为 16，且后两个采样间隔都有正的输入 token 增量，才开启窗口。启动压测后 180 秒内没有满足门槛时，回放继续到结束，采集状态记为无效。
- 服务端采集参数 `delay_iterations=0,max_iterations=0,ignore_frontend=true`。客户端收到启动响应后开始计时，30 秒时请求停止，AISBench 继续回放。采样日志分别记录启动接口、控制窗口、停止接口耗时和各 rank 设备实际录制时间。若 AISBench 在 30 秒前结束，窗口标记为无效。
- 输出保存在本实验 `data/formal/`。每次起服独立保存八份原始 trace，AISBench 保留 debug、逐 DP 采样、采集事件、原文校验和 `CAPTURE_STATUS.json`。采集脚本只保存原始数据，离线转换和阶段分析另行处理。
- `vllm:prompt_tokens_total` 是前端按 prefill 结果更新的累计值，可用于判断窗口内有 prefill 活动。不同 rank 的瞬时分块大小仍需从采集或附加统计确认。

## 审核后运行顺序

在 118 的 `xzx_moonep` 容器、确认 8 张卡空闲后，先运行 `bash start_off.sh`，服务就绪后在另一终端运行 `bash bench_profile.sh off`。完成后结束这次 Off 服务并确认卡释放，再运行 `bash start_slot.sh` 和 `bash bench_profile.sh slot`。各脚本参数在对应文件中修改。当前未起服，也未发送请求。


## start_off.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
PHASE='formal'
PAIR_LABEL='pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='off'
SERVED_MODEL='dsv4flash_longbench_dp8_4k_off'
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
# MoonEP disabled: no MoonEP or synthetic routing environment is exported.

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
    --additional-config '{"enable_moonep":false}' \
    --profiler-config "$PROFILER_CONFIG"
```

## start_slot.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
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
```

## bench_profile.sh

```bash
#!/usr/bin/env bash
# Awaiting review. Select the already running service with off or slot.
set -Eeuo pipefail

SCRIPT_REVIEWED=0
HOST='127.0.0.1'
PORT=8444
WINDOW_SECONDS=30
CONCURRENCY=16
POLL_SECONDS=1
SATURATED_SAMPLES=3
GATE_TIMEOUT_SECONDS=180
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'

if [[ $# != 1 || ( "$1" != off && "$1" != slot ) ]]; then
    echo 'Usage: bash bench_profile.sh off|slot' >&2
    exit 2
fi
if (( SCRIPT_REVIEWED != 1 )); then
    echo 'Draft awaiting review. No requests sent.' >&2
    exit 2
fi
MODE="$1"
CONFIG="$EXPERIMENT_DIR/aisbench_$MODE.py"
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export TMPDIR='/dev/shm/moonep_longbench_dp8_4k_30s_profile_118'
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"
test -r "$EXPERIMENT_DIR/data/longbenchv2_full_le128k.json"
test -r "$EXPERIMENT_DIR/data/selection_manifest.json"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$EXPERIMENT_DIR/data/formal/aisbench"

exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
flock -n 8 || { echo 'An AISBench client from this experiment is already active.' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/data/formal/aisbench/$MODE-$(date +%Y%m%d_%H%M%S)-XXXXXXXX")"
ROUND_DIR="$RUN_DIR/round_1"
mkdir "$ROUND_DIR"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$EXPERIMENT_DIR/profile_window.py" "$RUN_DIR/profile_window.py"
cp -- "$EXPERIMENT_DIR/data/selection_manifest.json" "$RUN_DIR/selection_manifest.json"
printf '%s\n' "$MODE" > "$RUN_DIR/mode.txt"
printf 'Output directory %s\n' "$RUN_DIR"

cd "$BENCHMARK_DIR"
"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" formal "$ROUND_DIR" \
    2>&1 | tee "$ROUND_DIR/mode_audit.log"
"$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$MODE" "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/preflight.log"
if [[ "$MODE" == slot ]]; then
    service_dir=$("$PYTHON" - "$ROUND_DIR" <<'FIND_SERVICE'
import json
import sys
from pathlib import Path

audit, = Path(sys.argv[1]).glob("*_mode_audit.json")
print(json.loads(audit.read_text())["service_dir"])
FIND_SERVICE
)
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$service_dir" "$ROUND_DIR" \
        2>&1 | tee "$ROUND_DIR/optimization_audit.log"
fi
"$PYTHON" "$EXPERIMENT_DIR/profile_window.py" "$MODE" "$ROUND_DIR" \
    "http://$HOST:$PORT" "$WINDOW_SECONDS" "$CONCURRENCY" "$POLL_SECONDS" \
    "$SATURATED_SAMPLES" "$GATE_TIMEOUT_SECONDS" -- \
    "$AIS_BENCH" "$CONFIG" --mode perf --debug \
    --max-num-workers 1 --num-warmups 0 --work-dir "$ROUND_DIR/aisbench" \
    2>&1 | tee "$ROUND_DIR/aisbench_debug.log"
"$PYTHON" "$EXPERIMENT_DIR/validate_round.py" \
    "$ROUND_DIR/aisbench" "$EXPERIMENT_DIR/data/selection_manifest.json" \
    2>&1 | tee "$ROUND_DIR/validation.log"
"$PYTHON" - "$ROUND_DIR/CAPTURE_STATUS.json" <<'CHECK_CAPTURE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = json.loads(path.read_text())
if not status["valid"]:
    raise SystemExit(f"Replay validated, capture invalid: {status['reason']}. See {path}")
print(f"Valid {status['window_seconds']}s window, eight original rank traces: {status['profile_dir']}")
CHECK_CAPTURE
```

## profile_window.py

```python
"""Capture a bounded profiling window during an existing AISBench replay."""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
API = "http://127.0.0.1:8444"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, payload=None, timeout=1800):
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(API+path, data=data, method="POST",
                                 headers={"Content-Type":"application/json"})
    with OPENER.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def event(output, step, **fields):
    row = dict(at=datetime.datetime.now().astimezone().isoformat(), step=step, **fields)
    with (output/"profile_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)


def warmup(mode):
    data = json.loads((ROOT/"data/longbenchv2_full_le128k.json").read_text())
    manifest = json.loads((ROOT/"data/selection_manifest.json").read_text())
    items = sorted(manifest["samples"],key=lambda v:int(v["data_id"]))
    assert len(items)==316
    path = WORKSPACE/"benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
    templates = [n.value.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.keyword)
                 and n.arg=="prompt" and isinstance(n.value,ast.Constant)]
    template, = templates

    def one(item):
        prompt = template.format(**data[int(item["data_id"])])
        assert hashlib.sha256(prompt.encode()).hexdigest()==item["prompt_sha256"]
        result = post("/v1/completions",dict(model=f"dsv4flash_longbench_dp8_4k_{mode}",prompt=prompt,max_tokens=1,
                                             temperature=0,ignore_eos=True,seed=1024,stream=False))
        usage = result["usage"]
        assert usage["prompt_tokens"]==item["prompt_tokens"] and usage["completion_tokens"]==1
        return {"data_id":item["data_id"],"prompt_tokens":usage["prompt_tokens"],"completion_tokens":1}

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(one,items[:16]))


def read_metrics(text, model):
    fields = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:prompt_tokens_total": "prompt_tokens",
    }
    ranks = {i: {} for i in range(8)}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            field = fields.get(sample.name)
            if not field or sample.labels.get("model_name") != model:
                continue
            rank = int(sample.labels["engine"])
            value = sample.value
            if rank not in ranks or field in ranks[rank] or not math.isfinite(value):
                raise ValueError(f"Unexpected metric {sample}")
            if value < 0 or value != int(value):
                raise ValueError(f"Expected a nonnegative integer metric: {sample}")
            ranks[rank][field] = int(value)
    if any(set(row) != set(fields.values()) for row in ranks.values()):
        raise ValueError(f"Expected running, waiting, and prompt tokens on all 8 engines: {ranks}")
    return ranks


def sample_metrics(output, model, phase, previous):
    record = dict(at=datetime.datetime.now().astimezone().isoformat(), phase=phase,
                  monotonic=time.monotonic())
    current = None
    try:
        with OPENER.open(API + "/metrics", timeout=2) as response:
            current = read_metrics(response.read().decode("utf-8"), model)
        record["ranks"] = current
        record["running"] = sum(row["running"] for row in current.values())
        record["waiting"] = sum(row["waiting"] for row in current.values())
        record["in_service"] = record["running"] + record["waiting"]
        if previous is not None:
            increments = {rank: row["prompt_tokens"] - previous[rank]["prompt_tokens"]
                          for rank, row in current.items()}
            if min(increments.values()) < 0:
                raise ValueError("Input token counter reset during the round")
            record["prompt_tokens_delta_by_engine"] = increments
            record["prompt_tokens_delta"] = sum(increments.values())
    except (OSError, ValueError, KeyError) as exc:
        record["error"] = str(exc)
        current = None
    with (output / "metrics_samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record, current


def wait_idle(output, model):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row, _ = sample_metrics(output, model, "before_replay", None)
        if row.get("in_service") == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"Service did not drain after warmup: {row}")


def gate_ready(samples, target, required):
    return (
        len(samples) == required
        and all("error" not in row and row["in_service"] == target for row in samples)
        and all(row.get("prompt_tokens_delta", 0) > 0 for row in list(samples)[1:])
    )


def wait_for_load(bench, output, model, target, required, interval, timeout):
    deadline = time.monotonic() + timeout
    samples = deque(maxlen=required)
    previous = None
    while bench.poll() is None and time.monotonic() < deadline:
        began = time.monotonic()
        row, previous = sample_metrics(output, model, "gate", previous)
        samples.append(row)
        if gate_ready(samples, target, required) and time.monotonic() < deadline:
            return row, previous
        time.sleep(max(0, min(began + interval, deadline) - time.monotonic()))
    return None, previous


def check_service_profile(output, mode):
    audit, = output.glob("*_mode_audit.json")
    report = json.loads(audit.read_text())
    if not report["valid"] or report["mode"] != mode:
        raise ValueError("Service mode audit failed")
    command = report["actual_command"]
    config = json.loads(command[command.index("--profiler-config") + 1])
    profile_dir = Path(report["service_dir"]) / "profiling"
    required = dict(profiler="torch", torch_profiler_dir=str(profile_dir),
                    delay_iterations=0, max_iterations=0, ignore_frontend=True,
                    torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                    torch_profiler_with_memory=False)
    if config != required or any(profile_dir.iterdir()):
        raise ValueError("Expected an empty, per-service profiling directory and the reviewed config")
    workers = {row["rank"]: row["pid"] for row in report["workers"]}
    return profile_dir, workers


def inspect_traces(profile_dir, workers):
    traces = {}
    for path in sorted(profile_dir.glob("dp*_ascend_pt")):
        match = re.fullmatch(r"dp(\d+)_pp\d+_tp\d+_dcp\d+_ep(\d+)_rank\d+_(\d+)_\d+_ascend_pt",
                             path.name)
        if not match:
            raise ValueError(f"Unexpected trace directory: {path}")
        rank, ep, pid = map(int, match.groups())
        if rank != ep or workers.get(rank) != pid or rank in traces:
            raise ValueError(f"Trace does not match the audited worker: {path}")
        info, = path.glob("profiler_info_*.json")
        meta = json.loads(info.read_text())
        device, = path.glob("PROF_*/device_*")
        if not any((device / "data").iterdir()) or not any(device.glob("end_info.*.done")):
            raise ValueError(f"Device capture missing data or end marker: {path}")
        if (path / "FRAMEWORK/torch.op_range").stat().st_size == 0:
            raise ValueError(f"CPU capture empty: {path}")
        seconds = (meta["end_info"]["MonotonicTimeEnd"] -
                   meta["start_info"]["start_monotonic"]) / 1e9
        if seconds <= 0 or meta["rank_id"] != rank:
            raise ValueError(f"Invalid rank or timing metadata: {info}")
        traces[rank] = dict(rank=rank, worker_pid=pid, seconds=seconds,
                            trace_dir=str(path), profiler_info=str(info))
    if set(traces) != set(workers) or len(traces) != 8:
        raise ValueError(f"Expected 8 original traces, found {sorted(traces)}")
    return list(traces.values())


def capture(bench, output, model, previous, window, interval, status):
    stop_polling = threading.Event()
    poller = None

    def poll_window():
        last = previous
        while not stop_polling.is_set():
            began = time.monotonic()
            _, last = sample_metrics(output, model, "window", last)
            stop_polling.wait(max(0, began + interval - time.monotonic()))

    try:
        request_start = time.monotonic()
        event(output, "profile_start_requested")
        post("/start_profile", timeout=90)
        opened = time.monotonic()
        status["start_api_seconds"] = opened - request_start
        event(output, "profile_started", window_seconds=window)
        poller = threading.Thread(target=poll_window, daemon=True)
        poller.start()
        try:
            bench.wait(timeout=max(0, opened + window - time.monotonic()))
        except subprocess.TimeoutExpired:
            status["window_completed"] = True
    finally:
        stop_polling.set()
        request_stop = time.monotonic()
        if "start_api_seconds" in status:
            status["controller_window_seconds"] = request_stop - opened
        event(output, "profile_stop_requested")
        post("/stop_profile", timeout=180)
        status["stop_api_seconds"] = time.monotonic() - request_stop
        event(output, "profile_stopped")
        if poller is not None:
            poller.join(timeout=3)


def stop_client(bench):
    if bench.poll() is None:
        try:
            os.killpg(bench.pid, signal.SIGTERM)
            bench.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(bench.pid, signal.SIGKILL)
            bench.wait(timeout=10)


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    global API
    mode, folder, api, window, target, interval, required, timeout, separator, *command = sys.argv[1:]
    API = api
    if mode not in ("off", "slot") or separator != "--" or not command:
        raise ValueError("Expected mode, output, API, window, target, interval, samples, timeout, --, command")
    window, target, interval, required, timeout = map(int, (window, target, interval, required, timeout))
    if min(window, target, interval, timeout) <= 0 or required < 2:
        raise ValueError("Invalid profiling timing or load threshold")
    output = Path(folder)
    model = f"dsv4flash_longbench_dp8_4k_{mode}"
    profile_dir, workers = check_service_profile(output, mode)
    status = dict(mode=mode, valid=False, reason="incomplete", window_seconds=window,
                  window_completed=False, profile_dir=str(profile_dir))
    bench = None
    try:
        event(output, "warmup_start", mode=mode, requests=16)
        warm = warmup(mode)
        (output / "warmup_validated.json").write_text(json.dumps(warm, indent=2) + "\n")
        event(output, "warmup_validated", mode=mode, requests=len(warm))
        wait_idle(output, model)
        replay_start = time.monotonic()
        bench = subprocess.Popen(command, start_new_session=True)
        event(output, "aisbench_started", mode=mode, pid=bench.pid)
        gated, previous = wait_for_load(bench, output, model, target, required, interval, timeout)
        if gated is None:
            status["reason"] = "saturation/prefill gate not reached"
            event(output, "capture_skipped", reason=status["reason"])
        else:
            event(output, "gate_reached", metrics=gated)
            capture(bench, output, model, previous, window, interval, status)
            try:
                status["ranks"] = inspect_traces(profile_dir, workers)
            except (OSError, ValueError, KeyError) as exc:
                status["reason"] = str(exc)
            else:
                status["valid"] = status["window_completed"]
                status["reason"] = "ok" if status["valid"] else "AISBench ended before the full window"
        if bench.wait(timeout=max(0, replay_start + 7200 - time.monotonic())) != 0:
            status.update(valid=False, reason=f"AISBench exited {bench.returncode}")
            raise subprocess.CalledProcessError(bench.returncode, command)
        event(output, "aisbench_complete", mode=mode)
    except BaseException as exc:
        if status["reason"] in ("incomplete", "ok"):
            status.update(valid=False, reason=str(exc))
        raise
    finally:
        if bench is not None:
            stop_client(bench)
        (output / "CAPTURE_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    main()
```

## aisbench_off.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base
from literal_longbench_prompt import LiteralLongBenchPrompt

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-dp8-4k-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_dp8_4k_off",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=1,  # This aisbench version counts total attempts; 1 means no retries.
        api_key="",
        host_ip="127.0.0.1",
        host_port=8444,
        url="",
        max_out_len=1,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0, ignore_eos=True, seed=1024),
    )
]
datasets = [
    dict(
        _base_datasets[0],
        abbr="longbenchv2-full-le128k",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261109_dp8_longbenchv2_4k_30s_profile_118/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.

# Approved original-text fix: preserve literal placeholders inside original context/code text.
datasets[0]["infer_cfg"] = dict(
    _base_datasets[0]["infer_cfg"],
    prompt_template=dict(_base_datasets[0]["infer_cfg"]["prompt_template"], type=LiteralLongBenchPrompt),
)
```

## aisbench_slot.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base
from literal_longbench_prompt import LiteralLongBenchPrompt

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-dp8-4k-slot",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_dp8_4k_slot",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=1,  # This aisbench version counts total attempts; 1 means no retries.
        api_key="",
        host_ip="127.0.0.1",
        host_port=8444,
        url="",
        max_out_len=1,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0, ignore_eos=True, seed=1024),
    )
]
datasets = [
    dict(
        _base_datasets[0],
        abbr="longbenchv2-full-le128k",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261109_dp8_longbenchv2_4k_30s_profile_118/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.

# Approved original-text fix: preserve literal placeholders inside original context/code text.
datasets[0]["infer_cfg"] = dict(
    _base_datasets[0]["infer_cfg"],
    prompt_template=dict(_base_datasets[0]["infer_cfg"]["prompt_template"], type=LiteralLongBenchPrompt),
)
```

## 沿用的辅助脚本

`audit_service_mode.py`、`audit_optimizations.py`、`preflight_api.py`、`validate_round.py` 和 `literal_longbench_prompt.py` 复制自上一轮。两份数据文件通过符号链接复用上一轮的原始数据和清单。
