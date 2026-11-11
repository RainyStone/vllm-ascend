# GSM8K 单机 DP8 实验审阅稿

状态为待审阅。四个入口均设置 SCRIPT_REVIEWED=0。

一对独立服务按 Off → Slot 排列。每组先核验模式和最长问题，再由 aisbench 预热 16 条，正式跑全部 1,319 条。两组都使用 GSM8K 标准 test.jsonl 的原始 question，流式 completions，并发 16，输出 1 token。

离线分词范围为 22–185 token，正式输入合计预期 77,756 token/组。开测前 API 分词与最长样本 usage 必须匹配，计时后逐条验证原文、输入和输出 token。离线计数用当前模型 tokenizer.json，服务侧校验负责确认口径。

共同服务配置为 DeepSeek-V4-Flash、DP8/TP1/EP8、eager、chunked prefill、预算 8192、最大长度 131073、KV 24 GiB/卡和关闭 prefix cache。Slot 开 MXFP8 dispatch、融合 publish 和副流档 2，强制 MTE，缓存及提前预取为 0。

Off 保留原生通信选择。如果最短或最长问题的校验阶段出现 MC2 建链失败，需先确认替代 Off 口径。实验脚本中的 preflight 会在 API 失败或 token 数不匹配时直接退出。

每组保留源码和已映射 MoonEP 库的 SHA、运行前后审计、aisbench_debug.log、逐请求 JSONL 和 SQLite。metrics.json 含平均 TTFT、P95 TTFT 和输入吞吐。报告按 (Slot/Off−1)×100% 比较。

脚本从 2026-11-06 LongBench-v2 版本派生，已通过 bash -n、Python 语法和两份 aisbench 配置加载检查。用上一轮留存 SQLite 核对过 TTFT 计算格式。GSM8K 在线校验与压测留待批准后的运行阶段。

## 六个入口及配置全文

### start_off.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_gsm8k_mxfp8_publish'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='off'
SERVED_MODEL='dsv4flash_gsm8k_off'
CARDS='0,1,2,3,4,5,6,7'
HOST='127.0.0.1'
PORT=8444
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB per card.
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
export TMPDIR="/dev/shm/moonep_gsm8k_20261107"
mkdir -p "$TMPDIR"
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
"$PYTHON" - "$RUN_DIR/npu_before.txt" "$CARDS" <<'CHECK_CARDS'
import re
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_text()
free = set(re.findall(r"No running processes found in NPU (\d+)", report))
requested = set(sys.argv[2].split(","))
if not requested <= free:
    raise SystemExit(f"Requested NPUs are occupied or unverified: {sorted(requested - free)}")
CHECK_CARDS
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

### start_slot.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_gsm8k_mxfp8_publish'
PYTHON='/usr/local/python3.11.10/bin/python'
CANN_ENV='/usr/local/Ascend/cann/set_env.sh'
MODEL='/mnt/share/weight/DeepSeek-V4-Flash'
MODE='slot'
SERVED_MODEL='dsv4flash_gsm8k_slot'
CARDS='0,1,2,3,4,5,6,7'
HOST='127.0.0.1'
PORT=8444
MAX_NUM_SEQS=16
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=131073
KV_CACHE_BYTES=25769803776  # 24 GiB per card.
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
export TMPDIR="/dev/shm/moonep_gsm8k_20261107"
mkdir -p "$TMPDIR"
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
"$PYTHON" - "$RUN_DIR/npu_before.txt" "$CARDS" <<'CHECK_CARDS'
import re
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_text()
free = set(re.findall(r"No running processes found in NPU (\d+)", report))
requested = set(sys.argv[2].split(","))
if not requested <= free:
    raise SystemExit(f"Requested NPUs are occupied or unverified: {sorted(requested - free)}")
CHECK_CARDS
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

### bench_off.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_gsm8k_mxfp8_publish'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='off'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_gsm8k_off'
CONFIG="$EXPERIMENT_DIR/aisbench_off.py"
WARMUPS=16
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
export TMPDIR="/dev/shm/moonep_gsm8k_20261107"
mkdir -p "$TMPDIR"
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
mkdir -p "$HF_DATASETS_CACHE"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"

# Keep the pair sequential and retain a fresh work directory for each service.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another client from this experiment is active.' >&2
    exit 1
fi

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
ROUND_DIR="$RUN_DIR/round_1"
MANIFEST="$EXPERIMENT_DIR/data/selection_manifest.json"
mkdir -p "$ROUND_DIR/precheck" "$ROUND_DIR/postcheck"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$MANIFEST" "$RUN_DIR/selection_manifest.json"

"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" auto "$ROUND_DIR/precheck" \
    2>&1 | tee "$ROUND_DIR/mode_audit.log"
SERVICE_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["service_dir"])' \
    "$ROUND_DIR/precheck/mode_audit.json")

"$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$HOST" "$PORT" "$SERVED_MODEL" "$MANIFEST" "$ROUND_DIR" \
    2>&1 | tee "$ROUND_DIR/preflight.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/precheck" \
        2>&1 | tee "$ROUND_DIR/optimization_audit.log"
fi
npu-smi info > "$ROUND_DIR/npu_before.txt"
cd "$BENCHMARK_DIR"
"$AIS_BENCH" "$CONFIG" --mode perf --debug \
    --max-num-workers 1 --num-warmups "$WARMUPS" \
    --work-dir "$ROUND_DIR/aisbench" \
    2>&1 | tee "$ROUND_DIR/aisbench_debug.log"

# Recheck the same service while its workers are still alive.
"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
    2>&1 | tee "$ROUND_DIR/mode_postcheck.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
        2>&1 | tee "$ROUND_DIR/optimization_postcheck.log"
fi
"$PYTHON" - "$ROUND_DIR" <<'CHECK_AUDIT'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "precheck/mode_audit.json").read_text())
after = json.loads((root / "postcheck/mode_audit.json").read_text())
for key in ("service_pid", "source_sha256"):
    if before[key] != after[key]:
        raise SystemExit(f"Service or audited source changed during measurement: {key}")
workers_before = {worker["rank"]: worker for worker in before["workers"]}
workers_after = {worker["rank"]: worker for worker in after["workers"]}
for rank, worker in workers_before.items():
    current = workers_after[rank]
    if worker["pid"] != current["pid"]:
        raise SystemExit(f"Worker changed during measurement: rank {rank}")
    for library, digest in worker["library_sha256"].items():
        if current["library_sha256"].get(library) != digest:
            raise SystemExit(f"Library changed during measurement: {library}")
CHECK_AUDIT
"$PYTHON" "$EXPERIMENT_DIR/validate_round.py" "$ROUND_DIR/aisbench" "$MANIFEST" \
    2>&1 | tee "$ROUND_DIR/validation.log"
npu-smi info > "$ROUND_DIR/npu_after.txt"
printf 'Raw logs and validated metrics: %s\n' "$ROUND_DIR"
```

### bench_slot.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261107_dp8_gsm8k_mxfp8_publish'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='slot'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_gsm8k_slot'
CONFIG="$EXPERIMENT_DIR/aisbench_slot.py"
WARMUPS=16
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
export TMPDIR="/dev/shm/moonep_gsm8k_20261107"
mkdir -p "$TMPDIR"
export HF_DATASETS_CACHE="$EXPERIMENT_DIR/data/hf_datasets_cache"
mkdir -p "$HF_DATASETS_CACHE"
export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
export NO_PROXY='127.0.0.1,localhost,141.61.45.118'
export no_proxy="$NO_PROXY"

# Keep the pair sequential and retain a fresh work directory for each service.
exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"
if ! flock -n 8; then
    echo 'Another client from this experiment is active.' >&2
    exit 1
fi

RUN_DIR="$EXPERIMENT_DIR/data/aisbench/$PAIR_LABEL-$MODE-$(date +%Y%m%d_%H%M%S)-$$"
ROUND_DIR="$RUN_DIR/round_1"
MANIFEST="$EXPERIMENT_DIR/data/selection_manifest.json"
mkdir -p "$ROUND_DIR/precheck" "$ROUND_DIR/postcheck"
cp -- "$0" "$RUN_DIR/bench_script.sh"
cp -- "$CONFIG" "$RUN_DIR/aisbench_config.py"
cp -- "$MANIFEST" "$RUN_DIR/selection_manifest.json"

"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" auto "$ROUND_DIR/precheck" \
    2>&1 | tee "$ROUND_DIR/mode_audit.log"
SERVICE_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["service_dir"])' \
    "$ROUND_DIR/precheck/mode_audit.json")

"$PYTHON" "$EXPERIMENT_DIR/preflight_api.py" "$HOST" "$PORT" "$SERVED_MODEL" "$MANIFEST" "$ROUND_DIR" \
    2>&1 | tee "$ROUND_DIR/preflight.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/precheck" \
        2>&1 | tee "$ROUND_DIR/optimization_audit.log"
fi
npu-smi info > "$ROUND_DIR/npu_before.txt"
cd "$BENCHMARK_DIR"
"$AIS_BENCH" "$CONFIG" --mode perf --debug \
    --max-num-workers 1 --num-warmups "$WARMUPS" \
    --work-dir "$ROUND_DIR/aisbench" \
    2>&1 | tee "$ROUND_DIR/aisbench_debug.log"

# Recheck the same service while its workers are still alive.
"$PYTHON" "$EXPERIMENT_DIR/audit_service_mode.py" "$MODE" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
    2>&1 | tee "$ROUND_DIR/mode_postcheck.log"
if [[ "$MODE" == slot ]]; then
    "$PYTHON" "$EXPERIMENT_DIR/audit_optimizations.py" "$SERVICE_DIR" "$ROUND_DIR/postcheck" \
        2>&1 | tee "$ROUND_DIR/optimization_postcheck.log"
fi
"$PYTHON" - "$ROUND_DIR" <<'CHECK_AUDIT'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "precheck/mode_audit.json").read_text())
after = json.loads((root / "postcheck/mode_audit.json").read_text())
for key in ("service_pid", "source_sha256"):
    if before[key] != after[key]:
        raise SystemExit(f"Service or audited source changed during measurement: {key}")
workers_before = {worker["rank"]: worker for worker in before["workers"]}
workers_after = {worker["rank"]: worker for worker in after["workers"]}
for rank, worker in workers_before.items():
    current = workers_after[rank]
    if worker["pid"] != current["pid"]:
        raise SystemExit(f"Worker changed during measurement: rank {rank}")
    for library, digest in worker["library_sha256"].items():
        if current["library_sha256"].get(library) != digest:
            raise SystemExit(f"Library changed during measurement: {library}")
CHECK_AUDIT
"$PYTHON" "$EXPERIMENT_DIR/validate_round.py" "$ROUND_DIR/aisbench" "$MANIFEST" \
    2>&1 | tee "$ROUND_DIR/validation.log"
npu-smi info > "$ROUND_DIR/npu_after.txt"
printf 'Raw logs and validated metrics: %s\n' "$ROUND_DIR"
```

### aisbench_off.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-gsm8k-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_gsm8k_off",
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
        abbr="gsm8k-test",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/gsm8k",
    )
]
# Use all of the test split with the stock {question} performance template.
```

### aisbench_slot.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-gsm8k-slot",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_gsm8k_slot",
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
        abbr="gsm8k-test",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/gsm8k",
    )
]
# Use all of the test split with the stock {question} performance template.
```

