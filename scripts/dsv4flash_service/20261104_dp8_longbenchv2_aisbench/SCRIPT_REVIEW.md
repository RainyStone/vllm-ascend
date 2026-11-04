# LongBench-v2 prefill 对照实验：脚本与配置审阅稿

日期：2026-11-04（Asia/Shanghai）。状态：**等待脚本/配置内容确认；未起服，未压测。**

## 已确认的输入与服务参数

- 权重：/mnt/share/weight/DeepSeek-V4-Flash（DSV4 Flash）。
- 使用完整 prompt 不超过 131072 token 的 LongBench-v2 原始全文样本，不截断。
- CPU 筛选得到 316 条，排除 187 条；保留原始顺序、context、问题和选项。
- 筛选后完整 prompt 长度：10285～130285 token。max_model_len=131073 为输出1 token留出空间。
- 8 卡 DP8 / TP1 / EP8；服务 max-num-seqs=16；chunked prefill；max-num-batched-tokens=8192。
- off / slot 两组统一 KV cache 24 GiB/卡（25769803776 bytes），这是待实际起服验证的候选预算。
- 每个服务对应一份完整 sh；不要求调用者 export 环境变量或传入参数。
- aisbench 并发16、max_out_len=1、流式性能测试、--debug。

## 本次提交确认的具体设计

1. 接口采用流式 /v1/completions，沿用 aisbench 的 LongBench-v2 原始 prompt 模板，使用模型原生专家路由。
2. Slot 开启 REAL_T、三项优化 PREFETCH_RELAXED / NO_RECONCILE / SKIP_DUP；持久专家缓存、提前预测预取、独立流预取、KEEP_MC2均关闭。每卡同形状层共享32个工作槽位。
3. 两组统一 enforce-eager，并关闭 prefix cache。模拟路由及上轮诊断环境变量由脚本清理。
4. 已确认重复方案：两对独立起服，pair01按 off→slot，pair02按 slot→off。每次服务先由 aisbench预热16条，再测3轮完整316条，保留第1轮但不计均值，取第2、3轮；预热不重复于后续轮次。
5. aisbench retry=0、temperature=0、ignore_eos=True、request_rate=0。只做性能测评，不以输出1token评估准确率。
6. 两组正式采样请求共 4服务 × 3轮 × 316 = 3792条，另有64条预热。若用户选择更轻方案，可修改每份压测脚本顶部 ROUNDS 和按计划调整起服次数。
7. 保留 TTFT均值/中位数/P90/P99、输入token吞吐、请求耗时、成功失败数和实际输入/输出token数。主要判读prefill占主导的端到端表现；TTFT仍包含排队与网络等开销。

第4项重复方案已确认；第2项及脚本/配置全文仍待用户确认。确认脚本内容不代表现在开始执行。

## 文件

| 文件 | 用途 |
|---|---|
| start_off.sh | off独立起服入口 |
| start_slot.sh | Slot独立起服入口 |
| bench_off.sh | 已启动off服务的aisbench性能入口 |
| bench_slot.sh | 已启动Slot服务的aisbench性能入口 |
| aisbench_off.py / aisbench_slot.py | 本实验隔离配置，公共预置配置不修改 |
| prepare_dataset.py | 按已完成token审计生成316条全文子集，已在CPU执行 |
| validate_round.py | 校验每轮316条成功、输出1token、输入token数与审计一致 |
| data/selection_manifest.json | 冻结样本ID、原索引、长度、数据hash |
| length_audit.csv / length_summary.json | 503条完整prompt的tokenizer审计 |
| VALIDATION.txt | 静态检查与CPU合成数据验证记录 |

四份sh顶部均有 SCRIPT_REVIEWED=0。内容确认后才会改为1，当前直接运行会在启动服务/发送请求前退出。其余调优参数都直接在对应sh中修改，aisbench的模型和数据配置在对应py文件中修改。

## 按已确认两对方案的手工执行顺序（现在不执行）

1. 确认8卡空闲、端口8444可用；把四份sh的PAIR_LABEL保持pair01。
2. 在服务终端运行 start_off.sh，在压测终端运行 bench_off.sh。
3. 停止该服务，确认其子进程和NPU资源释放，再运行 start_slot.sh 与 bench_slot.sh。
4. 将四份sh内 PAIR_LABEL 改为pair02，先Slot后off，再各测3轮。每次均为独立重启。
5. 两组共用8444端口、相同输入和参数，依次运行。两个 served-model-name 不同，用于防止客户端连接错模式；不改变权重或计算配置。

服务max-num-seqs=16是每个DP engine的调度上限；aisbench batch_size=16控制本实验全局在途请求数。DP8不把单条请求的KV分片到8卡。

## 原始输出保留方式

服务日志：data/services/<pair>-<mode>-<time>-<pid>/service.log。
压测结果：data/aisbench/<pair>-<mode>-<time>-<pid>/round_<n>/。
每轮 --debug 的 stdout/stderr 完整保存到 aisbench_debug.log。aisbench自身保存的配置、逐请求 *_details.jsonl、关联SQLite .db / db_data、失败明细、统计CSV/JSON/HTML均保留在整份 aisbench work_dir 中；只复制JSONL可能缺失被数据库引用的时间数组，因此不拆分原始目录。

validate_round.py 在每轮结束后校验完整性。服务实际token数与CPU审计不一致、存在失败、输出不为1、ID重复或缺失时，脚本停止并保留所有日志。正式采样前应先核对服务tokenize/usage口径及最长样本，确认131073与24GiB预算有效。

## 已完成与尚未完成的验证

已完成：四份sh的bash -n；Python AST；两份mmengine配置真实解析；316条数据的原文字段及顺序核对；合成CPU文件验证原始结果校验器的正反例。原始LongBench-v2/data.json未修改。

未执行：服务启动、动态库链接验证、24GiB KV分配、130285-token最长请求、并发16的运行时显存和aisbench实际请求。因此目前不承诺起服或运行时无OOM，也不自动降参数；后续实验若失败，保留错误并分析。

## 脚本与配置全文

以下内容由当前磁盘文件直接生成，供逐项确认。

### start_off.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
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
set -u

PLUGIN="$WORKSPACE/vllm-ascend"
MOONEP_PACKAGE="$WORKSPACE/ascend-moonep/ascend_moonep"
VENDOR="$PLUGIN/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$PLUGIN:$WORKSPACE/ascend-moonep"
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

### start_slot.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.
set -Ee -o pipefail

# ===== Edit this block directly after reviewing the experiment =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'
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
set -u

PLUGIN="$WORKSPACE/vllm-ascend"
MOONEP_PACKAGE="$WORKSPACE/ascend-moonep/ascend_moonep"
VENDOR="$PLUGIN/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$PLUGIN:$WORKSPACE/ascend-moonep"
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

### bench_off.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'  # Set pair02 for the reversed independent-service pair.
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='off'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_longbench_off'
CONFIG="$EXPERIMENT_DIR/aisbench_off.py"
ROUNDS=3
FIRST_ROUND_WARMUPS=16
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
export PYTHONPATH="$BENCHMARK_DIR"
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
printf 'Round 1 is retained but excluded; compare means of rounds 2 and 3.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
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

### bench_slot.sh

```bash
#!/usr/bin/env bash
# DRAFT FOR REVIEW. This script benchmarks an already running service.
set -Eeuo pipefail

# ===== Edit here, without supplying environment variables from outside =====
SCRIPT_REVIEWED=0
PAIR_LABEL='pair01'  # Set pair02 for the reversed independent-service pair.
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench'
BENCHMARK_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark'
PYTHON='/usr/local/python3.11.10/bin/python'
AIS_BENCH='/usr/local/python3.11.10/bin/ais_bench'
MODE='slot'
HOST='127.0.0.1'
PORT=8444
SERVED_MODEL='dsv4flash_longbench_slot'
CONFIG="$EXPERIMENT_DIR/aisbench_slot.py"
ROUNDS=3
FIRST_ROUND_WARMUPS=16
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
export PYTHONPATH="$BENCHMARK_DIR"
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
printf 'Round 1 is retained but excluded; compare means of rounds 2 and 3.\n' > "$RUN_DIR/MEASUREMENT_RULE.txt"
cd "$BENCHMARK_DIR"
for (( round=1; round<=ROUNDS; round++ )); do
    ROUND_DIR="$RUN_DIR/round_$round"
    mkdir -p "$ROUND_DIR"
    warmups=0
    if (( round == 1 )); then
        warmups="$FIRST_ROUND_WARMUPS"
    fi
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

### aisbench_off.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_off",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=0,
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
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.
```

### aisbench_slot.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-slot",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_slot",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=0,
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
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.
```

### prepare_dataset.py

```python
#!/usr/bin/env python3
"""Create the approved full-text subset on CPU, preserving original order."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/LongBench-v2/data.json")
MODEL = Path("/mnt/share/weight/DeepSeek-V4-Flash")
INPUT_CAP = 131072
MAX_MODEL_LEN = 131073
EXPECTED_SAMPLES = 316

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = json.loads((ROOT / "length_summary.json").read_text())
    assert sha256(SOURCE) == summary["dataset_sha256"]
    assert sha256(MODEL / "tokenizer.json") == summary["tokenizer_sha256"]
    model_config = json.loads((MODEL / "config.json").read_text())
    assert INPUT_CAP + 1 == MAX_MODEL_LEN <= model_config["max_position_embeddings"]
    records = json.loads(SOURCE.read_text())
    audit = list(csv.DictReader((ROOT / "length_audit.csv").open()))
    assert len(records) == len(audit) == 503
    selected, mapping = [], []
    for index, (record, entry) in enumerate(zip(records, audit)):
        assert record["_id"] == entry["id"] and index == int(entry["index"])
        tokens = int(entry["prompt_tokens"])
        if tokens <= INPUT_CAP:
            mapping.append(dict(data_id=len(selected), source_index=index,
                                id=record["_id"], prompt_tokens=tokens,
                                domain=record["domain"], difficulty=record["difficulty"]))
            selected.append(record)
    assert len(selected) == EXPECTED_SAMPLES
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    output = data_dir / "longbenchv2_full_le128k.json"
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    # This checks every field in every kept sample: no context truncation or editing.
    assert json.loads(output.read_text()) == [records[m["source_index"]] for m in mapping]
    manifest = dict(
        source=str(SOURCE), source_sha256=summary["dataset_sha256"],
        prepared=str(output), prepared_sha256=sha256(output),
        tokenizer_sha256=summary["tokenizer_sha256"],
        policy="filter_full_prompt_length_preserve_original_text_and_order",
        input_cap=INPUT_CAP, output_tokens=1, max_model_len=MAX_MODEL_LEN,
        selected=len(selected), excluded=len(records)-len(selected),
        shortest_prompt_tokens=min(m["prompt_tokens"] for m in mapping),
        longest_prompt_tokens=max(m["prompt_tokens"] for m in mapping),
        samples=mapping,
    )
    (data_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    assert sha256(SOURCE) == summary["dataset_sha256"]
    print("PREPARED", output)
    print("SELECTED", len(selected), "EXCLUDED", len(records)-len(selected))
    print("INPUT_TOKENS", manifest["shortest_prompt_tokens"], manifest["longest_prompt_tokens"])
    print("SHA256", manifest["prepared_sha256"])

if __name__ == "__main__":
    main()
```

### validate_round.py

```python
#!/usr/bin/env python3
"""Check retained aisbench request details; never starts a service."""
import json
import sys
from pathlib import Path

DATA_ABBR = "longbenchv2-full-le128k"

def rows_from_payload(payload):
    if isinstance(payload, list):
        for entry in payload:
            yield from rows_from_payload(entry)
    elif isinstance(payload, dict):
        if "success" in payload and "id" in payload:
            yield payload
        else:
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    yield from rows_from_payload(value)

def validate(result_dir, manifest_path):
    result_dir = Path(result_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    expected = {int(row["data_id"]): row for row in manifest["samples"]}
    files = [f for f in result_dir.rglob(DATA_ABBR + "_details.jsonl")
             if "warmup" not in str(f.relative_to(result_dir)).lower()]
    if not files:
        raise ValueError("No raw aisbench request details found; retain the entire work_dir.")
    records = []
    for path in files:
        for line in path.read_text().splitlines():
            if line.strip():
                records.extend(rows_from_payload(json.loads(line)))
    errors, ids = [], []
    for row in records:
        index = int(row["id"])
        ids.append(index)
        if row.get("success") is not True:
            errors.append(f"id={index}: request failed: {row.get('error_info')}")
        if index not in expected:
            errors.append(f"Unexpected id={index}")
            continue
        actual_in = int(row.get("input_tokens", 0))
        actual_out = int(row.get("output_tokens", 0))
        wanted = expected[index]["prompt_tokens"]
        if actual_in != wanted:
            errors.append(f"id={index}: service input_tokens={actual_in}, tokenizer audit={wanted}; stop and reconcile tokenization.")
        if actual_in > manifest["input_cap"]:
            errors.append(f"id={index}: prompt exceeds input cap")
        if actual_out != 1:
            errors.append(f"id={index}: output_tokens={actual_out}, expected 1")
    if len(ids) != len(expected) or set(ids) != set(expected) or len(set(ids)) != len(ids):
        errors.append(f"Expected {len(expected)} distinct ids, got {len(ids)} rows / {len(set(ids))} ids")
    for failed in result_dir.rglob("*_failed.jsonl"):
        if failed.stat().st_size:
            errors.append(f"Failed-request file is nonempty: {failed}")
    report = dict(valid=not errors, requests=len(records), expected=len(expected),
                  details=[str(f) for f in files], errors=errors)
    (result_dir.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValueError("\n".join(errors[:25]))
    print(f"VALID_ROUND: {len(records)} successful full-text requests; every output=1; input token counts match audit.")
    return report

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: validate_round.py RESULT_DIR SELECTION_MANIFEST")
    validate(sys.argv[1], sys.argv[2])
```
