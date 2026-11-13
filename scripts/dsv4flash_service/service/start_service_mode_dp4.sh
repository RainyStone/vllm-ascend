#!/bin/bash
# start_service_mode_dp4.sh —— DP4 版三模式起服脚本（real-T 重跑专用）
#
#   bash start_service_mode_dp4.sh off      # 非 MoonEP（标准 MC2/all-to-all EP）
#   bash start_service_mode_dp4.sh owner    # MoonEP owner（本矩阵未用，保留）
#   bash start_service_mode_dp4.sh slot     # MoonEP slot（real-T 默认开）
#   MOONEP_REAL_T=0 bash start_service_mode_dp4.sh slot   # slot 消融腿（旧行为）
#   BATCHED_TOKENS=8192 bash start_service_mode_dp4.sh slot  # mbt 8192 配置
#   MOONEP_MIN_DISPATCH_TOKENS=16 SIM_ROUTE=@<档案> ...    # min16 + 仿真路由（命令前缀）
#
# 派生自 start_service_mode.sh（DP8 版，幸存于本目录），差异（= H16 轮 min16 实验条件）：
#   DP=4、ASCEND_RT_VISIBLE_DEVICES=0,1,2,3、MAX_MODEL_LEN=20480、
#   --kv-cache-memory-bytes 9663676416（9 GiB）、BATCHED_TOKENS 可前缀（默认 16384）。
# 新增：MOONEP_REAL_T 透传 + 横幅回显（本脚本不 unset 它：未设 = 引擎默认 1；
#   slot0 消融腿用命令前缀 MOONEP_REAL_T=0）。
# 采集开关（PROFILE/SIM_ROUTE）语义与 DP8 版一致，默认全关，
#   性能跑**什么都不加**（脚本对每个都有无条件 unset 兜底）。
# MoE逐rank原始负载采集和BALANCE账本已移除，旧开关不再生效。
set -euo pipefail

# ---------------------------------------------------------- 模式变量 -- #
MODE="${1:-slot}"
case "$MODE" in
  off)
    unset MOONEP MOONEP_SLOT_MODE
    ENABLE_MOONEP=false
    MODE_DESC="非 MoonEP：标准 MC2 / all-to-all EP（对照组）"
    ;;
  owner)
    export MOONEP=1 MOONEP_SLOT_MODE=0
    ENABLE_MOONEP=true
    MODE_DESC="MoonEP owner：专家行常驻本 rank"
    ;;
  slot)
    export MOONEP=1 MOONEP_SLOT_MODE=1
    ENABLE_MOONEP=true
    MODE_DESC="MoonEP slot：按 slot 镜像 expert 行并预取（real-T 默认开）"
    ;;
  *)
    echo "用法: bash $0 {off|owner|slot}" >&2
    exit 2
    ;;
esac
# MOONEP_REAL_T：不 unset、不设默认——未设 = 引擎默认 1（real-T 新路径）；
# 消融腿用命令前缀 MOONEP_REAL_T=0 透传（回到旧的「pad 全量计算」路径）。
MOONEP_REAL_T_GIVEN="${MOONEP_REAL_T:-}"
# -------------------------------------------------- 产物路径（可前缀覆盖） -- #
HERE=$(cd "$(dirname "$0")" && pwd)
# cwd 必须在没有裸 vllm/ 目录的地方起服务（CANN set_env 的空 PYTHONPATH 条目 = cwd，
# 会把 $WS/vllm/ 当 namespace package 抢解析——见 DP8 版同段注释）。
cd "$HERE" || exit 1
# 只保留 BATCHED_TOKENS 的「外部给定」标注（其余 *_OVERRIDE 是死代码）；
# 必须 ${VAR:-} 兜底：set -u 下引用未定义变量会直接炸（本行是 DP4 版首跑暴露的第一个坑）
BATCHED_TOKENS_OVERRIDE=${BATCHED_TOKENS:-}
OUT_DIR=${OUT_DIR:-$HERE/serve_dp4_$MODE}
LEG_NAME=$(basename "${OUT_DIR%/}")
LOG_FILE=${LOG_FILE:-$HERE/start_service_dp4_$MODE.log}
PROFILE_DIR=$OUT_DIR/profiling
for p in "$OUT_DIR" "$LOG_FILE"; do
  case "$p" in
    /*) ;;
    *) echo "路径必须是绝对路径（要给 worker/profiler 用，那边 cwd 不可靠）：$p" >&2; exit 2 ;;
  esac
done
if [ -e "$LOG_FILE" ]; then
  echo "[服务日志] 已存在，不覆盖：$LOG_FILE —— 删掉它，或用前缀指到别处" >&2
  exit 2
fi

# ---------------------------------------------------------- 共用环境 -- #
unset ftp_proxy https_proxy http_proxy
source /usr/local/Ascend/cann/set_env.sh

WS=/home/x00893896/vllm_ascend_moonep_integration
export HCCL_IF_IP=90.90.93.34
export GLOO_SOCKET_IFNAME=ens6f1np1
export TP_SOCKET_IFNAME=ens6f1np1
export HCCL_SOCKET_IFNAME=ens6f1np1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"   # DP+EP 下必须关，否则会挂
export ASCEND_RT_VISIBLE_DEVICES=${SVC_CARDS:-0,1,2,3}   # DP4 用 4 张；SVC_CARDS 可覆盖（如后四卡 4,5,6,7）
export PYTHONPATH=$WS/vllm-ascend:$PYTHONPATH
#                ^ ascend_moonep 不再 vendored：由 ascend-moonep 仓的 pip install -e 提供，
#                  _C*.so/lib 就在其包目录内（重编后天然同版）
export ASCEND_CUSTOM_OPP_PATH=$WS/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset MOONEP_SHMEM_IPPORT
export MOONEP_SHMEM_BOOTSTRAP=ipport        # uid 走 HCCL 广播会死锁
export MOONEP_DISABLE_UDMA=1                # 关 UDMA 走 MTE（与 H16 轮一致）
export ASCEND_MOONEP_BUILD_DIR=$WS/ascend-moonep/ascend_moonep   # 含新编的 _C（real-T ABI）
export LD_LIBRARY_PATH=$ASCEND_MOONEP_BUILD_DIR/lib:$LD_LIBRARY_PATH

# -------------------------------------------------------- 仿真路由 -- #
unset VLLM_ASCEND_MOE_SIM_ROUTE VLLM_ASCEND_MOE_SIM_ROUTE_SEED \
      VLLM_ASCEND_MOE_SIM_ROUTE_EVERY VLLM_ASCEND_MOE_SIM_ROUTE_TABLE
if [ -n "${SIM_ROUTE:-}" ]; then
  case "$SIM_ROUTE" in
    @*) SPEC_FILE="${SIM_ROUTE#@}"
        if [ ! -r "$SPEC_FILE" ]; then
          echo "[仿真路由] 档案文件不可读：$SPEC_FILE（要绝对路径；worker 是 spawn 的，cwd 不可靠）" >&2
          exit 2
        fi ;;
  esac
  export VLLM_ASCEND_MOE_SIM_ROUTE="$SIM_ROUTE"
  export VLLM_ASCEND_MOE_SIM_ROUTE_SEED="${SIM_ROUTE_SEED:-0}"
  export VLLM_ASCEND_MOE_SIM_ROUTE_EVERY="${SIM_ROUTE_EVERY:-43}"
  export VLLM_ASCEND_MOE_SIM_ROUTE_TABLE="${SIM_ROUTE_TABLE:-1}"
fi

# --------------------------------------------------------- profiling -- #
PROFILE_ARGS=()
if [ "${PROFILE:-0}" = 1 ]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_ARGS=(--profiler-config.profiler=torch
                --profiler-config.torch_profiler_dir="$PROFILE_DIR"
                --profiler-config.torch_profiler_with_stack=false)
fi

# ---------------------------------------------------------- 服务参数 -- #
MODEL=/mnt/data/models/DeepSeek-V4-Flash
SERVED_NAME=test_auto
HOST=0.0.0.0
PORT=8444
API_SERVER_COUNT=1
DP=4
TP=1
MAX_NUM_SEQS=16
BATCHED_TOKENS=${BATCHED_TOKENS:-16384}     # 前缀可换（8192 配置用）；默认 16384
MAX_MODEL_LEN=20480
KV_CACHE_BYTES=9663676416                   # 9 GiB（H16 轮同值；两 mbt 档共用）
GPU_UTIL=0.80

# -------------------------------------------------------- 启动核对 -- #
if [ -n "$MOONEP_REAL_T_GIVEN" ]; then
  REAL_T_LINE="MOONEP_REAL_T=$MOONEP_REAL_T_GIVEN（外部给定，透传）"
else
  REAL_T_LINE="MOONEP_REAL_T=未设 → 引擎默认 1（real-T 新路径）"
fi
cat <<EOF | tee -a "$LOG_FILE"
══════════════════════ 启动核对（dp4）══════════════════════
 模式        mode=$MODE —— $MODE_DESC
 MoonEP      enable_moonep=$ENABLE_MOONEP   MOONEP=${MOONEP:-未设}   MOONEP_SLOT_MODE=${MOONEP_SLOT_MODE:-未设}
 real-T      $REAL_T_LINE
 min-tokens  MOONEP_MIN_DISPATCH_TOKENS=${MOONEP_MIN_DISPATCH_TOKENS:-未设 → 默认 128}
 引导/传输   MOONEP_SHMEM_BOOTSTRAP=$MOONEP_SHMEM_BOOTSTRAP   MOONEP_DISABLE_UDMA=$MOONEP_DISABLE_UDMA
 采集        PROFILE=${PROFILE:-0}
 仿真路由    SIM_ROUTE=${SIM_ROUTE:-未设}
 服务        http://$HOST:$PORT   api-server-count=$API_SERVER_COUNT   dp=$DP tp=$TP ep=on   served-model-name=$SERVED_NAME
 模型        $MODEL
 负载口径    max-num-seqs=$MAX_NUM_SEQS   max-num-batched-tokens=$BATCHED_TOKENS${BATCHED_TOKENS_OVERRIDE:+（外部给定）}   max-model-len=$MAX_MODEL_LEN   kv-cache-bytes=$KV_CACHE_BYTES   gpu-mem-util=$GPU_UTIL   chunked-prefill=on   seed=1024
 卡/关键env  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES   PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF   HCCL_BUFFSIZE=$HCCL_BUFFSIZE
 日志        $LOG_FILE
 产物        $OUT_DIR/
════════════════════════════════════════════════════════════
EOF

vllm serve "$MODEL" \
  --host "$HOST" \
  --port $PORT \
  --api-server-count $API_SERVER_COUNT \
  --data-parallel-size $DP \
  --tensor-parallel-size $TP \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-chunked-prefill \
  --additional-config "{\"enable_moonep\": $ENABLE_MOONEP}" \
  "${PROFILE_ARGS[@]}" \
  --served-model-name "$SERVED_NAME" \
  --seed 1024 \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --kv-cache-memory-bytes $KV_CACHE_BYTES \
  --trust-remote-code \
  --gpu-memory-utilization $GPU_UTIL \
  2>&1 | tee -a "$LOG_FILE"

# ------------------------------------------------------------- 收尾 -- #
set +x
{
  echo
  echo "──────── 这条腿结束：$LEG_NAME（mode=$MODE）────────"
  echo " 服务日志   $LOG_FILE"
} 2>&1 | tee -a "$LOG_FILE"
