#!/bin/bash
# run_opt_phase.sh —— off vs slot(+历史收益优化组合) 编排器（每形状一台服务）。
#   nohup bash run_opt_phase.sh <plan_file> > /tmp/mc2_probe/grid_opt/RUN_opt.log 2>&1 &
# plan 行格式（# 开头跳过）：
#   <run_name> <mode: off|slot> <mbt> <sim_route 绝对路径> <shape:L8k|L16k>
#   off  = 非 MoonEP（标准 MC2/EP 对照）
#   slot = MoonEP slot + real-T（默认）+ 历史收益优化组合前缀：
#          ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_SKIP_DUP=1
# 与 run_rt_phase.sh 的差异：plan 多一列 shape；bench 用 bench_rt_one.sh（单形状）；
# slot 腿固定带优化前缀（无需 slot0 消融腿）；GRID 默认 /tmp/mc2_probe/grid_opt（不与旧 rt 产物混）。
# 断点续跑：$GRID/<run>.done 存在即跳过。只杀自己起的进程组；占卡用 hold_cards_vllm.sh。
set -uo pipefail

# 本地回环的探活/bench 不走公司代理（本机环境有 http_proxy，no_proxy 默认空 → curl/httpx 全被代理吞）
export no_proxy="127.0.0.1,localhost,90.90.93.34${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

PLAN="${1:?用法: run_opt_phase.sh <plan_file>}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1  # 中性 cwd（无裸 vllm/）：bench 等子进程继承，避免 namespace 解析坑
SVC=$HERE/../service/start_service_mode_dp4.sh
BENCH=$HERE/bench_rt_one.sh
GRID="${GRID:-/tmp/mc2_probe/grid_opt}"
PORT=8444
HOLD_SH="${HOLD_SH:-$HERE/../service/hold_cards_vllm.sh}"   # 占卡一律用 vllm 进程（用户要求）
HOLD_CARDS="${HOLD_CARDS:-0 1 2 3}"              # 四张卡现在都归我们；被别人的进程占住时按需收窄
READY_TIMEOUT="${READY_TIMEOUT:-1500}"   # 模型加载 + 池构建，25 分钟
STOP_TIMEOUT=120

mkdir -p "$GRID"

stop_holder() {
  bash "$HOLD_SH" stop || true
}

start_holder() {
  # shellcheck disable=SC2086  # 卡号列表按空白分词
  bash "$HOLD_SH" start $HOLD_CARDS || true
}

hbm_used() {  # $1=卡号 → HBM 使用 MB（npu-smi 表；取不到输出空）
  npu-smi info 2>/dev/null | grep -A1 "^| $1 " | grep 0000 \
    | grep -oE "[0-9]+ */ 131072" | head -1 | grep -oE "^[0-9]+"
}

wait_hbm_free() {
  # 杀进程后驱动侧 HBM 释放有滞后（会让下一次 vllm 显存检查直接判死）。等卡 0-3 回落基线。
  local i ok c v
  for i in $(seq 1 30); do
    ok=1
    for c in 0 1 2 3; do
      v=$(hbm_used "$c") || v=""
      if [ -z "$v" ] || [ "$v" -ge 20000 ]; then ok=0; break; fi
    done
    if [ "$ok" = 1 ]; then echo "[hbm] 卡 0-3 已回落基线（等了 $((i * 10 - 10))s）"; return 0; fi
    sleep 10
  done
  echo "[hbm] 警告：300s 内卡 0-3 HBM 未全部回落，继续起服" >&2
  return 1
}

wait_port_free() {
  for _ in $(seq 1 $((STOP_TIMEOUT / 2))); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 || return 0
    sleep 2
  done
  return 1
}

while read -r run mode mbt simroute shape; do
  case "$run" in ''|'#'*) continue ;; esac
  [ -f "$GRID/$run.done" ] && { echo "[skip] $run（.done 存在）"; continue; }
  echo "════════ $(date '+%H:%M:%S') 运行 $run（mode=$mode mbt=$mbt route=$simroute shape=$shape）════════"

  stop_holder
  wait_hbm_free || true

  # 腿名映射：off 原样；slot → mode=slot + 历史收益优化组合前缀（三开关成组，全 rank 一致）
  svc_mode=$mode
  rt_prefix=()
  if [ "$mode" = slot ]; then
    svc_mode=slot
    rt_prefix=(ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_SKIP_DUP=1)
  fi

  # 端口残留检查（上一次异常退出的兜底）
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "[abort] 端口 $PORT 已被占用——先处理残留服务再续跑" >&2
    start_holder
    exit 4
  fi

  rm -rf "$GRID/$run" "$GRID/$run.log"   # 旧日志必须清：起服脚本见到已存在的日志文件会直接 exit 2
  setsid env "${rt_prefix[@]}" \
    OUT_DIR=$GRID/$run LOG_FILE=$GRID/$run.log \
    BATCHED_TOKENS=$mbt MOONEP_MIN_DISPATCH_TOKENS=16 \
    SIM_ROUTE="@$simroute" \
    bash "$SVC" "$svc_mode" > "$GRID/$run.boot.out" 2>&1 &
  SVC_PID=$!
  echo "[svc] 已启动 pgid=$SVC_PID"

  # 探活
  ready=0
  for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
    kill -0 "$SVC_PID" 2>/dev/null || break
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then ready=1; break; fi
    # 引擎初始化失败时 vllm 主进程会挂着不退（实测空等到 25 分钟超时）；日志一出现该错误立即判死
    if grep -q "Engine core initialization failed" "$GRID/$run.log" 2>/dev/null; then
      echo "[svc] 日志出现 Engine core initialization failed，提前判死" >&2
      break
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "[abort] $run 服务未就绪（${READY_TIMEOUT}s）——日志尾部：" >&2
    tail -30 "$GRID/$run.log" >&2 || true
    kill -TERM -- "-$SVC_PID" 2>/dev/null || true
    sleep 10; kill -KILL -- "-$SVC_PID" 2>/dev/null || true
    start_holder
    exit 5
  fi
  echo "[svc] $run 就绪"

  # bench（热身 + 单形状 × 3 轮）
  if ! bash "$BENCH" "$run" "$shape" 3; then
    echo "[abort] $run bench 失败" >&2
    kill -TERM -- "-$SVC_PID" 2>/dev/null || true
    sleep 10; kill -KILL -- "-$SVC_PID" 2>/dev/null || true
    wait_port_free || true
    start_holder
    exit 6
  fi

  # 停服（按进程组；先 TERM 让 API server 收尾，超时 KILL）
  kill -TERM -- "-$SVC_PID" 2>/dev/null || true
  if ! wait_port_free; then
    kill -KILL -- "-$SVC_PID" 2>/dev/null || true
  fi
  sleep 20   # 等 HBM 释放，下一条服务才能起
  echo "[svc] $run 已停"
  touch "$GRID/$run.done"
done < "$PLAN"

# 连续跑时运行间隙由服务自己占卡（避免反复起停占卡器）；全程结束才补占卡防被抢
start_holder
echo "════════ $(date '+%H:%M:%S') 全部完成（已补占卡 $HOLD_CARDS，约 1-2 分钟到位；放卡等用户发话）════════"
