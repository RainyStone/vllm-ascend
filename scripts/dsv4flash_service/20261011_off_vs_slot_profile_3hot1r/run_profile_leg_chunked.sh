#!/bin/bash
# run_profile_leg.sh —— 单腿「bench + profiler 同窗采集」：off 或 slot(+三件套)。
#   bash run_profile_leg.sh off     # off 腿
#   bash run_profile_leg.sh slot    # slot 腿（+历史收益优化组合前缀）
# 流程：停占卡→等 HBM→PROFILE=1 起服→探活→warmup→start_profile→
#       一轮 L16k bench（seed 21001，两腿同 seed=同流量）→stop_profile→等 trace 导出→
#       停服→bench json / 服务日志 / trace 归档进本实验目录。
# 注意：该轮 bench 含 profiler 开销，数据不用于性能统计（本实验目的是 trace 对比）。
set -uo pipefail

# 本地回环不走公司代理
export no_proxy="127.0.0.1,localhost,90.90.93.34${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

MODE="${1:?用法: run_profile_leg.sh <off|slot>}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1   # 中性 cwd（无裸 vllm/）
SVC=$HERE/../service/start_service_mode_dp4.sh
HOLD_SH=$HERE/../service/hold_cards_vllm.sh
GRID=/tmp/mc2_probe/profile_3hot1r
PORT=8444
SEED=21001          # 全新 seed（区别于 op2 的 12001-12003）；两腿相同 → 输入流量一致
READY_TIMEOUT=1500
MODEL=/mnt/data/models/DeepSeek-V4-Flash
mkdir -p "$GRID"

# 腿 → 起服模式与 env 前缀
svc_mode=$MODE
rt_prefix=()
[ "$MODE" = slot ] && rt_prefix=(ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_SKIP_DUP=1)

# ---- 停占卡、等卡干净 ----
bash "$HOLD_SH" stop || true
for i in $(seq 1 30); do
  ok=1
  for c in 0 1 2 3; do
    v=$(npu-smi info 2>/dev/null | grep -A1 "^| $c " | grep 0000 | grep -oE "[0-9]+ */ 131072" | head -1 | grep -oE "^[0-9]+")
    [ -z "$v" ] || [ "$v" -ge 20000 ] && { ok=0; break; }
  done
  [ "$ok" = 1 ] && { echo "[hbm] 卡 0-3 已回落基线"; break; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[abort] 端口被占"; exit 4; }

# ---- 起服（PROFILE=1 → 开 /start_profile /stop_profile，trace 落 $OUT_DIR/profiling）----
rm -rf "$GRID/$MODE" "$GRID/$MODE.log"
setsid env "${rt_prefix[@]}" \
  PROFILE=1 OUT_DIR=$GRID/${MODE}_chunked LOG_FILE=$GRID/${MODE}_chunked.log \
  BATCHED_TOKENS=8192 MOONEP_MIN_DISPATCH_TOKENS=16 \
  SIM_ROUTE="@$HERE/../sim/sim_route_3hot_1rank.txt" \
  bash "$SVC" "$svc_mode" > "$GRID/$MODE.boot.out" 2>&1 &
SVC_PID=$!
echo "[svc] $MODE 已启动 pgid=$SVC_PID"

ready=0
for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
  kill -0 "$SVC_PID" 2>/dev/null || break
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  grep -q "Engine core initialization failed" "$GRID/$MODE.log" 2>/dev/null && break
  sleep 5
done
[ "$ready" = 1 ] || { echo "[abort] $MODE 服务未就绪——日志尾部：" >&2; tail -30 "$GRID/$MODE.log" >&2; exit 5; }
echo "[svc] $MODE 就绪"

# ---- warmup（不采集）----
PYTHONPATH=$HERE/../../..:$PYTHONPATH vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --base-url http://127.0.0.1:$PORT --model $MODEL --served-model-name test_auto \
  --tokenizer $MODEL --dataset-name random --num-prompts 2 --max-concurrency 2 \
  --random-input-len 1024 --random-output-len 1 --random-range-ratio 0.0 \
  --request-rate inf --ignore-eos --seed 999 2>&1 | tail -2

# ---- 开采集 → 一轮 L16k bench → 停采集 ----
echo "[prof] start_profile"
curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" || { echo "[abort] start_profile 失败" >&2; exit 6; }
sleep 3
PYTHONPATH=$HERE/../../..:$PYTHONPATH vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --base-url http://127.0.0.1:$PORT --model $MODEL --served-model-name test_auto \
  --tokenizer $MODEL --dataset-name random --num-prompts 32 --max-concurrency 16 \
  --random-input-len 16384 --random-output-len 1 --random-range-ratio 0.0 \
  --request-rate inf --ignore-eos --seed $SEED \
  --percentile-metrics ttft,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir "$HERE/data" --result-filename "${MODE}_L16k_profiled.json" 2>&1 | tail -4
python3 - "$HERE/data/${MODE}_L16k_profiled.json" <<'PY' || exit 7
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("completed") != 32 or d.get("failed", 0):
    print(f"[bench] 结果异常 completed={d.get('completed')} failed={d.get('failed')}", file=sys.stderr)
    sys.exit(1)
print(f"[bench] completed=32 mean_ttft={d['mean_ttft_ms']:.0f}ms")
PY
echo "[prof] stop_profile"
curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" || echo "[warn] stop_profile 返回非 0" >&2

# ---- 等 trace 导出（文件数稳定 30s 或超时 600s）----
PROF_DIR=$GRID/${MODE}_chunked/profiling
prev=-1; stable=0
for i in $(seq 1 120); do
  n=$(ls "$PROF_DIR" 2>/dev/null | wc -l)
  if [ "$n" = "$prev" ] && [ "$n" -gt 0 ]; then stable=$((stable + 5)); else stable=0; fi
  prev=$n
  [ "$stable" -ge 30 ] && { echo "[prof] 导出完成：$n 个文件（等了 $((i * 5))s）"; break; }
  sleep 5
done
ls -la "$PROF_DIR" 2>/dev/null | head -8

# ---- 停服 ----
kill -TERM -- "-$SVC_PID" 2>/dev/null || true
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break; sleep 2; done
kill -KILL -- "-$SVC_PID" 2>/dev/null || true
sleep 20
echo "[svc] $MODE 已停"

# ---- 归档 ----
cp "$GRID/$MODE.log" "$HERE/data/" 2>/dev/null
mkdir -p "$HERE/profile/$MODE"
cp -r "$PROF_DIR"/. "$HERE/profile/$MODE/" 2>/dev/null
echo "[done] $MODE：bench=$HERE/data/${MODE}_L16k_profiled.json trace=$HERE/profile/$MODE/（$(du -sh $HERE/profile/$MODE 2>/dev/null | cut -f1)）"
