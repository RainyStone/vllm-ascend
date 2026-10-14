#!/bin/bash
# bench_rt_one.sh —— 单形状 bench（每形状一台服务的粒度用）：
#   热身 + 指定形状 × 3 轮（轮 1 视为热身，统计只取轮 2、3）
#   bash bench_rt_one.sh <run_name> <shape:L8k|L16k> [rounds]
# 口径与 bench_rt_grid.sh 完全一致（seed = 档位序号×1000+轮次），只是每次只跑一个形状。
# 产物：$GRID/<run_name>/{warmup_c0,<shape>_n32_c1..3}.json
set -euo pipefail

# 本地回环请求不走公司代理（环境有 http_proxy，no_proxy 默认空会吞掉对 127.0.0.1:8444 的请求）
export no_proxy="127.0.0.1,localhost,90.90.93.34${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

cd "$(dirname "$0")" || exit 1   # 中性 cwd（无裸 vllm/）：vllm bench serve 导入 vllm 用

RUN="${1:?用法: bench_rt_one.sh <run_name> <shape:L8k|L16k> [rounds]}"
SHAPE_KEY="${2:?用法: bench_rt_one.sh <run_name> <shape:L8k|L16k> [rounds]}"
ROUNDS="${3:-3}"
GRID="${GRID:-/tmp/mc2_probe/grid_opt}"
PORT="${PORT:-8444}"
MODEL=/mnt/data/models/DeepSeek-V4-Flash
WS=/home/x00893896/vllm_ascend_moonep_integration
OUT=$GRID/$RUN
mkdir -p "$OUT"

# 形状表（与 bench_rt_grid.sh 同源）：key -> name:num_prompts:max_concurrency:random_input_len:seed序号
case "$SHAPE_KEY" in
  L8k)  SHAPE=L8k_n32:32:16:8192:9 ;;
  L16k) SHAPE=L16k_n32:32:16:16384:12 ;;
  *) echo "形状只认 L8k / L16k，收到：$SHAPE_KEY" >&2; exit 2 ;;
esac
IFS=: read -r NAME NP MC LEN IDX <<<"$SHAPE"

run_bench() {  # np mc len seed outname
  local np=$1 mc=$2 len=$3 seed=$4 name=$5
  echo "[bench] $RUN $name (np=$np conc=$mc len=$len seed=$seed)"
  PYTHONPATH=$WS/vllm-ascend:${PYTHONPATH:-} \
  vllm bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --base-url http://127.0.0.1:$PORT \
    --model $MODEL \
    --served-model-name test_auto \
    --tokenizer $MODEL \
    --dataset-name random \
    --num-prompts $np \
    --max-concurrency $mc \
    --random-input-len $len \
    --random-output-len 1 \
    --random-range-ratio 0.0 \
    --request-rate inf \
    --ignore-eos \
    --seed $seed \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --result-dir "$OUT" \
    --result-filename "$name.json" 2>&1 | tail -4
  [ -f "$OUT/$name.json" ] || { echo "[bench] 结果文件缺失：$OUT/$name.json" >&2; exit 3; }
  # 结果自检：json 落盘 ≠ 请求成功（首跑曾因漏传 --served-model-name，32 个请求全 404 但 json 照写）
  if ! python3 - "$OUT/$name.json" "$np" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
np = int(sys.argv[2])
if d.get("completed") != np or d.get("failed", 0):
    print(f"[bench] 结果异常：completed={d.get('completed')} failed={d.get('failed')}（期望 {np}/0）", file=sys.stderr)
    sys.exit(1)
PY
  then exit 3; fi
}

# 热身（轮次记 0）
run_bench 2 2 1024 999 warmup_c0

for r in $(seq 1 "$ROUNDS"); do
  run_bench "$NP" "$MC" "$LEN" "$((IDX * 1000 + r))" "${NAME}_c${r}"
done
echo "[bench] $RUN 完成：$(ls "$OUT"/*.json | wc -l) 个 json"
