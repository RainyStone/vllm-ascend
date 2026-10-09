#!/bin/bash
# bench_rt_grid.sh —— 一条运行（同一服务进程）上的 bench 序列：热身 + 两形状 × 3 轮
#
#   bash bench_rt_grid.sh <run_name> [rounds]
# 与 H16 轮口径一致：
#   热身   num_prompts=2  max_concurrency=2  random_input_len=1024  seed=999  → warmup_c0.json
#   形状   L8k_n32  = 32 prompts / 并发 16 / input 8192  / seed 序号 9  → 9001-9003
#          L16k_n32 = 32 prompts / 并发 16 / input 16384 / seed 序号 12 → 12001-12003
#   轮内 seed = 序号×1000+轮次；轮 1 视为热身，统计只取轮 2、3（离线聚合时处理）。
# 产物：$GRID/<run_name>/{warmup_c0,L8k_n32_c1..3,L16k_n32_c1..3}.json
set -euo pipefail

# 本地回环请求不走公司代理（环境有 http_proxy，no_proxy 默认空会吞掉对 127.0.0.1:8444 的请求）
export no_proxy="127.0.0.1,localhost,90.90.93.34${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

cd "$(dirname "$0")" || exit 1   # 中性 cwd（无裸 vllm/）：vllm bench serve 导入 vllm 用

RUN="${1:?用法: bench_rt_grid.sh <run_name> [rounds]}"
ROUNDS="${2:-3}"
GRID="${GRID:-/tmp/mc2_probe/grid}"
PORT="${PORT:-8444}"
MODEL=/mnt/data/models/DeepSeek-V4-Flash
WS=/home/x00893896/vllm_ascend_moonep_integration
OUT=$GRID/$RUN
mkdir -p "$OUT"

# 形状表：name:num_prompts:max_concurrency:random_input_len:seed序号
SHAPES=(
  "L8k_n32:32:16:8192:9"
  "L16k_n32:32:16:16384:12"
)

run_bench() {  # np mc len seed outname
  local np=$1 mc=$2 len=$3 seed=$4 name=$5
  echo "[bench] $RUN $name (np=$np conc=$mc len=$len seed=$seed)"
  PYTHONPATH=$WS/vllm_ascend_moonep/third_party/ascend_moonep:$WS/vllm_ascend_moonep:${PYTHONPATH:-} \
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

for s in "${SHAPES[@]}"; do
  IFS=: read -r name np mc len idx <<<"$s"
  for r in $(seq 1 "$ROUNDS"); do
    run_bench "$np" "$mc" "$len" "$((idx * 1000 + r))" "${name}_c${r}"
  done
done
echo "[bench] $RUN 完成：$(ls "$OUT"/*.json | wc -l) 个 json"
