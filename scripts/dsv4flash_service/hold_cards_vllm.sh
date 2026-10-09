#!/bin/bash
# hold_cards_vllm.sh —— 用 vllm 进程占卡（用户要求：占卡一律用 vllm 进程，不用裸 torch 驻留）
#
#   bash hold_cards_vllm.sh start <卡号列表>      # 如: start 0 2（每卡一个 vllm serve，端口 8500+卡号）
#   bash hold_cards_vllm.sh stop                  # 按 pid 文件逐个停（只停本脚本起的进程组）
#   bash hold_cards_vllm.sh status                # 各 pid 存活与 HBM 驻留
# 模型用 DeepSeek-V2-Lite + --load-format dummy：不读盘秒起，权重零填充 + KV 池把
# gpu-memory-utilization 挡位吃满（0.85 → 每卡约 111 GiB 驻留）。
# 只用已安装环境（不带仓库 PYTHONPATH）：占卡进程与 moonep 代码改动物理隔离。
# pid 落盘 /tmp/mc2_probe/hold_cards_vllm.pid（每行 卡号:pid），stop 只按这些 pid 的进程组停。
set -uo pipefail

PID_FILE="${HOLD_PID_FILE:-/tmp/mc2_probe/hold_cards_vllm.pid}"
MODEL=/mnt/data/models/DeepSeek-V2-Lite
HOLD_UTIL="${HOLD_UTIL:-0.85}"
WS=/home/x00893896/vllm_ascend_moonep_integration   # 仅为自定义算子路径（与服务脚本一致）

case "${1:-}" in
start)
  shift
  mkdir -p /tmp/mc2_probe
  touch "$PID_FILE"
  # 清掉已死条目（保留仍存活的，避免重复起）
  tmp="$PID_FILE.tmp"; : > "$tmp"
  while IFS=: read -r _c _p; do
    [ -n "${_p:-}" ] && kill -0 "$_p" 2>/dev/null && echo "$_c:$_p" >> "$tmp"
  done < "$PID_FILE"
  mv "$tmp" "$PID_FILE"
  for c in "$@"; do
    p=$(grep "^$c:" "$PID_FILE" 2>/dev/null | head -1 | cut -d: -f2)
    if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
      echo "[hold] card$c 已有存活占卡 pid=$p，跳过"
      continue
    fi
    port=$((8500 + c))
    setsid nohup bash -c "
      source /usr/local/Ascend/cann/set_env.sh
      cd /tmp/mc2_probe || exit 1   # 不能在有裸 vllm/ 的目录起服（CANN set_env 的空 PYTHONPATH 条目 = cwd）
      export ASCEND_RT_VISIBLE_DEVICES=$c
      export VLLM_USE_V1=1
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
      export HCCL_BUFFSIZE=1024
      export HCCL_IF_IP=90.90.93.34
      export GLOO_SOCKET_IFNAME=ens6f1np1
      export TP_SOCKET_IFNAME=ens6f1np1
      export HCCL_SOCKET_IFNAME=ens6f1np1
      export OMP_PROC_BIND=false
      export OMP_NUM_THREADS=100
      export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:\$LD_LIBRARY_PATH
      export ASCEND_CUSTOM_OPP_PATH=$WS/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
      export LD_LIBRARY_PATH=\$ASCEND_CUSTOM_OPP_PATH/op_api/lib:\$LD_LIBRARY_PATH
      echo \"$c:\$\$\" >> $PID_FILE
      exec vllm serve $MODEL --load-format dummy --enforce-eager --max-model-len 4096 \
        --gpu-memory-utilization $HOLD_UTIL --trust-remote-code \
        --port $port --served-model-name hold$c
    " > "/tmp/mc2_probe/hold_card$c.log" 2>&1 &
    sleep 2
  done
  echo "[hold] pid（卡号:pid）:"; cat "$PID_FILE"
  ;;
stop)
  if [ ! -s "$PID_FILE" ]; then
    echo "[hold] 无 pid 文件/为空，无占卡器可停"; exit 0
  fi
  while IFS=: read -r c pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "$pid" 2>/dev/null; then
      # setsid 起的会话：pgid=pid；EngineCore 子进程同在组内（承载显存的就是它），必须整组停
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 2; done
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      for _ in $(seq 1 15); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 2; done
      echo "[hold] card$c pid=$pid 已停（整组退出）"
    else
      echo "[hold] card$c pid=$pid 已不在"
    fi
  done < "$PID_FILE"
  : > "$PID_FILE"
  ;;
status)
  if [ -s "$PID_FILE" ]; then
    while IFS=: read -r c pid; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "[hold] card$c pid=$pid 存活"
      else
        echo "[hold] card$c pid=$pid **不在**（tail /tmp/mc2_probe/hold_card$c.log）"
      fi
    done < "$PID_FILE"
  else
    echo "[hold] 无 pid 记录"
  fi
  npu-smi info 2>/dev/null | grep -A1 "^| [0-9] " | grep 0000 | grep -oE "[0-9]+ */ 131072" | head -8
  ;;
*)
  echo "用法: bash $0 {start <卡号...>|stop|status}" >&2
  exit 2
  ;;
esac
