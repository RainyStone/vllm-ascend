#!/bin/bash
# Online-serving leg for the T sweep: `vllm serve` with a VERIFIED MoE route.
#
#   bash serve_moonep.sh slot 32768 8000 2 4
#   bash serve_moonep.sh ag   32768 8001 2 4
#
# Route selection, and why the self-check exists
# ----------------------------------------------
# vllm_ascend never reads a bare `MOONEP` env var -- only MOONEP_* variants.
# MoonEP is turned on by the CONFIG field `enable_moonep`, which the offline
# bench injects via additional_config. Setting MOONEP=1 on the CLI does nothing,
# and the failure is silent: the server starts, answers 200, and measures the
# SoC default (MC2 / ALLTOALL) while you believe you are measuring MoonEP.
#
# So the route is asserted from RUNTIME EVIDENCE, not from the env:
# moonep_comm prints `[moonep-init r<rank>] stage=creating-buffers ...` only when
# the MoonEP runtime actually initialises, i.e. only once a MoonEP forward ran.
#   slot -> that line MUST appear      ag -> it MUST NOT
#
# MBT is fixed per server and T is swept from the client, so the whole sweep is
# two server starts. Cards 4-7 are one UB clique; do NOT narrow
# ASCEND_RT_VISIBLE_DEVICES (CANN mixes id spaces; shmem init fails with -100)
# -- MOONEP_DEV_BASE does the shift instead.
set -u
ROUTE=${1:?usage: serve_moonep.sh <slot|ag> <mbt> <port> [dp] [base]}
MBT=${2:?}
PORT=${3:?}
DP=${4:-2}
BASE=${5:-4}

export VISIBLE=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE
export MOONEP_DEV_BASE=$BASE
export MOONEP_DP_FULL_VISIBLE=1
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MOONEP_SHMEM_BOOTSTRAP=ipport
export MOONEP_TIME_STATS=1          # per-phase comm wall, printed at exit
export TOKENIZERS_PARALLELISM=false

# MoonEP is selected by the CONFIG flag, not by an env var.
if [ "$ROUTE" = slot ]; then
    export MOONEP_SLOT_MODE=1 MOONEP_DISABLE_UDMA=1
    unset MOONEP_FORCE_ALLGATHER
    EXTRA_CONFIG='{"enable_moonep": true}'
else
    export MOONEP_SLOT_MODE=0
    export MOONEP_FORCE_ALLGATHER=1
    EXTRA_CONFIG=''
fi

# gpu_memory_utilization from the free memory of the cards THIS server will use
# (BASE .. BASE+DP-1) -- not from gmu_now.py, which spans 4-7 and therefore
# reads the OTHER server's cards when two legs run side by side. That mismatch
# silently clamped a second server to the 0.30 floor while its own cards were
# empty, leaving it a third of the KV cache and an incomparable configuration.
# Capped at 0.92 so a nearly-empty card does not push the request to the edge.
_gmu=$(/usr/local/python3.12.13/bin/python3 -c "
import torch, torch_npu
worst = 1.0
for d in range($BASE, $BASE + $DP):
    torch.npu.set_device(d)
    free, total = torch.npu.mem_get_info(d)
    worst = min(worst, free / total)
print(f'{min(max(0.30, worst - 0.02), 0.92):.2f}')
" 2>/dev/null || echo 0.70)
export GMU=${GMU:-$_gmu}

# ROUTES=1 additionally returns the router's per-token expert assignment in
# every completion response (base64 numpy, ~50 KB/request). Only needed for the
# imbalance sweep; it inflates TTFT, so leave it off for the timing sweep.
ROUTES_ARG=""
[ "${ROUTES:-0}" = "1" ] && ROUTES_ARG="--enable-return-routed-experts"

# DETAIL=1 logs one line per engine iteration:
#   Iteration(n): X context requests, T context tokens, ... (dummy)
# The "(dummy)" marker is what separates real work from DP sync steps, which is
# the only way to tell why `steps` exceeds DP in the /metrics sweep.
DETAIL_ARG=""
[ "${DETAIL:-0}" = "1" ] && DETAIL_ARG="--enable-logging-iteration-details"

# Long-context runs must raise --max-model-len past the prompt length, and drop
# --max-num-seqs with it: vllm sizes the KV cache as max_num_seqs x
# max_model_len, so 512 x 20480 will not fit and the engine refuses to start.
export MAXLEN=${MAXLEN:-4096}
export MAXSEQS=${MAXSEQS:-512}

LOG=/home/tmp/serve_${ROUTE}_p${PORT}.log
echo "[serve] route=$ROUTE dp=$DP base=$BASE mbt=$MBT port=$PORT gmu=$GMU $(date +%H:%M:%S)"
echo "[serve] cards $BASE..$((BASE + DP - 1))"
echo "[serve] maxlen=$MAXLEN maxseqs=$MAXSEQS"
echo "[serve] log -> $LOG"

/usr/local/python3.12.13/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /home/Qwen3-30B-A3B \
    --served-model-name qwen3 \
    --data-parallel-size "$DP" \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS" \
    --max-num-batched-tokens "$MBT" \
    --gpu-memory-utilization "$GMU" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "$PORT" \
    --no-enable-log-requests \
    ${ROUTES_ARG:+$ROUTES_ARG} \
    ${DETAIL_ARG:+$DETAIL_ARG} \
    ${EXTRA_CONFIG:+--additional-config "$EXTRA_CONFIG"} \
    > "$LOG" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
    curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "[serve] FATAL: server exited during startup"; tail -25 "$LOG"; exit 1; }
    sleep 10
done
if ! curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[serve] FATAL: not healthy after 10 min"; tail -25 "$LOG"; exit 1
fi

# ---- self-check: assert the route from runtime evidence ----------------------
# The marker appears only if a MoonEP forward actually initialised the runtime,
# so neither a stale env var nor a typo'd flag can fake it. Count DISTINCT ranks,
# not log lines, and require every DP rank to have engaged.
N_RANKS=$(grep -oE "moonep-init r[0-9]+" "$LOG" 2>/dev/null | sort -u | wc -l)
N_RANKS=${N_RANKS:-0}
if [ "$ROUTE" = slot ]; then
    if [ "$N_RANKS" -ne "$DP" ]; then
        echo "[serve] FATAL: slot route requested, but MoonEP engaged on $N_RANKS/$DP ranks."
        echo "        MoonEP did NOT fully engage -- this server would not be measuring it."
        echo "        (check --additional-config / --enable-expert-parallel)"
        exit 1
    fi
    echo "[serve] SELF-CHECK OK: MoonEP engaged on all $N_RANKS/$DP ranks"
else
    if [ "$N_RANKS" -ge 1 ]; then
        echo "[serve] FATAL: ag route requested, but MoonEP engaged on $N_RANKS ranks."
        exit 1
    fi
    echo "[serve] SELF-CHECK OK: all-gather route confirmed (no MoonEP runtime)"
fi
echo "[serve] READY on :$PORT $(date +%H:%M:%S)"
wait $SRV
