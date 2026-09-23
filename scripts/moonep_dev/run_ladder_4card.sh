#!/bin/bash
# ABCD TTFT ladder on the 4-card UB clique (NPU 4-7).
#
# Why: the 8-card ladder (plot_ladder.py) measures across BOTH UB clusters, so
# four of every rank's seven peers sit on the cross-socket PCIe path. Cards 4-7
# are one UB clique with no cross-socket hop at all, which separates "MoonEP is
# slower" from "MoonEP's all-to-all crosses a slow link that all-gather's ring
# never uses". Result: the slot/ag TTFT ratio stops growing with T (flat
# 1.22-1.31 instead of 1.09 -> 2.47), i.e. the gap is the topology.
# See docs/dev_results_2026-09-20.md.
#
# Shape scaling: the 8-card ladder had T = (N_SEQ/DP)*69 with DP=8. Holding T
# fixed at DP=4 means halving both N_SEQ and MBT.
#
#   rung  N_SEQ  MBT     T/rank
#   A        4   256        69
#   B       16   512       276
#   C       64  2048      1104
#   D      256  6144      4416
#
#   bash scripts/moonep_dev/run_ladder_4card.sh      # 8 legs, ~2.5 min each
#   python3 scripts/moonep_dev/analyze_ladder4.py    # TTFT + per-layer comm
#
# Two pitfalls, both hit for real:
#
#   1. Do NOT narrow ASCEND_RT_VISIBLE_DEVICES to 4,5,6,7. CANN mixes id spaces
#      (VMM allocation properties carry the *physical* id, aclrtGetDevice the
#      in-process logical id), so shmem init fails with -100. Keep the full list
#      visible and shift each DP worker with MOONEP_DEV_BASE (the vendored dpshim
#      does the shift).
#   2. A leg killed mid-run leaves VLLM::EngineCore/Worker orphans holding ~20 GiB
#      per card, and a hard GMU=0.85 then fails the next worker with "Free memory
#      ... less than desired". GMU is derived from the actually-free HBM below.
set -u

export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE
export MOONEP_DEV_BASE=4
export MOONEP_DP_FULL_VISIBLE=1
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

export MOONEP_SHMEM_BOOTSTRAP=ipport
export MOONEP_TIME_STATS=1          # per-phase host wall (dispatch/combine/...)
export MOONEP_BALANCE_STATS=1

export MODEL=/home/Qwen3-30B-A3B
export DP=4 TP=1
export PROMPT_LEN=69
export PROMPT_FILE=/home/tmp/prompts_gsm8k_raw.jsonl
export GEN_TOKENS=1
export UNIQUE_PROMPTS=1
export MAX_NUM_SEQS=256
export MAXLEN=4096
export WARMUP=1

_gmu=$(npu-smi info 2>/dev/null | awk '$1==4 && /^\| 4 / {for(i=1;i<=NF;i++) if($i=="/"){gsub(/[^0-9]/,"",$(i-1)); gsub(/[^0-9]/,"",$(i+1)); printf "%.2f", ($(i-1)/$(i+1))*0.97}}' | head -1)
export GMU=${GMU:-${_gmu:-0.70}}
export ENFORCE_EAGER=1

run() {  # rung route n_seq mbt
    export TAG=lad4_$1_$2 N_SEQ=$3 MBT=$4
    if [ "$2" = slot ]; then
        export MOONEP=1 MOONEP_SLOT_MODE=1 MOONEP_DISABLE_UDMA=1
        unset MOONEP_FORCE_ALLGATHER
    else
        export MOONEP=0 MOONEP_SLOT_MODE=0
        export MOONEP_FORCE_ALLGATHER=1
    fi
    echo "=== rung=$1 route=$2 n_seq=$3 mbt=$4  start $(date +%H:%M:%S)"
    cd /home/tmp
    timeout 1800 /usr/local/python3.12.13/bin/python3 bench_dp.py \
        > "/home/tmp/lad4_$1_$2.log" 2>&1
    echo "    exit=$? $(date +%H:%M:%S)"
}

for spec in "A slot 4 256" "A ag 4 256" \
            "B slot 16 512" "B ag 16 512" \
            "C slot 64 2048" "C ag 64 2048" \
            "D slot 256 6144" "D ag 256 6144"; do
    run $spec
done
echo "[ladder-4card] done $(date +%H:%M:%S)"
