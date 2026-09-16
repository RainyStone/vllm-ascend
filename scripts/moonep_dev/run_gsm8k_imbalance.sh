#!/bin/bash
# GSM8K load-imbalance comparison: MC2 (no balancing) vs MoonEP slot (MTE).
#
# Workload (per user): GSM8K raw prompts, batch 64 per card (N_SEQ=256, DP4),
# MBT=8192, natural EOS stop (gen cap 1024), cards 4-7.
#
# Measurement: MOE_ROW_STATS=1 records the per-layer local-expert row counts on
# device inside the MLP entry (works for both routes) and dumps
# /tmp/moe_rows_<tag>_rank<rank>.json at exit. Cross-rank skew = max/mean of
# those per call index; within-rank skew = max group / mean group.
set -u
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export MOONEP_DEV_BASE=4
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export MOONEP_DP_FULL_VISIBLE=1 MOONEP_SHMEM_BOOTSTRAP=ipport
export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MOE_ROW_STATS=1 STEP_DUMP=1

export PROMPT_FILE=/home/tmp/prompts_gsm8k_raw.jsonl
export N_SEQ=256 PROMPT_LEN=1024 GEN=1024 MBT=8192 MAXLEN=2048
export DP=4 TP=1 N_SEQ=256
export GMU=$(/usr/local/python3.12.13/bin/python3 /tmp/gmu_now.py 2>/dev/null || echo 0.85)

for leg in mc2 slot; do
  case $leg in
    mc2)  export MOONEP=0 MOONEP_SLOT_MODE=0 ;;
    slot) export MOONEP=1 MOONEP_SLOT_MODE=1 MOONEP_DISABLE_UDMA=1 ;;
  esac
  echo "############ leg=$leg gmu=$GMU n_seq=$N_SEQ mbt=$MBT gen=$GEN $(date +%T)"
  TAG=gsm8k_$leg /tmp/run_multi.sh 2>&1 \
    | grep -E "dp-bench r0|moe-rows|moonep-init|Traceback|RuntimeError|Error|error:" | tail -30
done
echo "############ done $(date +%T)"
