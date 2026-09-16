#!/bin/bash
# A/B: MOONEP_RELEASE_WEIGHTS=0 vs 1, identical everything else, short gen.
# The release only re-points layer.w13_weight at the VMM home chunk (slot mode
# computes against the [epn+B] alias either way), so the token streams must be
# bit-identical. Any difference = a code path still reading the old tensor.
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export MOONEP_DEV_BASE=4
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export MOONEP_DP_FULL_VISIBLE=1 MOONEP_SHMEM_BOOTSTRAP=ipport
export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export DUMP_TOKENS=1 MOONEP_BALANCE_STATS=1
export PROMPT_FILE=/home/tmp/prompts_gsm8k_raw.jsonl
export PROMPT_LEN=1024 GEN=128 MBT=8192 MAXLEN=2048 DP=4 TP=1 N_SEQ=256 GMU=0.85
export MOONEP=1 MOONEP_SLOT_MODE=1 MOONEP_DISABLE_UDMA=1
for rel in 0 1; do
  export MOONEP_RELEASE_WEIGHTS=$rel
  echo "############ release=$rel $(date +%T)"
  TAG=ab_rel$rel /tmp/run_multi.sh 2>&1 | grep -E "moonep-release|moonep-balance|dp-bench r0|exit|Traceback|RuntimeError" | tail -12
  echo "############ release=$rel exit=$? $(date +%T)"
done
