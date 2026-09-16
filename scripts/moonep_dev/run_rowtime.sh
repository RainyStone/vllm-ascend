#!/bin/bash
# Does intra-rank group skew actually cost GEMM time?
# MC2 route on purpose: no MoonEP prefetch confound. The decode plateau has
# near-constant total rows per layer while the group skew varies 1.0-3.0, so
# the layer-to-layer variation isolates the skew effect.
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export MOONEP_DEV_BASE=4
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export MOONEP_DP_FULL_VISIBLE=1 MOONEP_SHMEM_BOOTSTRAP=ipport
export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MOE_ROW_STATS=1 MOE_ROW_TIMING=1 MOE_ROW_STATS_MAX=200000
export PROMPT_FILE=/home/tmp/prompts_gsm8k_raw.jsonl
export PROMPT_LEN=1024 GEN=256 MBT=8192 MAXLEN=2048 DP=4 TP=1 N_SEQ=256 GMU=0.85
export MOONEP=0 MOONEP_SLOT_MODE=0
echo "############ rowtime $(date +%T)"
TAG=rowtime_mc2 /tmp/run_multi.sh 2>&1 | grep -E "moe-rowtimes|dp-bench r0|Traceback|RuntimeError" | grep -v "^     " | head -8
echo "############ exit=$? $(date +%T)"
