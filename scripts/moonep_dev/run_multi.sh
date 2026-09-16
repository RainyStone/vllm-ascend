#!/bin/bash
# N-card DP x TP run (all knobs via env). Defaults: 8 cards, DP8 x TP1 -> EP=8.
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE
export PYTHONPATH=/home/tmp/dpshim:/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MOONEP_DP_FULL_VISIBLE=1 MOONEP_SHMEM_BOOTSTRAP=ipport MOONEP_TIME_STATS=1
export MODEL=/home/Qwen3-30B-A3B PROMPT_LEN=256 MBT=${MBT:-1024}
export DP=${DP:-8} TP=${TP:-1} N_SEQ=${N_SEQ:-1024} GEN_TOKENS=${GEN:-128} GMU=${GMU:-0.85}
export TAG=${TAG:-mte} MOONEP=${MOONEP:-1} MOONEP_BALANCE_STATS=${MOONEP_BALANCE_STATS:-1}
# MTE by default; URMA=1 keeps the URMA context (no DISABLE_UDMA)
if [ "${URMA:-0}" != "1" ]; then export MOONEP_DISABLE_UDMA=1; fi
cd /home/tmp && timeout ${TIMEOUT:-1800} /usr/local/python3.12.13/bin/python3 bench_dp.py
