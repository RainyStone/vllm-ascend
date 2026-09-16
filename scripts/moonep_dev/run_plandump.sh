#!/bin/bash
# Standalone MoonEP plan dump on 4 cards (default 4-7).
# full visible list + PLAN_DEV_BASE: keeps logical id == physical id for shmem
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export PLAN_DEV_BASE=${PLAN_DEV_BASE:-4}
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE
export PYTHONPATH=/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export MOONEP_SHMEM_BOOTSTRAP=ipport
export PLAN_S=${PLAN_S:-64} PLAN_T=${PLAN_T:-32} PLAN_SKEW=${PLAN_SKEW:-0.6}
timeout ${TIMEOUT:-300} torchrun --nproc_per_node=4 --master_port=${PORT:-29677} \
  /tmp/moonep_plan_dump.py
