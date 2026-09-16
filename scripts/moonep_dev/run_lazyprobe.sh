#!/bin/bash
export VISIBLE=${VISIBLE:-0,1,2,3,4,5,6,7}
export PLAN_DEV_BASE=${PLAN_DEV_BASE:-0}
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE
export PYTHONPATH=/home/vllm-ascend-moonep/third_party/ascend_moonep:/home/vllm-ascend-moonep:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export MOONEP_SHMEM_BOOTSTRAP=ipport
timeout 300 torchrun --nproc_per_node=4 --master_port=${PORT:-29685} /tmp/probe_lazy_buffer.py 2>&1 | grep -E "^\[r[0-9]\]|Error|Traceback|error" | head -20
