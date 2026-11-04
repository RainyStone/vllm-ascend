#!/usr/bin/env bash
set -Ee -o pipefail
ROOT='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/precision_fix/weighted_combine_v1/host124'
SDK='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep'
CARDS='0,1,2,3,4,5,6,7'
WORLD=8
MASTER_PORT=29880
SHMEM_PORT=29881
OUT="$ROOT/results/operator_dp8"
source /usr/local/Ascend/cann/set_env.sh
export PATH="/usr/local/python3.11.10/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/python3.11.10/lib:$LD_LIBRARY_PATH"
unset PYTHONPATH
export SHMEM_HOME_PATH='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep/3rdparty/shmem/install'
export ASCEND_MOONEP_KERNEL_LIB_DIR="$SDK/ascend_moonep/lib"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export ASCEND_MOONEP_SKIP_DUP=1
export ASCEND_MOONEP_PREFETCH_RELAXED=1
export ASCEND_MOONEP_NO_RECONCILE=1
export ASCEND_MOONEP_USE_UDMA=0
export ASCEND_MOONEP_PLANNING_USE_UDMA=0
export ASCEND_MOONEP_DISPATCH_USE_UDMA=0
export ASCEND_MOONEP_COMBINE_USE_UDMA=0
export ASCEND_MOONEP_PREFETCH_USE_UDMA=0
export OMP_NUM_THREADS=1
set -u
# Do not disturb a resident service; shell scripts only run on idle selected cards.
python - "$CARDS" <<'PY'
import re, subprocess, sys
cards=set(map(int,sys.argv[1].split(',')))
s=subprocess.check_output(['npu-smi','info'],text=True)
busy=[(int(a),int(b)) for a,b in re.findall(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|',s,re.M) if int(a) in cards]
if busy:
    raise SystemExit(f'Refusing operator timing on busy cards: {busy}')
PY
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT"
python -m torch.distributed.run --nnodes=1 --nproc-per-node="$WORLD" \
 --master-addr=127.0.0.1 --master-port="$MASTER_PORT" \
 "$ROOT/../tests/test_weighted_combine.py" --output "$OUT" \
 --shmem-port "$SHMEM_PORT" --performance 2>&1 | tee "$ROOT/logs/operator_dp8.log"
