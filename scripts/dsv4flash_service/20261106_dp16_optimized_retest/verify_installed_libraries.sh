#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='/mnt/share/x00893896/vllm_ascend_moonep_integration'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest'
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
unset PYTHONPATH LD_PRELOAD
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$WORKSPACE/vllm:$WORKSPACE/vllm-ascend:$WORKSPACE/ascend-moonep"
export LD_LIBRARY_PATH="$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
export PYTHONDONTWRITEBYTECODE=1
exec python - "$EXPERIMENT_DIR" "$WORKSPACE" <<'PY'
from pathlib import Path
import ctypes,hashlib,json,socket,sys
root,ws=map(Path,sys.argv[1:])
import torch,torch_npu,ascend_moonep
from ascend_moonep import _C
from ascend_moonep.combine_publish import CombinePublisher

assert Path(ascend_moonep.__file__).resolve()==ws/'ascend-moonep/ascend_moonep/__init__.py'
assert Path(_C.__file__).resolve().parent==ws/'ascend-moonep/ascend_moonep'
updates=json.loads((root/'LIBRARY_UPDATE.json').read_text())
for item in updates['changes']:
 assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['after']
main=ws/'ascend-moonep/ascend_moonep/lib/libascend_moonep_kernels.so'
lib=ctypes.CDLL(str(main));assert lib.moonep_udma_weight_reuse_fix_abi()==2
CombinePublisher()
wire=ctypes.CDLL(str(ws/"ascend-moonep/ascend_moonep/lib/libascend_moonep_mxfp8_wire.so"));assert wire.moonep_mxfp8_wire_abi()==1
mapped={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '.so' in line and ('moonep' in line or 'libshmem' in line)}
assert {p for p in mapped if Path(p).name==main.name}=={str(main)}
assert len([p for p in mapped if Path(p).name=='libshmem.so'])==1
node=socket.gethostbyname(socket.gethostname())
data=dict(valid=True,host=socket.gethostname(),ip=node,package=ascend_moonep.__file__,extension=_C.__file__,scalar_fence_abi=2,no_device_allocation=True,mapped_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in mapped})
dest=root/'data/installed_library_audit';dest.mkdir(exist_ok=True)
(dest/(socket.gethostname()+'.json')).write_text(json.dumps(data,indent=2))
print('ORIGINAL_INSTALL_VERIFIED',socket.gethostname(),str(main),hashlib.sha256(main.read_bytes()).hexdigest(),flush=True)
PY
