#!/usr/bin/env bash
set -Ee -o pipefail
RUN='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/precision_fix/weighted_combine_v1/main_repo_install_20261104'
SDK='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep'
source /usr/local/Ascend/cann/set_env.sh
export PATH="/usr/local/python3.11.10/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/python3.11.10/lib:$LD_LIBRARY_PATH"
export SHMEM_HOME_PATH="$SDK/3rdparty/shmem/install"
export ASCEND_MOONEP_KERNEL_LIB_DIR="$SDK/ascend_moonep/lib"
export MAX_JOBS=4
set -u
mkdir -p "$RUN/logs" "$RUN/before"
exec > >(tee "$RUN/logs/build_install_124.log") 2>&1
python - "$SDK" "$RUN" <<'PY'
import sys,shutil,hashlib,json
from pathlib import Path
sdk,run=map(Path,sys.argv[1:])
files=[*sdk.joinpath('ascend_moonep').glob('_C*.so'),sdk/'ascend_moonep/lib/libascend_moonep_kernels.so']
record={}
for p in files:
 dest=run/'before'/p.relative_to(sdk);dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(p,dest)
 record[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(run/'before_libraries.json').write_text(json.dumps(record,indent=2))
PY
echo STAGE=kernel_configure
cmake -S "$SDK" -B "$SDK/build/weighted_combine_20261104" \
 -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
 -DASCEND_MOONEP_BISHENG_ARCH=a5 \
 -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" \
 -DASCEND_MOONEP_SHMEM_SOURCE_DIR="$SDK/3rdparty/shmem" \
 -DASCEND_MOONEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/aarch64-linux/pkg_inc/runtime/runtime"
echo STAGE=kernel_build
cmake --build "$SDK/build/weighted_combine_20261104" --target ascend_moonep_kernels --parallel 4
python - "$SDK" <<'PY'
from pathlib import Path
import sys,shutil,os
sdk=Path(sys.argv[1])
src=sdk/'build/weighted_combine_20261104/lib/libascend_moonep_kernels.so'
dest=sdk/'ascend_moonep/lib/libascend_moonep_kernels.so'
tmp=dest.with_name(dest.name+'.new')
shutil.copy2(src,tmp)
os.replace(tmp,dest)
PY
echo STAGE=editable_install_original_repo
cd "$SDK"
python -m pip install --no-index --no-deps --no-build-isolation \
 --config-settings editable_mode=compat -e .
echo STAGE=main_install_complete
