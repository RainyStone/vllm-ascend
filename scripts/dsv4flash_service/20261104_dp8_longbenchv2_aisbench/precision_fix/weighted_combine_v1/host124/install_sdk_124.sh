#!/usr/bin/env bash
set -Ee -o pipefail
ROOT='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/precision_fix/weighted_combine_v1/host124'
SDK='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep'
source /usr/local/Ascend/cann/set_env.sh
export PATH="/usr/local/python3.11.10/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/python3.11.10/lib:$LD_LIBRARY_PATH"
export SHMEM_HOME_PATH='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep/3rdparty/shmem/install'
export ASCEND_MOONEP_KERNEL_LIB_DIR="$SDK/ascend_moonep/lib"
export MAX_JOBS=4
set -u
cd "$SDK"
python -m pip install --no-index --no-deps --no-build-isolation \
 --config-settings editable_mode=compat -e . > "$ROOT/logs/install_sdk.log" 2>&1
cd /tmp
# Validate ordinary import without a test-only PYTHONPATH override.
env -u PYTHONPATH python - "$SDK" "$ROOT/results/sdk_install_after.json" <<'PY'
import ascend_moonep, torch, json, sys, importlib.metadata as m
from pathlib import Path
sdk=Path(sys.argv[1])
assert Path(ascend_moonep.__file__).resolve().is_relative_to(sdk.resolve())
assert hasattr(torch.classes.AscendMoonEP.Combine(),"compute_weighted")
d=m.distribution("ascend-moonep")
record=dict(package=ascend_moonep.__file__,version=d.version,
            direct_url=d.read_text("direct_url.json"),
            compute_weighted=True)
Path(sys.argv[2]).write_text(json.dumps(record,indent=2))
print(json.dumps(record))
PY
