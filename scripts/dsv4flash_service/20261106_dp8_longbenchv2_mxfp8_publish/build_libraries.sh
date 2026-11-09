#!/usr/bin/env bash
set -Ee -o pipefail
source /usr/local/Ascend/cann/set_env.sh
set -u
SOURCE='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep'
BUILD='/tmp/moonep_longbench_latest_20261106'
export TMPDIR='/dev/shm/moonep_longbench_latest_20261106'
mkdir -p "$TMPDIR"
cmake -S "$SOURCE" -B "$BUILD" -DASCEND_MOONEP_BISHENG_ARCH=a5 -DASCEND_MOONEP_BUILD_MXFP8_WIRE=ON -DASCEND_MOONEP_BUILD_PUBLISH=ON
cmake --build "$BUILD" --target ascend_moonep_mxfp8_wire ascend_moonep_publish --parallel 2
sha256sum "$BUILD/lib/libascend_moonep_mxfp8_wire.so" "$BUILD/lib/libascend_moonep_publish.so"
