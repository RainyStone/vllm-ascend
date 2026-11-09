#!/usr/bin/env bash
set -Ee -o pipefail
SOURCE='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep'
BUILD='/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep/build/dp16_optimized_20261106'
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest'
[[ $# == 0 ]] || exit 2
source /usr/local/Ascend/cann/set_env.sh
set -u
export TMPDIR='/dev/shm/dp16_optimized_build_20261106'
mkdir -p "$TMPDIR" "$EXPERIMENT_DIR/data"
cmake -S "$SOURCE" -B "$BUILD" -DASCEND_MOONEP_BISHENG_ARCH=a5 -DASCEND_MOONEP_BUILD_MXFP8_WIRE=ON -DASCEND_MOONEP_BUILD_PUBLISH=ON
cmake --build "$BUILD" --target ascend_moonep_kernels ascend_moonep_mxfp8_wire ascend_moonep_publish --parallel 2
sha256sum "$BUILD/lib/libascend_moonep_kernels.so" "$BUILD/lib/libascend_moonep_mxfp8_wire.so" "$BUILD/lib/libascend_moonep_publish.so"
