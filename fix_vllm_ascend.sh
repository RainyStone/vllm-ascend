#!/bin/bash
# ============================================================================
# vllm-ascend 第三方依赖修复脚本
#
# 【背景】
#   构建 vllm-ascend 时，CMake 需要从 gitcode.com 下载多个第三方依赖，
#   但构建服务器所在网络无法访问 file-cdn.gitcode.com（TLS 握手被 reset），
#   导致依赖下载失败、解压为空目录、编译报错。
#
# 【用途】
#   1. 验证本地压缩包完整性
#   2. 将压缩包解压到正确的第三方目录
#   3. 修改 CMake 文件，将 gitcode.com 远程 URL 改为本地 file:/// 路径
#   4. 清理构建缓存并重新构建
#
# 【前置条件】
#   在有网络的机器上下载以下文件，上传到构建服务器的第三方目录：
#     目录: /sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/
#
#     1. nlohmann/json v3.11.3 (include.zip)
#        https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip
#
#     2. protobuf v25.1
#        https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz
#
#     3. googletest v1.14.0（可选，构建可能需要）
#        https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz
#
#     4. abseil-cpp 20230802.1（可选，已有目录可跳过）
#        https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz
#
# 【用法】
#   bash fix_vllm_ascend.sh
#
# 【注意事项】
#   - 脚本会在修改 cmake 文件前自动备份到 backup_<时间戳> 目录
#   - 每个步骤有交互确认，Ctrl+C 可随时中断
#   - 构建日志保存在 /sfs_turbo/hw/xiazhixiang/vllm-ascend/build.log
# ============================================================================

set -e

BASE_DIR="/sfs_turbo/hw/xiazhixiang/vllm-ascend"
THIRD_PARTY_DIR="${BASE_DIR}/csrc/third_party"
CMAKE_DIR="${BASE_DIR}/csrc/cmake/third_party"
BUILD_DIR="${BASE_DIR}/csrc/build"

echo "============================================"
echo "  vllm-ascend 第三方依赖修复脚本"
echo "============================================"
echo ""

# -------------------------------------------------------
# 辅助函数
# -------------------------------------------------------
# 输出文件存在信息，统一格式: [FOUND] <路径> (<大小>)
print_found() {
    local filepath="$1"
    local size=$(stat -c%s "${filepath}" 2>/dev/null || stat -f%z "${filepath}" 2>/dev/null || echo "unknown")
    local size_mb=$(echo "scale=2; ${size} / 1048576" | bc 2>/dev/null || echo "${size}")
    echo "  [FOUND] ${filepath} (${size_mb} MB)"
}

# 输出文件缺失信息，统一格式: [MISSING] <路径>
print_missing() {
    echo "  [MISSING] $1"
}

# 输出文件损坏信息，统一格式: [BROKEN] <路径> <原因>
print_broken() {
    echo "  [BROKEN] $1 - $2"
}

# -------------------------------------------------------
# 第一步: 验证所有压缩包完整性
# -------------------------------------------------------
echo "===== 第一步: 验证压缩包完整性 ====="

# --- 1. include.zip (nlohmann/json) ---
echo ""
echo "--- 1. include.zip (nlohmann/json v3.11.3) ---"
INCLUDE_ZIP="${THIRD_PARTY_DIR}/include.zip"
if [ -f "${INCLUDE_ZIP}" ]; then
    print_found "${INCLUDE_ZIP}"
    if unzip -t "${INCLUDE_ZIP}" > /dev/null 2>&1; then
        echo "  [OK] 完整性校验通过"
    else
        print_broken "${INCLUDE_ZIP}" "文件损坏，请重新上传"
        echo "  下载地址: https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip"
        exit 1
    fi
else
    print_missing "${INCLUDE_ZIP}"
    echo "  请上传后再运行本脚本"
    exit 1
fi

# --- 2. protobuf-25.1.tar.gz ---
echo ""
echo "--- 2. protobuf-25.1.tar.gz ---"
PROTOBUF_TAR="${THIRD_PARTY_DIR}/protobuf-25.1.tar.gz"
if [ -f "${PROTOBUF_TAR}" ]; then
    print_found "${PROTOBUF_TAR}"
    if tar tzf "${PROTOBUF_TAR}" > /dev/null 2>&1; then
        echo "  [OK] 完整性校验通过"
    else
        print_broken "${PROTOBUF_TAR}" "文件损坏，请重新上传"
        echo "  下载地址: https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz"
        exit 1
    fi
else
    print_missing "${PROTOBUF_TAR}"
    echo "  请上传后再运行本脚本"
    exit 1
fi

# --- 3. googletest-1.14.0.tar.gz (可选) ---
echo ""
echo "--- 3. googletest-1.14.0.tar.gz (可选) ---"
GTEST_TAR="${THIRD_PARTY_DIR}/googletest-1.14.0.tar.gz"
HAVE_GTEST=0
if [ -f "${GTEST_TAR}" ]; then
    print_found "${GTEST_TAR}"
    if tar tzf "${GTEST_TAR}" > /dev/null 2>&1; then
        echo "  [OK] 完整性校验通过"
        HAVE_GTEST=1
    else
        print_broken "${GTEST_TAR}" "文件损坏"
        echo "  下载地址: https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz"
    fi
else
    print_missing "${GTEST_TAR}"
    echo "  (可选，构建时报错再上传即可)"
fi

# --- 4. abseil-cpp-20230802.1.tar.gz (可选) ---
echo ""
echo "--- 4. abseil-cpp-20230802.1.tar.gz (可选) ---"
ABSEIL_TAR="${THIRD_PARTY_DIR}/abseil-cpp-20230802.1.tar.gz"
HAVE_ABSEIL_TAR=0
if [ -f "${ABSEIL_TAR}" ]; then
    print_found "${ABSEIL_TAR}"
    if tar tzf "${ABSEIL_TAR}" > /dev/null 2>&1; then
        echo "  [OK] 完整性校验通过"
        HAVE_ABSEIL_TAR=1
    else
        print_broken "${ABSEIL_TAR}" "文件损坏"
    fi
else
    print_missing "${ABSEIL_TAR}"
    echo "  (可选，abseil-cpp 目录已存在则可忽略)"
fi
if [ -f "${THIRD_PARTY_DIR}/abseil-cpp/CMakeLists.txt" ]; then
    echo "  [FOUND] ${THIRD_PARTY_DIR}/abseil-cpp/ (已解压，目录正常)"
else
    echo "  [INFO] abseil-cpp 目录缺失或不完整"
fi

# --- 5. makeself ---
echo ""
echo "--- 5. makeself (已修复，确认状态) ---"
MAKESELF_TAR="${THIRD_PARTY_DIR}/makeself-release-2.5.0-patch1.tar.gz"
if [ -f "${MAKESELF_TAR}" ]; then
    print_found "${MAKESELF_TAR}"
else
    print_missing "${MAKESELF_TAR}"
fi
if [ -f "${THIRD_PARTY_DIR}/makeself/makeself-header.sh" ]; then
    echo "  [FOUND] ${THIRD_PARTY_DIR}/makeself/ (已解压，目录正常)"
else
    echo "  [INFO] makeself 目录不完整"
fi

echo ""
read -p "以上检查是否正常？按回车继续，Ctrl+C 取消..."

# -------------------------------------------------------
# 第二步: 解压 json (nlohmann/json)
# -------------------------------------------------------
echo ""
echo "===== 第二步: 解压 json (nlohmann/json) ====="

JSON_DIR="${THIRD_PARTY_DIR}/json"
rm -rf "${JSON_DIR}/include"
mkdir -p "${JSON_DIR}/include"

TMP_JSON=$(mktemp -d)
unzip -o "${INCLUDE_ZIP}" -d "${TMP_JSON}" > /dev/null

# 查找 json.hpp 的位置
JSON_HPP=$(find "${TMP_JSON}" -name "json.hpp" | head -1)
if [ -n "${JSON_HPP}" ]; then
    if [ -d "${TMP_JSON}/include/nlohmann" ]; then
        cp -r "${TMP_JSON}/include/nlohmann" "${JSON_DIR}/include/"
    else
        NLH_PARENT=$(find "${TMP_JSON}" -type d -name "nlohmann" | head -1)
        if [ -n "${NLH_PARENT}" ]; then
            mkdir -p "${JSON_DIR}/include"
            cp -r "${NLH_PARENT}" "${JSON_DIR}/include/"
        fi
    fi
    rm -rf "${TMP_JSON}"
    echo "  [OK] json 解压完成"
else
    rm -rf "${TMP_JSON}"
    echo "  [ERROR] include.zip 中未找到 json.hpp"
    exit 1
fi

# 验证
if [ -f "${JSON_DIR}/include/nlohmann/json.hpp" ]; then
    echo "  [FOUND] ${JSON_DIR}/include/nlohmann/json.hpp"
else
    echo "  [ERROR] nlohmann/json.hpp 未找到，请检查解压结果:"
    find "${JSON_DIR}" -name "json.hpp" 2>/dev/null
    exit 1
fi

# -------------------------------------------------------
# 第三步: 解压 protobuf
# -------------------------------------------------------
echo ""
echo "===== 第三步: 解压 protobuf ====="

PROTOBUF_DIR="${THIRD_PARTY_DIR}/ascend_protobuf"
rm -rf "${PROTOBUF_DIR}"
mkdir -p "${PROTOBUF_DIR}"

TOP_DIR=$(tar tzf "${PROTOBUF_TAR}" | head -1)
echo "  tarball 顶层目录: ${TOP_DIR}"

if [[ "${TOP_DIR}" == */ ]]; then
    tar xzf "${PROTOBUF_TAR}" -C "${PROTOBUF_DIR}" --strip-components=1
else
    tar xzf "${PROTOBUF_TAR}" -C "${PROTOBUF_DIR}"
fi

if [ -f "${PROTOBUF_DIR}/CMakeLists.txt" ]; then
    echo "  [FOUND] ${PROTOBUF_DIR}/CMakeLists.txt"
else
    echo "  [ERROR] CMakeLists.txt 未找到"
    ls -la "${PROTOBUF_DIR}/"
    exit 1
fi

# -------------------------------------------------------
# 第四步: 处理 gtest（如果有）
# -------------------------------------------------------
echo ""
echo "===== 第四步: 处理 gtest ====="

if [ "${HAVE_GTEST}" = "1" ]; then
    GTEST_DIR="${THIRD_PARTY_DIR}/googletest"
    if [ ! -d "${GTEST_DIR}" ] || [ -z "$(ls -A ${GTEST_DIR} 2>/dev/null)" ]; then
        rm -rf "${GTEST_DIR}"
        mkdir -p "${GTEST_DIR}"
        TOP_DIR=$(tar tzf "${GTEST_TAR}" | head -1)
        if [[ "${TOP_DIR}" == */ ]]; then
            tar xzf "${GTEST_TAR}" -C "${GTEST_DIR}" --strip-components=1
        else
            tar xzf "${GTEST_TAR}" -C "${GTEST_DIR}"
        fi
        if [ -f "${GTEST_DIR}/CMakeLists.txt" ]; then
            echo "  [FOUND] ${GTEST_DIR}/CMakeLists.txt"
        else
            echo "  [WARN] googletest 解压后未找到 CMakeLists.txt"
        fi
    else
        echo "  [FOUND] ${GTEST_DIR}/ (目录已有内容，跳过)"
    fi
else
    echo "  [SKIP] googletest 未上传，跳过（构建时报错再处理）"
fi

# -------------------------------------------------------
# 第五步: 修改 CMake 文件，改用本地路径
# -------------------------------------------------------
echo ""
echo "===== 第五步: 修改 CMake 文件 ====="

BACKUP_DIR="${CMAKE_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"
cp -r "${CMAKE_DIR}"/*.cmake "${BACKUP_DIR}/"
echo "  已备份 cmake 文件到: ${BACKUP_DIR}"

# --- 5.1 修改 json.cmake ---
echo ""
echo "--- 修改 json.cmake ---"
JSON_CMAKE="${CMAKE_DIR}/json.cmake"
if grep -q "gitcode.com" "${JSON_CMAKE}" 2>/dev/null; then
    sed -i 's|set(REQ_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip")|set(REQ_URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/include.zip")|g' "${JSON_CMAKE}"
    echo "  [OK] json.cmake 已修改为本地路径"
else
    echo "  [SKIP] json.cmake 未找到 gitcode.com URL（可能已修改）"
fi

# --- 5.2 修改 ascend_protobuf.cmake ---
echo ""
echo "--- 修改 ascend_protobuf.cmake ---"
PROTO_CMAKE="${CMAKE_DIR}/ascend_protobuf.cmake"
if grep -q "gitcode.com" "${PROTO_CMAKE}" 2>/dev/null; then
    sed -i 's|set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")|set(REQ_URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/protobuf-25.1.tar.gz")|g' "${PROTO_CMAKE}"
    echo "  [OK] ascend_protobuf.cmake REQ_URL 已修改"
fi

# 修复 ExternalProject_Add 中的 URL（处理之前可能的不完整修改）
if grep -q 'URL.*\${_protobuf_url}' "${PROTO_CMAKE}"; then
    # 将 URL 行替换为只有本地路径
    sed -i '/URL.*\${_protobuf_url}/,/\//s|URL.*\${_protobuf_url}.*|URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/protobuf-25.1.tar.gz"|' "${PROTO_CMAKE}"
    echo "  [OK] ascend_protobuf.cmake ExternalProject URL 已修复"
fi

echo "  修改后 ascend_protobuf.cmake 中的 URL 相关行:"
grep -n "URL\|_protobuf_url\|REQ_URL\|gitcode\|file://" "${PROTO_CMAKE}" || echo "  (未匹配到)"

# --- 5.3 修改 gtest.cmake ---
echo ""
echo "--- 修改 gtest.cmake ---"
GTEST_CMAKE="${CMAKE_DIR}/gtest.cmake"
if grep -q "gitcode.com" "${GTEST_CMAKE}" 2>/dev/null; then
    if [ "${HAVE_GTEST}" = "1" ]; then
        sed -i 's|set(REQ_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz")|set(REQ_URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/googletest-1.14.0.tar.gz")|g' "${GTEST_CMAKE}"
        echo "  [OK] gtest.cmake 已修改为本地路径"
    else
        echo "  [WARN] gtest.cmake 有 gitcode URL 但未上传 gtest 压缩包，暂时跳过"
    fi
else
    echo "  [SKIP] gtest.cmake 未找到 gitcode.com URL"
fi

# --- 5.4 修改 abseil-cpp.cmake ---
echo ""
echo "--- 修改 abseil-cpp.cmake ---"
ABSEIL_CMAKE="${CMAKE_DIR}/abseil-cpp.cmake"
if grep -q "gitcode.com" "${ABSEIL_CMAKE}" 2>/dev/null; then
    if [ "${HAVE_ABSEIL_TAR}" = "1" ]; then
        sed -i 's|set(REQ_URL "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz")|set(REQ_URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/abseil-cpp-20230802.1.tar.gz")|g' "${ABSEIL_CMAKE}"
        echo "  [OK] abseil-cpp.cmake 已修改为本地路径"
    else
        echo "  [WARN] abseil-cpp.cmake 有 gitcode URL 但未上传 abseil-cpp 压缩包，暂时跳过"
    fi
else
    echo "  [SKIP] abseil-cpp.cmake 未找到 gitcode.com URL"
fi

# --- 5.5 修改 makeself-fetch.cmake ---
echo ""
echo "--- 修改 makeself-fetch.cmake ---"
MAKESELF_CMAKE="${CMAKE_DIR}/makeself-fetch.cmake"
if grep -q "gitcode.com" "${MAKESELF_CMAKE}" 2>/dev/null; then
    sed -i 's|set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")|set(MAKESELF_URL "file:///sfs_turbo/hw/xiazhixiang/vllm-ascend/csrc/third_party/makeself-release-2.5.0-patch1.tar.gz")|g' "${MAKESELF_CMAKE}"
    echo "  [OK] makeself-fetch.cmake 已修改为本地路径"
else
    echo "  [SKIP] makeself-fetch.cmake 未找到 gitcode.com URL"
fi

# -------------------------------------------------------
# 第六步: 确认还有没有遗漏的 gitcode.com URL
# -------------------------------------------------------
echo ""
echo "===== 第六步: 检查是否还有遗漏的 gitcode.com 引用 ====="
REMAINING=$(grep -rn "gitcode.com" "${CMAKE_DIR}/" --include="*.cmake" 2>/dev/null || true)
if [ -n "${REMAINING}" ]; then
    echo "  [WARN] 仍有文件引用 gitcode.com:"
    echo "${REMAINING}"
    echo "  请手动处理以上文件！"
else
    echo "  [OK] 所有 cmake 文件已不再引用 gitcode.com"
fi

# -------------------------------------------------------
# 第七步: 清理构建缓存
# -------------------------------------------------------
echo ""
echo "===== 第七步: 清理构建缓存 ====="

if [ -d "${BUILD_DIR}" ]; then
    rm -rf "${BUILD_DIR}"
    echo "  [OK] 已删除 csrc/build/"
else
    echo "  [SKIP] csrc/build/ 不存在"
fi

find "${THIRD_PARTY_DIR}" -name "*.stamp" -delete 2>/dev/null || true
find "${THIRD_PARTY_DIR}" -path "*/stamp/*" -type f -delete 2>/dev/null || true
echo "  [OK] 已清理 stamp 文件"

# -------------------------------------------------------
# 第八步: 重新构建
# -------------------------------------------------------
echo ""
echo "===== 第八步: 重新构建 vllm-ascend ====="
echo ""
echo "即将执行: pip install -e . --no-build-isolation -i https://pypi.antfin-inc.com/simple/"
echo "日志保存到: ${BASE_DIR}/build.log"
echo ""
read -p "确认执行构建？按回车继续，Ctrl+C 取消..."

cd "${BASE_DIR}"
pip install -e . --no-build-isolation -i https://pypi.antfin-inc.com/simple/ 2>&1 | tee "${BASE_DIR}/build.log"

echo ""
echo "===== 完成 ====="
echo "构建日志: ${BASE_DIR}/build.log"