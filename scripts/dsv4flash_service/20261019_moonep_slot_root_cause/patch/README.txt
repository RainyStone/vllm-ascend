MoonEP tokenizer patch 使用说明



补丁怎么用：检查 → 应用 → 启用 → 验证 → 回退


补丁和使用说明集中在报告根目录的直接子目录 patch/：tokenizer_window.patch · 独立使用说明（HTML） · 使用说明（纯文本）。


当前容器已经有补丁。2026-10-19 只读复查确认：vLLM HEAD 为 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665，两处源码修改和新增测试均存在，git apply --reverse --check 通过。此环境直接从第 3 步配置和重启；下面完整流程供其它同版本环境复用。




这是 vLLM 仓库补丁，应用目标为集成工程的 vllm/，不是保存报告的 vllm-ascend/。补丁以本报告记录的 commit 为基准；其它版本必须先检查，不能仅凭参数同名假定兼容。执行前保存自己在这三个目标文件上的其它改动。


1. 定位仓库和补丁，检查当前状态


# 在目标容器内执行；以下路径对应本报告的实际保存位置。
WORKSPACE=/home/x00893896/vllm_ascend_moonep_integration
VLLM_REPO="$WORKSPACE/vllm"
PATCH_FILE="$WORKSPACE/vllm-ascend/scripts/dsv4flash_service/20261019_moonep_slot_root_cause/patch/tokenizer_window.patch"

git -C "$VLLM_REPO" rev-parse HEAD
git -C "$VLLM_REPO" status --short
git -C "$VLLM_REPO" apply --stat "$PATCH_FILE"

# 只检查，不改文件：
if git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "补丁已存在：跳过应用，直接进行环境配置。"
elif git -C "$VLLM_REPO" apply --check "$PATCH_FILE"; then
    echo "补丁尚未应用且检查通过：可以执行下一步应用命令。"
else
    echo "版本或工作区内容不匹配：先人工处理差异，不要强制应用。"
fi


2. 只对尚未应用的工作区打补丁


# 仅在上一段显示“尚未应用且检查通过”时执行：
git -C "$VLLM_REPO" apply --check "$PATCH_FILE" &&
git -C "$VLLM_REPO" apply "$PATCH_FILE"

# 应用后核对；反向检查成功说明整份补丁的修改已在工作区。
git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE"
git -C "$VLLM_REPO" diff --check
git -C "$VLLM_REPO" diff --stat
git -C "$VLLM_REPO" status --short -- \
  vllm/envs.py vllm/utils/async_utils.py tests/utils_/test_async_tokenizer_window.py


新增测试是未跟踪文件，可能不会出现在普通 git diff --stat 中，因此同时检查 git status。不使用 git reset --hard 或覆盖整个源码文件来处理冲突。


3. 设置 20ms，并确认服务会导入这份源码


# 本次实验实际使用的 Python；其它部署改成启动 API 服务的那个解释器。
SERVICE_PYTHON="$WORKSPACE/scripts/20261017_scheduler_alignment/.venv/bin/python"
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$VLLM_REPO:$WORKSPACE/vllm-ascend:${PYTHONPATH:-}"
export VLLM_TOKENIZER_BATCH_WAIT_MS=20

# 从中性目录检查实际导入位置，避免同名目录影响解析。
cd /tmp
"$SERVICE_PYTHON" - <<'PY'
import inspect
from vllm import envs
from vllm.utils import async_utils

print("loaded:", async_utils.__file__)
print("window_ms:", envs.VLLM_TOKENIZER_BATCH_WAIT_MS)
assert envs.VLLM_TOKENIZER_BATCH_WAIT_MS == 20.0
source = inspect.getsource(async_utils.AsyncMicrobatchTokenizer.__init__)
assert "VLLM_TOKENIZER_BATCH_WAIT_MS" in source
print("patched tokenizer configuration: OK")
PY

# 然后使用原服务启动命令重启 API，让新进程继承以上环境。


预期 loaded 指向 .../vllm/vllm/utils/async_utils.py，window_ms 为 20.0。本次通过 PYTHONPATH 使用源码，Python 补丁无需重新编译 NPU 算子或重装 Python。若服务采用其它虚拟环境或安装包，应以其实际导入路径为准。


设置环境变量之后必须重启 API 服务。上面检查的是新 Python 进程，不能据此认定已经运行的 API 已更新；在另一个 shell 中 export 不会改变旧进程。把变量设置在真正启动 API 的脚本 / 进程环境里，保持其余 DP、KV、chunk 和 slot 参数一致。若调用方显式传入 batch_wait_timeout_s，该显式值仍优先。




4. 验证补丁行为，再验证调度相位


# 使用上一步已设置的 SERVICE_PYTHON、PYTHONPATH、CANN 环境：
cd "$WORKSPACE/scripts/20261017_scheduler_alignment"
ALIGNMENT_DISABLE_EXTRA=1 "$SERVICE_PYTHON" -m pytest \
  --confcutdir="$VLLM_REPO/tests/utils_" \
  "$VLLM_REPO/tests/utils_/test_async_tokenizer_window.py" \
  "$VLLM_REPO/tests/utils_/test_async_utils.py" -q


归档环境已有相关测试依赖，原验证结果为 8 passed。性能回归还需使用正文的 n32/c16、L16K/B8K 配置，检查请求 ID、每轮 token 与 dummy：总真实 token=524288，首轮偏移 [0,0,0,0]，每 rank 8 请求，16 全真实 + 7 全 dummy。单测通过本身不等于任意业务流量一定对齐。


5. 回退窗口，或移除补丁


# 优先做配置回退：恢复默认行为，保留补丁的配置能力。
export VLLM_TOKENIZER_BATCH_WAIT_MS=2
# 或：unset VLLM_TOKENIZER_BATCH_WAIT_MS
# 再使用相同服务启动流程重启 API。

# 如确实要移除源代码补丁，先确认没有后续重叠修改，再执行：
git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE" &&
git -C "$VLLM_REPO" apply --reverse "$PATCH_FILE"


反向检查失败时，应人工处理重叠修改；不要强制覆盖其它改动。移除代码后同样需要重启 API。仅把窗口回退为 2ms 就能恢复原默认聚合行为，通常足够进行 A/B 对照。


应用、反向检查和回退已在原始基准源码的隔离临时目录中验证通过，并确认恢复原文件；没有对当前服务源码再次打补丁。验证记录：patch_usage_validation.json。




