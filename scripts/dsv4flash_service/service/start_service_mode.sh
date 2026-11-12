#!/bin/bash
# start_service_mode.sh：按原服务参数起服；模式及参数见下面的配置。
# PROFILE=1 保留官方 profiling 接口；SLOT_STATS 由后续计划诊断清理处理。
# MoE逐rank负载采集、BALANCE累计账本和相应目录/报告入口已移除。
# 旧 MOE_PAR/BALANCE 及运行时采集变量不再开启这些功能。
# 保留 OUT_DIR、LOG_FILE、profiling/ 和正常服务日志；路径必须为绝对路径。
# 例：PROFILE=1 OUT_DIR=/abs/run LOG_FILE=/abs/run.log bash start_service_mode.sh slot
# 服务日志及 profiling 存在时拒绝覆盖；其他服务参数、设备选择和模式行为保持原样。

set -x

# 启动前先记下外部已经存在的相关环境变量，稍后在「启动核对」横幅里列出来。
# 只认真正的环境变量（printenv），shell 里没 export 的同名变量不算。
PRE_ENV=""
for v in MOONEP MOONEP_SLOT_MODE MOONEP_SLOT_STATS MOONEP_SHMEM_BOOTSTRAP MOONEP_SHMEM_IPPORT \
         MOONEP_TRANSPORT_TYPE; do
  val=$(printenv "$v") || continue
  PRE_ENV="$PRE_ENV $v=$val"
done

# ---------------------------------------------------------- 模式变量 -- #
# MoonEP 一共四个开关，三个在环境变量（库侧）、一个在命令行（引擎侧）：
#   MOONEP=1            让 ascend_moonep 库加载并初始化 shmem；off 腿不设，整套 MoonEP 不参与
#   MOONEP_SLOT_MODE=1  slot 模式（按 slot 镜像 expert 行并预取）；=0 即 owner 模式
#   MOONEP_SLOT_STATS=1 只对 slot 有意义：worker 退出时打印 [moonep-stats] 计数，M7 判据靠它
#                       （注意：本脚本的停服方式是 force-kill，worker 的退出钩子不执行，
#                        服务化这条腿**本来就拿不到这一行**；M7 要在 bench/离线跑法里看，见 docs）。
#                       采集点在 dispatch 里每层一次 .item()（host 同步），是**只有 MoonEP 腿付**
#                       的开销，而服务化这条腿既拿不到它的输出、又把它算进了性能——所以服务腿
#                       默认关掉（SLOT_STATS 默认 0，零损失）；要 M7 那个数就留在 bench/离线路径上开着。
#   --additional-config {"enable_moonep": ...}  引擎侧开关：MoE 层是否改用 MoonEP 通信
# 环境变量和引擎开关两个都要对：只设 MOONEP=1 而不给 enable_moonep，引擎走的还是普通 EP。
# off 腿要把三个变量都清掉——shell 里残留的值会静默让下一条腿跑错配置。
#
SLOT_STATS="${SLOT_STATS:-0}"   # 只对 slot 有意义；服务腿打不出它的输出（退出时才打），默认关
SLOT_STATS_DESC="不适用（off/owner 没有 slot 计数）"
MODE="${1:-owner}"
case "$MODE" in
  off)
    unset MOONEP MOONEP_SLOT_MODE MOONEP_SLOT_STATS
    ENABLE_MOONEP=false
    MODE_DESC="非 MoonEP：标准 MC2 / all-to-all EP（对照组）"
    ;;
  owner)
    export MOONEP=1 MOONEP_SLOT_MODE=0
    unset MOONEP_SLOT_STATS
    ENABLE_MOONEP=true
    MODE_DESC="MoonEP owner：专家行常驻本 rank"
    ;;
  slot)
    export MOONEP=1 MOONEP_SLOT_MODE=1
    if [ "$SLOT_STATS" = 1 ]; then
      export MOONEP_SLOT_STATS=1
      SLOT_STATS_DESC="on（每层一次 host 同步，只 MoonEP 腿付；SLOT_STATS=1 显式指定）"
    else
      unset MOONEP_SLOT_STATS
      SLOT_STATS_DESC="off（默认：服务腿本来就打不出这一行，开了只付每层同步、零收益）"
    fi
    ENABLE_MOONEP=true
    MODE_DESC="MoonEP slot：按 slot 镜像 expert 行并预取（M7 计数靠 MOONEP_SLOT_STATS）"
    ;;
  *)
    echo "用法: bash $0 {off|owner|slot}" >&2
    exit 2
    ;;
esac
# 日志与采集产物都落在本脚本自己所在目录（绝对路径：worker 那边 cwd 若变也不会跑偏）
HERE=$(cd "$(dirname "$0")" && pwd)
# 工作目录必须换到**没有裸 vllm/ 目录**的地方再起服务：CANN 的 set_env.sh 会在 PYTHONPATH 末尾留一个
# 空条目，而空条目 = 当前目录，于是 cwd 会进 sys.path。若从 $WS（vLLM 源码仓库的上一级）启动，
# $WS/vllm/ 这个目录（里面才是 vllm 包）会被当成 namespace package 抢在 editable finder 之前解析，
# import vllm 拿到的是没有 __file__ 的空壳，服务直接死在启动阶段：
#   ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
# $HERE（scripts/）下没有 vllm/，cd 过去即可绕开；真正的根因是那个空 PYTHONPATH 条目。
cd "$HERE" || exit 1
# -------------------------------------------------- 产物路径（可前缀覆盖） -- #
# 默认值就是文件头列的那三条；每个都能用命令前缀覆盖，而且**必须是绝对路径**——这些路径会交给
# worker 与 profiler，那里的 cwd 不可靠（脚本自己 cd 过、worker 是 spawn 出来的），相对路径
# 落在哪个目录全凭运气，所以这里直接拒绝。空串等于没给（用默认值）。
# 先记下「外部给过哪些」（前缀与 export 分不出来），启动横幅会标出来。
OUT_DIR_OVERRIDE=$OUT_DIR; LOG_FILE_OVERRIDE=$LOG_FILE
OUT_DIR=${OUT_DIR:-$HERE/serve_$MODE}
LEG_NAME=$(basename "${OUT_DIR%/}")
LOG_FILE=${LOG_FILE:-$HERE/start_service_$MODE.log}   # 服务日志（tee 的落点）
PROFILE_DIR=$OUT_DIR/profiling   # torch trace 目录（GB 级、只在 stop_profile 时写一次，留 NFS）：跟着根目录走
for p in "$OUT_DIR" "$LOG_FILE"; do
  case "$p" in
    /*) ;;
    *) echo "路径必须是绝对路径（要给 worker/profiler 用，那边 cwd 不可靠）：$p" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------- 产物存在性检查 -- #
# 一律不覆盖：命中就打印是哪一条、然后退出。要重跑就把它们删掉，或改上面的路径/换模式。
if [ -e "$LOG_FILE" ]; then
  echo "[服务日志] 已存在，不覆盖：$LOG_FILE —— 删掉它，或用前缀指到别处：" >&2
  echo "    LOG_FILE=/abs/start_service_$MODE.log bash $0 $MODE" >&2
  exit 2
fi
if [ "$PROFILE" = 1 ] && [ -e "$PROFILE_DIR" ]; then
  echo "[profiling] 目录已存在，不覆盖：$PROFILE_DIR（= OUT_DIR/profiling）—— 删掉它，或把整条腿挪走：" >&2
  echo "    OUT_DIR=/abs bash $0 $MODE" >&2
  exit 2
fi

# ---------------------------------------------------------- 共用环境 -- #
unset ftp_proxy https_proxy http_proxy
source /usr/local/Ascend/cann/set_env.sh


WS=/home/x00893896/vllm_ascend_moonep_integration # vllm、vllm-ascend、ascend-moonep 仓在该目录下
export HCCL_IF_IP=90.90.93.34          # 本机 IP，ifconfig 查
export GLOO_SOCKET_IFNAME=ens6f1np1           # 下面三个都是上面这个 IP 对应的网卡
export TP_SOCKET_IFNAME=ens6f1np1
export HCCL_SOCKET_IFNAME=ens6f1np1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024                # HCCL 缓冲大小；首次跨 engine 集合通信需要这个量级
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"   # DP+EP 下必须关，否则会挂
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7            # 服务可见的卡；DP8 要 8 张
export PYTHONPATH=$WS/vllm-ascend:$PYTHONPATH
#                ^ ascend_moonep 不再 vendored：由 ascend-moonep 仓的 pip install -e 提供，
#                  _C*.so/lib 就在其包目录内（重编后天然同版）
export ASCEND_CUSTOM_OPP_PATH=$WS/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # worker 用 spawn；fork 子进程走 os._exit()，退出钩子不执行
unset MOONEP_SHMEM_IPPORT                   # 清掉可能残留的上一次设置
export MOONEP_SHMEM_BOOTSTRAP=ipport        # MoonEP shmem 引导方式必须用 ipport(config-store)；
                                            # uid 走 HCCL 广播会死锁
export MOONEP_TRANSPORT_TYPE=mte            # mte（默认）或 udma；切换传输类型直接修改此处
export ASCEND_MOONEP_BUILD_DIR=$WS/ascend-moonep/ascend_moonep   # 编好的 ascend_moonep（含 _C.cpython-*.so）
export LD_LIBRARY_PATH=$ASCEND_MOONEP_BUILD_DIR/lib:$LD_LIBRARY_PATH   # libshmem.so 在这个 lib/ 下

# --------------------------------------------------------- profiling -- #
# 默认关：PROFILE=1 bash start_service_mode.sh <mode> 才开（命令前缀，别 export——见文件头）。
# 这里只加命令行参数，不带任何 env 开关——这套 vLLM 里 profiling 只能由 --profiler-config
# 打开，所以外部 export 什么都不会让它被打开。
# 不开就没有 /start_profile、/stop_profile 两个路由；开了之后服务照常跑，采集是两次 curl，
# 一次 HTTP 会广播到全部 8 个 DP engine core，不用逐 rank 操作：
#   curl -X POST http://127.0.0.1:8444/start_profile
#   ... 灌一段流量 ...
#   curl -X POST http://127.0.0.1:8444/stop_profile    # 各 rank 同步导出 trace 到 $PROFILE_DIR
# 查看：tensorboard --logdir $PROFILE_DIR（trace 名自带 rank 后缀，8 rank 共用一个目录安全）
# 两条注意：窗口是这段真实流量（各 rank 步数不齐、含空转），和离线 --profile-only 的锁步
# 窗口不是一个口径；导出期间该 rank 不跑 step，所以 start/stop 之间别让请求断流太久。
# with_stack 显式关掉：NPU 上它等价于 with_modules，开着的开销很大；目录必须是绝对路径。
PROFILE_ARGS=()
if [ "$PROFILE" = 1 ]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_ARGS=(--profiler-config.profiler=torch
                --profiler-config.torch_profiler_dir="$PROFILE_DIR"
                --profiler-config.torch_profiler_with_stack=false)
fi

# ------------------------------------------------------------ 起服务 -- #
# 服务参数集中在这里：下面命令行和「启动核对」横幅用的是同一批变量，改一处就够。
MODEL=/mnt/data/models/DeepSeek-V4-Flash   # 权重目录
SERVED_NAME=test_auto                           # 客户端要填的 model 名
HOST=0.0.0.0                                     # 监听地址（0.0.0.0 = 本机所有网卡）
PORT=8444                                        # 服务端口；curl 采集接口也是这个
API_SERVER_COUNT=1                               # 单前端 + DPCoordinator 做 DP 间负载均衡与 wave 对齐
DP=8                                             # DSV4-Flash 受 DSA 头数约束只能 TP=1，靠 DP 铺开
TP=1
MAX_NUM_SEQS=16                                  # 并发上限；与离线负载腿同一取点
BATCHED_TOKENS=16384                              # 每步 token 数上限（max_num_batched_tokens），同时是 MoonEP dispatch 的容量 S
MAX_MODEL_LEN=40960                               # 40k:40960
GPU_UTIL=0.80                                    # slot 模式的 non-torch 占用约 20 GiB，别直接上 0.90
# 命令行里写死的三个开关（不做成变量）：
#   --enable-expert-parallel  EP 必须开（off 腿也是 EP，只是不用 MoonEP）；EP size = dp*tp
#   --enforce-eager           MoonEP 硬要求：MoE 层不能进 ACL graph
#   --additional-config       引擎侧 MoonEP 开关，见上面「模式变量」
#   --profiler-config         由 $PROFILE_ARGS 追加，只在 PROFILE=1 时出现（见「profiling」）

# -------------------------------------------------------- 启动核对 -- #
# 起服务之前把这次的有效配置打出来：模式、MoonEP 与采集开关、服务参数、可见卡、产物路径，
# 以及「启动前 shell 里已经存在的相关变量」「哪些路径/标签是外部给的」——旧环境变量最容易
# 悄悄改行为，核对这几行即可。
# 用 cat <<EOF 而不是一堆 echo：set -x 只会多打一行 "+ cat"，横幅本身保持干净。
# 横幅同时 tee 进 LOG_FILE（它是这条腿的配置记录，不该只活在 /tmp 的 .out 里）：日志的创建点
# 就在这里，后面 vllm 那一段因此也用 tee -a 追加，别把它截掉。
if [ -n "$PRE_ENV" ]; then
  PRE_ENV_LINE=" ⚠ 启动前 shell 里已经有这些环境变量：$PRE_ENV（本脚本已按 mode 覆盖/清除，当前生效值看上面几行；若不是你要的，先 unset 再重跑）"
else
  PRE_ENV_LINE=" 环境干净：启动前没有上述 MoonEP 运行配置残留"
fi
# 外部给了哪些路径（前缀与 export 分不出来）：标出来，没标的就都是默认值
PATH_OVERRIDE=""
[ -n "$OUT_DIR_OVERRIDE" ] && PATH_OVERRIDE="$PATH_OVERRIDE OUT_DIR"
[ -n "$LOG_FILE_OVERRIDE" ] && PATH_OVERRIDE="$PATH_OVERRIDE LOG_FILE"
if [ -n "$PATH_OVERRIDE" ]; then
  PATH_OVERRIDE_LINE=" ⚠ 这些路径是外部给的（命令前缀或 export），已按它们生效：$PATH_OVERRIDE —— export 过的会留到你自己 unset"
else
  PATH_OVERRIDE_LINE=" 路径都用默认值（要换就用前缀：OUT_DIR=… LOG_FILE=…）"
fi
cat <<EOF | tee -a "$LOG_FILE"
════════════════════════════════════ 启动核对 ════════════════════════════════════
 模式        mode=$MODE —— $MODE_DESC
 MoonEP      enable_moonep=$ENABLE_MOONEP   MOONEP=${MOONEP:-未设}   MOONEP_SLOT_MODE=${MOONEP_SLOT_MODE:-未设}   MOONEP_SLOT_STATS=${MOONEP_SLOT_STATS:-未设}
 引导/传输   MOONEP_SHMEM_BOOTSTRAP=$MOONEP_SHMEM_BOOTSTRAP   MOONEP_SHMEM_IPPORT=${MOONEP_SHMEM_IPPORT:-未设}   MOONEP_TRANSPORT_TYPE=${MOONEP_TRANSPORT_TYPE:-未设}
 profiling   PROFILE=${PROFILE:-0} → $PROFILE_DIR（= OUT_DIR/profiling）
             SLOT_STATS=$SLOT_STATS —— $SLOT_STATS_DESC
 服务        http://$HOST:$PORT   api-server-count=$API_SERVER_COUNT   dp=$DP tp=$TP ep=on   served-model-name=$SERVED_NAME
 模型        $MODEL
 负载口径    max-num-seqs=$MAX_NUM_SEQS   max-num-batched-tokens=$BATCHED_TOKENS   max-model-len=$MAX_MODEL_LEN   gpu-mem-util=$GPU_UTIL   chunked-prefill=on
 卡/关键env  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES   PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF   HCCL_BUFFSIZE=$HCCL_BUFFSIZE
 日志        $LOG_FILE
 产物        $OUT_DIR/（profiling/ 只在 PROFILE=1 时有内容）
$PRE_ENV_LINE
$PATH_OVERRIDE_LINE
════════════════════════════════════════════════════════════════════════════════
EOF

vllm serve "$MODEL" \
  --host "$HOST" \
  --port $PORT \
  --api-server-count $API_SERVER_COUNT \
  --data-parallel-size $DP \
  --tensor-parallel-size $TP \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-chunked-prefill \
  --additional-config "{\"enable_moonep\": $ENABLE_MOONEP}" \
  "${PROFILE_ARGS[@]}" \
  --served-model-name "$SERVED_NAME" \
  --seed 1024 \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --gpu-memory-utilization $GPU_UTIL \
  2>&1 | tee -a "$LOG_FILE"      # -a：横幅已经写在里面了（见「启动核对」）

# ------------------------------------------------------------- 收尾 -- #
set +x
{
  echo
  echo "──────── 这条腿结束：$LEG_NAME（mode=$MODE）────────"
  echo " 服务日志   $LOG_FILE"
  [ "$PROFILE" = 1 ] && echo " trace      $PROFILE_DIR/    （看：tensorboard --logdir $PROFILE_DIR）"
} 2>&1 | tee -a "$LOG_FILE"
