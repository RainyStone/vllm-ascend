#!/bin/bash
# start_service_mode.sh —— 一个脚本拉起三种 MoE 通信模式的服务
#
#   bash start_service_mode.sh off     # 非 MoonEP（标准 MC2/all-to-all EP）
#   bash start_service_mode.sh owner   # MoonEP owner
#   bash start_service_mode.sh slot    # MoonEP slot
#
# 三条腿的差异只有「模式变量」那一段（3 个 MOONEP* 环境变量 + enable_moonep），
# 其余环境变量与命令行参数全部共用。日志按模式分开：start_service_<mode>.log。
#
# 五个可选开关，**全部默认关闭**：不加前缀 = 不采集、不加账、服务行为与以前完全一样。
# 要采什么就显式给什么，都用「命令前缀」：
#   PROFILE=1     bash start_service_mode.sh slot   # 开 profiling 接口，见「profiling」
#   MOE_PAR=1     bash start_service_mode.sh slot   # 采 MoE 负载（PAR），见「MoE 负载采集」
#   BALANCE=1     bash start_service_mode.sh slot   # 加账目行（[moe-par-ledger] / [moonep-balance]）
#   SLOT_STATS=1  bash start_service_mode.sh slot   # slot 的每层 slot 计数（默认关，见「模式变量」）
#   SIM_ROUTE=... bash start_service_mode.sh slot   # 注入指定的专家选择分布，见「仿真路由」
#   MOE_PAR=1 PROFILE=1 BALANCE=1 bash start_service_mode.sh slot
#   MOE_PAR=1 SIM_ROUTE=@/tmp/sim_skew.txt bash start_service_mode.sh off   # 倾斜 + 采负载对账
# 坑：bash 分不出「命令前缀」和「export」——两种写法在脚本里看到的环境一模一样，所以
# export 过的 MOE_PAR=1 会一直生效，直到你自己 unset。别 export，用完即弃；
# 要长期改就直接改本脚本（脚本不读外部同名变量来开关）。下面的路径变量（OUT_DIR 等）同理：
# 前缀和 export 分不出来，所以「外部到底给没给」由启动横幅标出来。
#
# BALANCE=1 打开后每条腿都会拿到账目行：
#   有 MOE_PAR=1 时 → [moe-par-ledger rN]（记录器自己落盘的三本账：rows / pads / compute，
#                     和 dump 的收尾行同源、按定义相等）
#   owner/slot      → [moonep-balance rN]（MoonEP 侧另一本：rows_local / pad_rows，比 dump 早一步）
# 默认关的原因：MoonEP 那本每层要一次 host 同步（**只有 MoonEP 腿付**），只做性能/只采 profiling 时
# 不该把它混进 trace；而 off 腿那本 [moe-par-ledger] 要 MOE_PAR=1 才有（没有 flush 就没有这行）。
# 做负载分析时它和 dump 是配套的账，那时显式加 BALANCE=1。脚本顺带把
# MOONEP_BALANCE_STATS_EVERY 设成一步的层数，于是每 rank 每步打一行累计账（原因见
# 「每 rank 负载计数」那一段：退出钩子在这条腿上不执行）。
# off 腿原来那本 [baseline-balance]（在 token_dispatcher 里数 MC2/all2all 的 recv_rows）已经删掉：
# 那里读的是发送/接收侧的几个量，和 dump 对不上（cumsum 被当成了计数、被复用的字段各说各话），
# 留着一本对不上的账只会误导。
#
# 产物默认落在本脚本所在目录的 serve_<mode>/ 下（与 start_service_<mode>.log 并排）：
#   start_service_<mode>.log                                服务日志（横幅 + vllm 服务输出 + 收尾）
#   serve_<mode>/moe_par/moe_par_serve_<mode>_rank<r>.log   负载 dump，8 rank 各一份（与 profiling/ 同级）
#   serve_<mode>/profiling/                                 torch trace，8 rank 共用
#   serve_<mode>/serve_<mode>_report.html                   负载报告，名字=$(腿目录名)_report.html
#                                                           （横幅与收尾都会打出现成的命令）
# 三个位置能用命令前缀换（写法与开关相同，**必须绝对路径**），不用改脚本；含义是：
#   OUT_DIR=/abs       本次运行（这条腿）的产物根目录，默认 $HERE/serve_<mode>：
#                      trace 在它下面的 profiling/，dump 在 moe_par/，报告也写回它
#   DUMP_DIR=/abs      负载 dump 目录（默认 $OUT_DIR/moe_par），analyzer 的输入；
#                      想让它别占 /mnt/share 的 io 就指到 /tmp/…（实测每步省约 1 ms，
#                      代价是 /tmp 重启即丢）——见下面「负载 dump 的位置」那段
#   LOG_FILE=/abs      服务日志（默认 $HERE/start_service_<mode>.log；不跟着 OUT_DIR 走）
#   例：OUT_DIR=$PWD/g1_off LOG_FILE=$PWD/g1_off.log bash start_service_mode.sh off
# 停服后脚本会打一段收尾：产物清单 + 出报告的命令（路径就是本次的，可直接拷走执行）。
# 谁写进哪个文件：LOG_FILE 收三段——「启动核对横幅」「vllm serve 的全部输出」「收尾」，都是 tee 进去的。
# 脚本自己的其余输出（set -x 的 + 行、报错）留在启动器的 stdout/stderr 上，不进 LOG_FILE。
#
# 起服务的写法（前缀 + 重定向）：
#   cd <本脚本所在目录>
#   nohup env MOE_PAR=1 PROFILE=1 OUT_DIR=$PWD/g2_slot LOG_FILE=$PWD/g2_slot.log \
#     bash start_service_mode.sh slot > /tmp/leg_g2_slot.out 2>&1 &
# 末尾三个符号就是 shell 的常规写法：`>` 把这条命令的 stdout 写进 /tmp/leg_g2_slot.out（覆盖式），
# `2>&1` 让 stderr 也去同一个地方（set -x 的 + 行、报错都在那），`&` 放后台、命令立刻返回；
# nohup 只是让它在终端关掉/断开后不被 SIGHUP 打断。不用 nohup 也行：终端一直开着就直接 `… &`，
# 要彻底脱离会话用 `setsid env … &`；前台跑（去掉 & 和重定向）则这个终端会被占住。
# 本脚本生成的日志和目录一律不覆盖：启动时逐个检查，已存在就打印是哪一条后退出（exit 2），
# 要重跑就把它们删掉，或用上面的前缀把它挪到别处。
#
# 起服务前会打一块「启动核对」横幅：模式、MoonEP 四个开关、采集开关、服务与负载参数、
# 可见卡、日志/产物路径，外加「启动前 shell 里已经存在的相关变量」「哪些路径是外部给的」——
# 环境变量的旧值最容易悄悄改行为，核对这块即可。
#
# 停服时只杀自己起的进程（vllm serve 主进程 + 它拉起的 EngineCore_DP*/Worker_DP*）；
# slot 的 M7 计数（[moonep-stats] ... total_slot_rows=）是 worker 退出时才打印的，
# 停服后再到日志里 grep。

set -x

# 启动前先记下外部已经存在的相关环境变量，稍后在「启动核对」横幅里列出来。
# 只认真正的环境变量（printenv），shell 里没 export 的同名变量不算。
PRE_ENV=""
for v in MOONEP MOONEP_SLOT_MODE MOONEP_SLOT_STATS MOONEP_SHMEM_BOOTSTRAP MOONEP_SHMEM_IPPORT \
         MOONEP_DISABLE_UDMA MOONEP_BALANCE_STATS MOONEP_BALANCE_STATS_EVERY \
         VLLM_ASCEND_MOE_PAR_STATS \
         VLLM_ASCEND_MOE_PAR_STATS_DIR \
         VLLM_ASCEND_MOE_PAR_STATS_TAG VLLM_ASCEND_MOE_PAR_FLUSH_EVERY; do
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
# MOONEP_BALANCE_STATS=1（BALANCE=1）时：有 MOE_PAR=1 的腿打记录器自己的 [moe-par-ledger rN]
# （rows/pads/compute），owner/slot 另有 MoonEP 侧的 [moonep-balance rN]（rows_local/pad_rows）。
# 两者都在回答「本 rank 真算了多少条目」：前者和 dump 按定义相等，后者覆盖到在飞的那一步、
# 比 dump 早一步。**默认关**（MoonEP 那本每层一次 host 同步，性能腿不该付；负载腿要账目就加
# BALANCE=1）。
#
# 配一个 MOONEP_BALANCE_STATS_EVERY=<MoE 层数>：计数器本身只在进程退出时打一行，而服务化的
# 停服路径是 force-kill（API server 用 shutdown_timeout=0 收 EngineCore，worker 跟着被杀），
# worker 的退出钩子根本不执行——不配这个的话服务日志里一行都不会有。配成一步的层数，就变成
# 「一步一行」的累计账，最后一行覆盖整段负载，和 dump 的 compute_pairs_total 直接对账。
BALANCE="${BALANCE:-0}"
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
# ------------------------------------------------------ 每 rank 负载计数 -- #
# 三条腿的账都从记录器来：moe_par_stats 每落一次盘就打一行 [moe-par-ledger rN]，
# 里面的 rows / pads / compute 就是 dump 里已经写下的三本账（rows + pads = compute，
# 少掉的部分会另打 dropped=），所以它和 dump 按定义相等，可以逐 rank 直接对。
# owner/slot 另有 MoonEP 侧的一行 [moonep-balance rN]（moonep_comm 在 plan 之后统计）：
# 它覆盖到「正在飞的那一步」，因此比 dump 早一步，最后一行与 dump 的差额应当只是一步的量。
# 开销：MoonEP 那本每层一次 host 同步（把行数读回主机）——**只有 MoonEP 腿付**，
# 所以它默认关；要性能/只采 profiling 时保持默认（不指定 BALANCE），做负载/搬迁分析时
# 显式 BALANCE=1（读的是同一份计数，不影响结论）。
# 产物：每 rank 每步一行累计账，grep：
#   grep -E '\[(moe-par-ledger|moonep-balance)' start_service_<mode>.log | tail -8
# 对账：ledger 的 steps / compute 应当等于 dump 的 compute_rows_total / compute_pairs_total。
if [ "$BALANCE" = 1 ]; then
  export MOONEP_BALANCE_STATS=1
  export MOONEP_BALANCE_STATS_EVERY=43   # 43 = 本模型的 MoE 层数 = 一步一行；0 = 只在退出时打
  BALANCE_DESC="on（BALANCE=1；[moe-par-ledger rN]（记录器，与 dump 同源）$([ "$MODE" = off ] || echo ' + [moonep-balance rN]（MoonEP，每层一次 host 同步）')；每 ${MOONEP_BALANCE_STATS_EVERY} 层打一行）"
else
  unset MOONEP_BALANCE_STATS MOONEP_BALANCE_STATS_EVERY
  BALANCE_DESC="off（默认；要账目行就显式 BALANCE=1）"
fi

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
OUT_DIR_OVERRIDE=$OUT_DIR; DUMP_DIR_OVERRIDE=$DUMP_DIR; LOG_FILE_OVERRIDE=$LOG_FILE
# OUT_DIR：本次运行（这条腿）的产物根目录——trace 在它的 profiling/、dump 在 moe_par/、报告也写回它。
# 默认目录名沿用 serve_<mode>（跟老归档一致）。LEG_NAME 取目录名，报告里就这么称呼这条腿。
OUT_DIR=${OUT_DIR:-$HERE/serve_$MODE}
LEG_NAME=$(basename "${OUT_DIR%/}")
# 负载 dump 的位置：默认 $OUT_DIR/moe_par（和 profiling/ 同级），analyzer 直接读这里，不用再拷来拷去。
# 位置换来的是 io：记录器每步都要 seek+truncate 重写尾部汇总行，而在 /mnt/share 上那一次 truncate
# 是一次同步 NFS 往返——本机实测 tmpfs 0.01 ms/步、NFS 1.0～1.3 ms/步（写+flush 两边都很便宜，
# 6 KiB 和 70 KiB 一步差不多）。这 1 ms 就落在被测的那一步里，所以要「零 io 进窗口」就把 dump
# 指到本地盘（/tmp 是内存盘，重启会丢，别把它当仓库）：
#   DUMP_DIR=/tmp/moe_par_$MODE bash …（用哪种横幅会标出来）
DUMP_DIR=${DUMP_DIR:-$OUT_DIR/moe_par}   # 8 个 rank 的文件都落在这一个目录里（记录器自己建）
LOG_FILE=${LOG_FILE:-$HERE/start_service_$MODE.log}   # 服务日志（tee 的落点）
PROFILE_DIR=$OUT_DIR/profiling   # torch trace 目录（GB 级、只在 stop_profile 时写一次，留 NFS）：跟着根目录走
for p in "$OUT_DIR" "$DUMP_DIR" "$LOG_FILE"; do
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
if [ "$MOE_PAR" = 1 ] && [ -e "$DUMP_DIR" ]; then
  echo "[MoE 负载采集] 目录已存在，不覆盖：$DUMP_DIR（一个目录只放一次运行的 dump）—— 删掉它，或换位置：" >&2
  echo "    OUT_DIR=/abs bash $0 $MODE   （只挪 dump 的话：DUMP_DIR=/tmp/moe_par_$MODE）" >&2
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
export MOONEP_DISABLE_UDMA=1                # 关 UDMA，走 MTE 内核；要用 URMA/UDMA 时去掉这行
export ASCEND_MOONEP_BUILD_DIR=$WS/ascend-moonep/ascend_moonep   # 编好的 ascend_moonep（含 _C.cpython-*.so）
export LD_LIBRARY_PATH=$ASCEND_MOONEP_BUILD_DIR/lib:$LD_LIBRARY_PATH   # libshmem.so 在这个 lib/ 下

# ------------------------------------------------------ MoE 负载采集 -- #
# 默认关：MOE_PAR=1 bash start_service_mode.sh <mode> 才采（命令前缀，别 export——见文件头）。
# 开的时候每个 worker 把逐 (step, layer) 的 MoE 路由负载写成一份本 rank 的 dump，
# 8 个 rank 的文件都落在同一个目录：$DUMP_DIR/
# 停服后离线聚合成报告（analyzer 读这个目录，腿名取 OUT_DIR 的目录名，报告名再带上这
# 个腿名，多份报告放一起也不打架）：
#   python <analyze_moe_par.py 路径（该聚合工具已随调测清理移出本仓；fork 仓在则用其副本）> \
#       --run $LEG_NAME=$DUMP_DIR -o $OUT_DIR/${LEG_NAME}_report.html
# 服务化的 dump 没有 bench.json，所以报告只出 PAR 面板：对账检查和 TTFT/TPOT/吞吐面板
# （这两块来自离线 driver 的 bench.json）不会出现——TTFT/TPOT 走 /metrics 另取。
# 三条注意：
#   1) 一个目录只放一次运行的 dump：文件是截断写的（open "w"），而 analyzer 把一个目录当成
#      一个 run 读——同目录重跑会覆盖上一次，所以上面有存在性检查，存在就直接退出；
#   2) 停服是 force-kill（见「每 rank 负载计数」），记录器没有第二次机会收尾，最后不足
#      FLUSH_EVERY 步的数据就永远留在内存里。本脚本因此把 FLUSH_EVERY 设成 1（每步落盘）：
#      负载腿跑得短，而丢半个窗口就是丢一半样本（实测 32 步的运行只落盘 16 步）。做性能测量
#      时别开（每步一次 device→host 拷贝），但那时也不该开 MOE_PAR；运行期间把 $DUMP_DIR
#      整个拷走同样可行——每次 flush 之后的文件都是自洽的；
#   3) 服务化的 dump 含真实流量的空转步（DP dummy/wave 气泡也走记录器），PAR 是「含气泡的
#      真实利用率」，只能和服务化同口径的腿比，别直接拿去和离线满负载的饱和稳态比。
# 这行无条件清掉外部可能继承的同名变量：不开 MOE_PAR 时，shell 里残留什么都不会让记录器启动。
unset VLLM_ASCEND_MOE_PAR_STATS VLLM_ASCEND_MOE_PAR_STATS_DIR \
      VLLM_ASCEND_MOE_PAR_STATS_TAG VLLM_ASCEND_MOE_PAR_FLUSH_EVERY
if [ "$MOE_PAR" = 1 ]; then
  export VLLM_ASCEND_MOE_PAR_STATS=1
  export VLLM_ASCEND_MOE_PAR_STATS_DIR=$DUMP_DIR      # 要换位置改上面的 DUMP_DIR
  export VLLM_ASCEND_MOE_PAR_STATS_TAG=serve_$MODE       # dump 文件名与 dump 头里的腿名
  export VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1               # 每步落盘，见注意 2
fi

# -------------------------------------------------------- 仿真路由 -- #
# 默认关：SIM_ROUTE=<档案> bash start_service_mode.sh <mode> 才注入（命令前缀，别 export——见文件头）。
# 用一份**指定的**专家选择分布替换 router 的输出，用来把「router 选专家的倾斜」变成自变量，
# 再去量它下游的 rank 级计算与通信（因变量）。实现见 ops/fused_moe/moe_sim_route.py。
# 三种写法（含 `:` 就是稀疏式，未列出的专家均分剩余）。**E / K / 层数都从模型现取**，
# 下面括号里的数字都是本 checkpoint（E=256, K=6, 43 层）的值，换模型自己换算：
#   SIM_ROUTE=@/tmp/sim_skew.txt        # 文件里是列表，推荐：E=256 手写列表不现实
#   SIM_ROUTE=0:16,7:10                 # 稀疏：专家 0 占 16%、专家 7 占 10%，其余均分
#   SIM_ROUTE=1.5,3.2,...               # 完整列表，E 项（本模型 256），和必须等于 100
# **硬上限 max(share) <= 100/top_k（本模型 K=6 → 16.67%；K=8 则是 12.5%）**：真实 topk 每个
# token 的 K 个专家互异，任何专家最多被 T 个 token 选中。超过这个值直接报错退出，不是告警。
# 该上限只由 K 决定，与 E 无关——换 E 不用改档案。
# 两种「步太小」的情形都不会打挂引擎，且都照常注入：T < top_k（如 dp=4 时每 rank 只剩 1 条序列
# 的 decode，拼批后 T=4）只打一行 warning（每进程一次），因为一行本来就 K 个槽、一个 token 就能
# 填满一行的互异专家，真正的约束是 count[e] <= T；DP 同步用的 dummy step（T 恒为 1，
# NPUWorker.execute_dummy_batch）也必须注入——挂起注入会让它走真实 router，而 DSv4 的 hash gate
# （MoeGatingTopKHash）正好在 N=1 这个形状上打挂 AI core（所有空闲 DP rank 同时，errcode 263,95），
# 同一个核在 startup profile 与真实步的成批尺寸上都正常。不带 SIM_ROUTE 的腿同样会撞到它。
# 日志：T 变化时打一张 E 行的完整表（req% / 期望条数 / 实际条数 / 偏差 + home-rank rows），
# 之后复用同一份分配（按 T 缓存，层间与同长步之间完全一致），每 SIM_ROUTE_EVERY 次调用打一行
# 复用确认。SUM_act 必须等于 T×K；max|dev| 应 <= 1（最大余数法保证）。
# home-rank rows 是**预测的** rank 级行数分布，正是要测的因变量，但假定无 EPLB（恒等 log2phy）。
# 真值拿 MOE_PAR=1 的 dump 对：off 腿的 counts_scope=local 行是 E/R 项（本模型 32），
# sum 即该 rank 实际计算的行数。
# home-rank rows 只在 E 能被 R 整除时打得出来（本模型 E=256：R=8 → 32/rank）。E % R != 0 时
# 那一行会明说「不可预测」而不是消失。
# 注意 owner 腿的 rank 负载会被 planning 核的 rebalancer 抹平，看真实 rank 级计算请用 off/slot 腿。
unset VLLM_ASCEND_MOE_SIM_ROUTE VLLM_ASCEND_MOE_SIM_ROUTE_SEED \
      VLLM_ASCEND_MOE_SIM_ROUTE_EVERY VLLM_ASCEND_MOE_SIM_ROUTE_TABLE
if [ -n "${SIM_ROUTE:-}" ]; then
  # 档案要在 worker 那边读（每个 rank 各自解析一次），所以文件不存在就地失败，
  # 别等模型加载完 5 分钟才在 worker 里报 FileNotFoundError。
  case "$SIM_ROUTE" in
    @*) SPEC_FILE="${SIM_ROUTE#@}"
        if [ ! -r "$SPEC_FILE" ]; then
          echo "[仿真路由] 档案文件不可读：$SPEC_FILE（SIM_ROUTE=$SIM_ROUTE）" >&2
          echo "    用绝对路径；worker 是 spawn 出来的，cwd 不可靠。" >&2
          exit 2
        fi ;;
  esac
  export VLLM_ASCEND_MOE_SIM_ROUTE="$SIM_ROUTE"
  export VLLM_ASCEND_MOE_SIM_ROUTE_SEED="${SIM_ROUTE_SEED:-0}"
  export VLLM_ASCEND_MOE_SIM_ROUTE_EVERY="${SIM_ROUTE_EVERY:-43}"   # 43 = MoE 层数 = 一步一次
  export VLLM_ASCEND_MOE_SIM_ROUTE_TABLE="${SIM_ROUTE_TABLE:-1}"
  SIM_ROUTE_DESC="on（SIM_ROUTE=$SIM_ROUTE  seed=${VLLM_ASCEND_MOE_SIM_ROUTE_SEED}  EVERY=${VLLM_ASCEND_MOE_SIM_ROUTE_EVERY}  TABLE=${VLLM_ASCEND_MOE_SIM_ROUTE_TABLE}；专家选择被整体替换，看 [sim-route rN]）"
else
  SIM_ROUTE_DESC="off（默认：走真实 router，行为与改动前完全一致）"
fi

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

# ---------------------------------------------------- 出报告的命令 -- #
# 这条腿的 dump 交给 analyzer 出负载报告：腿名取 OUT_DIR 的目录名，报告写回
# $OUT_DIR/<腿名>_report.html——报告名自带腿名，换目录收报告也不会重名。
# 横幅与收尾都会按**本次实际路径**把这条命令打出来，直接拷走执行即可。
ANALYZER=$WS/vllm_ascend_moonep/scripts/moe_par/analyze_moe_par.py   # 已移出本仓：fork 在则用之
[ -f "$ANALYZER" ] || ANALYZER="analyze_moe_par.py（已移出本仓，自备路径）"
REPORT_FILE=$OUT_DIR/${LEG_NAME}_report.html

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
  PRE_ENV_LINE=" 环境干净：启动前没有 MOONEP* / VLLM_ASCEND_MOE_PAR_* 残留"
fi
# 外部给了哪些路径（前缀与 export 分不出来）：标出来，没标的就都是默认值
PATH_OVERRIDE=""
[ -n "$OUT_DIR_OVERRIDE" ] && PATH_OVERRIDE="$PATH_OVERRIDE OUT_DIR"
[ -n "$DUMP_DIR_OVERRIDE" ] && PATH_OVERRIDE="$PATH_OVERRIDE DUMP_DIR"
[ -n "$LOG_FILE_OVERRIDE" ] && PATH_OVERRIDE="$PATH_OVERRIDE LOG_FILE"
if [ -n "$PATH_OVERRIDE" ]; then
  PATH_OVERRIDE_LINE=" ⚠ 这些路径是外部给的（命令前缀或 export），已按它们生效：$PATH_OVERRIDE —— export 过的会留到你自己 unset"
else
  PATH_OVERRIDE_LINE=" 路径都用默认值（要换就用前缀：OUT_DIR=… DUMP_DIR=… LOG_FILE=…）"
fi
# dump 落在哪，横幅里标一句：两者的差别是每步那一次落盘往返（见「负载 dump 的位置」）
# 只对认识的两类盘下判断：/tmp 是内存盘；/mnt/share 是 NFS、有实测数。别的路径不做假设。
DUMP_DIR_NOTE=""
if [ "$MOE_PAR" = 1 ]; then
  case "$DUMP_DIR" in
    /tmp/*) DUMP_DIR_NOTE="，本地盘" ;;
    /mnt/share/*) DUMP_DIR_NOTE="，NFS（每步约 1 ms 的落盘往返，实测）" ;;
    *) DUMP_DIR_NOTE="" ;;
  esac
fi
# 出报告的命令：路径按本次实际值填好，横幅里就能直接拷（没采 dump 就没有报告可出）
if [ "$MOE_PAR" = 1 ]; then
  REPORT_LINE=" 报告        停服后出负载报告（下面这条就是本次的实际路径，直接拷走执行）：
   python $ANALYZER --run $LEG_NAME=$DUMP_DIR -o $REPORT_FILE
             多腿并排看（G2 的 off + slot）就在后面再加 --run <另一腿名>=<另一腿的 moe_par 目录>"
else
  REPORT_LINE=" 报告        无（MOE_PAR=0：这条腿没采 dump；要报告就加前缀 MOE_PAR=1）"
fi
cat <<EOF | tee -a "$LOG_FILE"
════════════════════════════════════ 启动核对 ════════════════════════════════════
 模式        mode=$MODE —— $MODE_DESC
 MoonEP      enable_moonep=$ENABLE_MOONEP   MOONEP=${MOONEP:-未设}   MOONEP_SLOT_MODE=${MOONEP_SLOT_MODE:-未设}   MOONEP_SLOT_STATS=${MOONEP_SLOT_STATS:-未设}
 引导/传输   MOONEP_SHMEM_BOOTSTRAP=$MOONEP_SHMEM_BOOTSTRAP   MOONEP_SHMEM_IPPORT=${MOONEP_SHMEM_IPPORT:-未设}   MOONEP_DISABLE_UDMA=${MOONEP_DISABLE_UDMA:-未设}
 采集        MOE_PAR=${MOE_PAR:-0} → $DUMP_DIR$DUMP_DIR_NOTE（每 ${VLLM_ASCEND_MOE_PAR_FLUSH_EVERY:-16} 步落盘，文件 moe_par_serve_${MODE}_rank<r>.log）
             PROFILE=${PROFILE:-0} → $PROFILE_DIR（= OUT_DIR/profiling）
 负载计数    BALANCE=$BALANCE   MOONEP_BALANCE_STATS=${MOONEP_BALANCE_STATS:-未设}   EVERY=${MOONEP_BALANCE_STATS_EVERY:-0} —— $BALANCE_DESC
             SLOT_STATS=$SLOT_STATS —— $SLOT_STATS_DESC
 仿真路由    SIM_ROUTE=${SIM_ROUTE:-未设} —— $SIM_ROUTE_DESC
 服务        http://$HOST:$PORT   api-server-count=$API_SERVER_COUNT   dp=$DP tp=$TP ep=on   served-model-name=$SERVED_NAME
 模型        $MODEL
 负载口径    max-num-seqs=$MAX_NUM_SEQS   max-num-batched-tokens=$BATCHED_TOKENS   max-model-len=$MAX_MODEL_LEN   gpu-mem-util=$GPU_UTIL   chunked-prefill=on
 卡/关键env  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES   PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF   HCCL_BUFFSIZE=$HCCL_BUFFSIZE
 日志        $LOG_FILE
 产物        $OUT_DIR/（profiling/ 只在 PROFILE=1 时有内容；负载 dump 在 $DUMP_DIR）
$REPORT_LINE
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
# vllm serve 停掉、tee 收尾之后脚本走到这里：把这条腿的产物和出报告的命令再打一遍（路径就是本次
# 实际用的，直接拷走执行），并追加进服务日志——日志里已经有横幅和 vllm 那两段，这段是第三段。
# 如果连这个脚本一起被 kill（比如按进程组杀整个作业），这段就打不出来：那就回横幅里抄那条命令。
if [ "$MOE_PAR" = 1 ]; then
  REPORT_CMD=" 出报告     python $ANALYZER --run $LEG_NAME=$DUMP_DIR -o $REPORT_FILE"
else
  REPORT_CMD=" 出报告     无（这条腿没采 dump；MOE_PAR=1 才有）"
fi
set +x   # 这段是要给人看的，别混进 set -x 的 + 行（日志里也是这段最该干净）
{
  echo
  echo "──────── 这条腿结束：$LEG_NAME（mode=$MODE）────────"
  echo " 服务日志   $LOG_FILE"
  [ "$PROFILE" = 1 ] && echo " trace      $PROFILE_DIR/    （看：tensorboard --logdir $PROFILE_DIR）"
  [ "$MOE_PAR" = 1 ] && echo " 负载 dump  $DUMP_DIR/"
  echo "$REPORT_CMD"
  echo " 提醒       停服是 force-kill，dump 的最后一步可能没落盘（analyzer 会把这种 dump 标成不可知）"
} 2>&1 | tee -a "$LOG_FILE"
