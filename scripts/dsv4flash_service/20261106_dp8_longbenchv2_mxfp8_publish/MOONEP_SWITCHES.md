# 当前MoonEP开关完整清单（2026-11-06）

范围：118共享原仓vllm-ascend/vllm_ascend和ascend-moonep/{ascend_moonep,csrc,include,kernels}；不含第三方SHMEM/HCCL内部选项及实验脚本私有参数。扫描到57个字面配置名，另列4个SDK阶段派生名、SDK参数和构建选项。
默认值来自实际读取代码，不用历史注释替代；最近值来自20261106单对实验Slot环境快照。很多开关在导入或首次调用时缓存，修改后应重启服务；collective相关设置各rank须一致。
服务总入口为--additional-config '{"enable_moonep":true}'，默认false。脚本中的MOONEP=1不是当前总入口判断；ASCEND_MOONEP_BUILD_DIR在本次扫描范围内未发现消费点，不列作有效优化开关。

## 优化

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|VLLM_ASCEND_MOONEP_DISPATCH_DTYPE|bf16|mxfp8在发送端执行已有MXFP8量化并打包激活及scale；支持Slot/W4A8MXFP/REAL_T；combine仍BF16|mxfp8|vllm-ascend/vllm_ascend/envs.py:29|
|VLLM_ASCEND_MOONEP_FUSED_PUBLISH|0|融合有效行复制与SHMEM尾部清零，省去中间clean矩阵；需要publish库|1|vllm-ascend/vllm_ascend/envs.py:36|
|MOONEP_REAL_T|1|planning/dispatch/combine只计真实token，排除padding负载|1|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:133|
|ASCEND_MOONEP_PREFETCH_RELAXED|0|省去prefetch前后跨rank host barrier；副流必须配套|1|ascend-moonep/csrc/acl_shared_buffer.cpp:47|
|ASCEND_MOONEP_NO_RECONCILE|0|省去反复npu.synchronize和dist.barrier；应与PREFETCH_RELAXED=1配套|1|ascend-moonep/ascend_moonep/api.py:45|
|ASCEND_MOONEP_SKIP_DUP|0|跳过source-dup跨rank去重握手，换取更低planning开销；加权combine链路依赖此模式|1|ascend-moonep/csrc/ops/planning.cpp:394|
|MOONEP_PF_STREAM|0|0主流；1事件管道诊断；2副流prefetch掩盖dispatch；3额外提前Home GEMM1；3要求cache及early关闭，需range GEMM库|2|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:754|
|MOONEP_SLOT_CACHE|0|N>0表示每卡每层缓存N个远端专家权重与scale；此前cache8有DDR越界记录，本轮关闭|0|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:88|
|MOONEP_RELEASE_WEIGHTS|1|home镜像发布后尝试释放原权重副本；具体是否释放依布局/载体路径|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:163|
|MOONEP_NO_SYNC|1|避免把行数.item()读回host，设备侧counts与尾部处理|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1019|
|MOONEP_REUSE_OUT|1|BF16 dispatch复用输出buffer；MXFP8分支使用自身缓存及view，不能用此开关控制全部复用|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1017|
|MOONEP_POOL|1|按步骤token数选择容量buffer；0使用完整容量；不代表0时不创建容量池|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1022|
|MOONEP_EARLY_PREFETCH|0|按预测/历史计划提前预取，真实plan到来后修补不匹配槽位；实验性|0|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:737|
|MOONEP_EARLY_PF_DEV|1|EARLY_PREFETCH启用时，1在副流发出；0在主流发出作对照|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:738|
|ASCEND_MOONEP_BARRIER_V4|0|planning跨rank汇合由默认v3 pull改为v4 push；所有rank须一致|未显式设置；按默认/条件分支|ascend-moonep/csrc/ops/planning.cpp:564|

## 调参

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|MOONEP_MIN_DISPATCH_TOKENS|128|最小dispatch容量；优先于旧别名MIN_CAP；最近实验16|16|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:496|
|MOONEP_MIN_CAP|128（fallback）|MIN_DISPATCH_TOKENS未设置时的旧别名；已设置新名时不生效|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:497|
|MOONEP_POOL_CAPS|自动|逗号分隔容量档位；会补入最大容量|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:528|
|MOONEP_POOL_MAX_RUNGS|32|自动容量阶梯的档位数量预算|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:553|
|MOONEP_POOL_BUDGET_MB|2048|自动容量阶梯的内存预算MiB；不等于全部MoonEP显存硬上限|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:554|
|MOONEP_B|E/EP|远端专家临时槽数；当前planning强制B=E/EP，不能任意减小|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/envs.py:162|
|MOONEP_NUM_SMS|未设置→SDK32|dispatch/combine/prefetch等算子的核数；不是卡数|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:799|
|MOONEP_PLANNING_NUM_SMS|自动：小路由8，大路由通常32|planning单独核数；合法8..56；大路由默认max(8,min(num_sms,32))|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:152|
|MOONEP_NUM_SMS_DEDUP|设备可用核数|本地prologue/epilogue等相关核数|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:115|
|MOONEP_ASCEND_OP_WS_BYTES|16777216（16MiB）|通用op workspace预算，主要dispatch/combine；planning单独计算，prefetch不使用此workspace|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:173|
|MOONEP_SLOT_CACHE_LAYERS|64|缓存分配的层数维度；仅SLOT_CACHE>0有意义|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:438|

## 模式

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|MOONEP_SLOT_MODE|0|1为Slot专家借用/负载均衡；0为owner-only；仍需enable_moonep|1|vllm-ascend/vllm_ascend/envs.py:152|
|MOONEP_SLOT_QUANT|1|允许量化模型进入Slot；不是把模型切成量化/非量化|1|vllm-ascend/vllm_ascend/envs.py:167|
|MOONEP_DISABLE_UDMA|0|vLLM侧：1强制MTE；0请求UDMA，实际还取决于硬件；planning仍MTE|1|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:755|
|ASCEND_MOONEP_USE_UDMA|SDK读取默认0，Buffer初始化会重写|SDK运行时后端状态；vLLM应通过DISABLE_UDMA设置，勿与服务默认混淆|初始化派生为0；MTE|ascend-moonep/ascend_moonep/shmem_runtime.py:44|
|MOONEP_KEEP_MC2|0|允许小请求保留MC2，可能不走MoonEP；纯Slot对照应0|0|vllm-ascend/vllm_ascend/envs.py:159|
|MOONEP_FORCE_ALLGATHER|0|A5原生MoE路径强制ALLGATHER；非Slot加速开关|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ascend_forward_context.py:338|
|MOONEP_ALLOW_CAPTURE|0|只放开capture保护守卫，不自动开启图捕获，也不证明已支持|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:759|
|MOONEP_NO_PIN|0|1使用pageable host buffer绕过pinned分配问题；可能牺牲H2D性能|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/platform.py:902|

## 初始化

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|MOONEP_AICPU_WARMUP|1|跨设备映射建立前预热AICPU custom调度器，属于初始化顺序保障|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:650|
|MOONEP_DEV_BASE|0|诊断/辅助路径的device到rank编号偏移|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:341|
|MOONEP_SHMEM_BOOTSTRAP|ipport|SHMEM引导方式；影响建连，不是计算优化|ipport|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:865|
|MOONEP_SHMEM_IP_PORT|28765|SHMEM ipport引导端口|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:876|

## 诊断

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|MOONEP_TIME_STATS|0|插件阶段host耗时及MLP时间统计|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:375|
|MOONEP_TIME_STATS_EVERY|0|按调用间隔打印；0通常退出时打印|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:452|
|ASCEND_MOONEP_TIME_STATS|0|SDK阶段耗时统计|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:42|
|ASCEND_MOONEP_TIME_STATS_EVERY|0|SDK统计打印间隔|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:51|
|MOONEP_MLP_SYNC|0|为MLP计时增加设备同步|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:376|
|MOONEP_PACE_SYNC|0|dispatch后强制设备同步，排障/对照|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1020|
|MOONEP_PF_TIMING|0|同步测prefetch及后续GEMM，影响正常重叠|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py:1115|
|MOONEP_BALANCE_STATS|0|逐rank/层负载统计|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moe_par_stats.py:137|
|MOONEP_BALANCE_STATS_EVERY|0|负载统计输出间隔|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moe_par_stats.py:138|
|MOONEP_PLAN_STATS|0|读回plan与槽位分配；会增加host同步|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:159|
|MOONEP_SLOT_STATS|0|槽位/预取专家统计；涉及.item()读回|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:475|
|MOONEP_DUMP_PLAN|无|设置输出目录后转储plan|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:411|
|MOONEP_DUMP_PLAN_N|96|plan转储数量上限|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:412|
|MOONEP_CAP_TRACE|0|容量/图捕获跟踪|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1021|
|ASCEND_MOONEP_TRACE|0|SDK调试trace|未显式设置；按默认/条件分支|ascend-moonep/ascend_moonep/api.py:46|
|MOONEP_EARLY_PLAN|0|提前路由/计划预测探针；额外router计算，用于评估预测命中，不是免费优化|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:176|

## 旧实验

|名称|代码默认|作用/限制|最近Slot脚本值|源码|
|---|---|---|---|---|
|MOONEP_PAD_CUT|0|源码明确记录可能裁掉真实token并产生错误；勿当作可用优化|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:382|
|MOONEP_PAD_EXPERT0|0|旧padding模式把pad路由到专家0；REAL_T=1时不生效|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1023|
|MOONEP_PAD_ROTATE|0|旧padding模式逐层轮转pad专家；REAL_T=1时不生效|未显式设置；按默认/条件分支|vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py:1261|

## SDK阶段变量与API参数

ASCEND_MOONEP_{PLANNING,DISPATCH,COMBINE,PREFETCH}_USE_UDMA由Buffer初始化按设备能力及use_udma/udma_flags写入，当前planning固定非UDMA；本次四项均为0。不能把这些派生状态当成可随意覆盖的独立vLLM开关。
UDMA+PF_STREAM>=2触发专用QP隔离及校验，没有另一个环境变量；MTE不走该分支。
SDK直接调用还有dispatch(reuse_outputs=False)、dispatch/combine/combine_weighted(zero_copy=False)、inter_rank_sync=True及combine_weighted(fused_publish=False)等参数。zero_copy返回内部存储，生命周期受下一次调用约束；fused_publish与zero_copy互斥；async_finish是接口兼容参数，不能据此认定通信已异步。vLLM当前由前述服务配置控制，不是所有SDK参数都有env对应。

## 构建开关

以下CMake option默认均OFF；ON仅生成对应库，不等于服务启用该功能：
- ASCEND_MOONEP_BUILD_HOME_GMM：Home/Slot GEMM1范围内核。
- ASCEND_MOONEP_BUILD_MXFP8_WIRE：MXFP8打包/解包库。
- ASCEND_MOONEP_BUILD_PUBLISH：融合publish库。
- ASCEND_MOONEP_BUILD_UDMA_WEIGHT_REUSE_FIX：UDMA weighted-combine重用围栏候选独立库。
- ASCEND_MOONEP_BUILD_UDMA_WEIGHT_SCALAR_FIX：UDMA scalar-to-vector候选独立库。
- ASCEND_MOONEP_BUILD_UDMA_WEIGHT_CAPTURE_CONTROL：负对照独立库，不是性能优化。
后三项为当前排障候选/对照，不能当作已验证服务优化；本次MTE实验未启用。

## 没有独立优化开关的现有改动

Slot融合FP32加权combine精度修复、跳过未消费的GMM2事件、未启诊断时跳过冗余路由histogram等当前按源码路径自动生效；不能把所有代码改动都归于新增两个env开关。当前没有combine输出MXFP8量化开关，combine仍BF16传输。
