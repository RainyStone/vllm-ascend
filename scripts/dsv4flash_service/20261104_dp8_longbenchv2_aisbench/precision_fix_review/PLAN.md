# MoonEP Slot 精度修复方案（待用户确认）

本文件仅为方案与接口草案。未修改生产源码，未编译，未起服。用户追加要求：修复尽量不引入额外开销。

## 1. 本轮目标

先修复已经证实的数值不等价点：当前Slot把每个专家的FP32加权结果提前转BF16，再做K路求和。改成在combine内部用FP32乘权重与累加，最后只转一次BF16。

~~~text
z_k = FP32(expert_output_k) * FP32(route_weight_k)
q_k = BF16(z_k)
当前：output = BF16(sum_k(FP32(q_k)))
目标：output = BF16(sum_k(z_k))
~~~

专家输出仍为BF16；DSV4 Flash的FP4权重、E8M0 scale和FP8激活计算路径不变。本轮不同时修改激活通信格式、路由、缓存或GEMM，以便判断修复作用。

已完成的算子实验中，原生weighted-unpermute与目标公式在593,920个输出元素上完全相同，当前公式约39.7%元素不同。该证据确认了额外舍入，但尚未证明全部端到端差异只有这个原因。

## 2. 推荐实现：源rank本地权重 + 融合weighted combine

1. dispatch保留源rank的FP32 top-k权重。优先引用已有路由权重缓冲，不跨卡取回权重，也不从打包后的专家行反推top-k顺序。
2. 专家计算输出保持原始BF16；移除Slot当前在Python端执行的“整块乘权重→转BF16”。
3. combine按plan.dst定位每个token的第k路专家结果。将取回的BF16行在UB中转FP32，乘该token原始第k路的本地FP32权重，按确定的k顺序FP32累加，最后只转一次BF16。
4. 保留有效行/尾部掩码、REAL_T边界、当前stream顺序。必须测试零权重、重复expert/重复映射以及异步缓冲复用，避免漏项或重复计入。
5. 用独立的新绑定与新内核入口表达weighted语义，不更改旧combine默认语义。

### 接口草案

~~~python
# 概念接口，最终实现在确认后形成代码diff：
buffer.combine_weighted(
    plan=plan,
    hidden_nvsh=unweighted_bf16_expert_outputs,
    source_topk_weights_sk=local_fp32_topk_weights,
    num_tokens=real_t,
    inter_rank_sync=True,
)
# 返回 [S,H] BF16；不返回跨rank取回的route weights。
~~~

现有with_weight只控制权重回传，不是加权求和开关，不能直接复用其语义。C++ Combine类可增加compute_weighted方法，旧compute保持兼容；新方法使用独立launcher/导出符号，避免加载旧kernel时静默执行未加权路径。缺少新接口或加载路径不匹配必须报错。

MoonEPCombineTilingData.reserved已有packed-weight偏移语义，不占用它作为新开关；如需新tiling，使用独立结构。

## 3. 低开销约束

| 项目 | 约束 |
|---|---|
| 跨卡通信 | 不新增collective，不增加第二次gather/combine；路由权重在源rank本地读取 |
| 内核 | 融合到一次weighted combine，不拆成“取回全部K路输出→另起算子归并” |
| 大张量 | 不创建[NvS,H]的FP32临时张量，不把专家输出整块升到FP32后传输 |
| 权重缓冲 | 优先复用；如向量化读取需要K对齐，最多[S,align8(K)] FP32，S=8192、K=6时0.25MiB/卡，并复用 |
| FP32工作区 | 只在UB里保留当前行的累加器/临时向量，不存全量token的FP32副本 |
| CPU同步 | 不新增逐层.item()、.cpu()、npu.synchronize()或host barrier |
| 热路径日志 | 仅启动/首调用时记录版本和分支；验证探针和计时只在测试中启用 |

当前外部加权可能产生大块FP32临时结果；融合后有机会减少HBM中间读写。但不预先承诺提速，必须测旧/新整段路径。

首版只接入本轮Slot的non-UDMA、REAL_T=1、SKIP_DUP=1配置。UDMA和source-dup启用的其他路径不静默回退，未覆盖配置明确报错；旧API的其他调用方保持原行为。重复top-k/别名映射与source-dup是不同情形，前者必须正确处理并单测。

## 4. 修改范围

- vllm-ascend：moonep_comm.py的combine metadata、dispatch保留源权重、Slot token_combine调用；不改模型权重或量化方案。
- ascend-moonep Python：api.py、combine.py/launchers.py、ops.py等对应调用链，新增显式weighted接口及输入/版本检查。
- ascend-moonep C++：torch_classes.cpp、combine host op、launcher及导出声明；新增weighted方法/入口。
- AscendC：在现有native combine代码组织中增加weighted实现，保持原路由/有效行边界；旧入口ABI不变。
- 测试及脚本：受控算子回归、8rank通信/映射测试、小样本模型验证与单独性能检查。

不引入新的环境变量。服务继续用独立sh，参数写在脚本内；不靠外部环境临时改变行为。

## 5. 隔离与构建

- 以当前工作树及实际二进制快照为基线，记录源码、SDK、kernel库SHA256；保留现有子模块修改和未跟踪编译产物，不能只checkout HEAD而丢失当前环境。
- 在独立副本和build目录修改/编译；先不覆盖原ascend-moonep库或现有vllm-ascend源码。
- 不直接运行会全局pip install -e并覆盖原库的build.sh；使用隔离构建和包路径，第三方依赖只读复用。
- 每个服务进程只加载一套SDK，记录实际Python模块路径、映射.so路径和新增入口可用性。8worker必须都使用修复后的weighted路径。
- 原off/旧Slot仍可复现，原实验结果与原始日志不覆盖。

当前基线：
- ascend-moonep：urma_dev，ceabdc2ec2d9559a7757be266456ebb061306066；3rdparty/shmem存在子模块修改。
- vllm-ascend：moonep_v0.23.0，d2666bfb7d10f2704efeabc5888209f499e01183；csrc/third_party/catlass存在子模块修改。
- 真正实施前再次检查是否变化；如变化先保留新状态，不强行重置。

## 6. 分阶段验证（推荐审批范围）

### A. 算子与通信正确性

- 同一BF16专家输出、同一FP32权重、同一plan，对比FP32公式和原生weighted-unpermute。
- 覆盖T=1/16/128/1024/8192、K=1/6/8/16中的有效组合，S>T、rank间T不一致、uniform/skew、零权重、重复专家/别名、不同k权重、无有效行边界、反复复用缓冲。
- 本机2rank/8rank测试；记录逐元素差异、最大绝对误差、RMSE和ULP。受控同输入优先要求BF16输出逐元素一致；若不一致先核对乘加顺序和舍入，不通过随意放宽容差宣布修复成功。
- 验证旧未加权API回归、无新增条目丢失、SDK/内核版本不匹配会明确报错。

### B. 修复本身的开销

先对同一受控输入比较整段“旧外部FP32乘法+BF16转换+combine”与“新weighted combine”，包含同步完成的设备耗时和峰值显存，不只比较launch时间。

覆盖full 8K和短步、均衡/偏斜、rank间输入量不一致，包含源rank输入集中的情况。测试中的同步只用于计时，不进入生产热路径。

建议初始门槛：重复测量后，中位链路耗时不增加超过2%，峰值显存不明显上升。超过或噪声无法判定则停下给出证据，不能通过恢复旧的舍入公式换取性能。

### C. 小样本端到端验证

- 新鲜启动off、旧Slot、修复Slot三个对照服务，均保持DSV4 Flash、DP8/TP1/EP8、KV24GiB、max_model_len131073、8K chunk、并发上限16、输出1及三项优化。
- 首先复现现有8条固定样本，串行和固定并发集合重复；同prompt SHA/token数/seed/temperature，记录top-logprobs及token ID（本地API已支持return_tokens_as_token_ids，精度请求仅改变返回表示）。
- 比较旧Slot→修复Slot变化、off自身重复差异以及所有剩余跨模式差异。不能仅凭一致率变高就宣称全部精度问题解决。
- 小样本通过后，off/修复Slot各补测一次316条全文请求。该轮用于精度和功能检查，不替代原正式性能矩阵。
- 若仍有可复现差异，保存首个分叉证据，再提出按层中间值/权重字节校验方案；不自动改变权重精度、路由或批次参数掩盖差异。

本轮不自动重跑完整多轮性能矩阵。完整性能验收和是否替换主工作目录，在上述证据可审阅后另行确认。所有日志、脚本、diff和新数据保存在本实验目录的独立precision_fix子目录。

## 7. 待确认

推荐批准：在隔离副本实施上述低开销weighted combine修复、编译并执行A/B/C阶段验证。2%的门槛用于初步工程筛查，不作为统计显著性结论。原生产工作目录和库暂不替换。
