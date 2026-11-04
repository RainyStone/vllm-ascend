# 诊断进度与证据边界

- 原四服务矩阵仍在执行；性能统计仅取各独立服务 round2/3。补充诊断计时不得并入。
- 第一对已完成：off TTFT 14471.15 ms、输入吞吐62294.51135 token/s；Slot TTFT15363.35 ms、吞吐58827.8645 token/s。Slot TTFT增加6.165%、吞吐下降5.565%。第二对尚待完成。
- 原正式服务明确未开启 VLLM_ASCEND_MOE_PAR_STATS，因此不能回溯声称已记录其逐step/rank专家不均衡比。
- 用户要求补查各rank不均衡比与精度根因；独立诊断脚本DIAGNOSTIC_REVIEW.md已准备、拉到本地并发出确认，尚未获得答复。

## 源码已确认

1. vllm_ascend/ascend_forward_context.py：A5默认 num_tokens<=MC2容量时选MC2；超过容量时，EP8大于topK6，选ALLTOALL。Slot的enable_moonep=true且KEEP_MC2=0会覆盖为SHMEM，不静默回退。实际运行method需统计/trace证据确认。
2. moonep_balance.py run_slot_forward：本轮cache=0、early_prefetch=0、PF_STREAM=0，所以每层都会进入prefetch_projections调用，再执行统一MLP；是否有实际权重GET及其字节量由规划决定，不能从“调用”推出每次都有搬运。
3. moonep_comm.py：REAL_T=1，不注入容量padding；MOONEP_NO_SYNC默认1，不能把旧.item()同步路径当作本轮瓶颈。
4. moonep_balance.py：home按层保留，工作slot跨同shape层复用；W4A8MXFP搬运4类载荷(w13/w2及各自scale)，不是只搬权重不搬scale。
5. moonep_comm.py token_combine：专家输出乘FP32路由权重，转BF16后交给MoonEP combine；kernels/moonep_combine.cpp将BF16行转FP32按K累加，最后转BF16。原生AllToAll的token_dispatcher.py调用npu_moe_token_unpermute(...,probs=topk_weights)融合加权归并。是否实际存在不同舍入与其影响需算子实验验证，当前不是已证实根因。
6. 三优化源码：PREFETCH_RELAXED去prefetch两侧host barrier；NO_RECONCILE去兼容层设备同步+跨rank barrier（依赖RELAXED）；SKIP_DUP去source-dup扫描和握手。不能只凭这些优化开关已开启就推断负载均衡必有净收益。

## 已准备的诊断工具（尚未执行NPU诊断）

- start_off/slot_diagnostic.sh：与正式服务参数相同，独立PAIR标签；仅额外开启现有逐rank负载统计，flush_every=1。
- bench_off/slot_diagnostic.sh：同316条与并发16，单轮、无预热；用于负载采样，计时不纳入原性能。
- analyze_rank_load.py：按(step,layer)拼8rank，区分归属负载与实际compute；输出峰均比分布、每层/每rank数据与逐专家条目守恒；不把补采数据冒称原轮统计。
- probe_first_tokens.py：CPU已准备8条固定样本，含5条稳定跨模式不同、1条off自身重复不同、1条稳定相同、1条最长；拟每模式串行3次+同请求集合并发3次，流式输出1token并收集top-logprobs。精确engine batch仍受调度影响。
- probe_weighted_reduce.py：拟单卡比较原生weighted-unpermute与“逐专家先BF16舍入”和“FP32求和后舍入”两条公式；仅可证明算子数值差异，不自动证明端到端因果。
- off_host_snapshot.json：正式第二对off首轮的一秒辅助快照，384逻辑CPU、loadavg约30，worker约55%~65%单核占用。单点不能证明全程无主机瓶颈，且尚无对应Slot快照。

## 精度证据边界

已经观察到首token文本不完全一致，且off自身重复也有差异。完整prompt hash、输入token数、成功数与输出token数均受校验；这排除了已知输入模板问题，但不证明权重/通信/计算完全正确。不提前将其归因于普通数值误差，也不以1token性能实验评价完整LongBench准确率。

## 补充核查
- aisbench VLLMCustomAPI.get_request_body直接合并generation_kwargs；BaseAPIModel构造时复制配置字典，未发现过滤temperature/seed。配置快照为temperature=0、seed=1024、ignore_eos=True、max_tokens=1。现有debug日志没有完整HTTP报文记录，源码路径核查不等同于历史线上抓包。
- 本次保留原始aisbench LongBench-v2 completion模板，没有额外套chat模板，且只生成1token。样本205原生off两轮首token为两个引号，Slot为引号+句号+换行；prompt尾部是未闭合的格式说明引号。当前差异是首token数值/执行一致性问题，不能直接定义为答题准确率下降，也不应为追求首token一致而修改已批准的性能输入。
- 按长度分组的TTFT分析脚本已准备，从原JSONL的SQLite引用读取时间点，并逐轮与aisbench汇总CSV交叉校验；不从TTFT推导未记录的专家不均衡比。
