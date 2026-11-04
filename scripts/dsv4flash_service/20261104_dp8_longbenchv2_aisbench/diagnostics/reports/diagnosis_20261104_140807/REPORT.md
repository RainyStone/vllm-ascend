# LongBench-v2 off / MoonEP Slot 补充诊断

补充诊断已完成：off/Slot各一次独立起服，各316条全文负载补采、各48次固定样本请求，另有共6次边界标记请求。原服务参数不变，两模式实际状态均通过8个worker核验。诊断计时未并入正式性能结果。

原正式两对结果：Slot TTFT平均增加6.10%，输入吞吐平均下降5.53%。

## rank不均衡比与均衡效果

每step/layer的rank条目数记为L(i,r)，一个token选6个专家就计6个token–expert条目：

~~~text
s(i) = max_r L(i,r) / (sum_r L(i,r) / 8)
W(i) = sum_r L(i,r)
加权s = sum_i(W(i)*s(i))/sum_i W(i)
      = 8*sum_i(max_r L(i,r))/sum_i(sum_r L(i,r))
~~~

|阶段|加权rank峰均比|P90|最大值|
|---|---:|---:|---:|
|off实际计算|1.253942|1.411169|2.888970|
|Slot重分配前|1.254709|1.433695|2.899703|
|Slot实际计算|1.000028|1.000122|1.005040|

Slot将负载基本完全均衡。按条目数线性模型，累计最重rank工作量约下降20.30%；这不是总推理时延能直接下降的比例，attention、其他计算和通信未被该指标计入。

专家峰均比加权均值off约7.120、Slot约7.111；不能把专家峰均比当rank峰均比。也不能先累计每rank总量再算比值来替代上述指标，因为热点rank会跨层、跨步骤变化。

主阶段对齐：off以八rank的ALLTOALL collective调用顺序对齐，299个step×43层；Slot验证八rank活跃原始step ID完全相同，307个step×43层。启动、空闲与极小尾部记录保留在原始日志，未混入主阶段跨rank指标。所选阶段覆盖各rank已记录条目约99.997%以上。不同模式step数量不同，没有强行拼成相同批次。

Slot全部13,201组对齐step/layer的路由与实际计算全局专家向量逐项相等：共4,777,934,766条，无缺失组。这排除了该统计范围内的专家条目数量丢失，尚未证明权重搬运字节及每个token身份逐项正确。

记录条目除以43层和topK6，off/Slot分别相当于18,519,064/18,519,127个token；316条请求input token合计18,518,962。微量额外等价行可能来自DP占位/边界，日志未标明身份，不把它们冒称纯文本token。差额均不足0.001%。MoonEP容量padding统计为0。

每rank归一化负载及P90/最大值见RANK_SUMMARY.csv；逐step/layer/rank计数见各模式step_layer_rank_load.csv。

## 为什么均衡了仍然劣化

“不均衡完全不够”不是本轮数据支持的简单结论。当前最有力的结构性解释是通信与额外路径成本抵消了均衡带来的计算收益：

- 当前W4A8MXFP配置中，原生off ALLTOALL先量化，再发送FP8激活和e8m0 scale；MoonEP发送BF16激活，到达后才在MLP入口量化。每跨rank条目的激活+输出tensor payload，原生约12,416字节，MoonEP约16,384字节，未计协议元数据。
- Slot每层仍有规划和四类weight/scale payload预取；本轮持久专家缓存、early-prefetch及独立prefetch流均关闭。记录到至少111,301次有实际工作的远端专家分配，涉及321,945,782个计算条目。按H=4096/F=2048/group32，每专家weight+scale为12.75MiB，对应至少约1.49TB预取tensor payload。
- 对本次Slot观察到的同一路由作保守核算：重分配B条计算最多减少B条跨rank传输，因此C_slot >= C_off-B。原生feature payload约51.90TB，Slot feature+prefetch payload下界约64.69TB，约为1.247倍。
- 借用计算条目并不全源自当前计算rank，不能把全部借用条目当作实际省下的跨卡通信；上述下界不依赖该假设。
- 这是同一路由的tensor payload下界模型，不是实测链路流量。协议、缓存、并行重叠、local memory copy和GEMM效率尚未逐项计时，不能把24.7%的字节差换算成6.10%的时延差。
- 动态msprof采样未成功：容器PID和宿主机PID均尝试，工具未接受目标PID，未获得有效trace。不能声称已测出规划/预取/激活通信/GEMM各自耗时百分比。

## 首token一致性与logprobs

8条固定样本并非随机精度样本。每模式每条做3次串行、3次固定请求集合并发，共48次；固定集合阶段并发8，316条负载补采并发16。每模式内每条6次文本全部相同，跨模式6/8条不同。

例如样本205，off两个候选token相对logprob差为+1.125，Slot反转为-0.750，相对变化1.875；并非只有极小margin的并列翻转。详细logprobs与原始SSE保留在PRECISION_COMPARISON.json和各responses.jsonl。

## 已证实的加权归并数值不等价点

源码中，MoonEP先把每个专家的FP32加权结果转回BF16，再由combine转FP32相加；原生weighted-unpermute融合加权求和：

~~~text
z_k = FP32(x_k) * w_k
native: BF16(sum_k(z_k))
MoonEP当前公式: BF16(sum_k(FP32(BF16(z_k))))
~~~

本机单卡受控实验：H=4096、topK=6、T=1/16/128，路由权重和为1。

- 原生算子两次重复完全相同。
- 原生与“FP32加权求和后一次BF16舍入”在全部593,920个输出元素上完全相同。
- 原生与MoonEP当前公式约39.7%的元素不同，最大绝对差0.015625。
- 另保留权重和1.5的敏感性测试，结论相同。

这确认了提前BF16舍入会引入额外数值差异。实验比较原生算子与两条数学公式，并非完整8rank MoonEP通信算子回放；未逐层对齐模型中间值，也未实施修复后的端到端消融，因此尚不能证明全部首token差异都由此产生。量化GEMM分组/批次变化、权重与scale内容还需进一步排除。

本轮max_out_len=1，使用原始completions prompt，不能据此评价完整LongBench答题准确率。

## 建议修复验证点

优先让combine在FP32中完成乘权重与累加，最后仅舍入一次BF16：传入未加权BF16 expert output和FP32 route weights，在combine内做FP32乘加。只把Python的w_trim改成FP32不能解决问题，因为它已经是FP32，额外舍入发生在乘法结果转BF16时。

生产源码未修改。后续应先做分布式combine对照与条目/权重校验，再用相同固定prompt、相同batch形状和原316条请求复测，量化修复对首token和性能的影响。

## 数据与资源

远程目录：
/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/diagnostics/data/executions/diagnosis_20261104_140807

原始rank日志、aisbench debug/JSONL/SQLite索引、固定请求SSE、单卡测试张量、脚本和配置均保留。标有load_analysis_unaligned的早期分析因编号对齐问题作废，未用于以上结论；原始日志未修改。最终资源释放检查见CLEANUP_CHECK.json。

最终资源释放检查时间：2026-11-04T14:43:30.343079+08:00。
