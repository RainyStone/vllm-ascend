#!/usr/bin/env python3
"""Summarize only existing evidence; keep diagnostics distinct from formal timing."""
from pathlib import Path
import datetime as dt,json,statistics
r=Path(__file__).resolve().parent
state=json.loads((r/'STATE.json').read_text())
modes=('off','slot_mte','slot_udma')
profroot=r/'data/diagnostics/optimized_profiling_20261106'
prof=json.loads((profroot/'PF2_STAGE_BREAKDOWN/summary.json').read_text())
result=r/'results';result.mkdir(exist_ok=True)
lines=['DP16 优化复测：阶段耗时与负载分析', '更新时间：'+dt.datetime.now(dt.timezone.utc).isoformat(),
'正式轮：off→MTE→UDMA，各503条、独立起服、0预热。以下诊断与正式计时严格分开。',
'本轮纳入的既有原仓优化：MTE/UDMA MXFP8激活传输、BF16尾部清理与发布融合、FP32路由加权combine及最终一次BF16舍入、UDMA S→V权重缓冲区同步修复、PF_STREAM=2预取/dispatch重叠。',
'本次正式复测及诊断期间只新增实验和分析脚本，没有边测边改生产算子。Home GEMM拆分按已确认方案关闭。',
'卡上正确性前置验证已通过；模型回答精度按用户要求暂不复测，不能把请求成功或算子一致性当作LongBench答案精度得分。',
'共同参数包含 dp_allreduce_on_npu=true；这是DP同步配置，不能据此把专家通信称为AllReduce。',
'Slot参数：MXFP8 dispatch、融合BF16发布、FP32加权combine、PF_STREAM=2、关闭Home拆分。UDMA组planning=MTE，dispatch/combine/prefetch=UDMA。',
'',
'正式性能（本次每模式只有一轮，不代表多次重复的置信区间）',
'模式 | 平均TTFT ms | 相对off | 输入吞吐token/s | 相对off | 成功数']
base=state['formal']['off']['summary']
for mode in modes:
 s=state['formal'][mode]['summary']
 lines.append(f"{mode} | {s['ttft_ms']['mean']:.2f} | {(s['ttft_ms']['mean']/base['ttft_ms']['mean']-1)*100:+.2f}% | {s['input_tps']:.2f} | {(s['input_tps']/base['input_tps']-1)*100:+.2f}% | {s['requests']}")
lines+=['','阶段profiling：每个模式16个rank、1333个匹配的局部prefill MoE层调用；单位ms/调用。',
'匹配同一组16个长度分位样本，按每rank捕获的prefill序号对应，取各模式共同可用的前两次prefill。rank8有43条，其余每rank86条。',
'这些序号尚未证明是全局同步step ID。结果是独立代表样本的局部阶段均值，不能按层数相乘，声称精确分解正式503条TTFT。',
'没有使用未经校准的两机wall-clock直接叠加时间轴；阶段耗时和prefetch重叠都取同一rank的本地事件差值。只读时钟样本在results/CLOCK_SNAPSHOT.json，未修改系统时间。',
'主流各段互不重叠；dispatch/combine包括协议与跨rank等待，不等于纯传输时间。副流prefetch不能再加到总耗时。',
'阶段 | off | Slot MTE | Slot UDMA']
keys=[('TopK路由','topk_us'),('dispatch整段','dispatch_inclusive_us'),('GEMM1','gemm1_kernel_us'),('激活及GEMM2前量化','activation_quant_span_us'),('GEMM2','gemm2_kernel_us'),('combine整段','combine_inclusive_us'),('上述路由专家主链合计','core_wall_us')]
for label,key in keys:
 vals=[prof['data'][m]['stages'][key]['mean_us']/1000 for m in modes]
 lines.append(label+' | '+' | '.join(f'{v:.4f}' for v in vals))
lines+=['','MoE其余阶段（不包含在上述路由专家主链合计中）：','阶段 | off | MTE | UDMA']
for label,key in [('router GEMM','router_gemm_us'),('shared expert链','shared_span_us'),('routed/shared合并加法','merge_add_us'),('完整router到merge墙钟','total_router_to_merge_wall_us')]:
 vals=[prof['data'][m]['stages'][key]['mean_us']/1000 for m in modes]
 lines.append(label+' | '+' | '.join(f'{v:.4f}' for v in vals))
lines+=['','GEMM1调用耗时分布：','模式 | mean ms | P95 ms | max ms']
for mode in modes:
 z=prof['data'][mode]['stages']['gemm1_kernel_us']
 lines.append(mode+' | '+' | '.join(f'{z[k]/1000:.4f}' for k in ('mean_us','p95_us','max_us')))
lines+=['Slot在该采样中的GEMM1尾部更短，但均值略高。该分布不是每个全局step的最慢rank，不能据此直接计算负载均衡对TTFT的收益。']
lines+=['','Slot内部细分（属于上表整段，不能重复相加）：','阶段 | MTE | UDMA']
detail=[('planning内核','planning_kernel_us'),('MXFP8 pack','pack_kernel_us'),('dispatch内核','dispatch_kernel_us'),('MXFP8 unpack','unpack_kernel_us'),('融合BF16发布','fused_publish_kernel_us'),('weighted combine内核','combine_kernel_us'),('副流prefetch跨度','prefetch_span_us'),('dispatch结束后未完成的prefetch','prefetch_pending_after_dispatch_us')]
for label,key in detail:
 vals=[prof['data'][m]['stages'][key]['mean_us']/1000 for m in modes[1:]]
 lines.append(label+' | '+' | '.join(f'{v:.4f}' for v in vals))
lines+=['',
'阶段发现：',
'1. MTE路由专家主链比off增加约1.755ms/调用；combine增加约1.029ms，dispatch整段增加约0.517ms，是当前采样的主要增量。',
'2. UDMA比off增加约7.060ms/调用；dispatch整段增加约6.256ms。UDMA dispatch内核约6.959ms，MTE约1.244ms；UDMA的combine反而比MTE更快，不能把问题笼统归为全部UDMA阶段。',
'3. 两种Slot在采样的1333/1333次调用中，prefetch均在dispatch结束前完成；未观察到暴露在dispatch之后的预取尾巴。这不排除其与其他内核竞争带宽，需要消融才能测得竞争成本。',
'4. 相比旧Slot BF16，MXFP8减少了激活payload，但off本来就传输量化激活。这次Slot额外pack约0.210ms、unpack约1.05ms，格式整理和同步成本仍在，不能因量化就预期必然超过off。',
'以本模型H=4096计，每行MXFP8激活4096字节，加group32 scale128字节，共4224字节，是BF16 8192字节的51.5625%（未计协议和路由重复）。S=8192、K=6的buffer有49152行，完整unpack约读取198MiB wire并写出198MiB activation/scale，合计约396MiB显存流量。该数是代码几何估算，不是链路流量实测。',
'',
'原仓代码依据（只读分析；未在这轮计时过程中修改生产代码）：',
str(r.parents[2].parent/'ascend-moonep/ascend_moonep/mxfp8_transport.py'),
'dispatch调用顺序：planning→启动副流prefetch→对有效hidden量化一次→pack→launch_dispatch→unpack→专家GEMM。unpack按self.rows=ctx[NvS]整容量输出，额外读写应作为后续融合候选。',
str(r.parents[2].parent/'ascend-moonep/kernels/moonep_dispatch_udma.cpp'),
'约794-935行：按peer和chunk筛选primary route，逐route产生WQE并成批提交，peer之间还有credit/completion同步。这证明内核包含准备、提交和协议等待；目前尚未通过内核内部计时区分每项，不能把6.959ms全部解释为网络带宽瓶颈。',
str(r.parents[2].parent/'ascend-moonep/kernels/moonep_combine.cpp'),
'MTE combine读取发布在SHMEM上的专家输出，结合route权重进行FP32乘加，最后写BF16；这条路径与native AlltoAll的bulk传输及finalize实现不同。此次未恢复外部BF16加权矩阵。',
'',
'逐rank负载诊断（完整503条另跑；诊断计时不纳入正式性能）',
'定义A：全程跨层。先把每张卡所有层/步负载相加，再取max/mean。',
'定义B：每层全程。对每层分别累计16张卡负载，再算max/mean；表中层均值是43层比值的算术平均。',
'两者均不是逐step/窗口峰均比；不把不同行为的rank本地step序号硬凑成一个全局step。',
'owner表示按原专家归属累计的路由负载；compute表示实际在本rank计算的专家行数，包含Slot容量padding。',
'模式 | 全程owner MTM | 全程compute MTM | 各层owner MTM均值/范围 | 各层compute MTM均值/范围']
rankdata={}
for mode in modes:
 p=r/f'data/diagnostics/optimized_rankload_20261106/{mode}/rank_summary.json'
 if not p.exists():
  lines.append(mode+' | 尚未完成 | 尚未完成 | 尚未完成 | 尚未完成')
  continue
 s=json.loads(p.read_text());rankdata[mode]=s
 own=[x['routing_owner_peak_mean'] for x in s['per_layer']]
 comp=[x['compute_peak_mean'] for x in s['per_layer']]
 lines.append(f"{mode} | {s['routing_owner_peak_mean']:.6f} | {s['compute_peak_mean']:.6f} | {statistics.mean(own):.6f} / {min(own):.6f}..{max(own):.6f} | {statistics.mean(comp):.6f} / {min(comp):.6f}..{max(comp):.6f}")
lines+=['',
'off不同层的热点rank不同：跨层累计1.028不代表每层平衡。off各层累计MTM均值1.247、范围1.119..1.504，确有可均衡空间。',
'用真实数字举例：off第0层，把该诊断记录内该层每卡工作量累计，rank3最大为24,009,553对；16卡平均16,136,305.875对，峰均比为1.487921。',
'如果连43层也合并，则最忙变为rank1，总工作量713,558,847对，16卡平均693,861,152.625对，得到1.028389。后者合并了层间不同的热点，因此会更接近1。',
'不能仅凭跨层1.028就得出“最多只能获得2.8%收益”。层与步的关键路径不同，均衡收益还取决于实际调度与padding成本。',
'MTE完整计数：各层owner MTM均值1.246928698，而实际compute MTM均值1.000013237，范围1.000008141..1.000021961。总routing与compute均为11,102,045,214对，expert级最大差值0，warnings为空。此次确有几乎完全的按层累计均衡，性能仍下降，不能简单归因于负载没有分均匀。',
'UDMA完整计数：各层owner MTM均值1.246930798，实际compute MTM均值1.000011890，范围1.000005988..1.000023154。总routing与compute均为11,102,045,730对，每层每个逻辑专家的计数均一致，warnings为空。两后端均实现了几乎完全的按层累计均衡，额外开销仍使正式端到端性能劣化。',
'这些累计均衡结果也不能证明每个全局同步step都达到同样比例；需要具备全局step标识的额外统计才能作该判断。',
'原始计数包含服务实际执行的少量dummy工作。off总计11,101,778,442 token-expert pairs，比503输入×6×43多189,114 pairs（733 token-equivalent，约0.0017%）；未任意减除。',
'末步通过独立16条短控制请求触发落盘，并按控制前记录的每rank边界封存；这16条控制不进入负载统计和正式数据。',
'Slot compute中容量padding和丢弃可能使expert级compute与routing不相等；下方逐模式保留原始总数、差值与元数据，不把差值直接解释为丢token或精度错误。',
'']
for mode,s in rankdata.items():
 lines+=['模式 '+mode,
 f"total routing pairs={s['total_routing_pairs']}; total compute pairs={s['total_compute_pairs']}; difference={s['total_compute_pairs']-s['total_routing_pairs']}; expert max abs difference={s['expert_pair_max_abs_delta']}",
 'warnings='+json.dumps(s['warnings']),
 'rank0..15 routing owner pairs='+json.dumps(s['routing_owner_pairs']),
 'rank0..15 actual compute pairs='+json.dumps(s['compute_pairs']),
 '']
lines+=['后续优化候选（本次只记录，未混入正式结果）：',
'优先定位UDMA dispatch的WQE构造/提交、peer credit等待与实际传输占比；不要凭总内核时长直接改QP或移除同步。',
'两后端均考虑把MXFP8 receive/unpack融合进dispatch或GEMM输入准备，减少约1.05ms的独立解包读写；需要另做正确性验证。',
'继续优化weighted combine的远端读取与片上流水，保留FP32乘路由系数、FP32累加及最终一次BF16舍入。',
'',
'证据索引：',
'正式原始性能和debug路径：'+str(result/'PERFORMANCE.txt'),
'两机默认安装、库SHA与ABI：'+str(r/'LIBRARY_UPDATE.json'),
'阶段原始trace与逐调用表：'+str(profroot/'PF2_STAGE_BREAKDOWN'),
'负载原始封存与边界：'+str(r/'data/diagnostics/optimized_rankload_20261106'),
'状态：'+str(r/'STATE.json')]
(result/'ANALYSIS.txt').write_text('\n'.join(lines)+'\n')
(result/'rank_load_comparison.json').write_text(json.dumps(rankdata,indent=2))
print('ANALYSIS_WRITTEN',len(rankdata),'rankload modes complete')
