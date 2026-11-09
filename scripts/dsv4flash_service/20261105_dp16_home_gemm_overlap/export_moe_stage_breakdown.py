#!/usr/bin/env python3
"""Export measured MoE phase comparisons, preserving disjoint boundaries and overlap caveats."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_profiling_v2_20261106/MOE_STAGE_BREAKDOWN'
d=json.loads((b/'summary.json').read_text())['data'];shared=json.loads((b/'shared_experts.json').read_text())['data']
modes=('off','slot_mte','slot_udma')
def mean(m,key):return d[m]['stages'][key]['mean_us']/1000
def sh(m,key):return shared[m]['stages'][key]['mean_us']/1000
def route(m):return sum(mean(m,k) for k in ('router_gemm_us','router_to_gate_gap_us','topk_us'))
definitions=[
 ('Router GEMM + TopK','Router + TopK',lambda m:route(m)),
 ('分发完整阶段','Dispatch incl. preparation',lambda m:mean(m,'dispatch_inclusive_us')),
 ('专家GEMM1（Slot为Home+Slot）','Expert GEMM1',lambda m:mean(m,'gemm1_span_us')),
 ('专家激活与量化','Expert activation + quant',lambda m:mean(m,'activation_quant_span_us')),
 ('专家GEMM2','Expert GEMM2',lambda m:mean(m,'gemm2_us')),
 ('归并完整阶段','Combine incl. preparation',lambda m:mean(m,'combine_inclusive_us')),
 ('共享专家与输出相加','Shared experts + merge',lambda m:sh(m,'combine_end_to_merge_end_us'))]
values={m:{label:fn(m) for label,_,fn in definitions} for m in modes}
totals={m:mean(m,'router_through_combine_wall_us')+sh(m,'combine_end_to_merge_end_us') for m in modes}
for m in modes:assert abs(sum(values[m].values())-totals[m])<1e-7
text=['MoE各阶段耗时：off / Slot-MTE / Slot-UDMA','',
'仅离线解析已有profiling，无新模型、无新NPU请求。',
'统计口径：每rank最先捕获的2个prefill chunk，每chunk43个MoE层；16rank x2x43=1376条rank-层记录/模式。',
'单位ms；对这些rank-层执行记录取均值。不是16卡耗时之和，不是一次请求TTFT。',
'该prefill子集off为原生AlltoAll路径；三组使用相同的按rank固定输入。',
'主流区间不重叠且逐条通过加和核对；起点Router GEMM开始，终点共享专家与路由专家输出相加结束。',
'未纳入此前的RMSNorm、输入cast及之后的HyperConnections/Attention；预取副流单列，不再加到主流合计。','',
'阶段 | off | Slot-MTE | Slot-UDMA | MTE-off | UDMA-off']
for label,_,fn in definitions:
 a,bv,c=[fn(m) for m in modes]
 text.append(f'{label} | {a:.6f} | {bv:.6f} | {c:.6f} | {bv-a:+.6f} | {c-a:+.6f}')
text.append(f"所列MoE区间合计 | {totals['off']:.6f} | {totals['slot_mte']:.6f} | {totals['slot_udma']:.6f} | {totals['slot_mte']-totals['off']:+.6f} | {totals['slot_udma']-totals['off']:+.6f}")
text += ['','分发完整阶段边界：TopK结束 -> 专家GEMM1开始。',
'off包含计数/分组信息交换、CPU计数处理间隔、两次排列/重排、输入量化、scale和token AlltoAll及同步等待。',
'Slot包含输入/路由准备、planning、dispatch内核、结果复制/计数整理、输入量化。',
'之前2.014ms/6.798ms的数据是Slot dispatch内核时间；此次4.140ms/8.920ms是完整分发阶段。',
'归并完整阶段边界：专家GEMM2结束 -> 路由专家最终输出可用。',
'off包含本地逆排列 -> 输出AlltoAll及等待 -> 最终加权逆排列。',
'Slot包含尾部掩码/对称缓冲复制等准备 -> fused weighted combine内核。','',
'MoonEP细分（下列为上表子项，不能再与上表相加）','阶段 | Slot-MTE | Slot-UDMA']
detail=[
 ('分发前输入/路由准备','dispatch_prepare_us'),('planning内核','planning_kernel_us'),
 ('planning到dispatch间隔','planning_to_dispatch_gap_us'),('dispatch内核','dispatch_kernel_us'),
 ('dispatch后至Home前全部处理','dispatch_post_quant_span_us'),('其中TensorMove','post_dispatch_tensor_move_us'),
 ('其中输入量化','input_quant_kernel_us'),('Home GEMM1','home_gemm1_us'),('Slot GEMM1','slot_gemm1_us'),
 ('Home到Slot接续间隔','home_slot_gap_us'),('combine前准备','combine_prepare_us'),('weighted combine内核','combine_kernel_us')]
for label,key in detail:text.append(f"{label} | {mean('slot_mte',key):.6f} | {mean('slot_udma',key):.6f}")
text += ['','并行预取（四份权重/scale数据的整体时间跨度，不加入主流合计）',
 f"MTE={mean('slot_mte','prefetch_span_us'):.6f}ms; UDMA={mean('slot_udma','prefetch_span_us'):.6f}ms。",
'该对齐子集每次prefetch均在dispatch结束前完成，Home之后没有剩余prefetch等待。','',
'off原生路径的可直接识别子项（不把内核/等待嵌套相加为网络带宽成本）：']
for key in ('dispatch_histogram_kernel_us','dispatch_permute_kernel_us','input_quant_kernel_us','combine_local_unpermute_us','combine_collective_span_us','combine_final_unpermute_us'):
 text.append(key+'='+f"{mean('off',key):.6f}ms")
text += ['','共享专家细分（本次trace中位于routed combine后、同一主流，所有记录均验证算子顺序）：','阶段 | off | Slot-MTE | Slot-UDMA']
for key in ('shared_input_quant_us','shared_gemm1_us','shared_activation_quant_us','shared_gemm2_us','merge_add_kernel_us'):
 text.append(key+' | '+' | '.join(f'{sh(m,key):.6f}' for m in modes))
text += ['','劣化位置（所列MoE区间内的额外时间，不是整轮TTFT增加的百分比）：']
for mode in modes[1:]:
 delta=totals[mode]-totals['off']
 text.append(mode+f' 总增量{delta:.6f}ms')
 for key in ('分发完整阶段','归并完整阶段','专家GEMM1（Slot为Home+Slot）','专家激活与量化','专家GEMM2'):
  change=values[mode][key]-values['off'][key]
  text.append(f'  {key}: {change:+.6f}ms，占本采样区间额外时间{100*change/delta:.2f}%')
text += ['','解释边界：',
'1. MTE的最大增加项在combine，其次完整分发、GEMM1；UDMA的最大增加项在完整分发，combine次之。',
'2. UDMA的combine比MTE更快，但dispatch慢得更多。不能概括为UDMA所有阶段都更慢。',
'3. planning/dispatch/combine内核时长含跨rank等待，阶段落点不等于纯网络传输或纯FP32加权算术成本。',
'4. Slot GEMM1两段合计的增加，同时受范围内核tiling、专家分组/负载变化影响，不能全部归因于多一次kernel launch。',
'5. 输入/激活量化和combine准备等实际设备工作已明确计入；未记录设备任务的空隙不能直接判作CPU空转。',
'6. 这是独立诊断固定子集；每rank均值不是迭代最慢rank，也不能用来精确分摊完整503条冷测TTFT。',
'7. 原始逐条数据见per_call.json，均值/P50/P95等见summary.json，共享专家见shared_experts.json。']
(b/'MOE_STAGE_BREAKDOWN.txt').write_text('\n'.join(text)+'\n')
(b/'comparison.json').write_text(json.dumps(dict(units='ms',cohort_calls_per_mode=1376,stage_means=values,interval_totals=totals),ensure_ascii=False,indent=2))
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
bottom=np.zeros(3);colors=['#939da8','#4e79a7','#f28e2b','#edc949','#b36d6d','#59a14f','#b07aa1']
for (_,label,fn),color in zip(definitions,colors):
 a=np.array([fn(m) for m in modes]);ax.bar(modes,a,bottom=bottom,label=label,color=color);bottom+=a
for i,v in enumerate(bottom):ax.text(i,v+.12,f'{v:.3f} ms',ha='center',fontsize=10)
ax.set_ylim(0,max(bottom)*1.12);ax.set_ylabel('Mean local device interval (ms)')
ax.set_title('MoE stage breakdown / matched prefill cohort\n16 ranks x 2 chunks x 43 MoE layers; parallel prefetch excluded',fontsize=11)
ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(b/'moe_stage_breakdown.png',dpi=180);fig.savefig(b/'moe_stage_breakdown.svg')
print('MOE_BREAKDOWN_EXPORTED',b,flush=True)
