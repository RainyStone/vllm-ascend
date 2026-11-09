#!/usr/bin/env python3
"""Generate final review only from completed formal/diagnostic evidence."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;state=json.loads((r/'STATE.json').read_text())
formal=r/'data/formal/home_formal_20261106_after_quiescent_gate'
prof=r/'data/diagnostics/home_profiling_v2_20261106'
diag=r/'data/diagnostics/home_rankload_20261106'
modes=('off','slot_mte','slot_udma');names={'off':'off','slot_mte':'Slot-MTE + Home GEMM1','slot_udma':'Slot-UDMA + Home GEMM1'}
rows=[];loads={}
for m in modes:
 assert all(state['formal'][m].get(k) for k in ('benchmark_passed','post_audit_passed','cleanup_completed'))
 f=json.loads((formal/m/'round_summary.json').read_text());assert f['requests']==503 and f['input_tokens']==43029416 and f['output_tokens']==503
 rows.append(f)
 d=json.loads((diag/m/'RANK_LOAD.json').read_text());assert d['world']==16 and d['layer_count']==43 and not d['warnings']
 assert d['expert_pair_max_abs_delta']==0 and max(d['per_layer_expert_pair_max_abs_delta'].values())==0
 seal=json.loads((diag/m/'sealed/SEAL.json').read_text());assert seal['valid'] and len(seal['ranks'])==16
 audit=json.loads((diag/m/'post_audit/cluster_mode_audit_post.json').read_text());assert audit['valid']
 for node in ('118','124'):
  owner=json.loads((diag/m/f'node{node}_ownership.json').read_text());assert owner['cleanup_completed'] and not owner['remaining_owned']
 loads[m]=d
base=rows[0]
out=['Home GEMM1跨机DP16实现、复测及劣化分析','',
'结果：已在原仓完成MTE/UDMA适配与低开销优化，三组正式503条全部成功；本轮两种Slot仍劣于off。',
'方案：off -> Slot-MTE -> Slot-UDMA；各一次独立起服、一次503条、无预热。所有诊断计时均排除。','',
'正式性能（TTFT越低越好，输入吞吐越高越好）',
'模式 | 平均TTFT(ms) | 相对off | P95 TTFT(ms) | 输入token/s | 相对off | 成功']
for f in rows:
 out.append(f"{names[f['mode']]} | {f['ttft_ms']['mean']:.2f} | {(f['ttft_ms']['mean']/base['ttft_ms']['mean']-1)*100:+.2f}% | {f['ttft_ms']['p95']:.2f} | {f['input_tps']:.2f} | {(f['input_tps']/base['input_tps']-1)*100:+.2f}% | 503/503")
out += ['','每组输入43,029,416 token、输出503 token；逐请求prompt SHA/token数和输出1均已核验。',
'TTFT取官方AIS口径：每请求1000*(time_points[1]-time_points[0])，再对503请求作不加权平均。SQLite numpy时间点原始数组保留。',
'输入吞吐使用AIS原始统计，等于总输入token/Benchmark Duration。服务启动时间不计入该Duration。','',
'固定参数：DSV4 Flash原权重；DP16/TP1/EP16；118 API8445、124 headless；完整503条，仅context中间截断至完整prompt≤131072；',
'max_model_len131073、全局并发16、max_num_seqs16、输出1、8192 chunked prefill、24 GiB KV/卡、eager、prefix cache off、seed1024、dp_allreduce_on_npu=true。',
'Slot REAL_T=1，持久缓存和提前预取关闭；副流PF_STREAM=3，Home GEMM1掩盖开启。UDMA仅dispatch/combine/prefetch是UDMA，planning仍为MTE，QP[3,2,2]。',
'off保持原生vLLM-Ascend通信，没有开启MoonEP；三组前后16-worker实际配置、源文件SHA和运行库映射均通过审计。','',
'完整503诊断：全程负载峰均比',
'先分别累计每rank整个诊断范围的(token,expert)对数，再算16*max/总数。不累加每步峰值，也不把某张固定卡当作每步最忙卡。',
'逐层逐专家路由计数与计算计数已核对，三组均完全守恒；并非少算token获得均衡。',
'原始归属视角：按逻辑专家原owner归属。实际计算视角：按接收/实际执行的rank归属，含记录到的框架工作。',
'模式 | 原始归属全程峰均比 | 实际计算全程峰均比 | 逐层原始归属范围 | 逐层实际计算范围']
for m,d in loads.items():
 route=[x['routing_owner_peak_mean'] for x in d['per_layer']]
 comp=[x['compute_peak_mean'] for x in d['per_layer']]
 out.append(f"{names[m]} | {d['routing_owner_peak_mean']:.6f} | {d['compute_peak_mean']:.6f} | {min(route):.6f}..{max(route):.6f} | {min(comp):.6f}..{max(comp):.6f}")
out += ['','每rank完整计数（rank0..15）及总数：']
for m,d in loads.items():
 out += [names[m], 'routing_owner_pairs='+str(d['routing_owner_pairs']), 'compute_pairs='+str(d['compute_pairs']),
 f"route_total={d['total_routing_pairs']}; compute_total={d['total_compute_pairs']}; dataset_tokens*topk6*43layers={43029416*6*43}"]
out += ['','注意：数据集纯请求理论对数与运行时记录可因DP同步dummy等框架工作有小量差异；完整保留原计数，不假装全为原始文本token。',
'off每层比理论多7470对（1245个token等价），43层共321210对，约0.0029%；不对这部分做无法证实的逐rank扣除。',
'全程累计接近均衡不代表每层或每步均衡；逐层范围已另列。未验证16-rank逐步时间对齐，因此不伪造逐步P95峰均比。','',
'Profiling证据与解释（三组各16份、共48份逐rank原始trace，每组独立诊断16请求，非完整503）']
overlap=json.loads((prof/'ALL_RANK_OVERLAP.json').read_text())
for m in ('slot_mte','slot_udma'):
 a=overlap[m]['long_prefill'];out.append(f"{names[m]}: 捕获长输入MoE调用{a['calls']}次，{a['ready_before_dispatch_end']}次预取在dispatch结束前完成，Home实际预取重叠总量{a['prefetch_active_overlap_home_us']:.6f}us。")
out += ['1. 所有捕获长输入调用中，权重预取已由dispatch覆盖；把join推迟到Home之后没有额外剩余搬运可掩盖。',
'   全部调用（含短/同步dummy）存在少量非零重叠，不能把“长输入零重叠”推广为所有形状绝不可能重叠。',
'2. Home/Slot GEMM1额外拆分的两段合计时间高于原生off GMM1；原生大GMM的性能没有被本实现自动继承。',
'   这同时受范围内核tiling与搬移后group尺寸影响，现有三臂对比不能单独隔离Home特性的净开销。',
'3. UDMA长输入dispatch耗时显著高于MTE，而预取本身更短；UDMA的额外劣化主要线索位于dispatch及同步路径。',
'   内核总时长含跨rank等待，不能把整段当纯网络搬运，也不能把各rank/stream时长相加当墙钟占比。',
'补充rank0时间线：dispatch结束到Home开始约447us，其中TensorMove约249us、DynamicMxQuant约188us；主要是已记录的设备复制/量化，不应称为CPU空转或等待预取。',
'4. 原生AlltoAll在发送前做MXFP8量化；Slot发送BF16 hidden_state、在接收后量化。H=4096时，每逻辑行主数据+scale约4224B对8192B（约1.94倍）。',
'   这只是行载荷理论大小，不等于实测链路总流量倍数；去重、本地传输、布局与padding会影响总流量。两者专家权重均保留FP4/e8m0；off专家权重本地常驻，Slot额外拉取计划指定的远端专家。BF16 carrier是原量化字节的视图，不是把FP4权重转成BF16数值传输。',
'   Slot还增加规划和按层专家权重搬运；完整负载累计较均衡时，其收益空间更有限，但不能仅凭累计峰均比断言每一步都均衡。','',
'对齐profiling子集：每rank最先捕获的2个prefill chunk，每chunk43个MoE层（每组1376个MoE调用），均值us：']
matched=json.loads((prof/'MATCHED_MLP_SUBSET.json').read_text())
for m in modes:out.append(names[m]+': '+json.dumps(matched[m],ensure_ascii=False))
out += ['上述子集只有少量固定请求；并未精确分摊正式TTFT增加的各项百分比。','',
'完成的优化与正确性验证：见OPTIMIZATIONS.txt、CODE_GUIDE.txt、HOME_GMM_CHANGES.patch。',
'35项CPU测试；单卡59条数值记录无mismatch；MTE/UDMA各自2-rank跨机与16-rank完整链路通过，远端权重读回与参考一致。',
'FP32路由乘权重并累加、最终一次BF16舍入保持；语义模型精度按用户要求暂不复测。',
'库在原仓构建/原子安装，未建立SDK副本。没有终止其他或不明归属进程；有效正式与本轮诊断服务均已正常退出，失败启动独立记录。','',
'局限：单次冷启动快速探索，没有重复置信区间；没有同版PF_STREAM=2对照，不能用旧实验估计Home改动的净收益。',
'后续优先研究范围GMM tiling和UDMA dispatch等待；激活量化通信格式/路由策略/大显存缓存变更不在本次已授权优化中。','',
'完整路径：',str(r),str(formal),str(prof),str(diag),
'正式原始AIS stdout/stderr、JSONL和SQLite：'+str(r/'data/aisbench'),
'原始profiling：'+str(r/'diagnostics/profiling/data/profiling_v2'),
'失败/中止启动与旧profiling v1记录独立保留，不纳入正式三组结果。']
(r/'RESULTS.txt').write_text('\n'.join(out)+'\n')
(r/'FINAL_DATA.json').write_text(json.dumps(dict(formal=rows,rank_load=loads,profile_overlap=overlap,matched_mlp=matched),indent=2))
print('RESULTS_EXPORTED',r/'RESULTS.txt')
