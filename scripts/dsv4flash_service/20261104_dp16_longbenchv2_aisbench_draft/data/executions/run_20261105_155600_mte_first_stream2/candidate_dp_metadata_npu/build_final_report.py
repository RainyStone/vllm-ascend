#!/usr/bin/env python3
"""Offline five-configuration report; requires validated results for every case."""
import csv,json,time
from pathlib import Path
ROOT=Path('/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/data/executions/run_20261105_155600_mte_first_stream2/candidate_dp_metadata_npu')
SPECS=[
 ('off','off',None,'OFF_RESULT.json'),
 ('slot_mte_stream2','slot_mte',2,'SLOT_MTE_RESULT.json'),
 ('slot_udma_stream2','slot_udma',2,'SLOT_UDMA_RESULT.json'),
 ('slot_mte_stream0','slot_mte',0,'no_overlap/SLOT_MTE_RESULT.json'),
 ('slot_udma_stream0','slot_udma',0,'no_overlap/SLOT_UDMA_RESULT.json'),
]
cases={}
for key,mode,pf,name in SPECS:
    data=json.loads((ROOT/name).read_text())
    folder=next(p for p in Path(data['raw_jsonl']).parents if p.name=='round_1')
    validation=json.loads((folder/'validation.json').read_text())
    assert validation['valid'] and validation['requests']==503
    pre=json.loads((folder/'cluster_mode_audit.json').read_text())
    post=json.loads((folder/'cluster_mode_audit_post.json').read_text())
    assert pre['valid'] and post['valid'] and pre['mode']==post['mode']==mode
    assert sorted(r for node in post['nodes'] for r in node['global_ranks'])==list(range(16))
    for a,b in zip(pre['nodes'],post['nodes']):
        assert a['node']==b['node'] and a['workers']==b['workers'] and a['service_pid']==b['service_pid']
        assert a['dp_allreduce_on_npu'] and b['dp_allreduce_on_npu']
        env=dict(s.split('=',1) for s in (Path(b['service_dir'])/'environment.txt').read_text().splitlines() if '=' in s)
        if pf is not None:
            assert env['MOONEP_PF_STREAM']==str(pf)
            if pf==2:
                assert b['side_stream_activation_verified']
            else:
                assert b['prefetch_stream_mode']==0 and b['side_stream_disabled_verified'] and not b['side_stream_activation_verified']
        else:
            assert not any(k.startswith(('MOONEP','ASCEND_MOONEP')) for k in env)
            assert all(not w['libraries'] for w in b['workers'])
    assert data['requests']==503 and data['input_tokens']==43029416
    data.update(mode=mode,prefetch_stream_mode=pf,client_round=str(folder),audits_passed=True,
                dp_allreduce_on_npu=True,formal_rounds=1,benchmark_warmups=0)
    cases[key]=data
base=cases['off']
for key,data in cases.items():
    data['ttft_increase_pct_vs_off']=100*(data['mean_ttft_ms']/base['mean_ttft_ms']-1)
    data['input_tps_change_pct_vs_off']=100*(data['input_tps']/base['input_tps']-1)
overlap={}
for mode in ('slot_mte','slot_udma'):
    side=cases[mode+'_stream2'];main=cases[mode+'_stream0']
    overlap[mode]=dict(side_vs_main_ttft_reduction_pct=100*(1-side['mean_ttft_ms']/main['mean_ttft_ms']),
                       side_vs_main_input_tps_gain_pct=100*(side['input_tps']/main['input_tps']-1))
result=dict(updated_at=time.time(),cases=cases,overlap_comparisons=overlap,
            completed_configurations=5,formal_requests=2515,rounds_per_configuration=1,
            accuracy_evaluation='deferred by user',historical_six_service_matrix_complete=False,
            actual_execution_order=[s[0] for s in [SPECS[1],SPECS[0],SPECS[2],SPECS[3],SPECS[4]]],
            imbalance_measurement='Not collected in these timed rounds; no conclusion about rank imbalance is asserted.')
(ROOT/'RESULTS_BY_CONFIGURATION.json').write_text(json.dumps(result,indent=2))
with (ROOT/'RESULTS_BY_CONFIGURATION.csv').open('w',newline='') as stream:
    fields=['configuration','prefetch_stream_mode','requests','mean_ttft_ms','p90_ttft_ms','p99_ttft_ms','input_tps','ttft_increase_pct_vs_off','input_tps_change_pct_vs_off']
    w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
    for key,v in cases.items():w.writerow(dict(configuration=key,**{f:v[f] for f in fields if f!='configuration'}))
lines=['DP16 LongBench-v2：五种配置各一轮的探索性结果','',
'两机：141.61.45.118 / 141.61.45.124，均为 xzx_moonep；原仓源码与安装库。',
'DSV4 Flash；DP16/TP1/EP16；完整503条，context中间截断，完整prompt上限131072，输出1。',
'全局并发16；max-model-len=131073；max-num-batched-tokens=8192；KV24GiB/卡；chunked prefill；prefix cache关闭；eager。',
'全部采用 dp_allreduce_on_npu=true；每个配置一次独立起服，正式测一轮503条，AISBench不预热，保留perf+debug。',
'下表每行是独立配置，副流开启/关闭没有相互平均。off没有MoonEP预取副流。','',
'配置 | 副流模式 | TTFT均值(s) | 输入吞吐(token/s) | TTFT相对off | 吞吐相对off']
for key,v in cases.items():
    lines.append(f"{key} | {v['prefetch_stream_mode']} | {v['mean_ttft_ms']/1000:.4f} | {v['input_tps']:.4f} | {v['ttft_increase_pct_vs_off']:+.2f}% | {v['input_tps_change_pct_vs_off']:+.2f}%")
lines+=['','开启副流相对关闭副流（正值代表开启后改善）：']
for mode,v in overlap.items():lines.append(f"{mode}: TTFT下降 {v['side_vs_main_ttft_reduction_pct']:.2f}%，输入吞吐增加 {v['side_vs_main_input_tps_gain_pct']:.2f}%。")
lines+=['','验证与边界：',
'每组503成功、0失败、每条输出1；逐请求完整prompt SHA和token数核对通过。每轮总输入43029416 token。',
'两机16个真实worker启动/结束审计通过，worker PID保持一致；off没有MoonEP开关或映射库。',
'Slot-MTE实际planning/dispatch/combine/prefetch均为MTE；Slot-UDMA的planning为MTE，另外三阶段为UDMA，不能称为全阶段UDMA。',
'副流开启：PF_STREAM=2，16 rank均有实际启用日志；UDMA另有每机8份native队列查询(3,2,2)。日志交错的证据处理见QP_AUDIT_EVIDENCE.txt。',
'副流关闭：PF_STREAM=0，预取仍保留在主流；独立审计检查启动设置、无冲突worker exec设置、无副流安装/激活记录及SHA固定源码。没有把side_stream_activation_verified伪造为真。',
'关闭UDMA副流时，原代码同时关闭专用prefetch重叠队列布局；这属于该开关的正常行为，没有额外修改服务参数。',
'旧CPU元数据off基线、原失败Slot-MTE轮、历史未完成矩阵均不混入这五行。精度暂不评估。',
'各配置仅一轮、非交错重复的统计实验；小百分比差异不能据此断言稳定收益。','',
'错误分析：',
'原失败首先出现在Gloo CPU元数据同步的报文长度检查，随后其他rank连接重置。采用vllm-ascend已有NPU元数据同步开关后，这五轮完成。',
'这验证了规避方案的运行效果，未证明底层Gloo报文错误的唯一根因；该开关不是ascend-moonep新增能力，也未将MTE改成UDMA。','',
'性能分析：',
'OFFLINE_LENGTH_ANALYSIS.json按相同样本ID/输入长度分档；PER_REQUEST_TTFT.csv由原始SQLite时间点重算，均值必须与AISBench汇总相符。',
'源码中Slot发送BF16 hidden_states；原生All2AllV在FP8配置启用dispatch_with_quant时先量化再传输，并带scale。模型config声明动态FP8。Slot还需要规划与专家权重预取。',
'因此通信字节量、规划/权重搬运及副流资源竞争都是候选解释。这里只做源码与端到端对照，没有逐算子profiling，不能把其中任何一项宣称为已证实根因。',
'正式轮关闭逐层负载/计时诊断，没有本轮逐rank不均衡统计，不能沿用旧DP8数值或断言“不均衡比不足”。','',
'保存位置：',str(ROOT),'']
for key,v in cases.items():lines.extend([key, '  原始JSONL: '+v['raw_jsonl'], '  汇总: '+v['summary'], '  审计/调试/校验: '+v['client_round']])
(ROOT/'REPORT.txt').write_text('\n'.join(lines))
print(json.dumps({'cases':{k:{x:v[x] for x in ('mean_ttft_ms','input_tps','ttft_increase_pct_vs_off','input_tps_change_pct_vs_off')} for k,v in cases.items()},'overlap_comparisons':overlap}))
