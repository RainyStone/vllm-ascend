#!/usr/bin/env python3
import csv,hashlib,json,re,shutil,socket,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parent
run=Path(json.loads((D/'CURRENT_DIAGNOSIS.json').read_text())['execution'])
assert json.loads((run/'state.json').read_text())['event']=='all_diagnostics_complete'
cases={m:json.loads((run/m/'case_result.json').read_text()) for m in ['off','slot']}
assert all(c['valid'] and c['census_requests']==316 and c['precision_requests']==48 for c in cases.values())
loads={m:json.loads((run/m/'load_analysis/SUMMARY.json').read_text()) for m in cases}
assert loads['slot']['count_conservation_ok']
assert all(v['incomplete_step_layer_keys_excluded']==0 for v in loads.values())
reduction=json.loads((run/'weighted_reduce_sum1/RESULT.json').read_text())
assert all(x['native_vs_repeat']['mismatched_elements']==x['native_vs_sum_then_round']['mismatched_elements']==0 for x in reduction['results'])
rank_rows=[]
for mode in cases:
    with (run/mode/'load_analysis/step_layer_rank_load.csv').open() as f:
        rows=[x for x in csv.DictReader(f) if int(x['routed_pairs'])>768]
    weights=np.array([int(x['routed_pairs']) for x in rows])
    for rank in range(8):
        a=np.array([float(x[f'owner_rank{rank}_normalized_load']) for x in rows])
        b=np.array([float(x[f'compute_rank{rank}_normalized_load']) for x in rows])
        rank_rows.append(dict(mode=mode,rank=rank,step_layers=len(rows),
          owner_normalized_weighted_mean=float(np.average(a,weights=weights)),
          owner_normalized_p90=float(np.percentile(a,90)),owner_normalized_max=float(a.max()),
          compute_normalized_weighted_mean=float(np.average(b,weights=weights)),
          compute_normalized_p90=float(np.percentile(b,90)),compute_normalized_max=float(b.max())))
with (run/'RANK_SUMMARY.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rank_rows[0]));w.writeheader();w.writerows(rank_rows)
prec={}
for mode in cases:
    records=[json.loads(line) for line in (run/mode/'precision/responses.jsonl').open()]
    assert len(records)==48 and all(x['usage']['completion_tokens']==1 for x in records)
    prec[mode]={(x['id'],x['phase'],x['repeat']):x for x in records}
assert set(prec['off'])==set(prec['slot'])
comparisons=[]
for sample_id in sorted({k[0] for k in prec['off']}):
    item=dict(id=sample_id)
    for mode in cases:
        rows=[x for k,x in prec[mode].items() if k[0]==sample_id]
        predictions=[x['prediction'] for x in rows]
        top=sorted(rows[0]['top_logprobs'][0].items(),key=lambda x:x[1],reverse=True)
        item[mode]=dict(all_six_stable=len(set(predictions))==1,prediction=predictions[0],
                       top_two=top[:2],top_two_margin=top[0][1]-top[1][1])
    item['cross_mode_equal']=item['off']['prediction']==item['slot']['prediction']
    comparisons.append(item)
precision=dict(samples=8,requests_per_mode=48,selected_not_random_sample=True,
    all_mode_repeats_stable=all(x[m]['all_six_stable'] for x in comparisons for m in cases),
    matching_samples=sum(x['cross_mode_equal'] for x in comparisons),comparisons=comparisons,
    full_task_accuracy_not_measured=True)
(run/'PRECISION_COMPARISON.json').write_text(json.dumps(precision,indent=2,ensure_ascii=False)+'\n')
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert set(re.findall(r'No running processes found in NPU (\d+)',npu))==set(map(str,range(8)))
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',8444))
cleanup=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),eight_npus_released=True,port8444_bindable=True)
(run/'CLEANUP_CHECK.json').write_text(json.dumps(cleanup,indent=2)+'\n');(run/'npu_final.txt').write_text(npu)
byte=json.loads((run/'communication_byte_accounting.json').read_text())
before=loads['slot']['main_prefill']['rank_imbalance_before']['weighted_mean']
after=loads['slot']['main_prefill']['rank_imbalance_after']['weighted_mean']
result=dict(execution=str(run),diagnostic_only=True,formal_performance_results_unchanged=True,
    census_requests=632,precision_requests=96,marker_requests=6,cleanup=cleanup,
    off_main_prefill=loads['off']['main_prefill'],slot_main_prefill=loads['slot']['main_prefill'],
    slot_expert_count_conservation=True,critical_rank_work_reduction_linear_model_pct=100*(1-after/before),
    precision=precision,weighted_reduce=reduction,communication_payload_bound=byte)
(run/'DIAGNOSIS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
p=run/'REPORT.md';text=p.read_text()
text=text.replace('共48次。每模式内','共48次；固定集合阶段并发8，316条负载补采并发16。每模式内')
p.write_text(text+'\n最终资源释放检查时间：'+cleanup['at']+'。\n')
out=run/'result_export';out.mkdir(exist_ok=True)
for name in ['REPORT.md','DIAGNOSIS.json','RANK_SUMMARY.csv','PRECISION_COMPARISON.json','communication_byte_accounting.json','CLEANUP_CHECK.json','npu_final.txt','events.jsonl']:
    shutil.copy2(run/name,out/name)
for mode in cases:
    case=run/mode;dest=out/mode;dest.mkdir(exist_ok=True)
    for name in ['case_result.json','phase_boundaries.json']:shutil.copy2(case/name,dest/name)
    for name in ['SUMMARY.json','step_layer_rank_load.csv']:shutil.copy2(case/'load_analysis'/name,dest/name)
    for name in ['VALIDATION.json','responses.jsonl']:shutil.copy2(case/'precision'/name,dest/name)
    for source in case.glob('*_mode_audit.json'):shutil.copy2(source,dest/source.name)
    for source in case.glob('*profile*.log'):shutil.copy2(source,dest/source.name)
    service=Path(cases[mode]['service_dir']);client=Path(cases[mode]['client_dir'])
    pairs=[(service/'start_script.sh','start_script.sh'),(client/'bench_script.sh','bench_script.sh'),
           (client/'aisbench_config.py','aisbench_config.py'),(client/'round_1/validation.json','census_validation.json'),
           (client/'round_1/aisbench_debug.log','aisbench_debug.log')]
    for source,name in pairs:shutil.copy2(source,dest/name)
    index=dict(service_dir=str(service),client_dir=str(client),routing_capture=str(case/'routing_capture'),
               precision=str(case/'precision'),raw_test_tensors=str(run/'weighted_reduce_sum1'))
    (dest/'RAW_DATA_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
for folder in ['weighted_reduce','weighted_reduce_sum1']:
    dest=out/folder;dest.mkdir(exist_ok=True);shutil.copy2(run/folder/'RESULT.json',dest/'RESULT.json')
scripts=out/'scripts';scripts.mkdir(exist_ok=True)
for pattern in ['*.py','*.sh']:
    for source in D.glob(pattern):shutil.copy2(source,scripts/source.name)
for name in ['APPROVAL.json','DIAGNOSTIC_REVIEW.md']:shutil.copy2(D/name,out/name)
ws=ROOT.parents[3];dest=out/'source_evidence';dest.mkdir(exist_ok=True)
sources=['vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py',
 'vllm-ascend/vllm_ascend/ops/fused_moe/moonep_balance.py','vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py',
 'vllm-ascend/vllm_ascend/ops/fused_moe/moe_stage_params.py','vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py',
 'vllm-ascend/vllm_ascend/quantization/fp8_config.py','vllm-ascend/vllm_ascend/quantization/methods/fp8.py',
 'vllm-ascend/vllm_ascend/quantization/methods/w4a8_mxfp4.py','ascend-moonep/ascend_moonep/api.py',
 'ascend-moonep/kernels/moonep_combine.cpp']
for source in sources:shutil.copy2(ws/source,dest/Path(source).name)
hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='EXPORT_SHA256.json'}
(out/'EXPORT_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
(D/'CURRENT_DIAGNOSIS.json').write_text(json.dumps(dict(execution=str(run),status='completed',report=str(run/'REPORT.md')),indent=2)+'\n')
with (ROOT/'EXPERIMENT.md').open('a') as f:
    f.write('\n\n## 补充诊断完成\n\n用户已批准并完成off/Slot各316条负载补采和48次固定请求，诊断计时未并入正式性能。报告：'+str(run/'REPORT.md')+'。\n')
print(json.dumps(dict(export_dir=str(out),files=len(hashes),cleanup=cleanup,
    slot_rank_before=before,slot_rank_after=after,precision_matching_samples=precision['matching_samples'],
    tensor_payload_ratio_lower_bound=byte['ratio_lower_bound_tensor_payload']),ensure_ascii=False))
