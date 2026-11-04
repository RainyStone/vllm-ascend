#!/usr/bin/env python3
"""Export a compact, verifiable result package after all authorized cases finish."""
import csv,hashlib,json,re,shutil,socket,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
e=Path(sys.argv[1])
state=json.loads((e/'remaining_state.json').read_text())
assert state['event']=='all_remaining_complete',state
cases={}
for name in ['pair01_off','pair01_slot','pair02_slot','pair02_off']:
    result=json.loads((e/'cases'/name/'case_result.json').read_text())
    assert result['valid'] and result['formal_requests']==948
    cases[name]=result
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert set(re.findall(r'No running processes found in NPU (\d+)',npu))==set(map(str,range(8)))
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',8444))
cleanup=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             eight_npus_released=True,port_8444_bindable=True)
(e/'CLEANUP_CHECK.json').write_text(json.dumps(cleanup,indent=2)+'\n')
(e/'npu_final.txt').write_text(npu)
subprocess.run([sys.executable,str(ROOT/'summarize_results.py'),str(e)],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(ROOT/'analyze_request_metrics.py'),str(e)],check=True,stdout=subprocess.DEVNULL)
current=json.loads((ROOT/'CURRENT_EXECUTION.json').read_text())
current.update(status='completed',result=str(e/'RESULT.md'),cleanup=str(e/'CLEANUP_CHECK.json'))
(ROOT/'CURRENT_EXECUTION.json').write_text(json.dumps(current,indent=2)+'\n')
doc=ROOT/'EXPERIMENT.md';text=doc.read_text()
text=text.replace('修复已获用户确认，正在执行完整四服务矩阵。','修复已获用户确认，完整四服务、12轮、3792条正式请求均已完成并校验通过。')
text=text.replace('实验完成后在当前执行目录生成RESULT.md、results.json与rounds.csv。','已在当前执行目录生成RESULT.md、results.json与rounds.csv；八卡和8444端口释放检查见CLEANUP_CHECK.json。\n\n首token文本并非完全一致，同模式重复也有差异；本轮仅提供性能对照，不证明生成精度等价。')
doc.write_text(text)
out=e/'result_export';out.mkdir(exist_ok=True)
for name in ['RESULT.md','results.json','rounds.csv','CLEANUP_CHECK.json','npu_final.txt','selection_manifest.json','remaining_events.jsonl','request_metrics.csv','length_group_comparison.csv','LENGTH_ANALYSIS.md']:
    shutil.copy2(e/name,out/name)
for name in ['EXPERIMENT.md','PROVENANCE.json','CURRENT_EXECUTION.json','PROMPT_FIX_VALIDATION.json','PROMPT_FIX_REVIEW.md']:
    shutil.copy2(ROOT/name,out/name)
sources=['start_off.sh','start_slot.sh','bench_off.sh','bench_slot.sh','aisbench_off.py','aisbench_slot.py',
         'literal_longbench_prompt.py','audit_service_mode.py','validate_round.py','preflight_api.py',
         'summarize_results.py','run_confirmed_remainder.py','matrix_status.py','analyze_request_metrics.py','finalize_results.py']
scriptdir=out/'scripts';scriptdir.mkdir(exist_ok=True)
for name in sources:shutil.copy2(ROOT/name,scriptdir/name)
review=['# 实际运行脚本与配置','',
        '当前脚本的PAIR_LABEL为pair02。每次起服及压测时保存的原始快照另见case_evidence，pair01/pair02性能参数相同。',
        '脚本中的draft/候选注释属于初始草稿残留；SCRIPT_REVIEWED=1，24 GiB KV和131073长度已在本次四台服务验证。','']
for name in sources[:7]:
    language='bash' if name.endswith('.sh') else 'python'
    review += ['## '+name,'','```'+language,(ROOT/name).read_text().rstrip(),'```','']
(out/'ACTUAL_SCRIPTS.md').write_text('\n'.join(review)+'\n')
index=[]
for name,result in cases.items():
    target=out/'case_evidence'/name;target.mkdir(parents=True,exist_ok=True)
    for source in (e/'cases'/name).glob('*'):
        if source.suffix=='.json' or source.name.startswith('npu_'):
            shutil.copy2(source,target/source.name)
    service=Path(result['service_dir']);client=Path(result['client_dir'])
    for source,dest in [(service/'start_script.sh','start_script.sh'),(service/'MODE_AUDIT.json','MODE_AUDIT.json'),
                        (client/'bench_script.sh','bench_script.sh'),(client/'aisbench_config.py','aisbench_config.py')]:
        shutil.copy2(source,target/dest)
    for i in [1,2,3]:
        round_dir=client/f'round_{i}'
        round_target=target/f'round_{i}';round_target.mkdir(exist_ok=True)
        for source in round_dir.glob('*.json'):shutil.copy2(source,round_target/source.name)
        logs=list(round_dir.rglob('aisbench_debug.log'))
        assert len(logs)==1,(round_dir,logs)
        raws=list(round_dir.rglob('longbenchv2-full-le128k_details.jsonl'))
        assert len(raws)==1,round_dir
        summary=raws[0].with_name('longbenchv2-full-le128k.json')
        assert summary.exists()
        for source in [summary,summary.with_suffix('.csv')]:shutil.copy2(source,round_target/source.name)
        index.append(dict(case=name,round=i,debug_log=str(logs[0]),debug_bytes=logs[0].stat().st_size,
                          raw_jsonl=str(raws[0]),sqlite_directory=str(raws[0].parent/'db_data'),
                          round_directory=str(round_dir),service_log=str(service/'service.log')))
with (out/'RAW_LOG_INDEX.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
manifest={}
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name!='EXPORT_SHA256.json':manifest[str(p.relative_to(out))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'EXPORT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(export_dir=str(out),files=len(manifest),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                     cleanup=cleanup),ensure_ascii=False))
