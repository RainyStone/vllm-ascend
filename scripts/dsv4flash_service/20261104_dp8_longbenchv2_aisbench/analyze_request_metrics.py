#!/usr/bin/env python3
"""Read aisbench's raw SQLite timestamps, excluding round1 from comparisons."""
import csv,io,json,sqlite3,sys
from pathlib import Path
import numpy as np
from summarize_results import read_round
e=Path(sys.argv[1]);all_rows=[]
for name in ['pair01_off','pair01_slot','pair02_slot','pair02_off']:
    c=json.loads((e/'cases'/name/'case_result.json').read_text());assert c['valid']
    for i in [1,2,3]:
        metric=read_round(c['client_dir'],i);raw=Path(metric['raw_file']);connections={};round_rows=[]
        try:
            with raw.open() as f:
                for line in f:
                    r=json.loads(line);assert r['success'] and r['output_tokens']==1
                    db=raw.parent/'db_data'/r['db_name']
                    if db not in connections:connections[db]=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
                    blob=connections[db].execute('SELECT arr_blob FROM numpy_store WHERE id=?',(r['time_points']['__db_ref__'],)).fetchone()[0]
                    t=np.load(io.BytesIO(blob),allow_pickle=False)
                    assert len(t)==3 and np.all(np.diff(t)>=0),(name,i,r['id'])
                    row=dict(case=name,pair=name[:6],mode=name.split('_')[1],round=i,
                             included_in_comparison=i in [2,3],id=int(r['id']),input_tokens=r['input_tokens'],
                             ttft_ms=float(t[1]-t[0])*1000,e2e_ms=float(t[2]-t[0])*1000,
                             first_token_text=r.get('prediction',''))
                    round_rows.append(row)
        finally:
            for con in connections.values():con.close()
        assert len(round_rows)==316 and len({r['id'] for r in round_rows})==316
        # aisbench reports TTFT Average rounded to 0.1 ms; half-unit tolerance.
        assert abs(np.mean([r['ttft_ms'] for r in round_rows])-metric['mean_ttft_ms'])<0.051
        all_rows+=round_rows
with (e/'request_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
boundaries=[(0,32768),(32768,65536),(65536,98304),(98304,131072)]
groups=[]
for lo,hi in boundaries:
    result=dict(input_range=f'({lo},{hi}]',samples=None)
    for mode in ['off','slot']:
        rows=[r for r in all_rows if r['included_in_comparison'] and r['mode']==mode and lo<r['input_tokens']<=hi]
        ids={r['id'] for r in rows};assert len(rows)==4*len(ids)
        result['samples']=len(ids);result[mode+'_mean_ttft_ms']=float(np.mean([r['ttft_ms'] for r in rows]))
    result['slot_ttft_increase_pct']=100*(result['slot_mean_ttft_ms']/result['off_mean_ttft_ms']-1)
    groups.append(result)
with (e/'length_group_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
lines=['# 按输入长度分组的TTFT辅助分析','',
       '仅纳入每服务round2/3，共8轮；从逐请求JSONL指向的SQLite读取时间点，并核对每轮均值与aisbench CSV一致。','',
       '|输入token区间|样本数|off TTFT(ms)|Slot TTFT(ms)|Slot TTFT增加|',
       '|---|---:|---:|---:|---:|']
for r in groups:
    lines.append(f'|{r["input_range"]}|{r["samples"]}|{r["off_mean_ttft_ms"]:.2f}|{r["slot_mean_ttft_ms"]:.2f}|{r["slot_ttft_increase_pct"]:.2f}%|')
lines+=['','每个样本每模式有4次计入测量。请求并发及engine调度会影响单条TTFT，分组结果不直接等同于独立服务时间，也不能从中还原专家负载不均衡比。']
(e/'LENGTH_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(groups,ensure_ascii=False))
