#!/usr/bin/env python3
"""Offline comparison only. Reads saved JSONL and SQLite; sends no requests."""
import csv,io,json,sqlite3,statistics
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/data/executions/run_20261105_155600_mte_first_stream2/candidate_dp_metadata_npu')
results={}
rows={}
for mode,name in [('off','OFF_RESULT'),('slot_mte','SLOT_MTE_RESULT'),('slot_udma','SLOT_UDMA_RESULT'),('slot_mte_mainstream','no_overlap/SLOT_MTE_RESULT'),('slot_udma_mainstream','no_overlap/SLOT_UDMA_RESULT')]:
    result_file=ROOT/(name+'.json')
    if not result_file.exists():continue
    summary=json.loads(result_file.read_text())
    path=Path(summary['raw_jsonl'])
    connections={}
    values={}
    with path.open() as stream:
        for line in stream:
            r=json.loads(line)
            assert r['success'] and r['output_tokens']==1
            db=path.parent/'db_data'/r['db_name']
            if db not in connections:
                connections[db]=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
            blob=connections[db].execute('SELECT arr_blob FROM numpy_store WHERE id=?',(r['time_points']['__db_ref__'],)).fetchone()[0]
            points=np.load(io.BytesIO(blob),allow_pickle=False)
            assert points.shape==(3,) and points[2]>=points[1]>=points[0]
            key=int(r['data_id'])
            assert key not in values
            values[key]={'input_tokens':r['input_tokens'],'ttft_ms':float((points[1]-points[0])*1000)}
    for connection in connections.values():connection.close()
    assert len(values)==503
    mean=statistics.mean(v['ttft_ms'] for v in values.values())
    # Ensure the DB interpretation reproduces AISBench's published TTFT.
    assert abs(mean-summary['mean_ttft_ms'])<0.11,(mode,mean,summary['mean_ttft_ms'])
    rows[mode]=values
    results[mode]={'raw_mean_ttft_ms':mean,'buckets':[]}
    for low,high in [(0,32768),(32768,65536),(65536,131072)]:
        ids=[k for k,v in values.items() if low<v['input_tokens']<=high]
        results[mode]['buckets'].append({'tokens_gt':low,'tokens_le':high,'n':len(ids),
            'mean_ttft_ms':statistics.mean(values[k]['ttft_ms'] for k in ids)})
for mode in rows:
    assert rows[mode].keys()==rows['off'].keys()
    assert all(rows[mode][k]['input_tokens']==rows['off'][k]['input_tokens'] for k in rows[mode])
    if mode!='off':
        for b,base in zip(results[mode]['buckets'],results['off']['buckets']):
            assert b['n']==base['n']
            b['ttft_change_pct_vs_off']=100*(b['mean_ttft_ms']/base['mean_ttft_ms']-1)
(ROOT/'OFFLINE_LENGTH_ANALYSIS.json').write_text(json.dumps(results,indent=2))
with (ROOT/'PER_REQUEST_TTFT.csv').open('w',newline='') as stream:
    fields=['sample_id','input_tokens']+[m+'_ttft_ms' for m in rows]
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for k in sorted(rows['off']):
        writer.writerow(dict(sample_id=k,input_tokens=rows['off'][k]['input_tokens'],
            **{m+'_ttft_ms':rows[m][k]['ttft_ms'] for m in rows}))
print(json.dumps(results))
