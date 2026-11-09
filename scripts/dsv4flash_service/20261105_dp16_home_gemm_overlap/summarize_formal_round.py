#!/usr/bin/env python3
"""Read-only AIS raw timing extraction, matching default_perf.py TTFT definition."""
import io,json,os,sqlite3,sys,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
state_path=root/'STATE.json';state=json.loads(state_path.read_text());mode=sys.argv[1]
rounddir=Path(state['formal'][mode]['round_dir'])
paths=[]
for folder,dirs,files in os.walk(rounddir,followlinks=True):
 for name in files:
  if name.endswith('_details.jsonl'):paths.append(Path(folder)/name)
assert len(paths)==1,paths
details=paths[0];summary_path=details.with_name(details.name.removesuffix('_details.jsonl')+'.json')
statistics=json.loads(summary_path.read_text())
connections={};requests=[]
with details.open() as f:
 for line in f:
  row=json.loads(line);assert row['success'] and row['output_tokens']==1
  db=details.parent/'db_data'/row['db_name']
  if str(db) not in connections:connections[str(db)]=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
  ref=row['time_points']['__db_ref__']
  blob=connections[str(db)].execute('select arr_blob from numpy_store where id=?',(ref,)).fetchone()[0]
  times=np.load(io.BytesIO(blob),allow_pickle=False)
  assert times.ndim==1 and len(times)==3 and np.all(np.diff(times)>=0),(row['id'],times)
  requests.append(dict(id=row['id'],input_tokens=row['input_tokens'],output_tokens=1,ttft_ms=float((times[1]-times[0])*1000),latency_ms=float((times[-1]-times[0])*1000)))
assert len(requests)==503 and sum(x['input_tokens'] for x in requests)==43029416
ttft=np.array([x['ttft_ms'] for x in requests])
source=root.parents[3]/'benchmark/ais_bench/benchmark/summarizers/default_perf.py'
# Resolve workspace explicitly: experiment lives below vllm-ascend/scripts/dsv4flash_service.
source=Path('/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/benchmark/summarizers/default_perf.py')
result=dict(mode=mode,requests=503,input_tokens=43029416,output_tokens=503,ttft_ms=dict(mean=float(ttft.mean()),median=float(np.median(ttft)),p95=float(np.percentile(ttft,95)),p99=float(np.percentile(ttft,99)),max=float(ttft.max())),input_tps=float(statistics['Input Token Throughput']['total'].split()[0]),benchmark_duration_ms=float(statistics['Benchmark Duration']['total'].split()[0]),raw_statistics=statistics,ttft_rule='unweighted per-request mean of 1000*(time_points[1]-time_points[0]), as AIS default_perf.py',ttft_definition_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_details=str(details),raw_statistics_path=str(summary_path))
dest=root/'data/formal'/state['formal_run_id']/mode
(dest/'per_request_latency.json').write_text(json.dumps(requests,indent=2))
(dest/'round_summary.json').write_text(json.dumps(result,indent=2))
state['formal'][mode]['summary']=result
tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,state_path)
print(json.dumps({k:v for k,v in result.items() if k not in ('raw_statistics','ttft_definition_source_sha256','raw_details','raw_statistics_path')}))
