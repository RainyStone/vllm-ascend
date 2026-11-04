#!/usr/bin/env python3
"""Small fixed-request diagnosis, excluded from the formal performance matrix."""
import concurrent.futures,hashlib,json,sys,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prepare():
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('summary',ROOT/'summarize_results.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    execution=Path(json.loads((ROOT/'CURRENT_EXECUTION.json').read_text())['session'])
    rounds={}
    for mode in ['off','slot']:
        case=json.loads((execution/'cases'/('pair01_'+mode)/'case_result.json').read_text())
        rounds[mode]=[mod.read_round(case['client_dir'],i) for i in [2,3]]
    manifest=json.loads((execution/'selection_manifest.json').read_text())
    samples=manifest['samples'];lookup={i:x for i,x in enumerate(samples)}
    a,b=rounds['off'];c,d=rounds['slot']
    ordered=sorted(a['predictions'],key=lambda i:lookup[i]['prompt_tokens'])
    stable_diff=[i for i in ordered if a['predictions'][i]==b['predictions'][i] and c['predictions'][i]==d['predictions'][i] and a['predictions'][i]!=c['predictions'][i]]
    off_unstable=[i for i in ordered if a['predictions'][i]!=b['predictions'][i]]
    matched=[i for i in ordered if a['predictions'][i]==b['predictions'][i]==c['predictions'][i]==d['predictions'][i]]
    chosen=[]
    for ids,reason,limit in [(stable_diff,'stable_cross_mode_difference',5),(off_unstable,'off_repeat_difference',1),(matched,'stable_matching_control',1),([max(ordered,key=lambda i:lookup[i]['prompt_tokens'])],'longest_prompt',1)]:
        for i in ids[:limit]:
            if i not in [x[0] for x in chosen]:chosen.append((i,reason))
    assert len(chosen)>=6
    wanted={i for i,_ in chosen};raws={}
    with Path(a['raw_file']).open() as f:
        for line in f:
            r=json.loads(line)
            if int(r['id']) in wanted:raws[int(r['id'])]=r
    output=[]
    for i,reason in chosen:
        r=raws[i];prompt=r['input']
        assert hashlib.sha256(prompt.encode()).hexdigest()==lookup[i]['prompt_sha256']
        output.append(dict(id=i,reason=reason,prompt=prompt,prompt_tokens=lookup[i]['prompt_tokens'],
                           prompt_sha256=lookup[i]['prompt_sha256'],
                           previous_off=[r['predictions'][i] for r in rounds['off']],
                           previous_slot=[r['predictions'][i] for r in rounds['slot']]))
    p=ROOT/'diagnostics/precision_samples.json';p.write_text(json.dumps(output,ensure_ascii=False)+'\n')
    print(json.dumps([dict(id=x['id'],reason=x['reason'],prompt_tokens=x['prompt_tokens']) for x in output]))
def request(mode,sample,phase,repeat,barrier=None):
    if barrier:barrier.wait()
    body=dict(model='dsv4flash_longbench_'+mode,prompt=sample['prompt'],max_tokens=1,
              temperature=0,seed=1024,ignore_eos=True,stream=True,logprobs=5,
              stream_options=dict(include_usage=True))
    req=urllib.request.Request('http://127.0.0.1:8444/v1/completions',
          data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    began=time.monotonic();chunks=[]
    with opener.open(req,timeout=1200) as response:
        for physical_line in response:
            line=physical_line.decode().strip()
            if line.startswith('data:'):
                payload=line[5:].strip()
                if payload=='[DONE]':break
                chunks.append(json.loads(payload))
    usages=[c['usage'] for c in chunks if c.get('usage')]
    assert usages,('No stream usage',sample['id'])
    usage=usages[-1]
    assert usage['prompt_tokens']==sample['prompt_tokens'],usage
    assert usage['completion_tokens']==1,usage
    tokens=[];text='';probabilities=[]
    for c in chunks:
        for choice in c.get('choices',[]):
            text+=choice.get('text','')
            lp=choice.get('logprobs')
            if lp:
                tokens.extend(lp.get('tokens') or [])
                probabilities.extend(lp.get('top_logprobs') or [])
    assert len(tokens)==1,(sample['id'],tokens,chunks)
    return dict(mode=mode,id=sample['id'],phase=phase,repeat=repeat,
                prompt_sha256=sample['prompt_sha256'],prompt_tokens=sample['prompt_tokens'],
                prediction=text,tokens=tokens,top_logprobs=probabilities,usage=usage,
                elapsed_seconds=time.monotonic()-began,raw_sse_chunks=chunks)
def main():
    if sys.argv[1]=='prepare':prepare();return
    mode,folder=sys.argv[1:];assert mode in ['off','slot']
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    samples=json.loads((ROOT/'diagnostics/precision_samples.json').read_text())
    assert len(samples)<=8
    with (folder/'responses.jsonl').open('w') as out:
        for sample in samples:
            for repeat in range(3):
                record=request(mode,sample,'sequential_repeat',repeat)
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:record[k] for k in ['mode','id','phase','repeat','prediction']},ensure_ascii=False),flush=True)
        for repeat in range(3):
            barrier=threading.Barrier(len(samples))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(samples)) as pool:
                futures=[pool.submit(request,mode,sample,'same_request_set_concurrent',repeat,barrier) for sample in samples]
                for future in futures:
                    record=future.result()
                    out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
            print('concurrent_set_complete',repeat,flush=True)
    result=dict(valid=True,mode=mode,samples=len(samples),requests=6*len(samples),
                max_output_tokens=1,formal_performance_data=False,
                limit='Concurrent request sets are fixed; exact per-engine execution batches are scheduler-controlled.')
    (folder/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
