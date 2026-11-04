#!/usr/bin/env python3
"""Aggregate recorded per-rank expert load, keeping pre-plan and compute views distinct."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
def ratio(values):
    a=np.asarray(values,dtype=np.float64)
    return float(a.max()/a.mean()) if a.sum()>0 else None
def distribution(values,weights):
    a=np.asarray(values,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    return dict(samples=len(a),mean=float(a.mean()),weighted_mean=float(np.average(a,weights=w)),
                p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)),maximum=float(a.max()))
def parse_file(path):
    meta={};layers={};records={}
    pattern=re.compile(r'^step=(\d+) layer=(\d+) method=(\S+)(?: scope=(\S+))? counts=\[([^\]]*)\]')
    with path.open() as f:
        for line in f:
            if line.startswith('# meta '):
                meta.update(dict(re.findall(r'(\w+)=([^\s]+)',line)))
            elif line.startswith('# layer '):
                ordinal=int(re.search(r'layer_ord=(\d+)',line).group(1))
                expert_map=re.search(r'expert_map=(\[[^\]]*\]|None)',line)
                logical=re.search(r'log2phy=(\[[^\]]*\]|None)',line)
                layers[ordinal]=dict(expert_map=None if not expert_map or expert_map[1]=='None' else json.loads(expert_map[1]),
                                     log2phy=None if not logical or logical[1]=='None' else json.loads(logical[1]))
            elif line.startswith('step='):
                m=pattern.match(line);assert m,(path,line[:150])
                step,layer,method,scope,counts=m.groups()
                key=(int(step),int(layer),scope)
                assert key not in records,(path,key)
                records[key]=(method,np.fromstring(counts,sep=',',dtype=np.int64))
    assert meta and records,path
    primary=meta.get('counts_scope','local').split('+')[0]
    assert primary in ('local','global'),meta
    normalized={}
    for (step,layer,scope),value in records.items():
        key=(step,layer,scope or primary);assert key not in normalized
        normalized[key]=value
    return dict(path=str(path),meta=meta,layers=layers,records=normalized,primary=primary)
def local_mapping(parsed,layer,width):
    m=parsed['meta'];rank=int(m['ep_rank']);e=int(m['global_experts']);n=int(m['local_experts'])
    assert width==n and e==8*n and int(m['redundant_experts'])==0,m
    desc=parsed['layers'].get(layer,{})
    logical=desc.get('log2phy')
    assert logical is None or logical==list(range(len(logical))),'Nonidentity logical mapping needs explicit handling'
    emap=desc.get('expert_map')
    if emap is None:return list(range(rank*n,(rank+1)*n))
    assert len(emap)==e,(rank,len(emap),e)
    inverse={}
    for global_id,local_id in enumerate(emap):
        if local_id>=0:inverse[int(local_id)]=global_id
    assert set(inverse)==set(range(n)),(rank,inverse)
    mapping=[inverse[i] for i in range(n)]
    assert mapping==list(range(rank*n,(rank+1)*n)),'Noncontiguous placement requires owner-map aggregation'
    return mapping
def main():
    mode,source,target=sys.argv[1:4];assert mode in ['off','slot']
    bounds=json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv)>4 else None
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=False)
    files=sorted(source.glob('moe_par_*_rank*.log'));assert len(files)==8,files
    ranks={}
    for file in files:
        p=parse_file(file);rank=int(p['meta']['ep_rank']);assert rank not in ranks
        if bounds:
            span=bounds['ranks'][str(rank)]
            p['records']={key:value for key,value in p['records'].items() if span['start_step']<=key[0]<=span['end_step']}
        ranks[rank]=p
    assert set(ranks)==set(range(8))
    alignment={'method':'relative_to_phase_start','per_rank':{}}
    if mode=='off':
        alignment['method']='ALLTOALL collective invocation order; MC2 idle/small steps retained only in raw logs and rank totals'
        lengths=[]
        for rank,p in ranks.items():
            original=p['records']
            steps=sorted({key[0] for key,value in original.items() if key[2]=='local' and value[0]=='AlltoAllCommImpl'})
            index={step:i for i,step in enumerate(steps)}
            total=sum(int(value[1].sum()) for key,value in original.items() if key[2]=='local')
            selected={ (index[step],layer,scope):value for (step,layer,scope),value in original.items()
                       if step in index and value[0]=='AlltoAllCommImpl'}
            for step in steps:
                assert {layer for (rawstep,layer,scope),value in original.items() if rawstep==step and scope=='local' and value[0]=='AlltoAllCommImpl'}==set(range(int(p['meta']['moe_layers'])))
            kept=sum(int(value[1].sum()) for key,value in selected.items() if key[2]=='local')
            alignment['per_rank'][rank]=dict(original_step_ids=steps,phase_pairs=total,selected_pairs=kept,pair_coverage=kept/total)
            p['records']=selected;lengths.append(len(steps))
        assert len(set(lengths))==1,('ALLTOALL collective counts differ',lengths)
    elif bounds:
        alignment['method']='Identical raw SHMEM active-step IDs across all ranks; layer0 compute pairs >1024. Idle/tiny records retained in raw logs.'
        active_ids=[]
        for rank,p in ranks.items():
            original=p['records']
            steps=sorted({step for (step,layer,scope),value in original.items()
                          if layer==0 and scope=='compute' and int(value[1].sum())>1024})
            selected={key:value for key,value in original.items() if key[0] in set(steps)}
            total=sum(int(value[1].sum()) for key,value in original.items() if key[2]=='global')
            kept=sum(int(value[1].sum()) for key,value in selected.items() if key[2]=='global')
            alignment['per_rank'][rank]=dict(original_step_ids=steps,phase_pairs=total,selected_pairs=kept,pair_coverage=kept/total)
            p['records']=selected;active_ids.append(steps)
        assert all(ids==active_ids[0] for ids in active_ids),'SHMEM active step IDs do not align; no cross-rank metrics allowed'
    e={int(p['meta']['global_experts']) for p in ranks.values()};assert len(e)==1
    e=e.pop();assert e%8==0
    required=['global','compute'] if mode=='slot' else ['local']
    keysets=[]
    for rank,p in ranks.items():
        for scope in required:keysets.append({(s,l) for s,l,c in p['records'] if c==scope})
    common=set.intersection(*keysets);union=set.union(*keysets)
    assert common,'No complete cross-rank step/layer records'
    rows=[];before_values=[];after_values=[];expert_values=[];weights=[];layer_summary={}
    route_total=np.zeros(8,dtype=np.int64);compute_total=np.zeros(8,dtype=np.int64)
    mismatch_keys=[];methods={}
    for step,layer in sorted(common):
        expert=np.zeros(e,dtype=np.int64);compute=np.zeros(8,dtype=np.int64)
        borrowed_experts=np.zeros(8,dtype=np.int64);borrowed_pairs=np.zeros(8,dtype=np.int64)
        computed_expert=np.zeros(e,dtype=np.int64)
        for rank,p in ranks.items():
            if mode=='slot':
                method,gate=p['records'][(step,layer,'global')]
                _,work=p['records'][(step,layer,'compute')]
                assert len(gate)==len(work)==e and gate.min()>=0 and work.min()>=0
                expert+=gate;compute[rank]=work.sum();computed_expert+=work
                remote=work.copy();remote[rank*(e//8):(rank+1)*(e//8)]=0
                borrowed_experts[rank]=np.count_nonzero(remote);borrowed_pairs[rank]=remote.sum()
            else:
                method,work=p['records'][(step,layer,'local')]
                mapping=local_mapping(p,layer,len(work))
                assert work.min()>=0
                expert[mapping]+=work;compute[rank]=work.sum();computed_expert[mapping]+=work
            methods[method]=methods.get(method,0)+1
        before=expert.reshape(8,e//8).sum(axis=1)
        if expert.sum()==0:
            assert compute.sum()==0
            continue
        s_before=ratio(before);s_after=ratio(compute);s_expert=ratio(expert)
        l1=int(np.abs(expert-computed_expert).sum())
        if l1:mismatch_keys.append(dict(step=step,layer=layer,expert_count_l1=l1,
                                        routed_pairs=int(expert.sum()),computed_pairs=int(compute.sum())))
        row=dict(step=step,layer=layer,routed_pairs=int(expert.sum()),computed_pairs=int(compute.sum()),
                 rank_imbalance_before=s_before,rank_imbalance_after=s_after,
                 expert_imbalance=s_expert,expert_count_l1=l1,
                 distinct_remote_experts_computed=int(borrowed_experts.sum()),
                 remote_expert_pairs_computed=int(borrowed_pairs.sum()))
        for rank in range(8):
            row[f'owner_rank{rank}']=int(before[rank]);row[f'compute_rank{rank}']=int(compute[rank])
            row[f'owner_rank{rank}_normalized_load']=float(before[rank]/before.mean())
            row[f'compute_rank{rank}_remote_experts']=int(borrowed_experts[rank])
            row[f'compute_rank{rank}_remote_pairs']=int(borrowed_pairs[rank])
            row[f'compute_rank{rank}_normalized_load']=float(compute[rank]/compute.mean()) if compute.sum()>0 else None

        rows.append(row);before_values.append(s_before);after_values.append(s_after)
        expert_values.append(s_expert);weights.append(int(expert.sum()))
        route_total+=before;compute_total+=compute
        layer_summary.setdefault(layer,[]).append(row)
    assert rows,'All recorded rows are empty'
    per_layer={}
    for layer,items in layer_summary.items():
        w=[x['routed_pairs'] for x in items]
        per_layer[layer]={key:distribution([x[key] for x in items],w) for key in ['rank_imbalance_before','rank_imbalance_after','expert_imbalance']}
    report=dict(mode=mode,diagnostic_only=True,formal_performance_rounds_instrumented=False,
                definition='s_rank=max(rank token-expert pairs)/(sum(rank pairs)/8); s_expert=max(expert pairs)/mean(expert pairs)',
                phase_boundaries=bounds,alignment=alignment,complete_step_layers=len(common),nonempty_step_layers=len(rows),
                incomplete_step_layer_keys_excluded=len(union-common),
                excluded_keys=sorted(union-common),recorded_steps=len({r['step'] for r in rows}),
                rank_imbalance_before=distribution(before_values,weights),
                rank_imbalance_after=distribution(after_values,weights),
                expert_imbalance=distribution(expert_values,weights),
                owner_rank_totals=route_total.tolist(),compute_rank_totals=compute_total.tolist(),
                aggregate_owner_rank_imbalance=ratio(route_total),aggregate_compute_rank_imbalance=ratio(compute_total),
                count_conservation_ok=not mismatch_keys,conservation_failures=mismatch_keys,
                per_layer=per_layer,method_rank_records=methods,
                distinct_remote_experts_computed_total=sum(x['distinct_remote_experts_computed'] for x in rows),
                remote_expert_pairs_computed_total=sum(x['remote_expert_pairs_computed'] for x in rows),
                metadata={rank:p['meta'] for rank,p in ranks.items()},
                raw_files=[str(x) for x in files],
                limitations=['Same request corpus and concurrency; scheduler batches may differ from the uninstrumented performance runs.',
                             'Off records received local expert counts; its routing and compute views coincide and are not independent conservation checks.',
                             'Slot conservation compares per-expert routing and computed counts; this does not verify copied weight bytes or tensor values.'])
    # Exclude tiny idle / tail steps from the principal cross-mode comparison.
    # Threshold is 8 ranks * 16 tokens * topK6 = 768 token-expert pairs.
    main_rows=[x for x in rows if x['routed_pairs']>768]
    main_weights=[x['routed_pairs'] for x in main_rows]
    report['main_prefill']=dict(global_routed_pairs_threshold_exclusive=768,
        step_layers=len(main_rows),pair_coverage_of_aligned_rows=sum(main_weights)/sum(weights),
        rank_imbalance_before=distribution([x['rank_imbalance_before'] for x in main_rows],main_weights),
        rank_imbalance_after=distribution([x['rank_imbalance_after'] for x in main_rows],main_weights),
        expert_imbalance=distribution([x['expert_imbalance'] for x in main_rows],main_weights))
    with (target/'step_layer_rank_load.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (target/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['mode','complete_step_layers','incomplete_step_layer_keys_excluded','rank_imbalance_before','rank_imbalance_after','expert_imbalance','count_conservation_ok']},ensure_ascii=False))
if __name__=='__main__':main()
