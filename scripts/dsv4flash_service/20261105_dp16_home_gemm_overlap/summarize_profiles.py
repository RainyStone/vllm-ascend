#!/usr/bin/env python3
"""Aggregate 16 local trace inventories; never treat overlapping kernel sums as wall time."""
import json,collections
from pathlib import Path
r=Path(__file__).resolve().parent
base=r/'data/diagnostics/home_profiling_v2_20261106'
result={}
for mode in ('off','slot_mte','slot_udma'):
 ds=[json.loads((base/mode/f'rank{i:02d}_trace_inventory.json').read_text()) for i in range(16)]
 cats=collections.defaultdict(lambda:dict(count=0,sum_ms=0.))
 for d in ds:
  for item in d['hardware_top']:
   name=item['name'].lower()
   kind=('attention' if name=='kvquantsparseattnsharedkv' else
         'home_and_slot_gemm1' if 'home_gmm_mxfp4_kernel' in name else
         'native_grouped_gemm' if 'groupedmatmul' in name else
         'planning' if 'planning_kernel' in name else
         'dispatch' if 'moonep_dispatch' in name else
         'combine' if 'combine' in name and 'weighted' in name else
         'weight_prefetch' if 'prefetch_weight' in name else None)
   if kind:
    cats[kind]['count']+=item['count'];cats[kind]['sum_ms']+=item['sum_ms']
 result[mode]=dict(ranks=16,categories=dict(cats),
  attention_calls_per_rank=[sum(x['count'] for x in d['hardware_top'] if x['name']=='KvQuantSparseAttnSharedkv') for d in ds])
for v in result.values():
 for c in v['categories'].values():c['mean_us']=1000*c['sum_ms']/c['count']
result['limits']='All-rank task sums overlap across streams and ranks. Per-rank MoE forward counts differ across modes; long-prefill subset and per-call metrics are reported separately. These are independent diagnostic samples, not complete503 latency attribution.'
(base/'PROFILE_SUMMARY.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
