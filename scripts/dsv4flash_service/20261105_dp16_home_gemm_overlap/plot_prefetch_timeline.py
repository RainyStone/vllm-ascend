#!/usr/bin/env python3
"""Export measured representative rank0 timelines; axes local to each trace."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
base=r/'data/diagnostics/home_profiling_v2_20261106'
fig,axes=plt.subplots(2,1,figsize=(10,5),constrained_layout=True)
chosen=[]
for ax,mode in zip(axes,('slot_mte','slot_udma')):
 d=json.loads((base/mode/'rank00_prefetch_overlap.json').read_text())
 rows=sorted([x for x in d['rows'] if x['long_prefill']],key=lambda x:x['home_us'])
 x=rows[len(rows)//2];t=x['timeline_us'];offset=t['dispatch'][0];chosen.append(dict(mode=mode,row=x))
 for key,color in [('dispatch','#477bb4'),('home','#e49b35'),('slot','#6d8b45')]:
  a,b=t[key];ax.broken_barh([((a-offset)/1000,(b-a)/1000)],(0.8,0.4),facecolors=color,label=key)
  ax.text(((a+b)/2-offset)/1000,1,key,ha='center',va='center',fontsize=8)
 for i,(a,b) in enumerate(t['prefetch']):
  ax.broken_barh([((a-offset)/1000,(b-a)/1000)],(-0.2,0.4),facecolors='#a77bb5',label='weight prefetch' if i==0 else None)
 ready=t['prefetch'][-1][1]
 ax.axvline((ready-offset)/1000,linestyle='--',color='#7b3b85',linewidth=1)
 ax.set_yticks([0,1],['prefetch stream','main stream']);ax.set_ylim(-.5,1.6)
 ax.set_title(mode+' / rank0 / median Home GEMM1 duration among captured long-prefill calls',fontsize=10)
 ax.set_xlabel('Local GPU time relative to dispatch start (ms)');ax.grid(axis='x',alpha=.2)
fig.savefig(base/'prefetch_home_timeline.png',dpi=180);fig.savefig(base/'prefetch_home_timeline.svg')
(base/'TIMELINE_SELECTION.json').write_text(json.dumps(chosen,indent=2))
print('TIMELINE_EXPORTED')
