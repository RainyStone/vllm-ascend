#!/usr/bin/env python3
"""Full503 diagnostic rank/layer totals, no per-step peak summation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;b=r/'data/diagnostics/home_rankload_20261106'
modes=('off','slot_mte','slot_udma');colors=('#477bb4','#e49b35','#6d8b45')
fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
for j,(m,c) in enumerate(zip(modes,colors)):
 d=json.loads((b/m/'RANK_LOAD.json').read_text())
 v=np.array(d['compute_pairs'],float);v/=v.mean()
 axes[0].scatter(np.arange(16)+(j-1)*.16,v,s=24,label=m,color=c,zorder=3)
 axes[1].plot([x['layer'] for x in d['per_layer']],[x['compute_peak_mean'] for x in d['per_layer']],label=m,color=c,linewidth=1.5)
axes[0].axhline(1,color='gray',linewidth=.7);axes[0].set_xticks(range(16));axes[0].set_xlabel('Global rank');axes[0].set_ylabel('Whole-run compute / rank mean');axes[0].set_ylim(.94,1.05);axes[0].legend(ncol=3)
axes[1].set_xlabel('MoE layer ordinal');axes[1].set_ylabel('Per-layer whole-run rank max / mean');axes[1].set_ylim(.98,1.55);axes[1].legend(ncol=3)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('DP16 LongBench-v2 full503 diagnostics: actual expert-pair counts')
fig.savefig(b/'rank_load_comparison.png',dpi=180);fig.savefig(b/'rank_load_comparison.svg')
print('RANK_LOAD_FIGURE_EXPORTED')
