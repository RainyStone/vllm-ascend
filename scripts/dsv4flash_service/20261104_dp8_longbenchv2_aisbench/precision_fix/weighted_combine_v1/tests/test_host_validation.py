"""Host API validation only; does not initialize or allocate NPU resources."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import ascend_moonep
from ascend_moonep.api import Buffer
from ascend_moonep.planning import MoonEPCommPlan

S,H,K,R = 4,16,6,2
h=torch.zeros(S*K,H,dtype=torch.bfloat16)
w=torch.zeros(S,8,dtype=torch.float32)
plan=MoonEPCommPlan(dst=torch.zeros(S*K,dtype=torch.int32),
    experts_to_copy=torch.zeros(R,1,dtype=torch.int32),
    dup_groups=torch.zeros(S*K,3,dtype=torch.int32),
    dup_loffs=torch.zeros(S*K,dtype=torch.int32),
    dup_counts=torch.zeros(2,dtype=torch.int32), N=S*K,R=R,E=2,B=1,NvS=S*K,K=K)
ctx=dict(S=S,H=H,K=K,R=R,hidden_buf_local=h.clone())
fake=SimpleNamespace(_require_ctx=lambda:ctx, _ensure_shmem_views=lambda:None)
def call(hh=h,ww=w,pp=plan,**kw):
    return Buffer.combine_weighted(fake,pp,hh,ww,num_tokens=kw.pop("num_tokens",S),**kw)
cases=[
 ("zero_T",lambda:call(num_tokens=0),ValueError),
 ("excess_T",lambda:call(num_tokens=S+1),ValueError),
 ("BF16_weight_rejected",lambda:call(ww=w.bfloat16()),ValueError),
 ("unpadded_weight_rejected",lambda:call(ww=w[:,:K].contiguous()),ValueError),
 ("noncontiguous_hidden_rejected",lambda:call(hh=h.T.contiguous().T),ValueError),
 ("missing_plan_rejected",lambda:call(pp=None),TypeError),
 ("nonalias_zero_copy_rejected",lambda:call(zero_copy=True),ValueError),
]
results=[]
for name,fn,typ in cases:
    try:fn()
    except typ as exc:results.append(dict(test=name,passed=True,message=str(exc)))
    else:raise AssertionError(name)
os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"]="1"
try:call()
except NotImplementedError as exc:
    results.append(dict(test="UDMA_rejected",passed=True,message=str(exc)))
else:raise AssertionError("UDMA")
obj=torch.classes.AscendMoonEP.Combine()
assert hasattr(obj,"compute_weighted")
record=dict(status="passed",scope="host-only; no device execution",package=ascend_moonep.__file__,
            new_host_method_registered=True,tests=results)
Path(__file__).resolve().parents[1].joinpath("results/host_validation.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
