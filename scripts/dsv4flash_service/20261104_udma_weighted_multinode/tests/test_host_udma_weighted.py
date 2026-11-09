"""Host-only interface/dispatcher regression; no NPU allocation or kernel launch."""
import json
import os
from pathlib import Path
from load_staged import load_staged

loaded=load_staged()
import torch
import ascend_moonep.launchers as launchers
assert hasattr(torch.classes.AscendMoonEP.CombineUrma(),"compute_weighted")
assert hasattr(torch.classes.AscendMoonEP.Combine(),"compute_weighted")

class Recorder:
    def __init__(self):self.calls=[]
    def compute_weighted(self,*args):self.calls.append(args)

native,udma=Recorder(),Recorder()
launchers._COMBINE=native
launchers._COMBINE_UDMA=udma
objects=[object() for _ in range(8)]
hidden,weights,dst,groups,loffs,counts,out,sync=objects
args=(hidden,weights,dst,groups,loffs,counts,out,16,7,64,6,2,0)
os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"]="1"
launchers.launch_combine_weighted_op(*args,sync_buffer=sync)
assert len(udma.calls)==1 and not native.calls
assert udma.calls[0]==(hidden,weights,sync,dst,groups,loffs,counts,out,16,7,64,6,2,0)
try:launchers.launch_combine_weighted_op(*args)
except ValueError:pass
else:raise AssertionError("Missing UDMA symmetric sync buffer was accepted")
os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"]="0"
launchers.launch_combine_weighted_op(*args,sync_buffer=sync)
assert native.calls==[args] and len(udma.calls)==1

# Invalid API inputs reject before any copy/launch, on CPU tensors.
from types import SimpleNamespace
from ascend_moonep.api import Buffer
from ascend_moonep.planning import MoonEPCommPlan
S,H,K,R=4,16,6,2
h=torch.zeros(S*K,H,dtype=torch.bfloat16)
w=torch.zeros(S,8,dtype=torch.float32)
plan=MoonEPCommPlan(dst=torch.zeros(S*K,dtype=torch.int32),
 experts_to_copy=torch.zeros(R,1,dtype=torch.int32),
 dup_groups=torch.zeros(S*K,3,dtype=torch.int32),
 dup_loffs=torch.zeros(S*K,dtype=torch.int32),
 dup_counts=torch.zeros(2,dtype=torch.int32),N=S*K,R=R,E=2,B=1,NvS=S*K,K=K)
ctx=dict(S=S,H=H,K=K,R=R,hidden_buf_local=h.clone())
fake=SimpleNamespace(_require_ctx=lambda:ctx,_ensure_shmem_views=lambda:None)
checks=[]
for flag in ("0","1"):
 os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"]=flag
 for label,ww,T in [("zero_T",w,0),("invalid_weight_dtype",w.bfloat16(),S),
                    ("missing_padding",w[:,:K].contiguous(),S)]:
  try:Buffer.combine_weighted(fake,plan,h,ww,num_tokens=T)
  except ValueError:checks.append(dict(backend="UDMA" if flag=="1" else "MTE",test=label,passed=True))
  else:raise AssertionError((flag,label))
result=dict(status="passed",scope="host-only; no device work",loaded=loaded,
 native_and_udma_weighted_bindings=True,dispatcher_abi_and_sync_buffer=True,api_rejections=checks)
out=Path(__file__).resolve().parents[1]/"results";out.mkdir(exist_ok=True)
(out/"host_validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
