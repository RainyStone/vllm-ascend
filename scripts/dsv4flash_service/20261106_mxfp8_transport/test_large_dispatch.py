#!/usr/bin/env python3
"""Model-free production-size 8K scatter: BF16-then-quant vs quant-then-scatter."""
import argparse,datetime as dt,faulthandler,hashlib,json,os,socket
from pathlib import Path
p=argparse.ArgumentParser()
for key in ("backend","output","library","bootstrap-ip"):p.add_argument("--"+key,required=True)
p.add_argument("--shmem-port",type=int,required=True);a=p.parse_args()
faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
import torch,torch_npu
import torch.distributed as dist
rank,world,local=[int(os.environ[k]) for k in ("RANK","WORLD_SIZE","LOCAL_RANK")]
assert world==2
torch.npu.set_device(local);dev=torch.device("npu",local)
dist.init_process_group("gloo",timeout=dt.timedelta(seconds=180))
from ascend_moonep.api import Buffer
from ascend_moonep.shmem_runtime import ShmemRuntime
S,K,H,E,B=8192,6,4096,32,16
b=Buffer(S=S,H=H,K=K,E=E,B=B,num_ep_ranks=world,group=dist.group.WORLD,use_udma=a.backend=="udma")
ShmemRuntime.init_with_ip_port(group=dist.group.WORLD,ip_port=f"tcp://{a.bootstrap_ip}:{a.shmem_port}",enable_udma=a.backend=="udma",qp_count=2 if a.backend=="udma" else 0,prefetch_overlap=a.backend=="udma")
b._ensure_shmem_views();ctx=b._require_ctx()
import ascend_moonep.launchers as launchers
native=torch.classes.AscendMoonEP.Dispatch()
calls=[]
class Spy:
 def compute(self,*v):
  calls.append(dict(use_udma=bool(v[-1]),hidden_width=int(v[-5])))
  return native.compute(*v)
launchers._DISPATCH=Spy()
torch.manual_seed(331+rank)
hidden=torch.randn(S,H,device=dev,dtype=torch.bfloat16)*.5
checks=[];status="failed";out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
try:
 for name,T in (("balanced",8192),("hot",8192),("real_tail",8191)):
  topk=((torch.arange(S*K).reshape(S,K)+rank*16)%E if name!="hot" else torch.arange(K)[None,:].expand(S,K).clone()).int().to(dev)
  topk[T:].fill_(-1)
  tpe=torch.bincount(topk[:T].flatten().long(),minlength=E).int()
  old,_,cu,plan=b.dispatch(hidden,topk_experts_sk=topk,tokens_per_expert=tpe,num_tokens=T)
  torch.npu.synchronize();dist.barrier()
  allcounts=torch.cat((cu[:1],cu[1:]-cu[:-1])).long()
  valid=int(torch.cat((allcounts[rank*16:(rank+1)*16],allcounts[E:])).sum().cpu())
  expected_q,expected_s=torch_npu.npu_dynamic_mx_quant(old[:valid],dst_type=torch.float8_e4m3fn)
  cu_ref=cu.clone()
  del old
  q,s,_,cu2,plan2=b.dispatch_mxfp8(hidden,topk_experts_sk=topk,tokens_per_expert=tpe,num_tokens=T,wire_library=a.library)
  torch.npu.synchronize();dist.barrier()
  assert torch.equal(cu_ref.cpu(),cu2.cpu())
  assert torch.equal(expected_q.view(torch.uint8).cpu(),q[:valid].view(torch.uint8).cpu()),("activation",rank,name)
  assert torch.equal(expected_s.view(torch.uint8).cpu(),s[:valid].view(torch.uint8).cpu()),("scales",rank,name)
  checks.append(dict(pattern=name,T=T,valid_rows=valid,activation_bytes_equal=True,scale_bytes_equal=True))
  print("LARGE_CHECK",rank,json.dumps(checks[-1]),flush=True)
  del expected_q,expected_s
 assert len(calls)==6 and all(x["use_udma"]==(a.backend=="udma") for x in calls)
 status="passed"
finally:
 libs={ln.split()[-1] for ln in Path("/proc/self/maps").read_text().splitlines() if ".so" in ln and ("moonep" in ln or "libshmem" in ln)}
 (out/f"rank{rank}.json").write_text(json.dumps(dict(status=status,backend=a.backend,rank=rank,world=world,host=socket.gethostname(),model_loaded=False,cross_node=False,S=S,H=H,K=K,checks=checks,native_dispatch_calls=calls,mapped_libraries={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in libs}),indent=2))
 if status=="passed":torch.npu.synchronize();dist.barrier();ShmemRuntime.finalize();dist.destroy_process_group();faulthandler.cancel_dump_traceback_later()
print("LARGE_DISPATCH_PASSED",rank,a.backend,flush=True)
