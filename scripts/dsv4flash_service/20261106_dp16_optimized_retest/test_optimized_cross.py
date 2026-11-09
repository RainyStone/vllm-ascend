#!/usr/bin/env python3
"""Cross-node 2/16-rank integration: MXFP8, side-stream prefetch, unsplit native MLP, fused BF16 publication."""
import argparse,datetime as dt,faulthandler,hashlib,json,os,socket,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument("--backend",choices=["mte","udma"],required=True);p.add_argument("--output",required=True);p.add_argument("--library",required=True);p.add_argument("--shmem-port",type=int,required=True);p.add_argument("--bootstrap-ip",required=True);a=p.parse_args()
faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
import torch,torch_npu
import torch.distributed as dist
rank,world,local=[int(os.environ[k]) for k in ("RANK","WORLD_SIZE","LOCAL_RANK")]
assert world in (2,16)
torch.npu.set_device(local);dev=torch.device("npu",local)
dist.init_process_group("gloo",timeout=dt.timedelta(seconds=180))
import ctypes
ctypes.CDLL(str(Path(a.library).parent/"libascend_moonep_kernels.so"),mode=__import__("os").RTLD_NOW|ctypes.RTLD_GLOBAL)
from ascend_moonep.api import Buffer
from ascend_moonep.combine_publish import CombinePublisher
import ctypes
lib=ctypes.CDLL(str(Path(a.library).parent/"libascend_moonep_kernels.so"))
assert lib.moonep_udma_weight_reuse_fix_abi()==2
from ascend_moonep.shmem_runtime import ShmemRuntime
from ascend_moonep.mxfp8_transport import Mxfp8Transport,carrier_view,wire_geometry
from vllm_ascend.ops.fused_moe.moonep_balance import _LayerWeightState,_flat_bytes
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput,MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType
import vllm_ascend.vllm_ascend_C
sys.path.insert(0,"/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/tests")
from operator_context import operator_forward_context
S,K,H,F,epn=128,6,4096,2048,16
E,B=world*epn,epn
torch.manual_seed(1024+rank)
b=Buffer(S=S,H=H,K=K,E=E,B=B,num_ep_ranks=world,group=dist.group.WORLD,use_udma=a.backend=="udma")
def weight(k,n,salt):
 raw=torch.randint(0,256,(epn,n,k//2),dtype=torch.uint8).to(dev)
 w=torch_npu.npu_format_cast(raw,29,customize_dtype=torch.float8_e4m3fn,input_dtype=torch_npu.float4_e2m1fn_x2).transpose(1,2)
 sc=(torch.arange(epn*n*(k//32)).reshape(epn,n,k//32)%3+118+((rank+salt)%2)).to(torch.uint8).to(dev)
 return w,sc.reshape(epn,n,k//64,2).transpose(-3,-2)
w1,s1=weight(H,2*F,0);w2,s2=weight(F,H,1)
state=_LayerWeightState(SimpleNamespace(w13_weight=w1,w2_weight=w2,w13_weight_scale=s1,w2_weight_scale=s2),E,H,rank,local)
ShmemRuntime.init_with_ip_port(group=dist.group.WORLD,ip_port=f"tcp://{a.bootstrap_ip}:{a.shmem_port}",enable_udma=a.backend=="udma",qp_count=2 if a.backend=="udma" else 0,prefetch_overlap=a.backend=="udma")
b._ensure_shmem_views();ctx=b._require_ctx()
assert not ctx["_planning_use_udma"]
for st in ("dispatch","combine","prefetch"):assert bool(ctx["_"+st+"_use_udma"])==(a.backend=="udma")
state.publish();side=torch.npu.Stream()
node_ip=os.environ["HCCL_IF_IP"]
physical=[None]*world
dist.all_gather_object(physical,dict(rank=rank,ip=node_ip,host=socket.gethostname(),local_rank=local,visible=os.environ["ASCEND_RT_VISIBLE_DEVICES"]))
assert {x["ip"] for x in physical}=={"141.61.45.118","141.61.45.124"}
assert len({x["host"] for x in physical})==2
assert all(physical[(rr+world//2)%world]["ip"]!=physical[rr]["ip"] for rr in range(world))
b._combine_publisher=CombinePublisher(str(Path(a.library).parent/"libascend_moonep_publish.so"))
layout=ShmemRuntime.require_udma_prefetch_overlap(2) if a.backend=="udma" else None
quant=MoEQuantParams(quant_type=QuantType.W4A8MXFP,mxfp=MoEMxfpParams(act_quant_type=torch.float8_e4m3fn,weight_quant_type=torch_npu.float4_e2m1fn_x2,scale_dtype=torch.float8_e8m0fnu,per_token_scale_dtype=torch.float8_e8m0fnu,use_bf16=True))
import ascend_moonep.launchers as launchers
calls=[]
class Recorder:
 def __init__(self,op,label):self.op,self.label=op,label
 def __getattr__(self,name):
  method=getattr(self.op,name)
  if not name.startswith("compute"):return method
  def wrapped(*values,**kw):
   calls.append(dict(operator=self.label,method=name,stream=int(torch.npu.current_stream().npu_stream),bool_arguments=[v for v in values if isinstance(v,bool)]))
   return method(*values,**kw)
  return wrapped
launchers._PLANNING=Recorder(torch.classes.AscendMoonEP.Planning(),"Planning")
launchers._DISPATCH=Recorder(torch.classes.AscendMoonEP.Dispatch(),"Dispatch")
launchers._PREFETCH_WEIGHT=Recorder(torch.classes.AscendMoonEP.PrefetchWeight(0),"PrefetchWeight")
if a.backend=="udma":launchers._COMBINE_UDMA=Recorder(torch.classes.AscendMoonEP.CombineUrma(),"CombineUrma")
else:launchers._COMBINE=Recorder(torch.classes.AscendMoonEP.Combine(),"Combine")
def sync():torch.npu.synchronize();dist.barrier()
def bits(x):return x.contiguous().view(torch.uint8).cpu()
def payload(x,c,scale=None):return MoEMlpComputeInput(hidden_states=x.clone(),group_list=c,group_list_type=1,dynamic_scale=scale,topk_scales=None,weights=state.compute_weights(),quant=quant,fusion=True,activation="silu",swiglu_limit=10.)
def calc(x,c,ready,scale=None):
 torch.npu.current_stream().wait_event(ready)
 with operator_forward_context():
  return unified_apply_mlp(mlp_compute_input=payload(x,c,scale),record_before_gmm2=False)[0]
def hashes(t,row_bytes,count):
 arr=_flat_bytes(t,row_bytes*count).view(count,row_bytes).cpu().numpy()
 return [hashlib.sha256(memoryview(arr[i])).hexdigest() for i in range(count)]
homes={sp.name:hashes(sp.ref,sp.row_bytes,epn) for sp in state._specs}
peers=[None]*world;dist.all_gather_object(peers,homes)
checks=[];status="failed";quant_calls=[];raw_bytes_transport_passed=False;forced_evidence=[]
def quantize(x):
 quant_calls.append(list(x.shape));return torch_npu.npu_dynamic_mx_quant(x,dst_type=torch.float8_e4m3fn)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
try:
 for pattern,T in (("balanced",64),("hot",17),("remote_hot",1),("balanced_repeat",127),("remote_hot",65),("balanced_unequal",17+11*(rank%4))):
  sync()
  if pattern.startswith("balanced"):topk=(torch.arange(S*K).reshape(S,K)+rank*epn)%E
  elif pattern=="hot":topk=torch.arange(K)[None,:].expand(S,K).clone()
  else:topk=(((rank+world//2)%world)*epn+torch.arange(K))[None,:].expand(S,K).clone()
  topk=topk.int().to(dev);topk[T:].fill_(-1)
  hidden=torch.randn(T,H,dtype=torch.bfloat16,device=dev)*.125
  hidden_before=hidden.clone()
  probs=torch.zeros(S,8,dtype=torch.float32,device=dev);probs[:T,:K]=torch.softmax(torch.randn(T,K,device=dev),dim=-1)
  tpe=torch.bincount(topk[:T].flatten().long(),minlength=E).int()
  complete=[]
  def after(plan):
   side.wait_event(torch.npu.current_stream().record_event())
   with torch.npu.stream(side):
    b.prefetch_projections(plan,state.prefetch_views());done=side.record_event()
   complete.append(done)
  ctx["after_planning_hook"]=after
  old_input=torch.zeros(S,H,dtype=torch.bfloat16,device=dev);old_input[:T].copy_(hidden)
  old,_,cu,plan=b.dispatch(old_input,topk_experts_sk=topk,tokens_per_expert=tpe,num_tokens=T)
  old=old.clone();cu_ref=cu.clone()
  plan_ref={name:getattr(plan,name).clone() for name in ("dst","experts_to_copy","dup_counts")}
  counts_all=torch.cat((cu[:1],cu[1:]-cu[:-1])).long()
  counts=torch.cat((counts_all[rank*epn:(rank+1)*epn],counts_all[E:]))
  valid=int(counts.sum().cpu())
  expected_q,expected_sc=torch_npu.npu_dynamic_mx_quant(old[:valid],dst_type=torch.float8_e4m3fn) if valid else (None,None)
  old_y=calc(old,counts,complete[-1]);old_y=old_y.clone()
  padded=torch.zeros(S*K,H,dtype=torch.bfloat16,device=dev);padded[:old_y.shape[0]].copy_(old_y)
  old_final=b.combine_weighted(plan,padded,probs,num_tokens=T)[0][:T].clone()
  sync()
  q,sc,_,cu2,plan2=b.dispatch_mxfp8(hidden,topk_experts_sk=topk,tokens_per_expert=tpe,num_tokens=T,quantize=quantize,wire_library=a.library)
  assert len(complete)==2 and len(quant_calls)==len(checks)+1
  sync()
  assert torch.equal(cu_ref.cpu(),cu2.cpu())
  for name,val in plan_ref.items():
   current=getattr(plan2,name)
   # REAL_T planning intentionally leaves dst[T*K:] unspecified/stale.
   if name=="dst":val,current=val[:T*K],current[:T*K]
   assert torch.equal(val.cpu(),current.cpu()),name
  if valid:
   assert torch.equal(bits(q[:valid]),bits(expected_q)),("activation_bytes",rank,pattern)
   assert torch.equal(bits(sc[:valid]),bits(expected_sc)),("scale_bytes",rank,pattern)
  assert torch.equal(bits(hidden),bits(hidden_before))
  new_y=calc(q,counts,complete[-1],sc);sync()
  torch.testing.assert_close(new_y[:valid].float(),old_y[:valid].float(),rtol=0,atol=0)
  padded.zero_();padded[:new_y.shape[0]].copy_(new_y)
  new_final=b.combine_weighted(plan2,new_y.contiguous(),probs,num_tokens=T,fused_publish=True,row_counts=counts)[0][:T].clone();sync()
  assert torch.equal(bits(old_final),bits(new_final)),("weighted_combine",pattern)
  evidence=[]
  chosen=plan2.experts_to_copy[rank].cpu().tolist()
  for sp in state._specs:
   actual=hashes(state.slots[sp.name],sp.row_bytes,B)
   for slot,expert in enumerate(chosen):
    if expert<0:continue
    owner,index=divmod(expert,epn)
    assert actual[slot]==peers[owner][sp.name][index]
    evidence.append(dict(payload=sp.name,expert=expert,remote_rank=owner!=rank,sha256=actual[slot]))
  checks.append(dict(pattern=pattern,T=T,valid_rows=valid,fp8_bytes_equal=True,scale_bytes_equal=True,unsplit_mlp_bitwise_equal=True,bf16_weighted_combine_bitwise_equal=True,remote_prefetch=evidence))
  print("CHECK",rank,json.dumps(checks[-1]),flush=True)
  sync()
 # Adversarial all-256-byte payloads, including BF16-carrier NaN encodings.
 def raw_quant(x):
  rows=x.shape[0]
  q=(torch.arange(H,dtype=torch.int32,device=dev)%256).to(torch.uint8).expand(rows,H).contiguous().view(torch.float8_e4m3fn)
  sc=(torch.arange(H//32,dtype=torch.int32,device=dev)%256).to(torch.uint8).expand(rows,H//32).contiguous().reshape(rows,H//64,2)
  return q,sc
 q,sc,_,cu3,plan3=b.dispatch_mxfp8(hidden,topk_experts_sk=topk,tokens_per_expert=tpe,num_tokens=T,quantize=raw_quant,wire_library=a.library)
 torch.npu.current_stream().wait_event(complete[-1]);sync()
 exp_q,exp_sc=raw_quant(torch.empty(valid,H,dtype=torch.bfloat16,device=dev))
 assert torch.equal(bits(q[:valid]),bits(exp_q))
 assert torch.equal(bits(sc[:valid]),bits(exp_sc))
 raw_bytes_transport_passed=True
 # Force actual peer expert reads even if load planning needs no replicas.
 from ascend_moonep.planning import allocate_planning_outputs
 forced,_=allocate_planning_outputs(ctx)
 for r in range(world):forced.experts_to_copy[r].copy_(torch.arange(B,dtype=torch.int32,device=dev)+((r+world//2)%world)*epn)
 sync()
 side.wait_event(torch.npu.current_stream().record_event())
 with torch.npu.stream(side):
  b.prefetch_projections(forced,state.prefetch_views());done=side.record_event()
 torch.npu.current_stream().wait_event(done);sync()
 forced_evidence=[]
 for sp in state._specs:
  actual=hashes(state.slots[sp.name],sp.row_bytes,B)
  assert actual==peers[(rank+world//2)%world][sp.name]
  forced_evidence.append(dict(payload=sp.name,peer=(rank+world//2)%world,experts=B,sha256=actual))
 assert ctx["H"]==H
 dispatch_calls=[x for x in calls if x["operator"]=="Dispatch"]
 assert len(dispatch_calls)==2*len(checks)+1
 assert all(x["bool_arguments"][-1]==(a.backend=="udma") for x in dispatch_calls),dispatch_calls
 assert all(x["stream"]==int(side.npu_stream) for x in calls if x["operator"]=="PrefetchWeight")
 assert all(x["bool_arguments"][-1]==(a.backend=="udma") for x in calls if x["operator"]=="PrefetchWeight")
 mapped={ln.split()[-1] for ln in Path("/proc/self/maps").read_text().splitlines() if "libascend_moonep_kernels.so" in ln}
 assert mapped=={str(Path(a.library).parent/"libascend_moonep_kernels.so")},mapped
 status="passed"
finally:
 libs={ln.split()[-1] for ln in Path("/proc/self/maps").read_text().splitlines() if ".so" in ln and ("moonep" in ln or "libshmem" in ln)}
 record=dict(status=status,backend=a.backend,planning_backend="mte",world=world,rank=rank,host=socket.gethostname(),cross_node=True,prefetch_stream=2,home_gemm_split=False,fused_publish=True,udma_qp_layout=layout,model_loaded=False,visible=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),checks=checks,raw_bytes_transport_passed=raw_bytes_transport_passed,forced_prefetch=forced_evidence,quant_calls=quant_calls,actual_native_calls=calls,physical_peers=physical,mapped_libraries={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in libs})
 (out/f"rank{rank}.json").write_text(json.dumps(record,indent=2))
 if status=="passed":sync();ShmemRuntime.finalize();dist.destroy_process_group();faulthandler.cancel_dump_traceback_later()
print("TWO_CARD_MXFP8_PASSED",rank,a.backend,flush=True)
