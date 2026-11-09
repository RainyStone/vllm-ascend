#!/usr/bin/env python3
"""Small weighted-combine transport test; no vLLM, model weights or model service."""
import argparse
import faulthandler
import datetime
import json
import os
import socket
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--backend",choices=["mte","udma"],required=True)
parser.add_argument("--master-ip",default="141.61.45.118")
parser.add_argument("--shmem-port",type=int,required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--node-ip",required=True)
parser.add_argument("--installed",action="store_true")
args=parser.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
print("[stage] load_runtime", os.environ.get("RANK"), flush=True)
from load_staged import load_staged
if args.installed:
    loaded={"source":"/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep",
            "mode":"installed_original_repo"}
else:
    loaded=load_staged()
import torch
import torch.distributed as dist
import torch_npu
from ascend_moonep.api import Buffer
from ascend_moonep.planning import allocate_planning_outputs
from ascend_moonep.shmem_runtime import ShmemRuntime

rank,world=int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"])
assert world in (2,16),"Use one or eight ranks per physical node"
remote_rank=(rank+world//2)%world
torch.npu.set_device(int(os.environ.get("LOCAL_RANK","0")))
device=torch.device("npu",torch.npu.current_device())
print("[stage] device_ready", rank, torch.npu.current_device(), flush=True)
dist.init_process_group("gloo",timeout=datetime.timedelta(seconds=180))
print("[stage] gloo_ready", rank, flush=True)
outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
records=[]
buffers={}
# Native reference operator is initialized before symmetric peer mappings.
torch_npu.npu_moe_token_unpermute(torch.zeros(6,16,dtype=torch.bfloat16,device=device),
    torch.arange(6,dtype=torch.int32,device=device),
    probs=torch.ones(1,6,dtype=torch.float32,device=device),restore_shape=(1,16))
torch.npu.synchronize()
print("[stage] native_reference_ready", rank, flush=True)
for S,K,H in [(32,6,4096),(32,8,2048),(16,1,64),(16,16,128)]:
    buffers[S,K,H]=Buffer(S=S,H=H,K=K,E=256,B=256//world,num_ep_ranks=world,
                         group=dist.group.WORLD,use_udma=args.backend=="udma")
print("[stage] buffers_ready; shmem_init_start", rank, flush=True)
ShmemRuntime.init_with_ip_port(group=dist.group.WORLD,
    ip_port=f"tcp://{args.master_ip}:{args.shmem_port}",
    enable_udma=args.backend=="udma",qp_count=2 if args.backend=="udma" else 0)
print("[stage] shmem_ready", rank, flush=True)
for b in buffers.values():
    b._ensure_shmem_views()
    ctx=b._require_ctx()
    assert not ctx["_planning_use_udma"]
    for stage in ("dispatch","combine","prefetch"):
        assert bool(ctx["_"+stage+"_use_udma"]) is (args.backend=="udma")
import ascend_moonep.launchers as launchers
method_calls=[]
class BackendRecorder:
    def __init__(self,operator,label):self.operator,self.label=operator,label
    def __getattr__(self,name):return getattr(self.operator,name)
    def compute_weighted(self,*arguments):
        method_calls.append(self.label)
        return self.operator.compute_weighted(*arguments)
if args.backend=="udma":
    launchers._COMBINE_UDMA=BackendRecorder(
        torch.classes.AscendMoonEP.CombineUrma(),"CombineUrma.compute_weighted")
else:
    launchers._COMBINE=BackendRecorder(
        torch.classes.AscendMoonEP.Combine(),"Combine.compute_weighted")
assert launchers._env_use_udma("combine") is (args.backend=="udma")


def values(S,K,H,owner,version=0):
    rows=torch.arange(S*K)[:,None]
    channels=torch.arange(H)[None,:]
    return (((rows*7+channels*3+owner*19+version*23)%127-63).float()/32).bfloat16()
def synchronize():
    torch.npu.synchronize();dist.barrier()
def check(name,result,reference):
    actual=result.cpu()
    record=dict(name=name,rank=rank,elements=actual.numel(),
                mismatches=int((actual!=reference).sum()),
                finite=bool(torch.isfinite(actual.float()).all()),
                max_abs=float((actual.float()-reference.float()).abs().max()))
    records.append(record)
    gathered=[None]*world;dist.all_gather_object(gathered,record)
    if rank==0:print(json.dumps(dict(case=name,checks=gathered)),flush=True)
    assert all(d["mismatches"]==0 and d["finite"] for d in gathered),gathered

def run_case(S,K,H,T,kind="normal",version=0):
    print("[stage] case_start", rank, S,K,H,T,kind,version,flush=True)
    b=buffers[S,K,H];plan,_=allocate_planning_outputs(b._require_ctx())
    N=S*K;T=T if kind!="unequal" else (17 if rank==0 else 9)
    # Every valid route reads the other physical node. No local row can supply
    # the expected values, so a passing result cannot come from a self-copy.
    remote_T=17 if remote_rank==0 else 9
    indices=torch.arange(N) if kind!="unequal" else torch.arange(N)%(remote_T*K)
    raw=(remote_rank*N+indices).view(S,K).long()
    if kind=="alias" and K>1:raw[:,1]=-raw[:,0]-1
    if kind=="invalid":raw[:,-1]=world*N+13
    plan.dst.copy_(raw.flatten().int().to(device))
    plan.experts_to_copy.fill_(-1);plan.dup_groups.zero_();plan.dup_loffs.zero_();plan.dup_counts.zero_()
    hidden=values(S,K,H,rank,version)
    if kind=="unequal":hidden[T*K:]=float("nan")
    elif T<S:hidden[T*K:]=float("nan")
    nums=(torch.arange(S)[:,None]*3+torch.arange(K)[None,:]*5+rank*7)%17+1
    weights=nums.float()/nums.sum(1,keepdim=True)
    if kind=="zero":weights.zero_()
    padded=torch.full((S,(K+7)//8*8),float("nan"),dtype=torch.float32)
    padded[:T,:K]=weights[:T]
    hidden_npu=hidden.to(device);w_npu=padded.to(device)
    decoded=torch.where(raw[:T]<0,-raw[:T]-1,raw[:T])
    valid=(decoded>=0)&(decoded<world*N)
    rows=values(S,K,H,remote_rank,version)[(decoded%N).flatten()].view(T,K,H)
    rows[~valid]=0
    exact=torch.zeros(T,H,dtype=torch.float32)
    plain=torch.zeros_like(exact)
    for k in range(K):
        exact.add_(rows[:,k].float()*weights[:T,k:k+1])
        plain.add_(rows[:,k].float())
    native=torch_npu.npu_moe_token_unpermute(
        rows.reshape(T*K,H).to(device),
        torch.arange(T*K,dtype=torch.int32,device=device),
        probs=weights[:T].contiguous().to(device),restore_shape=(T,H))
    synchronize()
    result=b.combine_weighted(plan,hidden_npu,w_npu,num_tokens=T)[0][:T]
    torch.npu.synchronize()
    name=f"{args.backend}_S{S}_K{K}_H{H}_T{T}_{kind}_v{version}"
    check(name+"_fp32_reference",result,exact.bfloat16())
    check(name+"_native_reference",result,native.cpu())
    assert torch.equal(plan.dst.cpu(),raw.flatten().int()),"The caller plan must remain unchanged"
    if kind=="normal":
        synchronize()
        legacy=b.combine(plan,hidden_npu,num_tokens=T)[0][:T]
        check(name+"_legacy_unweighted",legacy,plain.bfloat16())
    # Same-stream reuse with changed coefficients.
    w_npu[:T,:K].mul_(0.5)
    synchronize()
    second=b.combine_weighted(plan,hidden_npu,w_npu,num_tokens=T)[0][:T]
    check(name+"_scratch_reuse",second,(exact*0.5).bfloat16())

status="failed"
try:
    for case in [(32,6,4096,1),(32,6,4096,17),(32,6,4096,32),
                 (32,8,2048,17),(16,1,64,7),(16,16,128,9)]:
        run_case(*case)
    for kind in ["alias","invalid","zero","unequal"]:
        run_case(32,6,4096,17,kind)
    run_case(32,8,2048,17,"alias",1)
    synchronize()
    assert method_calls and all(x==("CombineUrma.compute_weighted" if args.backend=="udma" else "Combine.compute_weighted") for x in method_calls)
    status="passed"
finally:
    record=dict(status=status,backend=args.backend,rank=rank,world_size=world,
                host=socket.gethostname(),node_ip=args.node_ip,master=args.master_ip,
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                model_loaded=False,all_valid_routes_remote=True,called_backend_methods=method_calls,loaded=loaded,checks=records)
    (outdir/f"rank{rank}.json").write_text(json.dumps(record,indent=2))
    if status=="passed":
        faulthandler.cancel_dump_traceback_later()
        synchronize();ShmemRuntime.finalize()
        for b in buffers.values():b.destroy()
        dist.destroy_process_group()
if rank==0:print("CROSS_NODE_SMALL_TEST_PASSED",args.backend,flush=True)
