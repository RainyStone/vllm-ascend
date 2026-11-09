#!/usr/bin/env python3
"""Forced-remote weighted UDMA regression on two physical hosts; no model."""
import argparse, ctypes, datetime, faulthandler, hashlib, json, os, socket
from pathlib import Path
import torch
import torch_npu
import torch.distributed as dist
from ascend_moonep.api import Buffer
from ascend_moonep.combine_publish import CombinePublisher
from ascend_moonep.planning import allocate_planning_outputs
from ascend_moonep.shmem_runtime import ShmemRuntime
parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
parser.add_argument("--shmem-port",type=int,required=True)
parser.add_argument("--library",required=True)
args=parser.parse_args()
faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
assert world==2 and int(os.environ["LOCAL_RANK"])==0
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"]=="1"
node_ip=["141.61.45.118","141.61.45.124"][rank]
remote_rank=1-rank
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
candidate=Path(os.environ["MOONEP_UDMA_TEST_LIBRARY"]).resolve()
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=="49008deff48dd883982f4ee3518b0a919f98fed6ec994d063dab33627b64ae90"
assert hashlib.sha256(Path(args.library).read_bytes()).hexdigest()=="e69aafba0bb53c235fa418409f41eeba7339d9a4e98bb6abe14bbc872856dacc"
global_symbols=ctypes.CDLL(None)
abi=global_symbols.moonep_udma_weight_reuse_fix_abi;abi.restype=ctypes.c_int
assert abi()==2
class DlInfo(ctypes.Structure):
    _fields_=[("dli_fname",ctypes.c_char_p),("dli_fbase",ctypes.c_void_p),
              ("dli_sname",ctypes.c_char_p),("dli_saddr",ctypes.c_void_p)]
dl=ctypes.CDLL("libdl.so.2");dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)]
proof=dict(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
           abi=abi(),symbols={},node_ip=node_ip,rank=rank,physical_card=1)
for name in ["moonep_udma_combine_weighted_kernel_do","moonep_udma_weight_reuse_fix_abi"]:
    info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(global_symbols,name),ctypes.c_void_p),ctypes.byref(info))
    assert Path(info.dli_fname.decode()).resolve()==candidate
    proof["symbols"][name]=info.dli_fname.decode()
for name in ["/etc/hccl_rootinfo.json","/usr/local/Ascend/driver/topo/950/atlas_850_1.json"]:
    p=Path(name);assert p.exists(),name
    proof[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print("BACKEND_LIBRARY_PROOF",json.dumps(proof),flush=True)
torch.npu.set_device(0);device=torch.device("npu",0)
dist.init_process_group("gloo",timeout=datetime.timedelta(seconds=180))
peers=[None]*2
dist.all_gather_object(peers,dict(ip=node_ip,hostname=socket.gethostname(),rank=rank,card=1))
assert {p["ip"] for p in peers}=={"141.61.45.118","141.61.45.124"}
assert len({p["hostname"] for p in peers})==2
proof["physical_peers"]=peers
(out/f"proof_rank{rank}.json").write_text(json.dumps(proof,indent=2))
print("PHYSICAL_PEERS",json.dumps(peers),flush=True)
# Initialize the independent native reference before symmetric peer mappings.
torch_npu.npu_moe_token_unpermute(torch.zeros(6,16,dtype=torch.bfloat16,device=device),
    torch.arange(6,dtype=torch.int32,device=device),
    probs=torch.ones(1,6,dtype=torch.float32,device=device),restore_shape=(1,16))
torch.npu.synchronize()
geometries=[(128,6,4096),(32768,6,128),(64,8,2048),(64,1,64),(64,16,128)]
buffers={g:Buffer(S=g[0],K=g[1],H=g[2],E=32,B=16,num_ep_ranks=world,
                  group=dist.group.WORLD,use_udma=True) for g in geometries}
ShmemRuntime.init_with_ip_port(group=dist.group.WORLD,
    ip_port=f"tcp://141.61.45.118:{args.shmem_port}",enable_udma=True,qp_count=2,prefetch_overlap=True)
for b in buffers.values():
    b._ensure_shmem_views();ctx=b._require_ctx()
    assert not ctx["_planning_use_udma"]
    assert all(ctx["_"+stage+"_use_udma"] for stage in ["dispatch","combine","prefetch"])
    b._combine_publisher=CombinePublisher(args.library)
import ascend_moonep.launchers as launchers
native=torch.classes.AscendMoonEP.CombineUrma()
calls=[]
class Recorder:
    def __getattr__(self,name):return getattr(native,name)
    def compute_weighted(self,*a,**kw):
        calls.append("CombineUrma.compute_weighted")
        return native.compute_weighted(*a,**kw)
launchers._COMBINE_UDMA=Recorder()
records=[];routes=[]
def values(S,K,H,owner,version):
    rows=torch.arange(S*K,dtype=torch.int32)[:,None]
    channels=torch.arange(H,dtype=torch.int32)[None,:]
    return (((rows*7+channels*3+owner*19+version*23)%127-63).float()/32).bfloat16()
def sync():
    torch.npu.synchronize();dist.barrier()
def check(name,result,reference):
    actual=result.cpu();reference=reference.contiguous()
    different=actual.view(torch.int16)!=reference.view(torch.int16)
    both_zero=(actual==0)&(reference==0)
    rec=dict(name=name,rank=rank,elements=actual.numel(),
             value_mismatches=int((actual!=reference).sum()),
             nonzero_bit_mismatches=int((different&~both_zero).sum()),
             signed_zero_differences=int((different&both_zero).sum()),
             finite=bool(torch.isfinite(actual.float()).all()),
             max_abs=float((actual.float()-reference.float()).abs().max()))
    records.append(rec)
    checks=[None]*2;dist.all_gather_object(checks,rec)
    if rank==0:print("CHECK",json.dumps(checks),flush=True)
    assert all(x["value_mismatches"]==0 and x["nonzero_bit_mismatches"]==0 and x["finite"] for x in checks),checks
    (out/f"checks_rank{rank}.json").write_text(json.dumps(records,indent=2))
def run_case(S,K,H,T,kind="normal",version=0):
    b=buffers[S,K,H];ctx=b._require_ctx();plan,_=allocate_planning_outputs(ctx)
    N=S*K
    local_T=(63 if rank==0 else 33) if kind=="unequal" else T
    remote_T=(63 if remote_rank==0 else 33) if kind=="unequal" else T
    indices=torch.arange(N,dtype=torch.int64)
    if kind=="unequal":indices%=remote_T*K
    raw=(remote_rank*N+indices).view(S,K)
    if kind=="alias" and K>1:raw[:,1]=-raw[:,0]-1
    if kind=="invalid":raw[:,-1]=world*N+13
    plan.dst.copy_(raw.flatten().int().to(device))
    plan.experts_to_copy.fill_(-1);plan.dup_groups.zero_();plan.dup_loffs.zero_();plan.dup_counts.zero_()
    hidden=values(S,K,H,rank,version)
    hidden[local_T*K:]=float("nan")
    hidden_npu=hidden.to(device)
    nums=(torch.arange(S)[:,None]*3+torch.arange(K)[None,:]*5+rank*7+version*11)%17+1
    weights=nums.float()/nums.sum(1,keepdim=True)
    if kind=="zero":weights.zero_()
    padded=torch.full((S,(K+7)//8*8),float("nan"),dtype=torch.float32)
    padded[:local_T,:K]=weights[:local_T]
    w_npu=padded.to(device)
    decoded=torch.where(raw[:local_T]<0,-raw[:local_T]-1,raw[:local_T])
    valid=(decoded>=0)&(decoded<world*N)
    assert bool(((decoded//N)[valid]==remote_rank).all())
    assert bool(((decoded%N)[valid]<remote_T*K).all())
    routes.append(dict(S=S,K=K,H=H,T=local_T,kind=kind,version=version,
                       valid_routes=int(valid.sum()),remote_routes=int(valid.sum()),
                       crosses_token_window=(local_T+31)//32 > 3072//K))
    rows=values(S,K,H,remote_rank,version)[(decoded%N).flatten()].view(local_T,K,H)
    rows[~valid]=0
    exact=torch.zeros(local_T,H,dtype=torch.float32);plain=torch.zeros_like(exact)
    for k in range(K):
        exact.add_(rows[:,k].float()*weights[:local_T,k:k+1])
        plain.add_(rows[:,k].float())
    native_reference=torch_npu.npu_moe_token_unpermute(
        rows.reshape(local_T*K,H).to(device),torch.arange(local_T*K,dtype=torch.int32,device=device),
        probs=weights[:local_T].contiguous().to(device),restore_shape=(local_T,H))
    sync()
    counts=torch.tensor([local_T*K],dtype=torch.int64,device=device)
    label=f"S{S}_K{K}_H{H}_T{local_T}_{kind}_v{version}"
    for mode in ["old","fused","trimmed"]:
        source=hidden_npu if mode!="trimmed" else hidden_npu[:local_T*K]
        result=b.combine_weighted(plan,source,w_npu,num_tokens=local_T,
            fused_publish=mode!="old",row_counts=counts if mode=="fused" else None)[0][:local_T]
        check(label+"_"+mode+"_fp32",result,exact.bfloat16())
        check(label+"_"+mode+"_native",result,native_reference.cpu())
        if mode=="fused":
            assert torch.equal(ctx["hidden_buf_local"][:local_T*K].cpu(),hidden[:local_T*K])
            assert bool((ctx["hidden_buf_local"][local_T*K:].cpu()==0).all())
    assert torch.equal(plan.dst.cpu(),raw.flatten().int()),"Caller plan modified"
    if kind=="normal" and version==0:
        sync();legacy=b.combine(plan,hidden_npu,num_tokens=local_T)[0][:local_T]
        check(label+"_legacy_unweighted",legacy,plain.bfloat16())
    # Reuse both symmetric and local gather workspaces with changed FP32 coefficients.
    w_npu[:local_T,:K].mul_(0.5);sync()
    second=b.combine_weighted(plan,hidden_npu,w_npu,num_tokens=local_T,
                             fused_publish=True,row_counts=counts)[0][:local_T]
    check(label+"_changed_coefficients",second,(exact*0.5).bfloat16())
status="failed"
try:
    cases=[(128,6,4096,t) for t in [1,17,63,65,127,128]]
    cases += [(32768,6,128,t) for t in [16385,32768]]
    cases += [(64,8,2048,63),(64,1,64,63),(64,16,128,63)]
    for case in cases:run_case(*case)
    for kind in ["alias","invalid","zero","unequal"]:run_case(128,6,4096,63,kind)
    # Refresh every source payload as well as coefficients, rather than reusing old bytes.
    run_case(128,6,4096,65,"normal",1)
    run_case(32768,6,128,16385,"normal",1)
    assert calls and all(x=="CombineUrma.compute_weighted" for x in calls)
    status="passed"
finally:
    mapped={line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and ("moonep" in line or "shmem" in line)}
    record=dict(status=status,rank=rank,world_size=2,node_ip=node_ip,host=socket.gethostname(),
                physical_card=1,model_loaded=False,backend="udma",planning_executed=False,
                dispatch_executed=False,prefetch_executed=False,all_valid_routes_remote=True,
                proof=proof,checks=records,routes=routes,called_backend_methods=calls,
                mapped_libraries={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in mapped})
    (out/f"rank{rank}.json").write_text(json.dumps(record,indent=2))
    if status=="passed":
        sync();ShmemRuntime.finalize()
        for b in buffers.values():b.destroy()
        dist.destroy_process_group();faulthandler.cancel_dump_traceback_later()
print("CROSS_NODE_UDMA_SYNC_PASSED",rank,flush=True)
