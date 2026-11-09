#!/usr/bin/env python3
"""Model-free single-card byte framing and quantization/permutation equivalence."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--library",required=True);p.add_argument("--output",required=True);a=p.parse_args()
import torch,torch_npu
torch.npu.set_device(0)
from ascend_moonep.mxfp8_transport import Mxfp8Wire,carrier_view,wire_geometry
wire=Mxfp8Wire(a.library);results=[]
for rows in (1,17,65,8192):
 h=4096
 # Adversarial byte patterns prove no BF16 carrier conversion/canonicalization.
 x=(torch.arange(rows*h).reshape(rows,h)%256).to(torch.uint8).npu().view(torch.float8_e4m3fn)
 s=(torch.arange(rows*128).reshape(rows,64,2)%256).to(torch.uint8).npu()
 storage=torch.full((rows,h),123.,dtype=torch.bfloat16,device="npu")
 packet=carrier_view(storage,rows,h)
 tail=storage.view(torch.uint8).reshape(-1)[rows*4224:].clone()
 wire.pack(x,s,packet)
 xo=torch.empty_like(x);so=torch.empty_like(s);wire.unpack(packet,xo,so);torch.npu.synchronize()
 assert torch.equal(xo.view(torch.uint8).cpu(),x.view(torch.uint8).cpu())
 assert torch.equal(so.cpu(),s.cpu())
 assert torch.equal(tail.cpu(),storage.view(torch.uint8).reshape(-1)[rows*4224:].cpu())
 results.append(dict(stage="all_raw_bytes_roundtrip",rows=rows,passed=True))
 # Same native MXFP quantizer before vs after row permutation/duplication.
 bf=(torch.randn(rows,h,device="npu",dtype=torch.bfloat16)*.5)
 bf[0].zero_()
 if rows>1:bf[1].fill_(256)
 q,scale=torch_npu.npu_dynamic_mx_quant(bf,dst_type=torch.float8_e4m3fn)
 oscale=torch.empty_like(scale)
 wire.pack(q,scale,packet);wire.unpack(packet,xo,oscale)
 # Index on CPU only to avoid extra NPU index type/format ambiguities.
 idx=torch.arange(rows-1,-1,-1).repeat_interleave(2)
 ref_bf=bf.cpu()[idx].npu()
 qr,sr=torch_npu.npu_dynamic_mx_quant(ref_bf,dst_type=torch.float8_e4m3fn)
 torch.npu.synchronize()
 assert torch.equal(q.cpu().view(torch.uint8)[idx],qr.cpu().view(torch.uint8))
 assert torch.equal(scale.view(torch.uint8).cpu().reshape(rows,-1)[idx],sr.view(torch.uint8).cpu().reshape(2*rows,-1))
 assert torch.equal(xo.view(torch.uint8).cpu(),q.view(torch.uint8).cpu())
 assert torch.equal(oscale.view(torch.uint8).cpu(),scale.view(torch.uint8).cpu())
 results.append(dict(stage="native_quant_commutes_with_row_duplication",rows=rows,passed=True))
 if rows==8192:
  for _ in range(3):wire.pack(q,scale,packet)
  torch.npu.synchronize();start=time.perf_counter()
  for _ in range(20):wire.pack(q,scale,packet)
  torch.npu.synchronize();pack=(time.perf_counter()-start)*1000/20
  oscale=torch.empty_like(scale)
  start=time.perf_counter()
  for _ in range(20):wire.unpack(packet,xo,oscale)
  torch.npu.synchronize();unpack=(time.perf_counter()-start)*1000/20
  results.append(dict(stage="rough_host_launch_inclusive_timing",rows=rows,pack_ms=pack,unpack_ms=unpack))
Path(a.output).write_text(json.dumps(dict(passed=True,model_loaded=False,visible=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),library=str(wire.library_path),checks=results),indent=2))
print("SINGLE_CARD_MXFP8_PASSED",json.dumps(results),flush=True)
