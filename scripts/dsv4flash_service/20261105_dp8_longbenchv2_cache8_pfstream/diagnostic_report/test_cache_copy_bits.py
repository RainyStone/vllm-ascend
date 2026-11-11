"""Synthetic raw-bit roundtrip through the same VMM BF16 carrier cache slices."""
import json
from pathlib import Path
import torch
import torch_npu
from ascend_moonep.buffer_c import create_vmm_physical, create_sym_tensor_from_ptr

ROOT=Path(__file__).resolve().parent
torch.npu.set_device(0)
device=torch.device("npu:0")
def raw(t):
    return create_sym_tensor_from_ptr(t.data_ptr(),[t.numel()*t.element_size()],torch.uint8,t.device)
pattern=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.uint8)
specs=[("w13",128,32768),("w13_scale",8,32768),("w2",64,32768),("w2_scale",4,32768)]
records=[];keep=[]
for name,rows,cols in specs:
    slots,_,_=create_vmm_physical([32,rows,cols],torch.bfloat16,0)
    cache,_,_=create_vmm_physical([64,8,rows,cols],torch.bfloat16,0)
    keep.extend([slots,cache])
    nbytes=rows*cols*2
    expected=pattern.repeat(nbytes//pattern.numel()).clone()
    src=slots[5]
    raw(src).copy_(expected.to(device))
    torch.npu.synchronize()
    assert torch.equal(raw(src).cpu(),expected),("source byte upload",name)
    for layer,index in [(0,0),(42,7),(63,3)]:
        cache[layer,index].copy_(src)
        slots[10].copy_(cache[layer,index])
        torch.npu.synchronize()
        middle=raw(cache[layer,index]).cpu()
        result=raw(slots[10]).cpu()
        bad_cache=int((middle!=expected).sum().item())
        bad_result=int((result!=expected).sum().item())
        records.append(dict(payload=name,layer=layer,cache_slot=index,bytes=nbytes,
                            cache_byte_mismatches=bad_cache,roundtrip_byte_mismatches=bad_result))
        print(json.dumps(records[-1]),flush=True)
        assert bad_cache==bad_result==0,(name,layer,index)
report=dict(passed=True,device="npu:0",synthetic=True,includes_all_bf16_bit_patterns=True,
            logical_cache_layers=64,cache_experts_per_layer=8,
            tests=records,note="Tests data-copy bit preservation and VMM slice offsets, not full model scheduling.")
(ROOT/"CACHE_COPY_BITS.json").write_text(json.dumps(report,indent=2)+"\n")
print("CACHE_COPY_BITS_OK",flush=True)
