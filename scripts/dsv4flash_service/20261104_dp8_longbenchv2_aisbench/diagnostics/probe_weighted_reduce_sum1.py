#!/usr/bin/env python3
"""Isolate weighted BF16 reduction semantics; no model weights or service involved."""
import json,sys
from pathlib import Path
import torch,torch_npu
torch.set_num_threads(1)
torch.manual_seed(1024)
torch.npu.set_device(0)
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
results=[]
for tokens in [1,16,128]:
    k,h=6,4096
    x_cpu=torch.randn(tokens,k,h,dtype=torch.float32).to(torch.bfloat16)
    p_cpu=torch.rand(tokens,k,dtype=torch.float32)
    p_cpu=p_cpu/p_cpu.sum(dim=1,keepdim=True)*1.0
    x=x_cpu.reshape(tokens*k,h).npu()
    probs=p_cpu.npu();index=torch.arange(tokens*k,dtype=torch.int32).npu()
    native=torch_npu.npu_moe_token_unpermute(permuted_tokens=x,sorted_indices=index,probs=probs,restore_shape=(tokens,h))
    native_repeat=torch_npu.npu_moe_token_unpermute(permuted_tokens=x,sorted_indices=index,probs=probs,restore_shape=(tokens,h))
    weighted=x.reshape(tokens,k,h).float()*probs.unsqueeze(-1)
    acc_before=torch.zeros(tokens,h,dtype=torch.float32,device=x.device)
    acc_after=torch.zeros_like(acc_before)
    for j in range(k):
        acc_before+=weighted[:,j]
        acc_after+=weighted[:,j].to(torch.bfloat16).float()
    pre_round_each_expert=acc_after.to(torch.bfloat16)
    fused_fp32_sum=acc_before.to(torch.bfloat16)
    torch.npu.synchronize()
    tensors=dict(native=native.cpu(),native_repeat=native_repeat.cpu(),
                 round_each_expert=pre_round_each_expert.cpu(),sum_fp32_then_round=fused_fp32_sum.cpu())
    row=dict(tokens=tokens,topk=k,hidden_size=h)
    for label,a,b in [('native_vs_repeat',tensors['native'],tensors['native_repeat']),
                      ('native_vs_round_each_expert',tensors['native'],tensors['round_each_expert']),
                      ('native_vs_sum_then_round',tensors['native'],tensors['sum_fp32_then_round'])]:
        delta=(a.float()-b.float()).abs()
        row[label]=dict(mismatched_elements=int((a!=b).sum()),elements=a.numel(),
                        max_abs=float(delta.max()),mean_abs=float(delta.mean()))
    results.append(row)
    torch.save(dict(x=x_cpu,probs=p_cpu,**tensors),out/f'tokens_{tokens}.pt')
payload=dict(route_weight_sum=1.0,results=results,scope='Isolated native weighted-unpermute vs two reduction formulas; not an end-to-end proof of the observed first-token cause.')
(out/'RESULT.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
