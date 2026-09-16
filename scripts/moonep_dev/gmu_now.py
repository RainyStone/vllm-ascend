import torch, torch_npu
worst = 1.0
for d in (4, 5, 6, 7):
    try:
        torch.npu.set_device(d); f, t = torch.npu.mem_get_info(d)
        worst = min(worst, f / t)
    except Exception:
        worst = 0.0
print(f"{max(0.30, worst - 0.02):.2f}")
