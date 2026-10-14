#!/usr/bin/env python3
"""占卡器：在指定卡上驻留显存、不跑计算、不碰别人的进程。

用法（只占卡 0-3，其它卡不管）：
  setsid nohup env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HOLD_GB=104 \
    python3 hold_cards.py > /tmp/mc2_probe/hold_cards.log 2>&1 &
  echo $! > /tmp/mc2_probe/hold_cards.pid      # 编排器只按这个 pid 停它
放卡：kill $(cat /tmp/mc2_probe/hold_cards.pid)
"""
import os
import time

import torch
import torch_npu  # noqa: F401

GB = int(os.environ.get("HOLD_GB", "104"))
n = GB * (1 << 30) // 2  # bf16 = 2 字节

held = []
for d in range(torch.npu.device_count()):
    torch.npu.set_device(d)
    held.append(torch.empty(n, dtype=torch.bfloat16, device=f"npu:{d}"))
    free, total = torch.npu.mem_get_info()
    print(f"[hold] npu:{d} 已驻留 {GB} GiB（free {free >> 30} / total {total >> 30} GiB）", flush=True)
print("[hold] 全部就位", flush=True)

while True:
    time.sleep(3600)
