#!/usr/bin/env python3
"""CPU offline profiling analysis. Does not create an NPU context."""
import sys
if __name__=='__main__':
 import torch,torch_npu
 from torch_npu.profiler.profiler import analyse
 assert not torch.npu.is_initialized()
 print('CPU_OFFLINE_ANALYSE_BEGIN',sys.argv[1],flush=True)
 analyse(sys.argv[1],max_process_number=int(sys.argv[2]) if len(sys.argv)>2 else 1)
 assert not torch.npu.is_initialized()
 print('CPU_OFFLINE_ANALYSE_DONE',sys.argv[1],flush=True)
