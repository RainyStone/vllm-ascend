#!/usr/bin/env python3
"""CPU-only exact all-rank trace inventory, capped at two workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,subprocess,sys,json
root=Path(__file__).resolve().parent
mode=sys.argv[1];assert mode in ("off","slot_mte","slot_udma")
out=root/"data/diagnostics/optimized_profiling_20261106"/mode
env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
def run(rank):
 p=subprocess.run([sys.executable,str(root/"inventory_profile_trace.py"),mode,str(rank)],env=env,text=True,capture_output=True)
 (out/f"rank{rank:02d}_inventory.log").write_text(p.stdout+p.stderr)
 assert p.returncode==0,(mode,rank,p.stderr)
 return rank
with ThreadPoolExecutor(max_workers=2) as pool:
 for rank in pool.map(run,range(16)):print("TRACE_INVENTORY_OK",mode,rank,flush=True)
