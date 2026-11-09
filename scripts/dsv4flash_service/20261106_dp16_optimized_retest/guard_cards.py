#!/usr/bin/env python3
"""Read-only scoped resource gate. Unknown occupancy never qualifies."""
import argparse,datetime,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--cards",required=True);p.add_argument("--output",required=True);a=p.parse_args()
cards=set(map(int,a.cards.split(",")));assert cards and cards<=set(range(8))
raw=subprocess.check_output(["npu-smi","info"],text=True)
idle=set(map(int,re.findall(r"No running processes found in NPU\s+(\d+)",raw)))
target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
data=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested=sorted(cards),idle_cards=sorted(idle),eligible=cards<=idle,raw=raw)
target.write_text(json.dumps(data,indent=2))
if not data["eligible"]:raise SystemExit("Requested cards are not confirmed idle")
print("CARDS_CONFIRMED_IDLE",sorted(cards),flush=True)
