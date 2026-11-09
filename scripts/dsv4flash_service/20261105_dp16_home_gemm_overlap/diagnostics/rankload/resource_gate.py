#!/usr/bin/env python3
"""Time gate before every driver query, plus read-only card occupancy check."""
import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from tests.test_home_gmm_npu import require_time

def main():
    require_time()
    p=argparse.ArgumentParser()
    p.add_argument("--cards",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    cards=set(map(int,args.cards.split(",")))
    assert cards and cards<=set(range(8))
    raw=subprocess.check_output(["npu-smi","info"],text=True)
    idle=set(map(int,re.findall(r"No running processes found in NPU\s+(\d+)",raw)))
    # Any parsing uncertainty is a refusal to allocate.
    data=dict(at=dt.datetime.now(dt.timezone.utc).isoformat(),requested=sorted(cards),
              idle_cards=sorted(idle),eligible=cards<=idle,raw=raw)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,indent=2))
    if not data["eligible"]:raise SystemExit("Requested cards are not confirmed idle")
    print("CARDS_CONFIRMED_IDLE",sorted(cards),flush=True)
if __name__=="__main__":main()
