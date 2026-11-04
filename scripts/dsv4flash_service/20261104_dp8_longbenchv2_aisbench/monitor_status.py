#!/usr/bin/env python3
"""Compact, read-only view of this experiment's latest client logs."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    pair,mode=sys.argv[1:]
    paths=list((ROOT/"data/aisbench").glob(f"{pair}-{mode}-*"))
    if not paths:
        print("NO_CLIENT");return
    run=max(paths,key=lambda p:p.stat().st_mtime)
    status={"client_dir":str(run),"rounds":[]}
    for path in sorted(run.glob("round_*")):
        log=path/"aisbench_debug.log"
        item={"round":path.name,"validated":False}
        val=path/"validation.json"
        if val.exists():
            obj=json.loads(val.read_text());item.update(validated=obj["valid"],requests=obj["requests"])
        if log.exists():
            with log.open("rb") as f:
                f.seek(max(0,log.stat().st_size-9000));s=f.read().decode(errors="replace")
            s=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]","",s)
            matches=re.findall(r"(\d+)/316",s)
            if matches:item["progress"]=int(matches[-1])
            lines=[x for x in s.replace("\r","\n").splitlines() if x.strip()]
            item["tail"]=lines[-4:]
            item["log_bytes"]=log.stat().st_size
        status["rounds"].append(item)
    print(json.dumps(status,ensure_ascii=False))
if __name__=="__main__":main()
