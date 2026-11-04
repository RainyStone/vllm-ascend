#!/usr/bin/env python3
"""Read-only, concise status of the current matrix."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
run=Path(json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())["session"])
state=json.loads((run/"remaining_state.json").read_text())
out={"event":state["event"],"case":state.get("case"),"completed_cases":len(list((run/"cases").glob("*/case_result.json")))}
if state["event"]=="benchmark_running":
    pair,mode=state["case"].rsplit("_",1)
    clients=list((ROOT/"data/aisbench").glob(f"{pair}-{mode}-*-{state['client_pid']}"))
    if clients:
        client=clients[0];out["client_dir"]=str(client);out["rounds"]=[]
        for folder in sorted(client.glob("round_*")):
            item={"round":folder.name}
            val=folder/"validation.json"
            if val.exists():
                obj=json.loads(val.read_text());item.update(valid=obj["valid"],requests=obj["requests"])
            log=folder/"aisbench_debug.log"
            if log.exists():
                with log.open("rb") as f:
                    f.seek(max(0,log.stat().st_size-10000));text=f.read().decode(errors="replace")
                counts=re.findall(r"(\d+)/316",text)
                if counts:item["completed_requests"]=int(counts[-1])
                failure=re.findall(r"FAIL=(\d+)",text)
                if failure:item["observed_failed_requests"]=int(failure[-1])
                if not counts and not val.exists():item["phase"]="preparing_or_warmup"
            out["rounds"].append(item)
elif state["event"]=="stopped_on_error":
    out["error"]=state.get("error")
print(json.dumps(out,ensure_ascii=False))
