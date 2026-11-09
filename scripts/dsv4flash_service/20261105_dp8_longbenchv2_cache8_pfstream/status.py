"""Read-only progress summary; does not probe devices or send model requests."""
import json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parent
current=json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())
session=Path(current["session"])
state=json.loads((session/"remaining_state.json").read_text())
out={"status":current["status"],"state":state,"validated_cases":[],"inflight_perf_records":[]}
for p in sorted((session/"cases").glob("pair*/case_result.json")):
    v=json.loads(p.read_text())
    if v.get("valid"):out["validated_cases"].append(p.parent.name)
for p in (ROOT/"data/aisbench").glob("*/round_*/aisbench/*/performances/*/tmp/*.db"):
    try:
        con=sqlite3.connect(p.resolve().as_uri()+"?mode=ro",uri=True,timeout=1)
        count=con.execute("SELECT COUNT(*) FROM numpy_store").fetchone()[0]
        con.close()
        out["inflight_perf_records"].append({"case":p.parents[6].name,"records":count})
    except sqlite3.Error:pass
if (session/"results.json").exists():
    result=json.loads((session/"results.json").read_text())
    out["completed_pairs"]=result["completed_pairs"]
    out["overall"]=result["overall"]
print(json.dumps(out,ensure_ascii=False))
