import json,re,sys
from pathlib import Path
service,out=map(Path,sys.argv[1:])
text=(service/"service.log").read_text(errors="replace")
assert "[pf-stream] issue failed" not in text and "[pf-stream] install failed" not in text
ranks=set(map(int,re.findall(r"\[pf-stream-active r(\d+)\] mode=2 cache_aware=1",text)))
assert ranks==set(range(8)),ranks
(out/"OPTIMIZATION_AUDIT.json").write_text(json.dumps({"valid":True,"diagnostic_only":True})+"\n")
print("DIAGNOSTIC_PATH_AUDIT_OK")
