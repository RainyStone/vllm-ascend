import json,re,sys
from pathlib import Path
service,out=map(Path,sys.argv[1:])
text=(service/"service.log").read_text(errors="replace")
assert "[pf-stream] issue failed" not in text and "[pf-stream] install failed" not in text
assert re.search(r"\[slot-cache-active r\d+\] layer=\d+ capacity=8 hits=[1-9]",text)
(out/"OPTIMIZATION_AUDIT.json").write_text(json.dumps({"valid":True,"diagnostic_only":True})+"\n")
print("DIAGNOSTIC_PATH_AUDIT_OK")
