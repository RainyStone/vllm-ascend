"""Audit the stream-only Slot profile after a real request, before measurement."""
import json,re,sys
from pathlib import Path
service,output=map(Path,sys.argv[1:])
environment=dict(line.split("=",1) for line in (service/"environment.txt").read_text().splitlines() if "=" in line)
expected={"MOONEP_SLOT_CACHE":"0","MOONEP_PF_STREAM":"2","MOONEP_EARLY_PREFETCH":"0","MOONEP_DISABLE_UDMA":"1"}
assert all(environment.get(k)==v for k,v in expected.items()),("Wrong Slot profile",environment)
log=(service/"service.log").read_text(errors="replace")
assert "[pf-stream] issue failed" not in log, "prefetch side-stream failed"
assert "[pf-stream] install failed" not in log, "prefetch hook installation failed"
assert "[slot-cache-active " not in log, "Persistent cache unexpectedly active"
ranks=set(map(int,re.findall(r"\[pf-stream-active r(\d+)\] mode=2 cache_aware=1",log)))
assert ranks==set(range(8)), ("side-stream path not confirmed on all ranks",ranks)
record={"valid":True,"profile":"stream_only","side_stream_ranks":sorted(ranks),
        "cache_capacity":0,"cache_hit_evidence":[],
        "note":"Confirms side-stream execution with persistent cache disabled; does not quantify overlap."}

# Verify newly selected transport and publication on the actual workers.
assert environment.get("VLLM_ASCEND_MOONEP_DISPATCH_DTYPE")=="mxfp8"
assert environment.get("VLLM_ASCEND_MOONEP_FUSED_PUBLISH")=="1"
transport_ranks=set(map(int,re.findall(r"\[moonep-activation-transport r(\d+)\] mode=mxfp8 backend=mte",log)))
assert transport_ranks==set(range(8)),("MXFP8 dispatch not confirmed on all ranks",transport_ranks)
assert "[home-gemm-active " not in log
workers={}
for dp,ep,pid in re.findall(r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)",log):
    if dp==ep:workers[int(dp)]=int(pid)
assert set(workers)==set(range(8))
mapped={}
for rank,pid in workers.items():
    maps=(Path("/proc")/str(pid)/"maps").read_text()
    for lib in ("libascend_moonep_mxfp8_wire.so","libascend_moonep_publish.so"):
        assert lib in maps,(rank,lib)
    mapped[rank]=sorted(set(line.split()[-1] for line in maps.splitlines() if "libascend_moonep_" in line))
record.update(dispatch_dtype="mxfp8",fused_publish=True,transport="mte",transport_ranks=sorted(transport_ranks),mapped_libraries=mapped,home_gemm_overlap=False)
print("MXFP8_AND_FUSED_PUBLISH_AUDIT_OK",sorted(transport_ranks))

(output/"OPTIMIZATION_AUDIT.json").write_text(json.dumps(record, indent=2)+"\n")
