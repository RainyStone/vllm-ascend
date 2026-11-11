"""Audit the requested dispatch transport on all eight workers before timing."""
import json
import re
import sys
from pathlib import Path

from verify_wire import STAGED, digest

service, output = map(Path, sys.argv[1:])
profile = "mxfp8"
expected = {
    "mxfp8": ("mxfp8", "0"),
    "bf16_copy": ("bf16", "0"),
    "bf16_zero_copy": ("bf16", "1"),
}
dtype, in_shard = expected[profile]
settings = {
    "MOONEP_SLOT_CACHE": "0",
    "MOONEP_PF_STREAM": "2",
    "MOONEP_EARLY_PREFETCH": "0",
    "MOONEP_DISABLE_UDMA": "1",
    "VLLM_ASCEND_MOONEP_FUSED_PUBLISH": "1",
    "VLLM_ASCEND_MOONEP_DISPATCH_DTYPE": dtype,
    "MOONEP_DISPATCH_IN_SHARD": in_shard,
}
environment = dict(line.split("=", 1) for line in (service / "environment.txt").read_text().splitlines())
assert all(environment.get(key) == value for key, value in settings.items()), (profile, environment)
log = (service / "service.log").read_text(errors="replace")
for forbidden in ("[pf-stream] issue failed", "[pf-stream] install failed", "[slot-cache-active ", "[home-gemm-active "):
    assert forbidden not in log, forbidden
ranks = set(map(int, re.findall(r"\[pf-stream-active r(\d+)\] mode=2 cache_aware=1", log)))
assert ranks == set(range(8)), ("Side stream not confirmed on all ranks", ranks)
transport_ranks = set(map(int, re.findall(r"\[moonep-activation-transport r(\d+)\] mode=mxfp8 backend=mte", log)))
assert transport_ranks == (ranks if dtype == "mxfp8" else set()), (dtype, transport_ranks)

workers = {int(dp): int(pid) for dp, ep, pid in re.findall(r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)", log) if dp == ep}
assert set(workers) == ranks, workers
mapped = {}
wire_hash = digest(STAGED) if dtype == "mxfp8" else None
for rank, pid in workers.items():
    proc = Path("/proc") / str(pid)
    worker_environment = dict(entry.split("=", 1) for entry in proc.joinpath("environ").read_bytes().decode().split("\0")
                              if "=" in entry)
    assert worker_environment.get("MOONEP_DISPATCH_IN_SHARD") == in_shard, (rank, in_shard)
    # /proc/environ is the exec-time snapshot. vLLM may populate VLLM_*
    # settings afterwards, so rely on the post-dispatch marker when absent.
    initial_dtype = worker_environment.get("VLLM_ASCEND_MOONEP_DISPATCH_DTYPE")
    if initial_dtype is not None:
        assert initial_dtype == dtype, (rank, initial_dtype, dtype)
    maps = proc.joinpath("maps").read_text().splitlines()
    names = sorted({line.split(maxsplit=5)[-1] for line in maps if "libascend_moonep_" in line})
    assert any(name.endswith("/libascend_moonep_publish.so") for name in names), (rank, names)
    wire = [Path(name) for name in names if "libascend_moonep_mxfp8_wire.so" in name]
    if dtype == "mxfp8":
        assert len(wire) == 1 and digest(wire[0]) == wire_hash, (rank, wire)
    else:
        assert not wire, (rank, wire)
    mapped[rank] = names

record = dict(profile=profile, valid=True, dispatch_dtype=dtype, backend="mte",
              dispatch_in_shard=(in_shard == "1"), bounded_wire_sha256=wire_hash,
              side_stream_ranks=sorted(ranks), fused_publish=True, mapped_libraries=mapped,
              bounded_unpack_after_dispatch_ranks=sorted(transport_ranks),
              runtime_markers=[line.strip() for line in log.splitlines()
                               if "[moonep-activation-transport r" in line])
(output / "OPTIMIZATION_AUDIT.json").write_text(json.dumps(record, indent=2) + "\n")
print("OPTIMIZATION_AUDIT_OK", profile, sorted(ranks))
