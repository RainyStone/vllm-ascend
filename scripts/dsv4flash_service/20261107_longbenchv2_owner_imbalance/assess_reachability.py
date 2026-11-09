import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import build_workloads as builder

root=Path(__file__).resolve().parent
data=root/"data"
manifest=json.loads((root.with_name("20261106_dp16_optimized_retest")/"data/full503_manifest.json").read_text())
scan=json.loads((data/"lbscan.json").read_text())
samples=scan["samples"]
expected={row["data_id"]:row["prompt_tokens"] for row in manifest["samples"]}
assert len(samples)==len(expected)==503
assert {row["data_id"]:row["prompt_tokens"] for row in samples}==expected
lengths=np.array([row["prompt_tokens"] for row in samples],dtype=np.int64)
counts=np.array([row["owner_pairs"] for row in samples],dtype=np.int64)
assert counts.shape==(503,16) and np.all(counts.sum(axis=1)==lengths*43*6)
ratios=16*counts.max(axis=1)/counts.sum(axis=1)
tolerance=0.01
ceiling=float(ratios.max())
top=int(ratios.argmax())
targets=(1.2,1.4,1.6)
report=dict(
    prompt_count=503,
    prompt_tokens=int(lengths.sum()),
    owner_pairs=counts.sum(axis=0).tolist(),
    total_pairs=int(counts.sum()),
    full_set_ratio=scan["ratio"],
    quantiles=dict(zip(("min","p50","p95","p99","max"),
                        np.quantile(ratios,[0,.5,.95,.99,1]).tolist())),
    highest_sample=samples[top],
    exact_length_counts=dict(sorted(Counter(lengths.tolist()).items())),
    tolerance=tolerance,
    method="Fixed measured natural-routing histograms, no synthetic routing",
    targets=[],
)
for target in targets:
    result={"target":target,"requests":503}
    lo=target-tolerance
    if ceiling<lo:
        result.update(status="unreachable",reason="single-sample upper bound below lower tolerance edge")
    else:
        builder.TARGETS=(target,)
        try:
            mult=builder.select_workloads(lengths,counts,503,tolerance,30)[0]
        except ValueError as error:
            message=str(error)
            status="unreachable" if "The problem is infeasible" in message else "undetermined"
            result.update(status=status,reason=message)
        else:
            total=mult@counts
            ratio=16*int(total.max())/int(total.sum())
            assert int(mult.sum())==503 and (mult>=0).all() and abs(ratio-target)<=tolerance+1e-9
            result.update(status="reachable",ratio=ratio,input_tokens=int(mult@lengths),
                          multiplicities=[{"data_id":s["data_id"],"requests":int(m)}
                                          for s,m in zip(samples,mult,strict=True) if m])
    print("TARGET",json.dumps(result),flush=True)
    report["targets"].append(result)
report["all_targets_individually_reachable"]=all(x["status"]=="reachable" for x in report["targets"])
with (data/"feasibility.json").open("x") as output:
    json.dump(report,output,indent=2)
print("FULL",report["prompt_count"],report["prompt_tokens"],report["total_pairs"],
      report["full_set_ratio"],report["quantiles"],"max_id",samples[top]["data_id"],flush=True)
