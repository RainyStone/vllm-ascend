#!/usr/bin/env python3
"""Issue a mode-specific gate only from successful real NPU evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from tests.source_manifest import source_manifest, W
from tests.test_home_gmm_npu import require_time

ROOT=Path(__file__).resolve().parent
def load(path):
    return json.loads(Path(path).read_text())
def main():
    require_time()
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["mte","udma"],required=True)
    p.add_argument("--single",required=True)
    p.add_argument("--cross2",required=True)
    p.add_argument("--cross16",required=True)
    args=p.parse_args()
    single=load(args.single)
    sources=source_manifest()
    assert single["status"]=="passed" and single["model_loaded"] is False
    assert single["source_unchanged"] and single["source_sha256"]==sources
    built=W/"ascend-moonep/build/home_gmm_20261105/lib/libascend_moonep_home_gmm.so"
    installed=W/"ascend-moonep/ascend_moonep/lib/libascend_moonep_home_gmm.so"
    assert single["library_sha256"]==hashlib.sha256(built.read_bytes()).hexdigest()
    assert installed.is_file(),"Atomically deploy validated library before service gating"
    assert hashlib.sha256(installed.read_bytes()).hexdigest()==single["library_sha256"]
    records=[]
    for world,folder in ((2,args.cross2),(16,args.cross16)):
        ranks=[]
        for rank in range(world):
            d=load(Path(folder)/f"rank{rank}.json")
            assert (d["status"],d["rank"],d["world_size"],d["backend"])==("passed",rank,world,args.backend)
            assert d["planning_backend"]=="mte" and d["prefetch_stream_mode"]==3
            assert d["source_unchanged"] and d["source_sha256"]==sources
            assert d["model_loaded"] is False and d["home_gemm1_before_prefetch_wait"]
            assert d["gemm2_unsplit"] and d["input_quant_once"] and d["cross_node_payload_reads"]>0
            expected={s:("mte" if s=="planning" else args.backend) for s in
                      ("planning","dispatch","combine","prefetch")}
            assert d["actual_backends"]==expected
            called={(x["operator"],x["method"]) for x in d["actual_backend_calls"]}
            assert ("Planning","compute") in called
            assert ("Dispatch","compute") in called
            assert ("PrefetchWeight","compute") in called
            assert ("CombineUrma" if args.backend=="udma" else "Combine","compute_weighted") in called
            stages={x["stage"] for x in d["checks"]}
            assert {"prefetch_bytes","real_quant_mlp","forced_remote_mlp",
                    "planning_dispatch_weighted_combine",
                    "planning_compatibility_histogram_independence"}<=stages
            if args.backend=="udma":
                assert d["layout"] is not None
            assert str(built) in d["runtime_binary_sha256"]
            assert d["runtime_binary_sha256"][str(built)]==single["library_sha256"]
            ranks.append(d)
        assert len({x["host"] for x in ranks})==2,"Both physical nodes must execute"
        records+=ranks
    binaries={}
    for d in records:
        for path,sha in d["runtime_binary_sha256"].items():
            path=str(installed) if path==str(built) else path
            assert binaries.get(path,sha)==sha
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
            binaries[path]=sha
    certificate=dict(backend=args.backend,use_udma=args.backend=="udma",
        world_size=16,valid_ranks=16,fused_weighted_combine=True,
        cross_node_validated=True,expert_prefetch_validated=True,
        planning_dispatch_validated=True,precision_passed=True,planning_backend="mte",
        prefetch_stream_mode=3,prefetch_side_stream_validated=True,
        home_gemm1_validated=True,gemm2_unsplit=True,
        source_sha256=sources,runtime_binary_sha256=binaries,
        validation_scope="synthetic real DSV4 shapes; native range/full MLP tolerance checks; exact fused-combine identity; not model semantic accuracy or timeline overlap proof",
        evidence=dict(single=args.single,cross2=args.cross2,cross16=args.cross16))
    target=ROOT/f"data/preflight/slot_{args.backend}_dp16_validation.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(),"Do not overwrite a certificate tied to a prior run"
    target.write_text(json.dumps(certificate,indent=2))
    print("HOME_GMM_MODE_CERTIFICATE_WRITTEN",target)
if __name__=="__main__":main()
