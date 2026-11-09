#!/usr/bin/env python3
"""Read-only worker audit, run inside each node's xzx_moonep after startup.

Usage: python capture_node_audit.py MODE NODE SERVICE_DIRECTORY PAIR [--post]
NODE is 118 or 124. Run once before timing and once with --post after timing.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap/diagnostics/profiling')
WS = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
REQUIRED = {
    "MOONEP": "1", "MOONEP_SLOT_MODE": "1", "MOONEP_REAL_T": "1",
    "MOONEP_KEEP_MC2": "0", "MOONEP_SLOT_CACHE": "0",
    "MOONEP_EARLY_PREFETCH": "0", "MOONEP_PF_STREAM": "3",
    "ASCEND_MOONEP_SKIP_DUP": "1", "ASCEND_MOONEP_PREFETCH_RELAXED": "1",
    "ASCEND_MOONEP_NO_RECONCILE": "1",
}

def audit(mode, node, directory, pair, phase="pre"):
    pid = int((directory / "service.pid").read_text())
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    for option, expected in {
        "--data-parallel-size": "16", "--data-parallel-size-local": "8",
        "--tensor-parallel-size": "1", "--max-num-seqs": "16",
        "--max-num-batched-tokens": "8192", "--max-model-len": "131073",
        "--kv-cache-memory-bytes": "25769803776",
        "--data-parallel-address": "141.61.45.118",
        "--data-parallel-rpc-port": "29560",
        "--served-model-name": "dsv4flash_dp16_longbench_" + mode,
    }.items():
        assert command[command.index(option)+1] == expected, option
    assert "--disable-log-stats" in command and "--disable-uvicorn-access-log" in command
    assert "--enable-log-requests" not in command and "--enable-log-outputs" not in command
    additional = json.loads(command[command.index("--additional-config")+1])
    assert additional == {"enable_moonep": mode != "off", "dp_allreduce_on_npu": True}
    if node == "118":
        assert "--headless" not in command and "--data-parallel-start-rank" not in command
    else:
        assert "--headless" in command and command[command.index("--data-parallel-start-rank")+1] == "8"
    log = (directory / "service.log").read_text(errors="replace")
    ranks = {}
    for dp, ep, worker in re.findall(r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)", log):
        ranks[int(ep)] = int(worker)
    expected = set(range(0, 8)) if node == "118" else set(range(8, 16))
    assert set(ranks) == expected, ("Expected global EP ranks on this node", ranks)
    parent_env = dict(line.split("=",1) for line in
                      (directory/"environment.txt").read_text().splitlines() if "=" in line)
    runtime = {}
    for dp, ep, worker, message in re.findall(
            r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)\s*\[moonep-env-audit\]\s+([^\n]+)", log):
        runtime[int(worker)] = dict(re.findall(r"([A-Z0-9_]+)=([^\s(]+)", message))
    certificate = None
    if mode != "off":
        certificate=json.loads((ROOT/"data/preflight"/(mode+"_dp16_validation.json")).read_text())
        assert certificate["prefetch_stream_mode"]==3 and certificate["prefetch_side_stream_validated"]
        for path,sha in certificate["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha, ("Source changed since validation",path)
        for path,sha in certificate["runtime_binary_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha, path
    workers = []
    for rank, worker in sorted(ranks.items()):
        proc = Path("/proc") / str(worker)
        env = dict(x.split("=",1) for x in (proc/"environ").read_bytes().decode().split("\0") if "=" in x)
        relevant = {k:v for k,v in env.items() if k.startswith(("MOONEP","ASCEND_MOONEP","VLLM_ASCEND_MOE_"))}
        assert not any(k.startswith("VLLM_ASCEND_MOE_SIM_ROUTE") for k in relevant)
        all_mapped_libraries = {line.split()[-1] for line in (proc/"maps").read_text().splitlines()
                                if line.split()[-1].startswith("/")}
        libraries = sorted(path for path in all_mapped_libraries
                           if "/ascend-moonep/" in path or "moonep" in Path(path).name.lower())
        if mode == "off":
            assert not relevant and not libraries
            assert "[moonep-init r" not in log
        else:
            assert all(parent_env.get(k)==v for k,v in REQUIRED.items())
            udma=mode=="slot_udma"
            assert parent_env.get("MOONEP_DISABLE_UDMA")==("0" if udma else "1")
            for stage in ("PLANNING","DISPATCH","COMBINE","PREFETCH"):
                assert parent_env.get("ASCEND_MOONEP_"+stage+"_USE_UDMA")==str(int(udma and stage!="PLANNING"))
            expected_backends=dict(planning="mte",dispatch="udma" if udma else "mte",
                                   combine="udma" if udma else "mte",prefetch="udma" if udma else "mte")
            messages=re.findall(r"\[moonep-backend r"+str(rank)+r"\] ([^\n]+)",log)
            assert messages, ("Missing effective backend proof",rank)
            for message in messages:
                flags=dict(re.findall(r"(planning|dispatch|combine|prefetch)=(mte|udma)",message))
                assert flags==expected_backends, (rank,flags,expected_backends)
            assert certificate["backend"]==("udma" if udma else "mte")
            # /proc/environ is the exec snapshot; vLLM injects worker env later.
            # Require actual mode=3 activation when that initial snapshot lacks it.
            if env.get("MOONEP_PF_STREAM")!="3":
                assert f"[pf-stream-active r{rank}] mode=3" in log, ("No runtime side-stream proof",rank)
            for path in certificate["runtime_binary_sha256"]:
                if phase=="post" or "libascend_moonep_home_gmm.so" not in path:
                    assert path in all_mapped_libraries, ("Validated runtime is not actually mapped",rank,path)
            if udma:
                # Worker stdout prefixes can interleave independently of messages.
                # Count all eight native-query records at node scope instead.
                # Every rank must also reach the backend records below the mandatory
                # require_udma_prefetch_overlap() check in the SHA-pinned source.
                qp=re.findall(r"\[moonep-udma-qp\] allocated=(\d+) dispatch_count=(\d+) prefetch_qp=(\d+)",log)
                assert len(qp)==8 and all(tuple(map(int,x))==(3,2,2) for x in qp), ("Missing actual isolated UDMA QP layouts for eight workers",qp)
            for key in ["MOONEP_SLOT_MODE","MOONEP_REAL_T","MOONEP_SLOT_QUANT",
                        "ASCEND_MOONEP_SKIP_DUP","ASCEND_MOONEP_PREFETCH_RELAXED","ASCEND_MOONEP_NO_RECONCILE"]:
                assert runtime.get(worker,{}).get(key) == "1", (rank,key)
            assert f"[moonep-init r{rank}] stage=shmem-init-done" in log
            assert f"[moonep-release r{rank}] home chunks now back the layers" in log
            if phase=="post":
                assert f"[pf-stream-active r{rank}] mode=3" in log, ("No actual side-stream activation",rank)
                assert f"[home-gemm-active r{rank}] mode=3 split=gemm1" in log, ("No actual Home GEMM1",rank)
                assert "[pf-stream] issue failed" not in log and "[pf-stream] install failed" not in log
            assert parent_env.get("SHMEM_LOG_LEVEL")=="ERROR"
            assert parent_env.get("VLLM_LOGGING_LEVEL")=="INFO"
            assert libraries and not any("/precision_fix/" in p for p in libraries)
        workers.append(dict(rank=rank,pid=worker,libraries=libraries,validated_runtime_libraries=sorted(set(certificate["runtime_binary_sha256"]) & all_mapped_libraries) if certificate else [],runtime=runtime.get(worker)))
    return dict(valid=True, mode=mode, node=node, pair=pair, phase=phase, dp_allreduce_on_npu=True,
                side_stream_activation_verified=(mode!="off" and phase=="post"), captured_at=time.time(),
                service_pid=pid, service_dir=str(directory), workers=workers,
                global_ranks=sorted(ranks), actual_command=command,
                source_sha256=hashlib.sha256(
                    (WS/"vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py").read_bytes()).hexdigest(),
                evidence_limit="process/config/source/init audit; not a per-layer kernel trace")

def main():
    mode,node,service,pair=sys.argv[1:5]
    if "--watch" in sys.argv:raise SystemExit("This experiment uses pre/post audits; no periodic audit during timing.")
    phase="post" if "--post" in sys.argv else "pre"
    assert mode in ("off","slot_mte","slot_udma") and node in ("118","124") and pair in ("pair01","pair02")
    directory=ROOT/"data/cluster_audit";directory.mkdir(parents=True,exist_ok=True)
    target=directory/f"{pair}_{mode}_{node}{'_post' if phase=='post' else ''}.json"
    while True:
        try:
            result=audit(mode,node,Path(service),pair,phase)
        except Exception as error:
            result=dict(valid=False,mode=mode,node=node,pair=pair,captured_at=time.time(),error=repr(error))
            temp=target.with_suffix(".tmp");temp.write_text(json.dumps(result,indent=2));os.replace(temp,target)
            raise
        temp=target.with_suffix(".tmp");temp.write_text(json.dumps(result,indent=2));os.replace(temp,target)
        print("NODE_AUDIT_OK",mode,node,result["global_ranks"],flush=True)
        if "--watch" not in sys.argv:break
        time.sleep(30)

if __name__=="__main__":
    main()
