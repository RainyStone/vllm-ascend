#!/usr/bin/env python3
"""Audit the running service's mode using actual /proc state and worker logs."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WS=Path('/mnt/share/x00893896/vllm_ascend_moonep_integration')
REQUIRED_SLOT={
    "VLLM_ASCEND_MOONEP_DISPATCH_DTYPE":"mxfp8","VLLM_ASCEND_MOONEP_FUSED_PUBLISH":"1",
    "MOONEP_DISPATCH_IN_SHARD":"0",
    "MOONEP":"1","MOONEP_SLOT_MODE":"1","MOONEP_REAL_T":"1",
    "MOONEP_SLOT_CACHE":"0","MOONEP_EARLY_PREFETCH":"0","MOONEP_PF_STREAM":"2",
    "MOONEP_DISABLE_UDMA":"1","MOONEP_KEEP_MC2":"0","ASCEND_MOONEP_PREFETCH_RELAXED":"1",
    "ASCEND_MOONEP_NO_RECONCILE":"1","ASCEND_MOONEP_SKIP_DUP":"1",
}

def main():
    mode,phase,session=sys.argv[1:]
    assert mode in ("off","slot") and phase in ("formal","diagnostic")
    candidates=[]
    for candidate in (ROOT/"data"/phase/"services").glob(f"*-{mode}-*"):
        pid_file=candidate/"service.pid"
        if pid_file.is_file() and (Path("/proc")/pid_file.read_text().strip()/"cmdline").exists():
            candidates.append(candidate)
    assert len(candidates)==1, ("Expected exactly one matching live service", phase, mode, candidates)
    directory=candidates[0]
    log=(directory/"service.log").read_text(errors="replace")
    parent=int((directory/"service.pid").read_text())
    cmd=(Path("/proc")/str(parent)/"cmdline").read_bytes().decode().rstrip("\0").split("\0")
    additional=json.loads(cmd[cmd.index("--additional-config")+1])
    assert additional["enable_moonep"] is (mode=="slot"),additional
    assert additional == {"enable_moonep": mode=="slot"}, additional
    for key, value in {"--max-num-seqs":"16","--max-num-batched-tokens":"4096",
                       "--data-parallel-size":"8","--tensor-parallel-size":"1",
                       "--served-model-name":"dsv4flash_gsm8k_4k_"+mode}.items():
        assert cmd[cmd.index(key)+1] == value, (key,cmd)
    for flag in ["--enable-expert-parallel","--enable-chunked-prefill","--enforce-eager"]:
        assert flag in cmd
    assert "--no-enable-prefix-caching" in cmd
    assert cmd[cmd.index("--max-model-len")+1]=="131073"
    assert cmd[cmd.index("--kv-cache-memory-bytes")+1]=="25769803776"
    ranks={}
    for dp,ep,pid in re.findall(r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)",log):
        assert dp==ep,(dp,ep,pid)
        ranks[int(dp)]=int(pid)
    assert set(ranks)==set(range(8)),ranks
    runtime_audits={}
    pattern=r"\(Worker_DP(\d+)_EP(\d+) pid=(\d+)\)\s*\[moonep-env-audit\]\s+([^\n]+)"
    for dp,ep,pid,message in re.findall(pattern,log):
        if dp==ep:
            runtime_audits[int(pid)]=dict(re.findall(r"([A-Z0-9_]+)=([^\s(]+)",message))
    parent_environment={}
    for line in (directory/"environment.txt").read_text().splitlines():
        if "=" in line:
            k,v=line.split("=",1);parent_environment[k]=v
    stats_keys = {"VLLM_ASCEND_MOE_PAR_STATS", "VLLM_ASCEND_MOE_PAR_FLUSH_EVERY",
                  "VLLM_ASCEND_MOE_PAR_STATS_TAG", "VLLM_ASCEND_MOE_PAR_STATS_DIR"}
    rank_dir = None
    if phase == "diagnostic":
        assert parent_environment.get("VLLM_ASCEND_MOE_PAR_STATS") == "1", parent_environment
        assert parent_environment.get("VLLM_ASCEND_MOE_PAR_FLUSH_EVERY") == "1", parent_environment
        assert parent_environment.get("VLLM_ASCEND_MOE_PAR_STATS_TAG") == "dp8_4k_"+mode, parent_environment
        rank_dir = Path(parent_environment["VLLM_ASCEND_MOE_PAR_STATS_DIR"]).resolve()
        assert rank_dir.parent == (ROOT/"data/diagnostic/rank_raw"/mode).resolve(), rank_dir
    else:
        assert not (stats_keys & parent_environment.keys()), parent_environment
    workers=[]
    for rank,pid in sorted(ranks.items()):
        proc=Path("/proc")/str(pid)
        environment=dict(pair.split("=",1) for pair in proc.joinpath("environ").read_bytes().decode().split("\0") if "=" in pair)
        relevant={k:v for k,v in environment.items() if k.startswith(("MOONEP","ASCEND_MOONEP","VLLM_ASCEND_MOE_","VLLM_ASCEND_MOONEP_"))}
        synthetic=[k for k in relevant if k.startswith("VLLM_ASCEND_MOE_SIM_ROUTE")]
        assert not synthetic,(rank,synthetic)
        maps=proc.joinpath("maps").read_text()
        libraries=sorted(set(line.split()[-1] for line in maps.splitlines() if "/ascend-moonep/" in line or "moonep" in Path(line.split()[-1]).name.lower()))
        if mode=="off":
            allowed = stats_keys if phase == "diagnostic" else set()
            assert not (set(relevant)-allowed),(rank,relevant)
            assert not libraries,(rank,libraries)
        else:
            # /proc/environ is the exec-time snapshot, not Python's current os.environ.
            # vLLM reinjects worker variables after exec; use the runtime audit emitted
            # inside ensure_runtime_initialized, plus the actual Slot-only publish path.
            assert all(parent_environment.get(k)==v for k,v in REQUIRED_SLOT.items()),parent_environment
            for key,value in REQUIRED_SLOT.items():
                if key in environment:
                    assert environment[key]==value,(rank,key,environment[key])
            runtime=runtime_audits.get(pid,{})
            for key in ["MOONEP_SLOT_MODE","MOONEP_REAL_T","MOONEP_SLOT_QUANT",
                        "ASCEND_MOONEP_PREFETCH_RELAXED","ASCEND_MOONEP_NO_RECONCILE",
                        "ASCEND_MOONEP_SKIP_DUP"]:
                assert runtime.get(key)=="1",(rank,key,runtime)
            assert runtime.get("MOONEP_MIN_DISPATCH_TOKENS")=="16",(rank,runtime)
            assert libraries,("MoonEP library not mapped in worker",rank)
            assert f"[moonep-init r{rank}] stage=shmem-init-done" in log,rank
            assert f"[moonep-release r{rank}] home chunks now back the layers" in log,rank
        workers.append(dict(rank=rank,pid=pid,initial_exec_environment=relevant,
                            runtime_initialization_environment=runtime_audits.get(pid),
                            mapped_moonep_libraries=libraries))
    if mode=="off":
        assert not any(k.startswith(("MOONEP", "ASCEND_MOONEP", "VLLM_ASCEND_MOONEP_"))
                       for k in parent_environment), parent_environment
        assert "[moonep-env-audit]" not in log and "[moonep-init r" not in log
    sources=[WS/"vllm-ascend/vllm_ascend/ascend_forward_context.py",
             WS/"vllm-ascend/vllm_ascend/ops/fused_moe/moonep_comm.py",
             WS/"vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py"]
    report=dict(mode=mode,phase=phase,rank_stats_dir=str(rank_dir) if rank_dir else None,valid=True,service_pid=parent,service_dir=str(directory),
                actual_command=cmd,actual_additional_config=additional,workers=workers,
                native_router=True,prefix_cache_enabled=False,
                source_sha256={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                execution_contract=("enable_moonep=false: SHMEM method not registered; SoC communication selection applies"
                   if mode=="off" else "enable_moonep=true + KEEP_MC2=0: SHMEM override, no silent fallback; SLOT_MODE=1 selects run_slot_forward"),
                evidence_limit="This is actual process/configuration/library/initialization verification plus source execution contract; no profiler trace is claimed.")
    (directory/"MODE_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    output=Path(session)/(directory.name+"_mode_audit.json")
    output.write_text(json.dumps(report,indent=2)+"\n")
    print("MODE_AUDIT_OK",mode,"8/8 actual worker processes; native router; expected MoonEP flags and libraries.")
    print("AUDIT_FILE",output)

if __name__=="__main__": main()
