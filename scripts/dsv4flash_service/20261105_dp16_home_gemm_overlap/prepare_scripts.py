from pathlib import Path
import ast
W=Path("/mnt/share/x00893896/vllm_ascend_moonep_integration")
E=W/"vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap"
D=W/"vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft"
C=D/"data/executions/run_20261105_155600_mte_first_stream2/candidate_dp_metadata_npu"
E.joinpath("data/preflight").mkdir(parents=True,exist_ok=True)
for name in ("full503_manifest.json","longbenchv2_all503_midcut128k.json"):
    target=E/"data"/name
    if not target.exists():target.symlink_to(D/"data"/name)
for name in ("validate_round_dp16.py","verify_cluster_audit.py","observe_owned_resources.py"):
    E.joinpath(name).write_text(D.joinpath(name).read_text())
audit=C.joinpath("capture_node_audit.py").read_text().replace(str(D),str(E))
audit=audit.replace('"MOONEP_PF_STREAM": "2"','"MOONEP_PF_STREAM": "3"')
audit=audit.replace('certificate["prefetch_stream_mode"]==2','certificate["prefetch_stream_mode"]==3')
audit=audit.replace('env.get("MOONEP_PF_STREAM")!="2"','env.get("MOONEP_PF_STREAM")!="3"')
audit=audit.replace('mode=2','mode=3')
audit=audit.replace('assert path in libraries, ("Validated runtime is not actually mapped",rank,path)',
'''if phase=="post" or "libascend_moonep_home_gmm.so" not in path:
                    assert path in libraries, ("Validated runtime is not actually mapped",rank,path)''')
audit=audit.replace('assert f"[pf-stream-active r{rank}] mode=3" in log, ("No actual side-stream activation",rank)',
'''assert f"[pf-stream-active r{rank}] mode=3" in log, ("No actual side-stream activation",rank)
                assert f"[home-gemm-active r{rank}] mode=3 split=gemm1" in log, ("No actual Home GEMM1",rank)''')
E.joinpath("capture_node_audit.py").write_text(audit)
for mode in ("off","slot_mte","slot_udma"):
    config=D.joinpath(f"aisbench_{mode}_draft.py").read_text()
    config=config.replace("DRAFT FOR REVIEW.","Home GEMM overlap experiment.")
    E.joinpath(f"aisbench_{mode}.py").write_text(config)
    bench=C.joinpath(f"bench_{mode}.sh").read_text()
    bench=bench.replace(str(D),str(E)).replace(f"aisbench_{mode}_draft.py",f"aisbench_{mode}.py")
    bench=bench.replace("metadata_npu-","home_gemm3-").replace("/dev/shm/dsv4_dp16_20261105","/dev/shm/home_gmm_dp16_20261106")
    bench=bench.replace("One measured round per independent service, zero benchmark warmups; average the two service runs for each mode.",
                        "One measured round per service, zero warmups. Quick order: off -> MTE -> UDMA; one run per mode.")
    bench=bench.replace("# DRAFT FOR REVIEW. This script benchmarks an already running service.","# One approved cold 503-request performance round; raw debug retained.")
    # Independent experiments share the previous client lock, preventing a second client.
    bench=bench.replace('exec 8>"$EXPERIMENT_DIR/.aisbench_client.lock"',f'exec 8>"{D}/.aisbench_client.lock"')
    E.joinpath(f"bench_{mode}.sh").write_text(bench)
    for node in ("118","124"):
        s=C.joinpath(f"start_{mode}_{node}.sh").read_text()
        s=s.replace(str(D),str(E)).replace("metadata_npu_pair01","home_gemm_pair01")
        s=s.replace("export MOONEP_PF_STREAM=2","export MOONEP_PF_STREAM=3")
        s=s.replace('prefetch_stream_mode\']==2','prefetch_stream_mode\']==3')
        s=s.replace("# DRAFT FOR REVIEW. All tuning values are defined here, without caller-supplied env.",
                    "# Approved Home GEMM1 experiment. Edit all tuning values inside this file.")
        # Share original service locks so separate experiments cannot overlap.
        s=s.replace(f'exec 9>"$EXPERIMENT_DIR/.service_node{node}_8445.lock"',
                    f'exec 9>"{D}/.service_node{node}_8445.lock"')
        # Gate before CANN setup or any driver query. Existing idle check remains.
        s=s.replace("# Remove inherited experimental settings",f'''# Time authorization is checked before any accelerator query.
"$PYTHON" "$EXPERIMENT_DIR/resource_gate.py" --cards "$CARDS" \\
    --output "$EXPERIMENT_DIR/data/idle_{mode}_{node}.json"

# Remove inherited experimental settings''')
        if mode!="off":
            a=s.index("SLOT_MULTINODE_VALIDATED=1")
            b=s.index('"$PYTHON" - "$PLUGIN"',a)
            s=s[:a]+"# Require the real backend-specific Home-GEMM certificate; no boolean override.\n"+s[b:]
            check="assert d['prefetch_stream_mode']==3 and d['prefetch_side_stream_validated']"
            s=s.replace(check,check+"\nassert d['home_gemm1_validated'] and d['gemm2_unsplit']")
        E.joinpath(f"start_{mode}_{node}.sh").write_text(s)
# Operator scripts: every invocation is fully configured inside its own file.
common='''#!/usr/bin/env bash
set -Ee -o pipefail
WORKSPACE='WORKSPACE_VALUE'
EXPERIMENT_DIR='EXPERIMENT_VALUE'
PYTHON='/usr/local/python3.11.10/bin/python'
CARDS='CARDS_VALUE'
RUN_ID='RUN_VALUE'
"$PYTHON" "$EXPERIMENT_DIR/resource_gate.py" --cards "$CARDS" \\
    --output "$EXPERIMENT_DIR/data/$RUN_ID/idle_NODE_VALUE.json"
while IFS= read -r name; do
    case "$name" in MOONEP*|ASCEND_MOONEP*|VLLM*|SHMEM*) unset "$name";; esac
done < <(compgen -e)
unset ASCEND_VISIBLE_DEVICES PYTHONPATH LD_PRELOAD http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export LD_LIBRARY_PATH='/usr/local/python3.11.10/lib'
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKSPACE/vllm:$WORKSPACE/vllm-ascend:$WORKSPACE/ascend-moonep:$PYTHONPATH"
VENDOR="$WORKSPACE/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$WORKSPACE/ascend-moonep/ascend_moonep/lib:$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export HCCL_IF_IP='141.61.45.NODE_VALUE'
export GLOO_SOCKET_IFNAME='enp34s0f1' TP_SOCKET_IFNAME='enp34s0f1' HCCL_SOCKET_IFNAME='enp34s0f1'
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false SHMEM_LOG_LEVEL=ERROR
export TMPDIR='/dev/shm/home_gmm_test_20261106'
mkdir -p "$TMPDIR"
export MOONEP_SLOT_MODE=1 MOONEP_SLOT_QUANT=1 MOONEP_SLOT_CACHE=0
export MOONEP_EARLY_PREFETCH=0 MOONEP_PF_STREAM=3 MOONEP_KEEP_MC2=0
export ASCEND_MOONEP_SKIP_DUP=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_PREFETCH_RELAXED=1
export MOONEP_BALANCE_STATS=0 MOONEP_PLAN_STATS=0 MOONEP_TIME_STATS=0
export ASCEND_MOONEP_TRACE=0 ASCEND_MOONEP_TIME_STATS=0
LIBRARY="$WORKSPACE/ascend-moonep/build/home_gmm_20261105/lib/libascend_moonep_home_gmm.so"
'''
for world in (2,16):
    for backend in ("mte","udma"):
        for node in ("118","124"):
            text=common.replace("WORKSPACE_VALUE",str(W)).replace("EXPERIMENT_VALUE",str(E))
            text=text.replace("CARDS_VALUE","1" if world==2 else "0,1,2,3,4,5,6,7")
            text=text.replace("RUN_VALUE",f"home_cross{world}_{backend}_20261106_00").replace("NODE_VALUE",node)
            udma=backend=="udma"
            text+=f"export MOONEP_DISABLE_UDMA={0 if udma else 1}\n"
            for stage in ("PLANNING","DISPATCH","COMBINE","PREFETCH"):
                text+=f"export ASCEND_MOONEP_{stage}_USE_UDMA={int(udma and stage!='PLANNING')}\n"
            text+=f'''exec "$PYTHON" "$EXPERIMENT_DIR/rank_group_runner.py" \\
    --world {world} --node {0 if node=="118" else 1} --backend {backend} \\
    --output "$EXPERIMENT_DIR/data/$RUN_ID" --library "$LIBRARY" \\
    --master-port {29671 if backend=="mte" else 29672} --shmem-port {29771 if backend=="mte" else 29772}
'''
            E.joinpath(f"cross{world}_{backend}_{node}.sh").write_text(text)
text=common.replace("WORKSPACE_VALUE",str(W)).replace("EXPERIMENT_VALUE",str(E))
text=text.replace("CARDS_VALUE","1").replace("RUN_VALUE","home_single_20261106_00").replace("NODE_VALUE","124")
text+='''exec "$PYTHON" "$EXPERIMENT_DIR/tests/test_home_gmm_npu.py" \\
    --library "$LIBRARY" --output "$EXPERIMENT_DIR/data/$RUN_ID/single.json"
'''
E.joinpath("single_124.sh").write_text(text)
for p in E.glob("*.py"):ast.parse(p.read_text())
for p in E.joinpath("tests").glob("*.py"):ast.parse(p.read_text())
print("PREPARED",E)
