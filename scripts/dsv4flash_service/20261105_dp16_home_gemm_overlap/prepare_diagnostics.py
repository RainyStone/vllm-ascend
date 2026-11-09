from pathlib import Path
import ast,json,subprocess
E=Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp16_home_gemm_overlap")
for kind in ("rankload","profiling"):
    root=E/"diagnostics"/kind
    root.mkdir(parents=True,exist_ok=True)
    (root/"data").mkdir(exist_ok=True)
    for name in ("full503_manifest.json","longbenchv2_all503_midcut128k.json","preflight"):
        dst=root/"data"/name
        if not dst.exists():dst.symlink_to(E/"data"/name,target_is_directory=name=="preflight")
    dst=root/"tests"
    if not dst.exists():dst.symlink_to(E/"tests",target_is_directory=True)
    for name in ("resource_gate.py","validate_round_dp16.py","verify_cluster_audit.py"):
        (root/name).write_text((E/name).read_text())
    audit=(E/"capture_node_audit.py").read_text().replace(str(E),str(root))
    if kind=="rankload":
        audit=audit.replace('if k.startswith(("MOONEP","ASCEND_MOONEP","VLLM_ASCEND_MOE_"))}',
                            'if k.startswith(("MOONEP","ASCEND_MOONEP","VLLM_ASCEND_MOE_")) and not k.startswith("VLLM_ASCEND_MOE_PAR_")}')
        audit=audit.replace('    runtime = {}','''    assert parent_env.get("VLLM_ASCEND_MOE_PAR_STATS")=="1"
    runtime = {}''')
    (root/"capture_node_audit.py").write_text(audit)
    for mode in ("off","slot_mte","slot_udma"):
        (root/f"aisbench_{mode}.py").write_text((E/f"aisbench_{mode}.py").read_text())
        for node in ("118","124"):
            text=(E/f"start_{mode}_{node}.sh").read_text().replace(str(E),str(root))
            text=text.replace("home_gemm_pair01",f"{kind}_pair01")
            if kind=="rankload":
                settings=f'''export VLLM_ASCEND_MOE_PAR_STATS=1
export VLLM_ASCEND_MOE_PAR_STATS_DIR="$EXPERIMENT_DIR/data/rank_load/{mode}"
export VLLM_ASCEND_MOE_PAR_STATS_TAG='home3_{mode}'
export VLLM_ASCEND_MOE_PAR_FLUSH_EVERY=1
'''
                text=text.replace("# Local resource protection.",settings+"\n# Local resource protection.")
            else:
                config=json.dumps(dict(profiler="torch",torch_profiler_dir=str(root/"data/profiling"/mode),
                    torch_profiler_with_stack=False,torch_profiler_record_shapes=False,
                    torch_profiler_with_memory=False,ignore_frontend=True,
                    delay_iterations=2,max_iterations=3),separators=(",",":"))
                # Append a separate immutable profiler option to the final CLI.
                text=text.rstrip()+" \\\n    --profiler-config '"+config+"'\n"
            (root/f"start_{mode}_{node}.sh").write_text(text)
        if kind=="rankload":
            text=(E/f"bench_{mode}.sh").read_text().replace(str(E),str(root))
            text=text.replace("home_gemm3-","DIAGNOSTIC_RANKLOAD-")
            text=text.replace("One measured round per service, zero warmups. Quick order: off -> MTE -> UDMA; one run per mode.",
                "DIAGNOSTIC ONLY: full503 per-rank load collection. Exclude all performance timings from formal results.")
            (root/f"bench_{mode}.sh").write_text(text)
    for path in root.glob("*.py"):ast.parse(path.read_text())
    for path in root.glob("*.sh"):subprocess.run(["bash","-n",str(path)],check=True)
print("DIAGNOSTIC_SCRIPTS_PREPARED_NOT_EXECUTED")
