from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ctypes,hashlib,json,subprocess,shutil
ws=Path.cwd();r=ws/"vllm-ascend/scripts/dsv4flash_service/20261106_dp8_longbenchv2_mxfp8_publish"
build=Path("/tmp/moonep_longbench_latest_20261106/lib")
installed=[]
for name,abi in [("libascend_moonep_mxfp8_wire.so","moonep_mxfp8_wire_abi"),("libascend_moonep_publish.so","moonep_publish_abi")]:
    src=build/name;dest=ws/"ascend-moonep/ascend_moonep/lib"/name
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    try:
        with dest.open("xb") as f:f.write(src.read_bytes())
        created=True
    except FileExistsError:
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest,("Unreviewed existing library",dest)
        created=False
    lib=ctypes.CDLL(str(dest));fun=getattr(lib,abi);fun.restype=ctypes.c_int;assert fun()==1
    installed.append(dict(source=str(src),target=str(dest),sha256=digest,newly_installed=created,abi=1))
(r/"data/LIBRARY_INSTALL.json").write_text(json.dumps(installed,indent=2)+"\n")
stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
session=r/"data/executions"/("run_"+stamp);session.mkdir(parents=True)
paths=set()
for directory in [ws/"ascend-moonep/ascend_moonep",ws/"vllm-ascend/vllm_ascend/ops/fused_moe"]:
    paths.update(directory.rglob("*.py"))
paths.update([ws/"vllm-ascend/vllm_ascend/envs.py",ws/"vllm-ascend/vllm_ascend/ascend_forward_context.py"])
paths.update((ws/"ascend-moonep/ascend_moonep/lib").glob("*.so"))
paths.add(ws/"ascend-moonep/ascend_moonep/_C.cpython-311-aarch64-linux-gnu.so")
for name in ["longbenchv2_full_le128k.json","selection_manifest.json"]:paths.add(r/"data"/name)
source=dict(sha256={str(p.relative_to(ws)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
    original_repositories=True,editable_install_changed=False,libraries=installed,
    profile=dict(dispatch_dtype="mxfp8",fused_publish=True,prefetch_stream=2,slot_cache=0,early_prefetch=0,transport="MTE",home_gemm_overlap=False))
(session/"SOURCE_PROVENANCE.json").write_text(json.dumps(source,indent=2)+"\n")
for repo in ["vllm-ascend","ascend-moonep"]:
    (session/(repo+".diff")).write_bytes(subprocess.check_output(["git","-C",str(ws/repo),"diff","--binary"]))
    (session/(repo+".status")).write_bytes(subprocess.check_output(["git","-C",str(ws/repo),"status","--short"]))
review=[];fence=chr(96)*3
for name in ["start_off.sh","start_slot.sh","bench_off.sh","bench_slot.sh","aisbench_off.py","aisbench_slot.py"]:
    p=r/name;shutil.copy2(p,session/name)
    review+=["## "+name,"",fence+("bash" if name.endswith(".sh") else "python"),p.read_text(),fence,""]
(r/"SCRIPT_CONTENTS.md").write_text("\n".join(review))
current=dict(session=str(session),status="starting",requested_pairs=1,schedule="off-slot",rounds_per_service=1,
    completed_pairs=0,profile=source["profile"],auto_start_when_cards_free=False)
(r/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
log=(session/"controller.log").open("w")
proc=subprocess.Popen(["/usr/local/python3.11.10/bin/python",str(r/"run_single_pair.py"),str(session)],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
current.update(controller_pid=proc.pid,controller_log=str(session/"controller.log"))
(r/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
(session/"controller.pid").write_text(str(proc.pid)+"\n")
print(json.dumps(dict(session=str(session),pid=proc.pid,libraries=installed),indent=2))

