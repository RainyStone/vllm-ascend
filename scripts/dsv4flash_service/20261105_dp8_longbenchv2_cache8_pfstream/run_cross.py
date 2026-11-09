"""Off1-Slot1-Off2-Slot2-Off3-Slot3; retain the initial live Off1."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import run_confirmed_remainder as engine
ROOT=Path(__file__).resolve().parent
SESSION=Path(sys.argv[1])
PYTHON="/usr/local/python3.11.10/bin/python"

def verify_sources():
    ws=ROOT.parents[3]
    for name,expected in json.loads((ROOT/"PROVENANCE.json").read_text())["source_sha256"].items():
        assert hashlib.sha256((ws/name).read_bytes()).hexdigest()==expected,("Source changed during experiment",name)

def adopt_off1():
    target=SESSION/"cases/pair01_off"
    if (target/"case_result.json").exists():return
    deadline=time.monotonic()+3600
    while True:
        gate=json.loads((ROOT/"SWITCH_TO_CROSSOVER.json").read_text())
        current=json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())
        if gate.get("round1_validated") and current.get("status")=="stopped_on_error":break
        if time.monotonic()>deadline:raise TimeoutError("Initial Off1 did not stop at requested round boundary")
        time.sleep(3)
    engine.assert_free()
    validation=Path(gate["client_dir"])/"round_1/validation.json"
    check=json.loads(validation.read_text())
    assert check["valid"] and check["requests"]==316
    assert not (Path(gate["client_dir"])/"round_2").exists(),"Unexpected Off2 in the original batch"
    shutil.copytree(SESSION/"cases/rerun01_off",target)
    case=dict(case="pair01_off",service_dir=gate["service_dir"],client_dir=gate["client_dir"],
              service_pid=gate["service_pid"],client_pid=gate["client_pid"],
              rounds=[check],valid=True,formal_requests=316,
              note="Adopted validated Off1 after user changed schedule; exit75 was intentional, not a model failure.")
    (target/"case_result.json").write_text(json.dumps(case,indent=2)+"\n")
    current.update(status="running",schedule="off-slot-off-slot-off-slot",completed_pairs=0)
    (ROOT/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
    engine.log("adopted_off1",formal_requests=316)

if __name__=="__main__":
    try:
        adopt_off1()
        for n in [1,2,3]:
            pair=f"pair{n:02}"
            for mode in ["off","slot"]:
                result=SESSION/"cases"/(pair+"_"+mode)/"case_result.json"
                if result.exists():
                    assert json.loads(result.read_text())["valid"]
                    continue
                verify_sources()
                engine.run_case(pair,mode)
            subprocess.run([PYTHON,str(ROOT/"finalize_cross.py"),str(SESSION)],check=True)
            engine.log("pair_complete",pair=pair,completed_pairs=n,formal_requests=n*632)
            current=json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())
            current.update(completed_pairs=n,result=str(SESSION/"RESULT.md"))
            if (ROOT/"STOP_AFTER_PAIR").exists() and n<3:
                current.update(status="paused_by_user")
                (ROOT/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
                engine.log("paused_at_pair_boundary",completed_pairs=n)
                break
            if n==3:current.update(status="completed")
            (ROOT/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
        (SESSION/"npu_final.txt").write_text(engine.assert_free())
        (SESSION/"CLEANUP_CHECK.json").write_text(json.dumps(dict(eight_npus_released=True,port_8444_bindable=True),indent=2)+"\n")
    except Exception as error:
        engine.log("stopped_on_error",error=repr(error))
        current=json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())
        current.update(status="stopped_on_error",error=repr(error))
        (ROOT/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
        raise
