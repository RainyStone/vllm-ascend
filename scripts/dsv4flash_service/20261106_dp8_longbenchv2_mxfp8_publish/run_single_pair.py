"""One newly authorized independent Off/Slot pair; no old results adopted."""
import fcntl,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import run_confirmed_remainder as engine
ROOT=Path(__file__).resolve().parent
SESSION=Path(sys.argv[1])
WS=ROOT.parents[3]
def state(**kw):
    p=ROOT/"CURRENT_EXECUTION.json"
    d=json.loads(p.read_text())
    d.update(kw)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
def verify():
    expected=json.loads((SESSION/"SOURCE_PROVENANCE.json").read_text())["sha256"]
    changed=[n for n,h in expected.items() if hashlib.sha256((WS/n).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError("Experiment source/library changed: "+repr(changed))
if __name__=="__main__":
    lock=(ROOT/".controller.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        state(status="running",active_case="pair01_off")
        for mode in ["off","slot"]:
            verify()
            state(active_case="pair01_"+mode)
            engine.run_case("pair01",mode)
            verify()
            state(last_completed_case="pair01_"+mode)
            if mode=="off" and (ROOT/"STOP_AFTER_OFF").exists():
                state(status="paused_by_user")
                raise SystemExit(0)
        subprocess.run([engine.PYTHON,str(ROOT/"finalize_cross.py"),str(SESSION)],check=True)
        result_path=SESSION/"results.json"
        result=json.loads(result_path.read_text())
        result.update(status="completed_single_pair",requested_pairs=1)
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        (SESSION/"npu_final.txt").write_text(engine.assert_free())
        (SESSION/"CLEANUP_CHECK.json").write_text(json.dumps(
            dict(eight_npus_released=True,port_8444_bindable=True),indent=2)+"\n")
        state(status="completed",active_case=None,completed_pairs=1,
              result=str(SESSION/"RESULT.md"),resource_status="eight_npus_released_and_port_free",
              completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        engine.log("experiment_complete",completed_pairs=1,formal_requests=632)
    except Exception as error:
        state(status="stopped_on_error",error=repr(error))
        engine.log("stopped_on_error",error=repr(error))
        raise

