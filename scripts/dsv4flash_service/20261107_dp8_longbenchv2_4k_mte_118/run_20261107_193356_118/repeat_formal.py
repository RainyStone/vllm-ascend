import fcntl
import hashlib
import importlib.util
import json
import signal
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_provenance():
    manifest = json.loads((ROOT/"SOURCE_PROVENANCE.json").read_text())
    for name, expected in manifest["sha256"].items():
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream,"sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source or library changed since the first pair: {name}")


def verify_diagnostic_done():
    state=json.loads((ROOT/"STATE.json").read_text())
    assert state["stage"]=="runs_complete", state
    for mode in ("off","slot"):
        proof,=list((ROOT/"diagnostic/data/diagnostic/aisbench").glob(f"*-{mode}-*/round_1/validation.json"))
        report=json.loads(proof.read_text())
        assert report["valid"] and report["requests"]==316,(mode,report)
        boundary=json.loads(proof.with_name("rank_boundary_after.json").read_text())
        assert set(boundary["ranks"])==set(map(str,range(8))),boundary


def current_host():
    with socket.socket() as sock:
        request=struct.pack("256s",b"enp34s0f1")
        reply=fcntl.ioctl(sock.fileno(),0x8915,request)
    return socket.inet_ntoa(reply[20:24])


def run_assigned(engine, pair, assignment):
    engine.ROOT=ROOT/assignment["directory"]
    engine.STATE=engine.ROOT/"STATE.json"
    try:
        verify_provenance()
        for mode in ("off","slot"):
            engine.run_case("formal",mode)
            verify_provenance()
        engine.log("runs_complete",pair=pair,host=assignment["host"])
    except BaseException as error:
        engine.log("stopped",pair=pair,host=assignment["host"],error=repr(error))
        traceback.print_exc()
        return 1
    return 0


def main():
    engine=load("experiment_controller","run_experiment.py")
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,engine.on_signal)
    verify_diagnostic_done()
    host=current_host()
    assignment_path=ROOT/"PAIR_HOSTS.json"
    assignment=json.loads(assignment_path.read_text())
    if host not in ("141.61.45.118","141.61.45.124"):
        raise RuntimeError(f"Unexpected experiment host {host}")
    lock_path=ROOT.parent/f".dp8_4k_{host.rsplit('.',1)[-1]}_controller.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if host=="141.61.45.124":
            pair="pair03"
            expected=assignment[pair]
            assert expected["host"]==host, expected
            engine.ROOT=ROOT/expected["directory"]
            engine.STATE=engine.ROOT/"STATE.json"
            try:
                engine.resources_free()
            except RuntimeError as error:
                engine.log("unavailable",pair=pair,host=host,error=repr(error))
                return 0
            return run_assigned(engine,pair,expected)

        status=run_assigned(engine,"pair02",assignment["pair02"])
        if status:
            return status
        state_124=ROOT/assignment["pair03"]["directory"]/"STATE.json"
        if state_124.is_file() and json.loads(state_124.read_text())["stage"]=="unavailable":
            assignment["pair03"]={"host":host,"directory":"repeat_formal/pair03",
                                  "fallback":"124 was occupied at launch"}
            tmp=assignment_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(assignment,indent=2)+"\n")
            tmp.replace(assignment_path)
            return run_assigned(engine,"pair03",assignment["pair03"])
    return 0


if __name__=="__main__":
    sys.exit(main())

