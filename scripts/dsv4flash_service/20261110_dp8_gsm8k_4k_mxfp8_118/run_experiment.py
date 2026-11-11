"""Run isolated 4K GSM8K formal and diagnostic pairs on 118."""
import argparse
import fcntl
import json
import signal
import socket
import struct
import subprocess
from pathlib import Path

import controller_base as engine
from verify_wire import STAGED, digest

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[3]
LOCK = ROOT.parent / "20261107_dp8_longbenchv2_4k_mte_118/.dp8_4k_118_controller.lock"
WIRE = WS / "ascend-moonep/ascend_moonep/lib/libascend_moonep_mxfp8_wire.so"
engine.ROOT = ROOT
engine.STATE = ROOT / "STATE.json"
engine.NAMES = {mode: f"dsv4flash_gsm8k_4k_{mode}" for mode in ("off", "slot")}


def verify():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    with socket.socket() as sock:
        info = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", b"enp34s0f1"))
    assert socket.inet_ntoa(info[20:24]) == provenance["host"]
    for repo, expected in provenance["commits"].items():
        actual = subprocess.check_output(["git", "-C", WS / repo, "rev-parse", "HEAD"], text=True).strip()
        assert actual == expected, (repo, actual, expected)
    for path, expected in provenance["inputs"].items():
        assert digest(Path(path)) == expected, path
    assert digest(WIRE) == digest(STAGED), WIRE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "formal", "diagnostic"), default="all")
    parser.add_argument("--mode", choices=("both", "off", "slot"), default="both")
    args = parser.parse_args()
    phases = ("formal", "diagnostic") if args.phase == "all" else (args.phase,)
    modes = ("off", "slot") if args.mode == "both" else (args.mode,)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, engine.on_signal)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in phases:
            if phase == "diagnostic":
                for mode in ("off", "slot"):
                    valid = list((ROOT / "formal/data/formal/aisbench").glob(f"*-{mode}-*/round_1/metrics.json"))
                    assert len(valid) == 1 and json.loads(valid[0].read_text())["valid"], valid
            for mode in modes:
                verify()
                engine.resources_free()
                engine.run_case(phase, mode)
            engine.log("phase_complete", phase=phase, host="141.61.45.118")
        engine.log("runs_complete", host="141.61.45.118")


if __name__ == "__main__":
    main()
