"""Run the explicitly selected 124 comparison after the eight cards are free."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
from pathlib import Path

import controller_base as engine
from report import summarize
from verify_wire import STAGED, digest

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[3]
ORIGINAL_WIRE = WS / "ascend-moonep/ascend_moonep/lib" / STAGED.name
LOCK = ROOT.parent / "20261107_dp8_longbenchv2_4k_mte_118/.dp8_4k_124_controller.lock"
OFF = ("mxfp8", "off")
MXFP8 = ("mxfp8", "slot")
BF16_COPY = ("bf16_copy", "slot")
BF16_ZERO = ("bf16_zero_copy", "slot")
PLANS = {
    "mxfp8": (OFF, MXFP8),
    "zero-copy-pair": (OFF, BF16_ZERO),
    "zero-copy": (OFF, BF16_COPY, BF16_ZERO),
    "both": (OFF, MXFP8, BF16_COPY, BF16_ZERO),
}


def verify_provenance():
    record = json.loads((ROOT / "PROVENANCE.json").read_text())
    with socket.socket() as sock:
        info = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", b"enp34s0f1"))
    host = socket.inet_ntoa(info[20:24])
    if host != record["host"]:
        raise RuntimeError(f"Expected {record['host']}, found {host}")
    for repo, expected in record["commits"].items():
        actual = subprocess.check_output(["git", "-C", WS / repo, "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"{repo} changed since preparation: {actual} != {expected}")
    for name, expected in record["files"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Changed since preparation: {name}")


def install_wire():
    expected = (STAGED.parent / "SHA256SUMS").read_text().split()[0]
    if digest(STAGED) != expected:
        raise RuntimeError(f"Staged library checksum mismatch: {STAGED}")
    old = digest(ORIGINAL_WIRE)
    if old == expected:
        return
    backup = STAGED.parent / f"previous_{old}.so"
    if not backup.exists():
        shutil.copy2(ORIGINAL_WIRE, backup)
    if digest(backup) != old:
        raise RuntimeError(f"Original wire backup checksum mismatch: {backup}")
    fd, temp = tempfile.mkstemp(prefix=".dispatch124_wire_", dir=ORIGINAL_WIRE.parent)
    os.close(fd)
    path = Path(temp)
    try:
        shutil.copy2(STAGED, path)
        if digest(path) != expected:
            raise RuntimeError(f"Copied wire checksum mismatch: {path}")
        os.replace(path, ORIGINAL_WIRE)
    finally:
        path.unlink(missing_ok=True)
    engine.log("wire_installed", sha256=expected, backup=str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=PLANS, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Add --execute only after the 124 eight-card run is authorized")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, engine.on_signal)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_provenance()
        engine.resources_free()
        for profile, mode in PLANS[args.plan]:
            engine.ROOT = ROOT / profile
            engine.STATE = engine.ROOT / "STATE.json"
            if (engine.ROOT / "formal" / mode).exists():
                raise RuntimeError(f"Existing result or partial case: {profile}/{mode}")
        for profile, mode in PLANS[args.plan]:
            verify_provenance()
            engine.ROOT = ROOT / profile
            engine.STATE = engine.ROOT / "STATE.json"
            if (profile, mode) == MXFP8:
                engine.resources_free()
                install_wire()
            engine.run_case("formal", mode)
            summarize()
        engine.log("runs_complete", plan=args.plan, host="141.61.45.124")


if __name__ == "__main__":
    main()
