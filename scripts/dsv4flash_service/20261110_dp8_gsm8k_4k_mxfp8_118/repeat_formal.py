"""Run two additional independent Off -> MXFP8 Slot pairs on 118."""
import fcntl
import json
import shutil
import signal
import subprocess
from pathlib import Path

import report
import run_experiment as experiment
from verify_wire import digest

ROOT = experiment.ROOT
ENGINE = experiment.engine
PAIRS = ("pair02", "pair03")
SOURCE = ROOT / "formal"


def check_first_pair():
    for mode in ("off", "slot"):
        results = ROOT / "formal/data/formal/aisbench"
        metrics, = results.glob(f"*-{mode}-*/round_1/metrics.json")
        data = json.loads(metrics.read_text())
        assert data["valid"] and data["requests"] == 1319
        audit, = metrics.parent.glob("*_mode_audit.json")
        proof = json.loads(audit.read_text())
        for source, expected in proof["source_sha256"].items():
            assert digest(experiment.WS / Path(source)) == expected, source


def prepare(pair):
    target = ROOT / "repeat_formal" / pair / "formal"
    target.mkdir(parents=True)
    (target / "data").mkdir()
    (target / "data/selection_manifest.json").symlink_to((SOURCE / "data/selection_manifest.json").resolve(strict=True))
    for item in SOURCE.iterdir():
        if item.suffix == ".sh":
            path = target / item.name
            text = item.read_text().replace(str(SOURCE), str(target))
            text = text.replace("PAIR_LABEL='mxfp8'", f"PAIR_LABEL='{pair}'")
            text = text.replace("moonep_gsm8k4k118_formal", f"moonep_gsm8k4k118_{pair}")
            assert str(target) in text and f"PAIR_LABEL='{pair}'" in text
            path.write_text(text)
            path.chmod(0o755)
            subprocess.run(["bash", "-n", path], check=True)
        elif item.suffix == ".py":
            if item.is_symlink():
                (target / item.name).symlink_to(item.resolve(strict=True))
            else:
                shutil.copyfile(item, target / item.name)
    return target.parent


def main():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, ENGINE.on_signal)
    with experiment.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment.verify()
        check_first_pair()
        ENGINE.resources_free()
        directories = {pair: prepare(pair) for pair in PAIRS}
        for pair in PAIRS:
            ENGINE.ROOT = directories[pair]
            ENGINE.STATE = ENGINE.ROOT / "STATE.json"
            for mode in ("off", "slot"):
                experiment.verify()
                check_first_pair()
                ENGINE.run_case("formal", mode)
            report.ROOT = ENGINE.ROOT
            cases, change = report.formal_results()
            assert set(cases) == {"off", "slot"}
            (ENGINE.ROOT / "FORMAL_RESULTS.json").write_text(json.dumps(
                {"host": "141.61.45.118", "pair": pair, "formal": cases, "slot_vs_off_pct": change},
                ensure_ascii=False, indent=2) + "\n")
            ENGINE.log("pair_complete", pair=pair, host="141.61.45.118")


if __name__ == "__main__":
    main()
