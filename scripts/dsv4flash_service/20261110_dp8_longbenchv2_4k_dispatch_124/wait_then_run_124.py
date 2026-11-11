"""Wait for the host and ports to become free, then run the single MXFP8 pair."""
import datetime
import subprocess
import sys
import time
from pathlib import Path

import controller_base as engine
from run_experiment import verify_provenance

ROOT = Path(__file__).resolve().parent


def main():
    verify_provenance()
    prior_error = None
    while True:
        try:
            engine.resources_free()
        except RuntimeError as error:
            if str(error) != prior_error:
                prior_error = str(error)
                print(datetime.datetime.now().astimezone().isoformat(), "waiting", prior_error, flush=True)
            time.sleep(20)
        else:
            break
    print(datetime.datetime.now().astimezone().isoformat(), "124 eight NPUs and ports available", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "run_experiment.py"), "--plan", "mxfp8", "--execute"])


if __name__ == "__main__":
    sys.exit(main())
