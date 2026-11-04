#!/usr/bin/env python3
"""Run the approved full matrix using the existing service and benchmark scripts."""
import json
import subprocess
import sys
from pathlib import Path
import run_confirmed_remainder as engine

ROOT = Path(__file__).resolve().parent
SESSION = Path(sys.argv[1])
ORDER = [("pair01", "off"), ("pair01", "slot"), ("pair02", "slot"), ("pair02", "off")]

if __name__ == "__main__":
    try:
        engine.log("full_matrix_begin", order=ORDER, implementation="original repos; fused FP32 weighted combine")
        for pair, mode in ORDER:
            engine.run_case(pair, mode)
        engine.log("all_remaining_complete", valid_cases=4)
        subprocess.run([sys.executable, str(ROOT / "finalize_results.py"), str(SESSION)], check=True)
    except Exception as error:
        engine.log("stopped_on_error", error=repr(error))
        raise
