"""Compatibility entry point for the validated, snapshot-free results collector."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("recompute_operator_results.py")), run_name="__main__")
