#!/usr/bin/env python3
"""Reuse the reviewed validator in a private module instance for the new dataset."""
import importlib.util
import sys
from pathlib import Path

source=Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/validate_round.py")
spec=importlib.util.spec_from_file_location("dp16_private_request_validator",source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Only this new process/module instance is parameterized. The source file and the
# running single-node benchmark are unchanged.
module.DATA_ABBR="longbenchv2-all503-midcut128k"
if __name__=="__main__":
    if len(sys.argv)!=3:
        raise SystemExit("Usage: validate_round_dp16.py AISBENCH_DIRECTORY MANIFEST")
    module.validate(sys.argv[1],sys.argv[2])
