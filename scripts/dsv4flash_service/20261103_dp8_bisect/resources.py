"""Resource/provenance helpers reused from the non-executed crossover design."""
from __future__ import annotations
import datetime as dt
import os
from pathlib import Path
import re
import runtime
import study
SOURCE_FILES = ("runtime.py", "study.py", "measurements.py", "resources.py", "bisect_logic.py", "run_bisect.py", "settings.json", "strategy.json", "EXPERIMENT.md")

def source_hashes():
    return {name: runtime.sha256(study.HERE / name) for name in SOURCE_FILES}



def environment_signature(cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    runtime.inventory(cfg, directory)
    record = study.read(directory / "environment.json")
    record.pop("at", None)
    if any(not item["head"] for item in record["repos"].values()):
        raise RuntimeError("Cannot determine repository versions.")
    record["pip_freeze_sha256"] = runtime.sha256(directory / "pip_freeze.txt")
    relevant = {k: v for k, v in os.environ.items() if k.startswith((
        "VLLM_", "MOONEP", "ASCEND_", "HCCL_", "GLOO_", "TP_", "OMP_"))
        or k in ("LD_LIBRARY_PATH", "PYTHONPATH")}
    record["ambient_runtime_environment_sha256"] = study.digest(relevant)
    version_files = [Path(cfg["cann_env"]), Path("/usr/local/Ascend/driver/version.info"),
                     Path("/usr/local/Ascend/ascend-toolkit/latest/share/info/runtime/version.info")]
    record["cann_driver_files"] = {str(p): runtime.sha256(p) for p in version_files if p.is_file()}
    workspace = Path(cfg["workspace"])
    binaries = []
    for package in (workspace / "ascend-moonep/ascend_moonep",
                    workspace / "vllm-ascend/vllm_ascend"):
        binaries.extend(package.glob("*cpython-311-aarch64*.so"))
        binaries.extend(package.glob("lib/*.so"))
        binaries.extend(package.glob("_cann_ops_custom/vendors/custom_transformer/op_api/lib/*.so"))
    record["binary_sha256"] = {str(path): runtime.sha256(path) for path in sorted(set(binaries))}
    if not record["binary_sha256"]:
        raise RuntimeError("Missing compiled extension provenance.")
    return study.digest(record)



def archived_route(root, descriptor):
    path = root / "sim" / descriptor["filename"]
    if not path.is_file() or runtime.sha256(path) != descriptor["sha256"]:
        raise ValueError("Archived route content changed.")
    return {key: value for key, value in descriptor.items() if key != "text"} | {"path": str(path)}



def archive_partial_pair(root, pair):
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    for case in pair["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]):
            raise ValueError("Invalid case ID.")
        path = (root / case["id"]).resolve()
        if path.parent != root.resolve():
            raise ValueError("Case path escapes session.")
        if path.exists():
            destination = root / "attempts" / stamp / case["id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)  # Preserve both old modes; never mix half-pairs across retries.
