"""Load newly built original-repo binaries in this test process only.

No package copy, editable install, binary replacement or device allocation.
Call before importing ascend_moonep anywhere else.
"""
import ctypes
import importlib.util
import os
import sys
from pathlib import Path

SDK = Path("/mnt/share/x00893896/vllm_ascend_moonep_integration/ascend-moonep")
BUILD = SDK / "build/weighted_udma_20261104"

def load_staged():
    if "ascend_moonep" in sys.modules:
        raise RuntimeError("Staged loader must run before importing ascend_moonep")
    import torch
    import torch_npu
    mode = getattr(os,"RTLD_NOW",0) | getattr(ctypes,"RTLD_GLOBAL",0)
    # Load one SHMEM runtime before _C resolves its dependencies.
    for name in ("libshmem_utils.so","aclshmem_bootstrap_config_store.so",
                 "libshmem.so","aclshmem_bootstrap_uid.so"):
        p=SDK/"ascend_moonep/lib"/name
        if p.exists():ctypes.CDLL(str(p),mode=mode)
    kernel=BUILD/"device/lib/libascend_moonep_kernels.so"
    ctypes.CDLL(str(kernel),mode=mode)
    extensions=list((BUILD/"host-lib/ascend_moonep").glob("_C*.so"))
    assert len(extensions)==1,extensions
    spec=importlib.util.spec_from_file_location("ascend_moonep._C",extensions[0])
    module=importlib.util.module_from_spec(spec)
    sys.modules["ascend_moonep._C"]=module
    spec.loader.exec_module(module)
    sys.path.insert(0,str(SDK))
    import ascend_moonep
    assert Path(ascend_moonep.__file__).resolve()==SDK/"ascend_moonep/__init__.py"
    libs={line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
          if ".so" in line and ("moonep" in line or "libshmem" in line)}
    assert str(kernel) in libs,libs
    assert len([p for p in libs if Path(p).name=="libshmem.so"])==1,libs
    assert all("/ascend_moonep/_C" not in p or "/host-lib/" in p for p in libs),libs
    return dict(source=str(SDK),extension=str(extensions[0]),kernel=str(kernel),
                libraries=sorted(libs))
