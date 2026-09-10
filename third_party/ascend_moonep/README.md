# third_party/ascend_moonep — vendored ascend-moonep Python layer

`ascend_moonep` (SHMEM symmetric-memory MoE runtime) is developed in an external
repository that this project cannot push to, so the Python layer is **vendored
here** together with the fixes this integration needs. Everything the
integration changes lives in this Python layer; the compiled pieces stay in the
external build tree.

## What is (and is not) vendored

| | |
|---|---|
| vendored | `ascend_moonep/*.py` (the whole Python layer, ~200 KB) |
| **not** vendored | `ascend_moonep/_C*.so` (the C++ extension) and `ascend_moonep/lib/*.so` (CANN SHMEM libraries) — build artifacts of the external project |

At import time the vendored `__init__.py` looks for the extension next to
itself and, when it is absent, appends the external build directory to the
package path (`ASCEND_MOONEP_BUILD_DIR`, default
`/home/ascend-moonep/ascend_moonep`). So `import ascend_moonep` from this
directory yields the vendored Python code **plus** the external `.so` files,
without copying 60 MB of binaries into git.

## Using it

Put the vendored directory *first* on `PYTHONPATH` (it shadows an editable
install of the external package, which is what we want):

```bash
export PYTHONPATH=/path/to/vllm-ascend-moonep/third_party/ascend_moonep:$PYTHONPATH
```

Sanity check:

```bash
python3 -c "
import ascend_moonep, ascend_moonep._C as _C
print(ascend_moonep.__file__); print(_C.__file__)
from ascend_moonep import Buffer; print('Buffer ok')"
```

## Local changes carried here

1. **`buffer_c.py` — `acl_vmm_access_device_id()` (VMM device-id fix).**
   CANN 9.2 (Ascend950) mixes id spaces: `aclrtMallocPhysical` /
   `aclrtGetDevice` take the in-process *logical* id (index into
   `ASCEND_RT_VISIBLE_DEVICES`) while `aclrtMemSetAccess`'s `location.id` — and
   a VMM allocation property's `location.id` — carry the *physical* device id.
   The two coincide only while a process sees the full device list, so this is
   invisible in single-node TP runs but breaks as soon as a process is narrowed
   to one non-zero device (vLLM DP narrows each rank to a single card):
   `set_access(0)` then targets physical device 0 and fails with `507899`,
   which killed MoonEP init on every rank but rank 0. All three `set_access`
   call sites now translate, and the allocation path keeps using the original
   C++ implementation whenever the ids are equal.
   (Repro/probes: `/home/tmp/probe_vmm_id2.py`, `/home/tmp/probe_vmm_visible.py`.)

2. **`api.py` — `Buffer.dispatch(reuse_outputs=True)`** (off by default): keeps
   the plan/`cu_seqlens` pair and the non-zero-copy output tensors on the Buffer
   instead of allocating them per call. Contract: valid until the next
   `dispatch` on that Buffer; tails past the dispatched rows are stale by
   design, so consumers must key off `group_list`/`cu_seqlens`.

## Updating the vendored copy

```bash
cp /path/to/external/ascend-moonep/ascend_moonep/*.py \
   third_party/ascend_moonep/ascend_moonep/
git diff third_party/ascend_moonep   # keep the __init__ bootstrap + local fixes
```

The `__init__.py` bootstrap block is marked with `vllm-ascend-moonep vendored
copy`; re-apply it if upstream overwrites the file.
