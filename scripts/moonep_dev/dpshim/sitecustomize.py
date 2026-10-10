"""Harness-only DP shim for the MoonEP dp2ep2 benchmark (env-gated).

Why: vLLM's DP engine core narrows ``ASCEND_RT_VISIBLE_DEVICES`` to one card per
rank. MoonEP's symmetric-memory init then breaks on every rank whose card is not
0, because CANN mixes id spaces (``aclrtMemSetAccess`` and VMM allocation
properties carry the *physical* id, ``aclrtMallocPhysical``/``aclrtGetDevice``
the *in-process logical* id, which is 0 after narrowing).

Instead of narrowing, keep the full device list in every DP rank process and
bind each rank's worker to its own card -- the DP shift that vLLM v0.24 does in
``gpu_worker.py::init_device`` and that vllm-ascend's ``_init_device`` only
applies for versions != 0.23.0.

Enable with MOONEP_DP_FULL_VISIBLE=1. Benchmark-only; not a product fix.
"""
import os
import sys
from importlib.machinery import PathFinder

_ENABLED = os.environ.get("MOONEP_DP_FULL_VISIBLE") == "1"

def _patch_engine_utils(module):
    """Stop the DP engine-core *subprocess* launch from narrowing the env.

    CoreEngineProcManager wraps the spawn in set_device_control_env_var(), which
    rewrites ASCEND_RT_VISIBLE_DEVICES to this DP shard before the child starts;
    patching the child in-process is too late.
    """
    import contextlib

    if getattr(module, "_moonep_dp_shim", False):
        return

    def _set_device_control_env_var(vllm_config, local_dp_rank):
        return contextlib.nullcontext()

    module.set_device_control_env_var = _set_device_control_env_var
    module._moonep_dp_shim = True
    print("[dp-shim] set_device_control_env_var disabled", flush=True)


def _patch_engine_core(module):
    cls = getattr(module, "DPEngineCoreProc", None)
    if cls is None:
        return
    if getattr(cls, "_moonep_dp_shim", False):
        return

    def _set_visible_devices(self, vllm_config, local_dp_rank):
        # Keep the full base list: the worker picks its own device below.
        return

    cls._set_visible_devices = _set_visible_devices
    cls._moonep_dp_shim = True
    print("[dp-shim] DPEngineCoreProc._set_visible_devices disabled", flush=True)


def _patch_worker(module):
    cls = getattr(module, "NPUWorker", None)
    if cls is None:
        return
    if getattr(cls, "_moonep_dp_shim", False):
        return
    orig = cls._init_device

    def _init_device(self):
        pc = self.parallel_config
        if (
            pc.data_parallel_size > 1
            and pc.distributed_executor_backend not in ("ray", "external_launcher")
            and pc.data_parallel_backend != "ray"
            and pc.nnodes_within_dp == 1
            and getattr(pc, "assigned_physical_gpu_ids", None) is None
        ):
            try:
                from vllm_ascend.utils import vllm_version_is

                needs_shift = vllm_version_is("0.23.0")
            except Exception:  # noqa: BLE001
                needs_shift = True
            if needs_shift:
                dp_local_rank = pc.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = pc.data_parallel_index
                tp_pp_world_size = pc.pipeline_parallel_size * pc.tensor_parallel_size
                self.local_rank += dp_local_rank * tp_pp_world_size
                print(f"[dp-shim] worker local_rank -> {self.local_rank}", flush=True)
        # MOONEP_DEV_BASE: start at an absolute device id (e.g. 4 for cards 4-7
        # with the *full* visible list). Keeping logical == physical is what lets
        # the vendored shmem's buffer-property check (location.id vs
        # aclrtGetDevice) pass -- see README limitation #7. Applies to TP-only
        # runs too, which have no DP shift above.
        base = int(os.environ.get("MOONEP_DEV_BASE", "0"))
        if base:
            self.local_rank += base
            print(f"[dp-shim] worker local_rank -> {self.local_rank} (base {base})", flush=True)
        return orig(self)

    # vllm-ascend hard-codes "hccl" in _init_worker_distributed_environment, so a
    # platform-attribute patch is not enough: wrap the imported function.
    if os.environ.get("MOONEP_FORCE_GLOO") == "1":
        real_init = getattr(module, "init_distributed_environment", None)
        if real_init is not None and not getattr(real_init, "_moonep_forced", False):
            def forced(world_size, rank, init_method, local_rank, backend="nccl", **kw):
                print(f"[dp-shim] init_distributed_environment backend {backend} -> gloo", flush=True)
                return real_init(world_size, rank, init_method, local_rank, "gloo", **kw)
            forced._moonep_forced = True
            module.init_distributed_environment = forced

    cls._init_device = _init_device
    cls._moonep_dp_shim = True
    print("[dp-shim] NPUWorker._init_device DP shift installed", flush=True)


def _patch_platform(module):
    """MOONEP_FORCE_GLOO=1: use the Gloo backend for the process group.

    8-card HCCL cannot initialise on this box: the 0-3 and 4-7 fabrics disagree
    on TLS (Config_Error EI0016) and HCCL error code 4 follows. With DP x TP=1
    the engine's only collective is the DP token-count sync (CPU tensors), so a
    Gloo group is sufficient and MoonEP's own MoE traffic goes over SHMEM.
    """
    if os.environ.get("MOONEP_FORCE_GLOO") != "1":
        return
    for name in ("NPUPlatform", "AscendPlatform"):
        cls = getattr(module, name, None)
        if cls is not None and getattr(cls, "dist_backend", None) != "gloo":
            cls.dist_backend = "gloo"
            print(f"[dp-shim] {name}.dist_backend -> gloo", flush=True)


_PATCHERS = {
    "vllm_ascend.platform": _patch_platform,
    "vllm.platforms.ascend": _patch_platform,
    "vllm.v1.engine.core": _patch_engine_core,
    "vllm.v1.engine.utils": _patch_engine_utils,
    "vllm_ascend.worker.worker": _patch_worker,
}


class _Patcher:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHERS:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module, _real=real_exec, _name=fullname):
            _real(module)
            try:
                _PATCHERS[_name](module)
            except Exception as exc:  # noqa: BLE001
                print(f"[dp-shim] patch {_name} failed: {exc}", flush=True)
                raise

        spec.loader.exec_module = exec_module
        return spec


if _ENABLED:
    sys.meta_path.insert(0, _Patcher())


def _install_backend_fallback():
    """MOONEP_FORCE_GLOO=1: tolerate vllm-ascend's _get_backend(torch.device("npu")).

    With a Gloo process group there is no NPU backend registered, so that call
    raises "No backend type associated with device type npu" during model init.
    Fall back to the group's default backend.
    """
    if os.environ.get("MOONEP_FORCE_GLOO") != "1":
        return
    try:
        from torch.distributed.distributed_c10d import ProcessGroup
    except Exception:  # noqa: BLE001
        return
    if getattr(ProcessGroup, "_moonep_backend_fallback", False):
        return
    orig = ProcessGroup._get_backend

    class _StubBackend:
        """Placeholder for _get_backend(torch.device("npu")) under Gloo.

        vllm-ascend's dispatchers only use the result to fetch an HCCL comm name
        for the MC2/all2all ops; MoonEP legs construct them but never call them,
        so a placeholder is enough (exact-name lookups would fail on Gloo).
        """

        def __init__(self, name: str):
            self._name = name

        def get_hccl_comm_name(self, rank, *args, **kwargs):
            return f"moonep-gloo-stub-{self._name}-{rank}"

    def _get_backend(self, device=None):
        try:
            return orig(self, device)
        except RuntimeError:
            # no backend for that device (npu under Gloo): try cpu, else stub
            try:
                return orig(self, torch.device("cpu"))
            except Exception:  # noqa: BLE001
                return _StubBackend("gloo")

    ProcessGroup._get_backend = _get_backend
    ProcessGroup._moonep_backend_fallback = True
    print("[dp-shim] ProcessGroup._get_backend fallback installed", flush=True)


_install_backend_fallback()
