#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Per-rank MoE expert-load instrumentation for offline peak-to-average analysis.

Every MoE communication method already reports its per-local-expert (token, expert)
route counts through ``FusedExpertsResult.expert_tokens`` -- the same quantity EPLB
accumulates into ``layer.moe_load``. This module taps that value and appends it,
once per (step, layer), to a per-rank text dump.

Nothing here communicates. The peak-to-average ratio is a cross-rank quantity, but
the aggregation is offline in ``scripts/analyze_moe_par.py``: each rank writes only
its own load and the script stitches the ranks together. That keeps the hot path
free of collectives -- one device-to-device copy per layer per step, plus a single
``.cpu()`` per ``MOE_PAR_STATS_FLUSH_EVERY`` steps.

Enable with ``VLLM_ASCEND_MOE_PAR_STATS=1``. The module is inert otherwise, and it
is independent of EPLB (``dynamic_eplb`` may stay off).

``VLLM_ASCEND_MOE_PAR_STATS_DIR``         output directory, default ``/tmp/moe_par_stats``
``VLLM_ASCEND_MOE_PAR_STATS_TAG``         run label in the filename, default ``run``
``VLLM_ASCEND_MOE_PAR_FLUSH_EVERY``       steps buffered per ``.cpu()``, default ``16``

Dump format, one file per rank named ``moe_par_{TAG}_rank{RANK}.log``::

    # meta model=... rank=0 ep_rank=0 ep_size=4 ... moe_layers=48 topk=8 ...
    # layer layer_ord=0 instance_id=3 log2phy=[...] expert_map=[...]
    step=0 layer=0 method=MoonEPCommImpl counts=[12,0,45,...]

The ``# meta`` line carries every dimension the analysis needs, so the script stays
model-agnostic, and ``# layer`` maps local expert slots back to global expert ids.
A second header line, ``# meta partial_steps=N pad_pairs_subtracted=...``, records
how many steps closed without every known layer running -- those rows hold real
zeros for the layers that did not run, so the load in them is understated rather
than absent -- plus the MoonEP pad total, which is not per step. It is written at
the *end* of the file and rewritten after every flush, so it is already there when
a run is cut short; the exit hook only gets to add the final, in-flight one.

Relying on that hook alone was a mistake worth recording: ``atexit`` does not run
in a ``fork``ed worker. ``multiprocessing``'s fork child ends in ``os._exit()`` and
skips the handlers, so the tail of the dump -- up to ``VLLM_ASCEND_MOE_PAR_FLUSH_EVERY``
steps, and with them every pad number -- went missing on a run started without
``VLLM_WORKER_MULTIPROC_METHOD=spawn``. ``scripts/moe_par_bench.py`` therefore
defaults that variable to ``spawn``; the per-flush rewrite below is the belt to
that pair of braces. A step still in flight at exit is dropped outright when it is
incomplete.

Two counting scopes are dumped, named by ``counts_scope`` in the header:

``local`` (every method except MoonEP)
    the row is index by this rank's local expert slot, as the method reports it.

``global`` (MoonEP)
    the row is index by *global* (logical) expert id and spans every expert of the
    model, so summing the ranks of one (step, layer) recovers the run's real
    routing; the analysis then credits each expert's load to the rank that holds
    it. MoonEP uses this scope because both of its own numbers are unusable for
    load accounting: the dispatcher's receive-side counts carry the capacity pads,
    and the pad rows sit in the same slot as real rows of the same expert, so they
    cannot be told apart offline (and rows the dispatch drops are missing from them
    entirely). The sending-side histogram the dispatcher hands over through
    ``note_route_counts()`` is exact by construction -- one histogram per layer,
    counted before the plan -- and the pads are subtracted where they were counted.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

# Read once at import: the switches are process-wide and the hot path must not pay
# an env lookup. ``enabled()`` is the only gate callers need.
_ENABLED = envs.VLLM_ASCEND_MOE_PAR_STATS
_DIR = envs.VLLM_ASCEND_MOE_PAR_STATS_DIR
_TAG = envs.VLLM_ASCEND_MOE_PAR_STATS_TAG
_FLUSH_EVERY = max(1, envs.VLLM_ASCEND_MOE_PAR_FLUSH_EVERY)

# Layers are numbered by first sight, so neither the layer count nor the layer
# order is assumed anywhere.
_ord_of: dict[int, int] = {}
_layers: dict[int, Any] = {}
_valid: set[int] = set()

# Layers already recorded for the step in flight: seeing one twice means the
# previous step ended. Every MoE layer runs exactly once per forward.
_step_seen: set[int] = set()
_current_method = "?"

_buf: torch.Tensor | None = None  # [window, layers, local_experts] int32, on device
_slot = 0  # window-local index of the step in flight
_step_base = 0  # global step index of _buf[0]
_methods: list[str] = []  # comm method name per completed step in the window

_out = None
_meta_written = False
# Byte offset at which the trailing summary line starts. The rows are appended
# before it and the summary is rewritten in place, so the file stays line-oriented
# and the summary is never stale -- see _rewrite_tail.
_tail_at: int | None = None
_warned: set[str] = set()

# Steps closed after running fewer layers than are known to exist. Their missing
# layers would be written as real zeros, understating that rank's load, so the
# count goes into the dump header for the analysis script to warn about.
_partial_steps = 0

# The routing histogram of the layer whose dispatch just finished, announced by the
# dispatcher through ``note_route_counts()`` and consumed by the ``record()`` that
# follows it. It replaces the method's own per-expert counts (the receiving side of
# the dispatch, which carries MoonEP's capacity pads and misses rows the dispatch
# dropped) and carries the pad block to remove: (histogram, pad_pairs, pad_expert).
_route_note: tuple[Any, int, int] | None = None
_pad_pairs_subtracted = 0
_pad_notes_used = 0
_pad_notes_misplaced = 0
_pad_expert_used: int | None = None
_negative_cells = 0

# ``global`` for a run whose rows are index by global expert id, ``local`` for one
# index by local slot. A run that used both is recorded as ``mixed``: the analysis
# would be reading one of the two widths wrong, so it must not guess.
_scopes: set[str] = set()


def enabled() -> bool:
    """Master switch. Callers short-circuit on this so the disabled path is one call."""
    return _ENABLED


def note_route_counts(histogram: torch.Tensor, pad_pairs: int, pad_index: int) -> None:
    """Announce the routing histogram of the layer currently being dispatched.

    ``histogram`` counts this step-layer's (token, expert) pairs by *global*
    (logical) expert id -- MoonEP's routing space -- with the capacity pads
    included, and ``pad_pairs`` of those sit in ``pad_index`` and are not real
    traffic. The recording side subtracts them, so what lands in the dump is the
    layer's real routing per logical expert: the quantity that stays meaningful
    when an expert is replicated (MoonEP's prefetch slots, EPLB redundancy) and
    that sums across ranks into the run's global routing.

    Called on every MoonEP dispatch, so it overwrites rather than accumulates: a
    dispatch with no padding clears the pad count instead of leaving a stale one
    behind. The dispatcher and the ``record()`` that consumes this run inside the
    same layer's forward, in that order.
    """
    global _route_note
    if not _ENABLED:
        return
    if histogram is None or histogram.numel() == 0:
        _route_note = None
        return
    _route_note = (histogram, max(0, int(pad_pairs)), int(pad_index))


def _in_profile_run() -> bool:
    """Whether this forward is a profile/dummy pass.

    Unreadable means "no": the flag only matters during startup, and treating a
    missing forward context as a profile run would throw the whole dump away.
    """
    try:
        return bool(_EXTRA_CTX.in_profile_run)
    except Exception:
        return False


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def _layer_key(layer: Any) -> int:
    # ``moe_instance_id`` is a process-unique construction-order counter, so it keys
    # a layer even when more than one LLM instance lives in the process.
    instance_id = getattr(layer, "moe_instance_id", None)
    return id(layer) if instance_id is None else int(instance_id)


def _to_counts(expert_tokens: torch.Tensor, group_list_type: int) -> torch.Tensor:
    flat = expert_tokens.reshape(-1)
    if group_list_type == 1:  # count mode: already per-expert
        return flat
    # cumulative mode: difference back to per-expert counts
    return torch.cat([flat[:1], flat[1:] - flat[:-1]])


def _take_route_note() -> tuple[Any, int, int] | None:
    """Consume the pending routing note, so a note is never applied twice."""
    global _route_note
    note, _route_note = _route_note, None
    return note


def _counts_from_route_note(note: tuple[Any, int, int] | None) -> torch.Tensor | None:
    """The note's histogram with this rank's pad block removed, or None.

    The histogram is this rank's exact sending side, so the pads come off where
    they were counted and the subtraction can never leave a negative amount of real
    traffic -- unlike a subtraction against the receiving side, whose pad rows
    share a slot with real rows of the same expert. One device-side subtraction per
    layer: no host sync, no collective.
    """
    global _pad_pairs_subtracted, _pad_notes_used, _pad_notes_misplaced, _pad_expert_used
    if note is None:
        return None
    histogram, pairs, index = note
    # ``copy=True`` is load-bearing, not defensive: the dispatcher's histogram is
    # already int32, and ``to()`` returns the *same* tensor when the dtype matches,
    # so the subtraction below would otherwise land on the engine's ``tpe``. The
    # plan does not read it back, but the recorder must not be the thing that finds
    # out whether that stays true.
    counts = histogram.reshape(-1).to(torch.int32, copy=True)
    _pad_notes_used += 1
    _scopes.add("global")
    if pairs:
        if 0 <= index < counts.numel():
            counts[index] -= pairs
            _pad_pairs_subtracted += pairs
            _pad_expert_used = index
        else:
            # The dispatcher names the pad expert in the histogram's own space, so
            # this means the two disagreed about the model's expert count.
            _pad_notes_misplaced += 1
    return counts


def record(layer: Any, expert_tokens: torch.Tensor | None, group_list_type: int, method_name: str) -> None:
    """Record this layer's real routing for the step in flight.

    MoonEP announces its sending-side histogram through ``note_route_counts()``;
    that histogram wins over ``expert_tokens`` because it is the *sending* side of
    the dispatch, exact by construction. Every other method reports its own real
    counts here, index by local expert slot.
    """
    if not _ENABLED:
        return
    # Consumed even on the early returns below: a note must not outlive the layer
    # that produced it and be applied to an unrelated one.
    note = _take_route_note()
    if expert_tokens is None and note is None:
        return
    try:
        if _in_profile_run():
            # Profile and dummy forwards at startup would otherwise pollute the dump.
            return
        counts = _counts_from_route_note(note)
        if counts is None:
            counts = _to_counts(expert_tokens, group_list_type)
            _scopes.add("local")
        _record(layer, counts, method_name)
    except Exception:
        if "record" not in _warned:
            _warned.add("record")
            logger.warning("[moe-par] dropped a sample: failed to record MoE load", exc_info=True)


def _ensure_buf(counts: torch.Tensor) -> None:
    """Allocate the device window on first use and widen it as new layers appear."""
    global _buf
    width = int(counts.numel())
    if _buf is None:
        _buf = torch.zeros((_FLUSH_EVERY, max(len(_ord_of), 1), width), dtype=torch.int32, device=counts.device)
        return
    if _buf.shape[2] != width or _buf.shape[1] >= len(_ord_of):
        return
    grown = torch.zeros((_FLUSH_EVERY, len(_ord_of) * 2, _buf.shape[2]), dtype=torch.int32, device=_buf.device)
    grown[:, : _buf.shape[1], :] = _buf
    _buf = grown


def _record(layer: Any, counts: torch.Tensor, method_name: str) -> None:
    global _slot, _current_method, _partial_steps
    key = _layer_key(layer)
    layer_ord = _ord_of.setdefault(key, len(_ord_of))
    _ensure_buf(counts)

    width = _buf.shape[2]
    if counts.numel() != width:
        _warn_once(
            f"width:{counts.numel()}",
            "[moe-par] layer %d routes %d local experts, buffer holds %d; skipping this layer",
            layer_ord,
            counts.numel(),
            width,
        )
        return
    _valid.add(layer_ord)
    _layers[layer_ord] = layer

    if key in _step_seen:
        # This layer already ran this step, so the previous step is complete.
        if len(_step_seen) < len(_valid):
            # The step ended without every known layer running, so the layers that
            # did not run are about to be written as zeros. Flag it rather than let
            # a short step look like a lightly loaded one.
            _partial_steps += 1
        _methods.append(_current_method)
        _slot += 1
        _step_seen.clear()
        if _slot >= _FLUSH_EVERY:
            _flush()
    if not _step_seen:
        # First layer of a step: the whole step shares one comm method, because
        # select_moe_comm_method keys on the step's token count only.
        _current_method = method_name
    _step_seen.add(key)

    _buf[_slot, layer_ord, :] = counts.to(torch.int32)


def _clamp_negatives(host: np.ndarray) -> np.ndarray:
    """Keep impossible counts out of the dump, and count them.

    A (token, expert) pair count cannot be negative. MoonEP's pad subtraction is
    exact by construction -- the dispatcher subtracts the block it wrote itself,
    in the histogram's own space -- and every other method reports counts straight
    from the kernel, so a negative cell means two sides disagreed about a layer.
    Clamping keeps the dump readable; the tally in the exit header keeps the
    disagreement visible.
    """
    global _negative_cells
    bad = int(np.count_nonzero(host < 0))
    if bad:
        _negative_cells += bad
        np.clip(host, 0, None, out=host)
    return host


def _summary_line() -> str:
    """The trailing ``# meta`` line: what is known only once the run has run."""
    pad = f" pad_pairs_subtracted={_pad_pairs_subtracted}"
    if _pad_expert_used is not None:
        pad += f" pad_expert_global={_pad_expert_used}"
    if _pad_notes_misplaced:
        pad += f" pad_notes_misplaced={_pad_notes_misplaced}"
    if _negative_cells:
        pad += f" negative_cells_clamped={_negative_cells}"
    return f"# meta partial_steps={_partial_steps}{pad}\n"


def _rewrite_tail(out) -> None:
    """Write the summary line at the end of the dump, replacing the previous one.

    The counters it carries (incomplete steps, subtracted pads) only reach their
    final value at the end of the run, but the process may never get an exit hook:
    a ``fork``ed worker leaves through ``os._exit()``, which skips ``atexit``. So
    the line is refreshed after every flush and the exit path only has to refresh
    it once more. Rewriting needs a truncate because the line's width grows with
    its numbers.

    The line must be the last thing in the file when this is called -- the rows of
    a flush are appended only after :func:`_rewind_tail` has dropped it, and the
    exit path adds no rows of its own.
    """
    global _tail_at
    if _tail_at is None:
        # Nothing was written after the rows, so the summary starts here.
        _tail_at = out.tell()
    else:
        out.seek(_tail_at)
        out.truncate(_tail_at)
    out.write(_summary_line())
    out.flush()


def _rewind_tail(out) -> None:
    """Drop the summary line, leaving the file ended by its rows.

    Drops the offset with it: a stale offset would truncate a *new* file, or rows
    appended since, at a position that has nothing to do with them.
    """
    global _tail_at
    if _tail_at is not None:
        out.flush()
        out.seek(_tail_at)
        out.truncate(_tail_at)
        _tail_at = None


def _flush() -> None:
    """Copy the completed steps to host and append them to this rank's dump."""
    global _slot, _step_base, _meta_written
    if _slot == 0 or _buf is None:
        return
    host = _clamp_negatives(_buf[:_slot].cpu().numpy())
    out = _open()
    if not _meta_written:
        _write_meta(out)
        _meta_written = True
    _rewind_tail(out)
    for i in range(_slot):
        method = _methods[i] if i < len(_methods) else "?"
        step = _step_base + i
        for layer_ord in sorted(_valid):
            row = host[i, layer_ord].tolist()
            out.write(f"step={step} layer={layer_ord} method={method} counts=[{','.join(str(v) for v in row)}]\n")
    out.flush()
    _step_base += _slot
    _slot = 0
    _methods.clear()
    _rewrite_tail(out)


def _open():
    global _out, _tail_at
    if _out is not None:
        return _out
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"moe_par_{_TAG}_rank{_rank_id()}.log")
    _out = open(path, "w")
    # The file starts empty, so any offset from a previous one is meaningless.
    _tail_at = None
    logger.info("[moe-par] writing per-rank MoE load dump to %s", path)
    return _out


def _attr(obj: Any, name: str, default: Any) -> Any:
    """getattr that also survives a property raising, for swappable models."""
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return default if value is None else value


def _fmt_tensor(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return "[" + ",".join(str(int(v)) for v in value.reshape(-1).cpu().tolist()) + "]"
    except Exception:
        return "None"


def _model_name(layer: Any = None) -> str:
    """The served model's name, for the dump header.

    Read off the layer first: on a worker ``get_current_vllm_config()`` is not
    reliably set, which used to leave every header saying ``model=unknown``.
    """
    model = _attr(_attr(_attr(layer, "vllm_config", None), "model_config", None), "model", None)
    if model:
        return str(model)
    try:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        if config is not None:
            return str(config.model_config.model)
    except Exception:
        pass
    return os.environ.get("MODEL", "unknown")


def _rank_id() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "-1"))


def _world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return -1


def _scope_name() -> str:
    """What this dump's rows are index by: ``global``, ``local`` or ``mixed``."""
    if len(_scopes) == 1:
        return next(iter(_scopes))
    return "mixed" if _scopes else "local"


def _write_meta(out) -> None:
    """Emit the self-describing header so the analysis script assumes no dimensions."""
    probe = next(iter(_layers.values()), None)
    topk = _attr(_attr(probe, "moe_config", None), "experts_per_token", None)
    if topk is None:
        topk = _attr(probe, "top_k", -1)
    fields = {
        "model": _model_name(probe),
        "tag": _TAG,
        "rank": _rank_id(),
        "world_size": _world_size(),
        "ep_rank": _attr(probe, "ep_rank", -1),
        "ep_size": _attr(probe, "ep_size", -1),
        "moe_layers": len(_valid),
        "topk": topk,
        # How many steps one host copy covers. The analysis uses it to size the
        # shortfall a dump can carry when the process had no exit hook.
        "flush_every": _FLUSH_EVERY,
        "counts_scope": _scope_name(),
        "global_experts": _attr(probe, "global_num_experts", -1),
        "local_experts": _attr(probe, "local_num_experts", -1),
        "redundant_experts": _attr(probe, "global_redundant_expert_num", 0),
    }
    out.write("# meta " + " ".join(f"{k}={v}" for k, v in fields.items()) + "\n")
    for layer_ord, layer in sorted(_layers.items()):
        out.write(
            f"# layer layer_ord={layer_ord} instance_id={_layer_key(layer)} "
            f"log2phy={_fmt_tensor(_attr(layer, 'log2phy', None))} "
            f"expert_map={_fmt_tensor(_attr(layer, '_expert_map', None))}\n"
        )


def _shutdown() -> None:
    global _slot
    if not _ENABLED:
        return
    try:
        # A run normally stops mid-step. Close that step only when every known layer
        # ran in it; a partial row would understate the load and skew the ratio.
        if _step_seen and len(_step_seen) >= len(_valid):
            _methods.append(_current_method)
            _slot += 1
        _flush()
        if _out is not None:
            if _meta_written:
                # One last refresh: this hook does not run in a forked worker, which is
                # why _flush leaves the line behind as well.
                _rewrite_tail(_out)
            _out.close()
        if _pad_notes_misplaced:
            logger.warning(
                "[moe-par] %d MoonEP routing note(s) named a pad expert outside the histogram's "
                "%d experts; those pads were left in the dump. The dispatcher and the recorder "
                "disagree about the model's expert count.",
                _pad_notes_misplaced,
                _buf.shape[2] if _buf is not None else -1,
            )
        if _negative_cells:
            logger.warning(
                "[moe-par] %d row cell(s) came out negative and were clamped to zero. A (token, "
                "expert) pair count cannot be negative, so the dispatcher and this recorder "
                "disagreed about those layers; the dump is readable but those rows are suspect.",
                _negative_cells,
            )
        if _partial_steps:
            logger.warning(
                "[moe-par] %d step(s) ran fewer than the %d known MoE layers and were recorded "
                "with zeros for the missing ones; the load in those steps is understated",
                _partial_steps,
                len(_valid),
            )
    except Exception:
        logger.warning("[moe-par] failed to flush the MoE load dump on exit", exc_info=True)


atexit.register(_shutdown)
