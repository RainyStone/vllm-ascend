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
the aggregation is offline in ``scripts/moe_par/analyze_moe_par.py``: each rank writes only
its own load and the script stitches the ranks together. That keeps the hot path
free of collectives and of device arithmetic: a layer costs one device-to-device
copy per row it writes and nothing else -- the pad subtraction and the folding of
MoonEP's receive-side rows back onto global expert ids are host arithmetic at the
flush -- and one ``.cpu()`` per ``MOE_PAR_STATS_FLUSH_EVERY`` steps pays for the
whole window.

Enable with ``VLLM_ASCEND_MOE_PAR_STATS=1``. The module is inert otherwise, and it
is independent of EPLB (``dynamic_eplb`` may stay off).

``VLLM_ASCEND_MOE_PAR_STATS_DIR``         output directory, default ``/tmp/moe_par_stats``
``VLLM_ASCEND_MOE_PAR_STATS_TAG``         run label in the filename, default ``run``
``VLLM_ASCEND_MOE_PAR_FLUSH_EVERY``       steps buffered per ``.cpu()``, default ``16``
``MOONEP_BALANCE_STATS=1``                also log the running ledger every
``MOONEP_BALANCE_STATS_EVERY=N``          ``N`` layers (0 = exit only). The ledger
line comes from this module's own host-side totals, so it can never disagree with
the dump, and it is the only ledger a force-killed service leaves in the log.

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
``VLLM_WORKER_MULTIPROC_METHOD=spawn``. ``scripts/moe_par/moe_par_bench.py`` therefore
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

MoonEP additionally writes a second row per (step, layer), marked ``scope=compute``,
and the header then says ``counts_scope=global+compute``. Both rows span the model's
experts; they differ in *what they are attributed to*:

* the ``global`` row is the routing: which logical experts the gate chose. The
  analysis attributes it to the ranks that *hold* those experts, so a row a slot
  moved to another rank still counts against the owner -- that is the view that
  answers "where would the load have gone without relocation", and it is an upper
  bound on the real imbalance.
* the ``compute`` row is the work: which rows this rank actually ran. It is the
  dispatcher's receive-side counts (``local_counts``, including the slot groups a
  prefetch created) handed over raw through ``note_compute_load()`` and folded back
  onto global expert ids by the flush, so a row kept local and computed against a
  prefetched weight lands on the rank that computed it -- "who is busy", which is
  what a peak-to-average ratio over ranks is supposed to measure.

One (token, expert) pair is counted once in each row: the expert's own column sums
across the ranks give the load of that logical expert, and a rank's row sums give
what that rank computed, so ``sum_experts == sum_ranks == computed pairs`` holds
inside the compute view. The two views differ only in the rank dimension (the
expert dimension agrees up to the rows the capacity envelope drops). The compute
row keeps the capacity pads, because on the receiving side a pad row is
indistinguishable from a real row of the same expert and may be computed by any
rank that prefetched that expert; the pad block is accounted for by the ``global``
row's ledger (``pad_pairs_subtracted``), and the analysis reports both numbers.
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

# MOONEP_BALANCE_STATS=1: log a ``[moe-par-ledger]`` line every
# MOONEP_BALANCE_STATS_EVERY layers (0 = never, the dump is the only output). The
# line mirrors the dump's own host-side totals -- see _ledger_line -- so the served
# leg gets the same per-step ledger the MoonEP dispatcher prints, from the one
# source both legs already agree on.
_BALANCE_ON = os.environ.get("MOONEP_BALANCE_STATS", "0") == "1"
_BALANCE_EVERY = int(os.environ.get("MOONEP_BALANCE_STATS_EVERY", "0") or 0)

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
# The block is named here and taken off at the flush, in the histogram's own space.
_route_note: tuple[Any, int, int] | None = None
# Ledger totals, accrued in _flush from the rows actually written rather than from
# the notes that announced them: the note of the layer that closes a step arrives
# before that step is flushed, so a note-time total runs one layer ahead of the
# dump and the analysis cross-check (pad rows vs pad total) would fire on it.
_pad_pairs_subtracted = 0
_route_pairs_total = 0
_pad_notes_used = 0
_pad_notes_misplaced = 0
_pad_expert_used: int | None = None
_negative_cells = 0

# The rows this rank is about to *compute* for the layer whose dispatch just
# finished, announced through ``note_compute_load()`` and consumed by the
# ``record()`` that follows it -- same one-shot contract as the routing note. The
# payload is (counts, home_first, epn, slot_experts), staged by _stage_compute and
# folded onto the model's global expert ids at the flush, once the model's expert
# count is known from the routing rows.
_compute_note: tuple[Any, int, int, Any] | None = None
# The staged compute rows, as MoonEP handed them over: [home(epn) | slots(B) |
# mapping(B)] int32, width ``epn + 2B`` (``epn`` alone in owner mode, which has no
# slots). Raw rather than folded because the fold is a scatter-add: on the device
# it cost ~10 small kernels per layer, in the layer's critical path, and only the
# MoonEP legs would have paid them -- the one asymmetry the two accounting views
# must not introduce. ``_raw_geo`` is the (home_first, epn, B) that fixes the
# layout; the first row sets it and a later row that disagrees is not accommodated.
_buf_raw: torch.Tensor | None = None
_raw_geo: tuple[int, int, int] | None = None
_compute_ords: list[set[int]] = []  # window slot -> layer ords that have a compute row
_compute_rows_total = 0
_compute_pairs_total = 0

# The pad block the routing notes carried, one (slot, layer_ord, expert, pairs) per
# layer that padded. Host-side ints only -- the note knows them already, so pads
# need no device buffer -- and they let the dump show padding as its own ledger
# instead of only as a scalar in the tail.
_pad_rows: list[tuple[int, int, int, int]] = []
_pad_rows_total = 0

# ``global`` for a run whose rows are index by global expert id, ``local`` for one
# index by local slot, ``compute`` for the extra rows MoonEP writes for the work
# this rank actually ran. A run that used both ``global`` and ``local`` is recorded
# as ``mixed``: the analysis would be reading one of the two widths wrong, so it
# must not guess. ``global`` + ``compute`` is not a mix -- both rows span the model
# and each has a single meaning -- so it gets its own name.
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


def note_compute_load(
    counts: torch.Tensor,
    home_first: int,
    epn: int,
    slot_experts: torch.Tensor | None = None,
) -> None:
    """Announce the rows this rank is about to *compute*, by global expert id.

    MoonEP hands over its own receive-side counts -- what it is going to run, as
    opposed to what the gate routed. They are laid out as ``[epn home experts |
    B prefetch slot groups]``: the home block covers the contiguous global expert
    range starting at ``home_first``, and group ``b`` holds the experts this rank
    pulled into slot ``b`` (``slot_experts[b]``, a global expert id, ``-1`` for an
    empty slot). Owner mode has no slots and passes ``None``.

    Recording this row is what makes "who is actually busy" measurable: the row is
    attributed to the rank that wrote it, so a row the plan relocated to another
    rank counts against *that* rank, unlike the routing row, which is attributed to
    the expert's owner. The two rows are the two accounting views of the same
    dispatch, and the analysis reports both.

    The counts include the capacity pads, because on the receiving side a pad row
    is a row like any other: it sits in the same slot as the expert's real rows and
    the plan may hand it to any rank that holds that expert, so it cannot be told
    apart here. The pads are therefore also announced *apart* -- the routing note
    carries them, and the recorder writes them as their own rows (``scope=pad``) --
    so the report can split "computed" into real work and padding.

    The payload is stored exactly as announced -- the slot mapping copied along with
    the counts, for the same reason -- and folded onto global expert ids at flush
    time, when the model's expert count is known from the routing rows. Folding
    there is what keeps the two accounting views from costing different amounts to
    measure: see :func:`_fold_compute_row`.

    Overwrites rather than accumulates, exactly like ``note_route_counts()``: the
    ``record()`` of the same layer consumes it once.
    """
    global _compute_note
    if not _ENABLED:
        return
    if counts is None or counts.numel() == 0 or epn <= 0:
        _compute_note = None
        return
    # Snapshot, don't hold: the caller goes on to hand these counts to the grouped
    # matmul as its `group_list`, and the recorder reads the note only after that.
    # A count vector that a kernel rewrote on the way would silently misreport the
    # load -- the one failure a measuring tool must not have. A few dozen ints, so
    # the copy is a rounding error next to the layer it measures.
    _compute_note = (counts.detach().clone(), int(home_first), int(epn), slot_experts)


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


def _take_compute_note() -> tuple[Any, int, int, Any] | None:
    """Consume the pending compute note, so a note is never applied twice."""
    global _compute_note
    note, _compute_note = _compute_note, None
    return note


def _stage_compute(slot: int, layer_ord: int, note: tuple[Any, int, int, Any], width: int) -> None:
    """Stage one compute note's raw payload in the window for the flush to fold.

    The payload goes in exactly as MoonEP handed it over -- the home counts, the
    slot counts, and the slot-to-expert mapping -- so nothing is lost or guessed;
    the fold onto global expert ids happens in :func:`_fold_compute_row`, on the
    host, where it costs a few dozen ints of numpy instead of ~10 small kernels per
    layer in the half of the comparison that runs MoonEP.

    The mapping is copied here rather than held: it is ``plan.experts_to_copy``, the
    dispatch's own tensor, and the next layer's plan rewrites it -- the same reason
    the counts are snapshotted by the note.

    Staging the *first* row fixes the run's layout (``_raw_geo``). A later row that
    disagrees is rejected rather than written at a width of its own: every compute
    row of a dump has to be read as one expert space, and a geometry that moves
    means the two sides disagree about the model, not that a new kind of row exists.
    """
    global _buf_raw, _raw_geo
    counts, home_first, epn, slot_experts = note
    if home_first < 0 or home_first + epn > width:
        _warn_once(
            "compute-geometry",
            "[moe-par] MoonEP compute note: home block [%d,%d) does not fit a %d-expert row; "
            "skipping the compute row of that layer",
            home_first,
            home_first + epn,
            width,
        )
        return
    b = int(counts.numel()) - epn
    mapping = None
    if b < 0:
        _warn_once(
            "compute-width",
            "[moe-par] MoonEP compute note carries %d counts for %d home experts; "
            "skipping the compute row of that layer",
            int(counts.numel()),
            epn,
        )
        return
    if b:
        if slot_experts is None or int(slot_experts.numel()) != b:
            # One count per slot group and one expert id per slot group: the two
            # come from the same plan, so a mismatch means a scatter that would put
            # some rows nowhere.
            _warn_once(
                "compute-slots",
                "[moe-par] MoonEP compute note carries %d slot groups but %d slot experts; "
                "skipping the compute row of that layer",
                b,
                0 if slot_experts is None else int(slot_experts.numel()),
            )
            return
        mapping = slot_experts.reshape(-1)
    geo = (int(home_first), int(epn), b)
    if _raw_geo is not None and _raw_geo != geo:
        _warn_once(
            "compute-layout",
            "[moe-par] MoonEP compute rows moved from layout %s to %s; "
            "skipping the compute row of that layer",
            _raw_geo,
            geo,
        )
        return
    want = epn + 2 * b
    buf = _ensure_raw_buf(want, counts.device)
    if buf.shape[2] != want or buf.shape[1] <= layer_ord:
        _warn_once(
            f"compute-fit:{want}",
            "[moe-par] staged compute row is %d wide but the buffer holds %d; skipping this layer",
            want,
            buf.shape[2],
        )
        return
    _raw_geo = geo
    # Counts first, mapping behind them, in one write each: two device-to-device
    # copies per layer, against the ~10 the on-device fold needed.
    buf[slot, layer_ord, : epn + b] = counts.to(torch.int32)
    if b:
        buf[slot, layer_ord, epn + b :] = mapping
    _scopes.add("compute")
    _compute_ords[slot].add(layer_ord)


def _fold_compute_row(raw_row: np.ndarray, home_first: int, epn: int, b: int, width: int) -> np.ndarray:
    """One staged compute row in global expert coordinates, on the host.

    The home block is a contiguous global range, so it is a slice copy; the slot
    groups are scattered -- each holds whichever experts the plan pulled in -- so
    they need an accumulating add. Nothing here depends on the analysis guessing the
    plan: the mapping was recorded with the row.
    """
    row = np.zeros(width, dtype=np.int32)
    row[home_first : home_first + epn] = raw_row[:epn]
    if b:
        slots = raw_row[epn : epn + b]
        targets = raw_row[epn + b : epn + 2 * b].astype(np.intp)
        # A slot the plan left empty (-1) received nothing and must not land on an
        # expert; the index is clamped in range and the value is zeroed with it.
        # ``add.at`` is the unbuffered one: two slots may hold the same expert (a
        # replicated one), and their counts have to accumulate, not overwrite.
        keep = (targets >= 0) & (targets < width)
        np.add.at(row, np.where(keep, targets, 0), np.where(keep, slots, 0))
    return row


def _route_note_row(note: tuple[Any, int, int] | None) -> tuple[torch.Tensor | None, tuple[int, int] | None]:
    """The note's histogram as it stands, plus the pad block this rank must remove.

    The histogram is this rank's exact sending side, so the pads come off where
    they were counted -- in the histogram's own space, which is why the subtraction
    can never take away another expert's real traffic, unlike one against the
    receiving side, where a pad row shares a slot with real rows of the same expert.

    The subtraction itself waits for the flush (:func:`_flush`): it is one host add
    on a row that is already on the host, and doing it here would put a device copy
    and a device subtraction on the layer's critical path to no purpose. The
    histogram is *not* copied here either -- the window write a moment later is the
    snapshot, and an extra copy would land on the engine's own ``tpe``.

    Returns ``(histogram, pad)`` where ``pad`` is the ``(expert, pairs)`` block to
    remove, or None when the layer did not pad, so the caller can write the pads as
    their own row instead of losing them with the subtraction.
    """
    global _pad_notes_used, _pad_notes_misplaced
    if note is None:
        return None, None
    histogram, pairs, index = note
    counts = histogram.reshape(-1)
    _pad_notes_used += 1
    _scopes.add("global")
    pad = None
    if pairs:
        if 0 <= index < counts.numel():
            pad = (int(index), int(pairs))
        else:
            # The dispatcher names the pad expert in the histogram's own space, so
            # this means the two disagreed about the model's expert count.
            _pad_notes_misplaced += 1
    return counts, pad


def record(layer: Any, expert_tokens: torch.Tensor | None, group_list_type: int, method_name: str) -> None:
    """Record this layer's real routing for the step in flight.

    MoonEP announces its sending-side histogram through ``note_route_counts()``;
    that histogram wins over ``expert_tokens`` because it is the *sending* side of
    the dispatch, exact by construction. Every other method reports its own real
    counts here, index by local expert slot.

    The same dispatch may also have announced its receive-side counts through
    ``note_compute_load()``. Those become the layer's second row -- what this rank
    computed, as opposed to what was routed -- and the pad block the routing note
    carried becomes a third, so the dump separates real routing, padding and work
    instead of leaving the reader to guess which is which.
    """
    if not _ENABLED:
        return
    # Consumed even on the early returns below: a note must not outlive the layer
    # that produced it and be applied to an unrelated one.
    note = _take_route_note()
    compute_note = _take_compute_note()
    if expert_tokens is None and note is None:
        return
    try:
        if _in_profile_run():
            # Profile and dummy forwards at startup would otherwise pollute the dump.
            return
        counts, pad = _route_note_row(note)
        if counts is None:
            counts = _to_counts(expert_tokens, group_list_type)
            _scopes.add("local")
        _record(layer, counts, method_name, compute_note, pad)
    except Exception:
        if "record" not in _warned:
            _warned.add("record")
            logger.warning("[moe-par] dropped a sample: failed to record MoE load", exc_info=True)


def _ensure_window(buf: torch.Tensor | None, width: int, device, layers: int) -> torch.Tensor:
    """Allocate a device window on first use and widen it as new layers appear.

    A row of a different width is not accommodated: the caller rejects that layer
    (see :func:`_record`), because every row of a dump is read as one expert space.
    """
    if buf is None:
        return torch.zeros((_FLUSH_EVERY, max(layers, 1), width), dtype=torch.int32, device=device)
    if buf.shape[2] != width or buf.shape[1] >= layers:
        return buf
    grown = torch.zeros((_FLUSH_EVERY, layers * 2, buf.shape[2]), dtype=torch.int32, device=buf.device)
    grown[:, : buf.shape[1], :] = buf
    return grown


def _ensure_buf(counts: torch.Tensor) -> None:
    global _buf
    _buf = _ensure_window(_buf, int(counts.numel()), counts.device, len(_ord_of))


def _ensure_raw_buf(width: int, device) -> torch.Tensor:
    global _buf_raw
    _buf_raw = _ensure_window(_buf_raw, width, device, len(_ord_of))
    return _buf_raw


def _record(
    layer: Any,
    counts: torch.Tensor,
    method_name: str,
    compute_note: tuple[Any, int, int, Any] | None = None,
    pad: tuple[int, int] | None = None,
) -> None:
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

    # Every slot of the window gets an entry, whether or not this layer announced a
    # compute row: _flush walks the slots, and a step whose layers computed nothing
    # (a rank that was sent no work) must not be the one that indexes past the end.
    while len(_compute_ords) <= _slot:
        _compute_ords.append(set())
    _buf[_slot, layer_ord, :] = counts.to(torch.int32)
    if compute_note is not None:
        # Staged raw, folded at the flush. The routing row's width is the width the
        # fold targets -- a layer is what gives the row its expert count -- and a
        # compute note whose geometry does not fit it is dropped here rather than
        # written at a width of its own.
        _stage_compute(_slot, layer_ord, compute_note, width)
    if pad is not None:
        _pad_rows.append((_slot, layer_ord, int(pad[0]), int(pad[1])))


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
    if _pad_rows_total:
        # Written rows, not notes: a dump whose pads were subtracted but not written
        # (an old dump) has the pairs and no rows, and the two are cross-checked.
        pad += f" pad_rows_total={_pad_rows_total}"
    if _compute_rows_total:
        # Rows and the pairs they hold: the pair total is what reconciles with the
        # whole-run counters the dispatch itself prints (``rows_local``), and the
        # row count catches a duplicate or a lost line the pairs alone would not.
        pad += f" compute_rows_total={_compute_rows_total} compute_pairs_total={_compute_pairs_total}"
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
    """Copy the completed steps to host, fold the compute rows and append them."""
    global _slot, _step_base, _meta_written, _compute_rows_total, _compute_pairs_total, _pad_rows_total
    global _pad_pairs_subtracted, _route_pairs_total, _pad_expert_used
    if _slot == 0 or _buf is None:
        return
    width = int(_buf.shape[2])
    # The step's one device-to-host wait, and everything below reads these copies.
    # The staged compute rows come over on the same sync: issued after the routing
    # window, whose copy has already drained the stream, the second one transfers
    # and returns without waiting on anything -- which is why the two windows do not
    # have to share an allocation to cost one wait.
    host = _buf[:_slot].cpu().numpy()
    raw = _buf_raw[:_slot].cpu().numpy() if _buf_raw is not None else None
    # Pads come off here, where they were counted: the routing note named the expert
    # and the amount in the histogram's own space, so no other expert's traffic is
    # touched. Doing it in the layer instead would be a device copy, a device
    # subtraction and a device-to-device write back, all inside the critical path.
    for slot, layer_ord, expert, pairs in _pad_rows:
        if slot < _slot and 0 <= expert < width:
            host[slot, layer_ord, expert] -= pairs
            _pad_expert_used = expert
    host = _clamp_negatives(host)
    out = _open()
    if not _meta_written:
        _write_meta(out)
        _meta_written = True
    _rewind_tail(out)
    # ``scope=`` on the primary rows only when the dump carries more than one kind of
    # row: a single-scope dump keeps the format it has always had, and a reader that
    # predates this field sees exactly what it saw before.
    mark = _scope_marker()
    # One numpy reduction over the host window, not a second device read: the rows
    # written below are exactly this sum, split per layer.
    _route_pairs_total += int(host[:_slot].sum())
    for i in range(_slot):
        method = _methods[i] if i < len(_methods) else "?"
        step = _step_base + i
        for layer_ord in sorted(_valid):
            row = host[i, layer_ord].tolist()
            out.write(
                f"step={step} layer={layer_ord} method={method}{mark} "
                f"counts=[{','.join(str(v) for v in row)}]\n"
            )
        if raw is not None and _raw_geo is not None:
            home_first, epn, b = _raw_geo
            for layer_ord in sorted(_compute_ords[i]):
                row = _clamp_negatives(_fold_compute_row(raw[i, layer_ord], home_first, epn, b, width)).tolist()
                out.write(
                    f"step={step} layer={layer_ord} method={method} scope=compute "
                    f"counts=[{','.join(str(v) for v in row)}]\n"
                )
                _compute_rows_total += 1
                _compute_pairs_total += int(sum(row))
    # Pads last: one sparse row per padded layer, so the block the routing rows had
    # subtracted stays in the dump as its own ledger instead of only as a total.
    for slot, layer_ord, expert, pairs in _pad_rows:
        if slot >= _slot:
            continue
        step = _step_base + slot
        method = _methods[slot] if slot < len(_methods) else "?"
        row = [0] * _buf.shape[2]
        row[expert] = pairs
        out.write(
            f"step={step} layer={layer_ord} method={method} scope=pad "
            f"counts=[{','.join(str(v) for v in row)}]\n"
        )
        _pad_rows_total += 1
        _pad_pairs_subtracted += pairs
    out.flush()
    _step_base += _slot
    _slot = 0
    _methods.clear()
    _compute_ords.clear()
    _pad_rows.clear()
    _rewrite_tail(out)
    _print_ledger()


def _ledger_line() -> str:
    """The ``[moe-par-ledger]`` line: the dump's totals, in the dump's own units.

    Fields follow the dump's scopes, so they mean the same thing in every leg:
    ``rows`` is the primary row -- the real routing for a MoonEP dump (pads already
    subtracted), this rank's computed rows for a ``local``-scope one -- ``pads``
    the pad block taken out of it (0 outside MoonEP) and ``compute`` the work rows
    MoonEP writes alongside. ``rows + pads == compute`` closes the ledger; the
    residual is printed as ``dropped`` rather than hidden, because a dispatch that
    lost rows to its capacity envelope would otherwise look like a bookkeeping bug.
    """
    layers = _step_base * max(1, len(_valid))
    per = lambda v: f" ({v / layers:.0f}/layer)" if layers else ""
    parts = [
        f"[moe-par-ledger r{_rank_id()} tag={_TAG}] steps={_step_base}",
        f"rows={_route_pairs_total}{per(_route_pairs_total)}",
        f"pads={_pad_pairs_subtracted}{per(_pad_pairs_subtracted)}",
    ]
    if _compute_pairs_total:
        parts.append(f"compute={_compute_pairs_total}{per(_compute_pairs_total)}")
        dropped = _compute_pairs_total - (_route_pairs_total + _pad_pairs_subtracted)
        if dropped:
            parts.append(f"dropped={dropped}")
    return " ".join(parts)


def _print_ledger() -> None:
    """Log the running ledger every ``MOONEP_BALANCE_STATS_EVERY`` layers.

    A served run is stopped by a force-kill of the worker tree, so no exit hook
    runs and this line is the log's only ledger; it is printed from _flush, i.e.
    for the steps already in the dump, and its numbers come from the same totals
    the dump's ``# meta`` line carries -- the two cannot disagree.
    """
    if not (_BALANCE_ON and _BALANCE_EVERY and _step_base):
        return
    every = max(1, _BALANCE_EVERY // max(1, len(_valid)))
    if _step_base % every:
        return
    print(_ledger_line(), flush=True)


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


def _primary_scope() -> str:
    """What the *primary* rows -- the ones each method reported -- are index by."""
    kinds = _scopes & {"global", "local"}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed" if kinds else "local"


def _scope_name() -> str:
    """What this dump's rows are index by, as a whole.

    ``global``/``local`` alone when a dump carries one kind of row, ``mixed`` for a
    dump that used both (see the note on ``_scopes``), and ``global+compute`` for
    MoonEP's two-view dump: both of its kinds of row span the model's experts and
    each has one meaning, so the pair is a nameable combination, not a mix.
    """
    kinds = _scopes & {"global", "local"}
    if not kinds:
        return "compute" if "compute" in _scopes else "local"
    if len(kinds) == 1:
        name = next(iter(kinds))
        return f"{name}+compute" if "compute" in _scopes else name
    return "mixed"


def _scope_marker() -> str:
    """The ``scope=`` field the primary rows carry, or nothing.

    Only written when the dump holds more than one kind of row -- otherwise the
    header already says what every row is, and the dump keeps its old shape. A mixed
    dump gets no marker at all: its primary rows are not one scope, and ``mixed`` as
    a row value would read as if they were.
    """
    if len(_scopes) <= 1:
        return ""
    primary = _primary_scope()
    return f" scope={primary}" if primary in ("global", "local") else ""


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
