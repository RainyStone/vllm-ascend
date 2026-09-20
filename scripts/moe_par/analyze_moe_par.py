#!/usr/bin/env python
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
"""Analyze the per-rank MoE load dumps written by ``moe_par_stats`` and emit a report.

The recorder (``vllm_ascend/ops/fused_moe/moe_par_stats.py``, enabled with
``VLLM_ASCEND_MOE_PAR_STATS=1``) writes one text file per rank holding the
per-local-expert route counts of every (step, layer). This script stitches those
files back together across ranks -- the one thing the recorder deliberately does
not do, because the peak-to-average ratio is a cross-rank quantity and the hot
path has to stay free of collectives.

Three ratios are reported, and they differ only in how much is summed before the
peak-over-mean:

``PAR (per-layer)``
    EPLB's own definition, the one ``EplbWorker._compute_imbalance`` computes:
    per layer ``max_r(load[r]) / mean_r(load[r])``, then averaged over the layers
    of a step. One value per step. The right unit when layers are scheduled
    independently.

``PAR (per-step, whole step)``
    ``max_r(sum_l load[r, l]) / mean_r(sum_l load[r, l])`` -- each rank's *total*
    work for one step, which is what decides who waits at the barrier. One value
    per step; the layer dimension is summed, time is not.

``PAR (whole run)``
    The same ratio over each rank's load summed across every step and layer. One
    value for the entire run: a figure of merit for the placement, not a
    predictor of any single wait.

All three live in ``[1, num_ranks]``; 1 is perfectly balanced, and they shrink as
the summing scope widens because layer-to-layer and step-to-step tilts cancel.
The gate routes the same tokens to the same experts whichever all-to-all carries
them, so the *routing* view of a run is a property of the batch, not of the
method. A ratio that is meant to describe how busy the ranks were, however, has
to be read on the *compute* view: MoonEP's slot mode relocates rows, and that
relocation is exactly what changes who waits at the barrier.

The load itself is defined once, and the report says so before any chart: an
*expert* load counts the (token, expert) pairs that **logical** expert received
(so a replicated expert stays one expert), and a *rank* load counts the pairs
that rank is answerable for -- which pairs those are is what the two views
disagree about (the holder of the expert for 路由归属, the rank that ran the row
for 实际计算). Padding is counted in the compute view and carved out of the
routing view; both are reported side by side rather than netted off, so the
reader can see how much of a difference is work and how much is padding. How a
dump delivers a row depends on its ``counts_scope``:

``local``
    The row is index by this rank's own expert slots (every method but MoonEP),
    so a row is one rank's own work and the rows are summed after being mapped
    back onto the global expert grid.

``global``
    The row already spans the model's whole expert space, index by logical id
    (MoonEP: the dispatcher's sending-side histogram). The ranks of one (step,
    layer) are summed into the step's real routing, and only then is each
    expert's load credited to the rank that holds it. This is the scope that
    survives MoonEP's capacity padding -- the pads are subtracted where they were
    counted, before the row is written -- and a replicated expert, which a single
    rank can no longer see, is still counted once.

MoonEP runs carry **two views of the same dispatch**, and this script reports both:

* 路由归属 (routing / owner attribution) -- where the gate sent the tokens. Built
  from the ``global`` rows above, credited to the rank that *holds* the expert.
  Exact, and an upper bound on the real imbalance: relocation can only flatten it.
* 实际计算 (actual work) -- which ranks ran the rows. Built from the extra
  ``scope=compute`` rows MoonEP writes alongside, credited to the rank that
  *wrote* them, so a row the plan moved to another rank counts against the rank
  that computed it. This is the view a peak-to-average ratio over ranks is trying
  to measure, and the one the report leads with.

Both views separate the same three ledgers, so a difference can be attributed
instead of guessed (see ``_finalize_ledger``): 真实路由 (the gate's pairs), pad
(MoonEP's capacity padding), and 丢弃 (rows the capacity envelope dropped). The
``scope=pad`` rows carry the pads -- they are subtracted from the routing rows to
keep those exact, and written separately so the report can show padding as its
own quantity rather than folding it into either view. The compute rows keep their
pads, because on the receiving side a pad row is indistinguishable from a real
row of the same expert and may be computed by any rank that holds that expert.

Usage::

    python scripts/moe_par/analyze_moe_par.py \\
        --run mc2=/tmp/par_mc2 --run moonep=/tmp/par_moonep \\
        -o /tmp/moe_par_report.html

Every dimension comes from the dump's ``# meta`` header, so the report adapts to
whatever model produced it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_META_RE = re.compile(r"^#\s*meta\s+(.*)$")
_LAYER_RE = re.compile(r"^#\s*layer\s+(.*)$")
# ``scope=`` is optional: a dump that writes one kind of row does not mark them,
# and one that writes several marks every non-primary row. A row without the field
# is the dump's primary kind (see RankData.primary_scope).
_ROW_RE = re.compile(r"^step=(\d+)\s+layer=(\d+)\s+method=(\S+)\s+(?:scope=(\S+)\s+)?counts=\[(.*)\]\s*$")

# The kinds of row a dump can carry, and what each one means. ``compute`` and ``pad``
# only exist in a MoonEP dump with the two-view recorder behind it.
_SCOPE_COMPUTE = "compute"
_SCOPE_PAD = "pad"
_SCOPE_ROUTE = "route"

# The comm-method class names the recorder sees, shown as something readable.
_METHOD_LABELS = {
    "AllGatherCommImpl": "all-gather",
    "MC2CommImpl": "MC2",
    "All2AllCommImpl": "all-to-all",
    "FusedMC2CommImpl": "fused-MC2",
    "MoonEPCommImpl": "MoonEP (SHMEM)",
}

# Latency measures the bench can report per request, and the labels the report uses.
_PERF_NAMES = {
    "ttft_ms": "TTFT",
    "tpot_ms": "TPOT",
    "e2e_ms": "端到端时延",
    "decode_ms": "decode 段",
    "queue_ms": "排队",
    "prefill_ms": "prefill 段",
}

_SRC_EXPERT_MAP = "expert_map"
_SRC_IDENTITY = "ep_rank*local_experts + slot"
_SRC_WHOLE = "every expert is local"

# The two accounting views MoonEP makes measurable, and what each one says.
_VIEW_LABELS = {_SCOPE_ROUTE: "路由归属", _SCOPE_COMPUTE: "实际计算"}
_VIEW_HINTS = {
    _SCOPE_ROUTE: "按专家属主记（谁持有这个专家，负载就算谁的）",
    _SCOPE_COMPUTE: "按实际计算记（谁算了这些行，负载就算谁的）",
}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _parse_kv(text: str) -> dict[str, str]:
    """Parse ``k=v k=v`` allowing values that contain spaces.

    ``model=/path/with space`` would otherwise split into three tokens; a token
    without ``=`` is appended to the value it follows.
    """
    out: dict[str, str] = {}
    last: str | None = None
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
            last = key
        elif last is not None:
            out[last] += " " + token
    return out


def _parse_int_list(text: str) -> list[int] | None:
    """``[1,2,3]`` -> [1, 2, 3]; ``None``, ``[]`` and junk come back as None.

    The recorder renders the list with brackets and an absent map as the literal
    ``None``; neither would survive a bare ``int()``.
    """
    text = text.strip().lstrip("[").rstrip("]")
    if not text or text == "None":
        return None
    try:
        return [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        return None


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class RankData:
    """One rank's dump: the header, the layer maps, and the (step, layer) counts."""

    __slots__ = ("path", "meta", "layers", "steps", "layer_ids", "methods", "counts", "row_scopes")

    def __init__(self, path: str):
        self.path = path
        self.meta: dict[str, str] = {}
        self.layers: dict[int, dict] = {}
        self.steps: list[int] = []
        self.layer_ids: list[int] = []
        self.methods: list[str] = []
        self.counts: list[list[int]] = []
        # Per row, the ``scope=`` field if it had one, else "" -- which means "the
        # dump's primary kind". Parallel to ``counts``.
        self.row_scopes: list[str] = []

    @property
    def rank(self) -> int:
        return _as_int(self.meta.get("rank"), -1)

    @property
    def ep_rank(self) -> int:
        """The expert-parallel rank, which is what the expert map is indexed by.

        Distinct from ``rank``: with tensor parallelism alongside EP the two
        differ, and the contiguous fallback layout is expressed in EP ranks.
        """
        return _as_int(self.meta.get("ep_rank"), self.rank)

    @property
    def row_width(self) -> int:
        """Width of this rank's dump rows, taken from the data rather than the header.

        What it means depends on ``scope``: local slots for a ``local`` dump, the
        model's whole expert space for a ``global`` one.
        """
        return len(self.counts[0]) if self.counts else -1

    @property
    def scope(self) -> str:
        """What this dump's rows are index by (``local`` / ``global`` / ``mixed``).

        MoonEP's two-view dumps add ``+compute`` (e.g. ``global+compute``): the
        primary rows still mean what they always meant, and a second row per (step,
        layer) carries the work this rank ran. Absent means ``local``: that is what
        every dump written before MoonEP started announcing its sending side carries.
        """
        return self.meta.get("counts_scope", "local")

    @property
    def primary_scope(self) -> str:
        """What this dump's *unmarked* rows are index by."""
        name = self.scope.split("+", 1)[0]
        return name if name in ("global", "local") else "local"

    def row_kind(self, index: int) -> str:
        """The kind of row ``index``: ``global``/``local``, ``compute`` or ``pad``."""
        return self.row_scopes[index] or self.primary_scope


def parse_dump(path: str) -> RankData:
    data = RankData(path)
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                if (match := _META_RE.match(line)) is not None:
                    data.meta.update(_parse_kv(match.group(1)))
                elif (match := _LAYER_RE.match(line)) is not None:
                    fields = _parse_kv(match.group(1))
                    data.layers[_as_int(fields.get("layer_ord"), -1)] = {
                        "instance_id": fields.get("instance_id", "?"),
                        "expert_map": _parse_int_list(fields.get("expert_map", "")),
                        "log2phy": _parse_int_list(fields.get("log2phy", "")),
                    }
                continue
            match = _ROW_RE.match(line)
            if match is None:
                continue
            inner = match.group(5).strip()
            counts = [int(v) for v in inner.split(",")] if inner else []
            if not counts:
                continue
            data.steps.append(int(match.group(1)))
            data.layer_ids.append(int(match.group(2)))
            data.methods.append(match.group(3))
            data.row_scopes.append(match.group(4) or "")
            data.counts.append(counts)
    return data


def find_dumps(directory: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, "moe_par_*rank*.log")))
    return paths or sorted(glob.glob(os.path.join(directory, "*.log")))


# --------------------------------------------------------------------------- #
# local -> global expert mapping
# --------------------------------------------------------------------------- #
def _loc2glob(
    layer: dict | None, local_experts: int, global_experts: int, ep_rank: int
) -> tuple[np.ndarray, str]:
    """Map this rank's local expert slots to global expert ids.

    ``_expert_map`` is vLLM's ``expert_map``: indexed by *global* expert id,
    holding the local slot that owns it, ``-1`` for experts living elsewhere.
    Inverting it is the only reliable reading -- it is what
    ``determine_expert_map`` returns and what gets handed to the routing op. The
    inverse is accepted only when it covers every local slot exactly once; under
    EPLB's redundant placement the map is indexed by logical ids and does not, so
    the contiguous fallback is used and flagged rather than silently attributing
    load to the wrong expert.

    ``log2phy`` is deliberately unused: its values are global *physical* ids
    (``slot + rank * experts_per_rank``), a different space entirely, and it is
    ``None`` on every non-EPLB path anyway.
    """
    if local_experts >= global_experts and global_experts > 0:
        # Expert parallelism is off (or one rank holds the whole table), so the local
        # slot *is* the global id. There is nothing to infer and nothing to warn about.
        return np.arange(local_experts, dtype=np.int64), _SRC_WHOLE
    identity = np.arange(local_experts, dtype=np.int64) + ep_rank * local_experts
    expert_map = (layer or {}).get("expert_map")
    if expert_map and global_experts > 0:
        inverse = np.full(local_experts, -1, dtype=np.int64)
        seen = np.zeros(local_experts, dtype=bool)
        ok = True
        for global_id, local_slot in enumerate(expert_map):
            if local_slot < 0:
                continue
            if local_slot >= local_experts or seen[local_slot]:
                ok = False
                break
            inverse[local_slot] = global_id
            seen[local_slot] = True
        if ok and (seen.all() or not seen.any()):
            # ``not seen.any()`` is a rank that owns nothing; an all -1 map is honest.
            return inverse, _SRC_EXPERT_MAP
    return identity, _SRC_IDENTITY


def _owned_mask(
    layer: dict | None, local_experts: int, global_experts: int, ep_rank: int
) -> tuple[np.ndarray, str]:
    """Which global experts this rank holds, read off the same ``expert_map``.

    The map is indexed by global (logical) expert id and holds the local slot that
    serves it, ``-1`` for experts living elsewhere, so "this rank holds it" is
    simply "the entry is not -1". That reading stays correct under redundancy --
    a replicated expert shows up as held by two ranks, which is exactly the fact
    the rank-level split has to respect -- while the inverse used for local-scope
    dumps cannot be built at all (two global ids would claim one slot).

    The contiguous fallback (rank ``ep_rank`` holds slot ``ep_rank`` of the
    partition) is the only sane guess when the map is absent, and it is flagged.
    """
    expert_map = (layer or {}).get("expert_map")
    if expert_map and len(expert_map) == global_experts:
        return np.array([slot >= 0 for slot in expert_map], dtype=bool), _SRC_EXPERT_MAP
    if local_experts >= global_experts and global_experts > 0:
        return np.ones(global_experts, dtype=bool), _SRC_WHOLE
    mask = np.zeros(global_experts, dtype=bool)
    if local_experts > 0:
        mask[ep_rank * local_experts : (ep_rank + 1) * local_experts] = True
    return mask, _SRC_IDENTITY


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _nanmax(array: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax without the all-NaN RuntimeWarning; all-NaN stays NaN."""
    out = np.max(np.where(np.isnan(array), -np.inf, array), axis=axis)
    return np.where(np.isneginf(out), np.nan, out)


def _nanmean(array: np.ndarray, axis: int) -> np.ndarray:
    valid = ~np.isnan(array)
    count = np.sum(valid, axis=axis)
    total = np.sum(np.where(valid, array, 0.0), axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _nansum(array: np.ndarray, axis: int) -> np.ndarray:
    """Sum treating NaN as absent, so a missing cell never poisons a run total."""
    return np.sum(np.where(np.isnan(array), 0.0, array), axis=axis)


def _percentile(ordered: list[float], p: float) -> float:
    """Linear-interpolated percentile of an already-sorted list; ``p`` in [0, 1]."""
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _quantiles(values: list[float]) -> dict:
    """One latency measure condensed to the numbers the report quotes."""
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p90": round(_percentile(ordered, 0.90), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def _par_pack(
    load: np.ndarray, present: np.ndarray, n_steps: int, n_layers: int, n_ranks: int, topk: int
) -> dict:
    """The three PAR scopes and the rank-load tables of one ``[R, T, L]`` load view.

    One function for both views on purpose: 路由归属 and 实际计算 differ only in the
    load they are handed, and running the two through the same arithmetic is what
    makes a difference between the reported series a difference in the data rather
    than a difference in how it was reduced.
    """
    # EPLB's per-layer balance: max over ranks of one layer's load, over the
    # mean of that layer. Averaged over layers, exactly as _compute_imbalance does.
    per_layer = np.full((n_steps, n_layers), np.nan)
    for si in range(n_steps):
        column = load[:, si, :]  # [R, L]
        count = np.sum(~np.isnan(column), axis=0)
        mean = _nanmean(column, axis=0)
        peak = _nanmax(column, axis=0)
        good = (count > 1) & (mean > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_layer[si, good] = peak[good] / mean[good]

    # Each rank's total for the step, then the same ratio over that total.
    step_sum = np.where(present, load, 0.0).sum(axis=2)  # [R, T]
    step_sum = np.where(present.any(axis=2), step_sum, np.nan)
    peak = _nanmax(step_sum, axis=0)
    mean = _nanmean(step_sum, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        par_agg = np.where(mean > 0, peak / np.maximum(mean, 1e-12), np.nan)

    # Third scope: the whole run in one number. Each rank's load summed over
    # every step and layer, then the same peak-over-mean. This answers "who did
    # more work overall", which is *not* what decides a barrier wait -- that is
    # per step -- but it is the figure of merit for a placement or a plan.
    rank_total = _nansum(step_sum, axis=1)
    whole_peak = _nanmax(rank_total, axis=0)
    whole_mean = _nanmean(rank_total, axis=0)
    step_pairs = np.where(present.any(axis=2), step_sum, 0.0).sum(axis=0)
    # A token is routed at *every* MoE layer, so it contributes topk pairs in each of
    # them: the pair total is tokens * topk * n_layers. Dividing by topk alone
    # overstates the batch by exactly the layer count (48x on Qwen3-30B-A3B).
    per_token = topk * max(n_layers, 1)
    pack = {
        "load": load,
        "par_layer": per_layer,
        "par_by_step": _nanmean(per_layer, axis=1),
        "par_agg": par_agg,
        "par_whole": float(whole_peak / whole_mean) if whole_mean and whole_mean > 0 else float("nan"),
        "rank_step_load": step_sum,
        "step_pairs": step_pairs,
        "tokens_per_step": step_pairs / per_token if topk > 0 else np.full_like(step_pairs, np.nan),
        "per_rank_mean": _nanmean(step_sum, axis=1),
        "per_rank_max": _nanmax(step_sum, axis=1),
        "rank_total": rank_total,
    }
    if n_ranks < 2:
        # With one rank the peak over the mean is 1/1, which would render as
        # "perfectly balanced" when in fact there is no spread to measure.
        pack["par_layer"][:] = np.nan
        pack["par_by_step"][:] = np.nan
        pack["par_agg"][:] = np.nan
        pack["par_whole"] = float("nan")
    return pack


class Run:
    """A named directory of per-rank dumps, aggregated into the report's numbers."""

    def __init__(self, name: str, directory: str):
        self.name = name
        self.dir = directory
        self.paths = find_dumps(directory)
        if not self.paths:
            raise SystemExit(f"[moe-par] no dump files found under {directory!r}")
        self.warnings: list[str] = []
        self._build()

    # -- construction ------------------------------------------------------- #
    def _build(self) -> None:
        dumps = sorted((parse_dump(p) for p in self.paths), key=lambda d: d.rank)
        self.dumps = dumps
        self.rank_ids = [d.rank for d in dumps]
        self.steps = sorted({s for d in dumps for s in d.steps})
        self.layer_ids = sorted({layer for d in dumps for layer in d.layer_ids})
        self.step_ix = {s: i for i, s in enumerate(self.steps)}
        self.layer_ix = {layer: i for i, layer in enumerate(self.layer_ids)}
        self.rank_ix = {r: i for i, r in enumerate(self.rank_ids)}

        head = dumps[0].meta
        self.model = head.get("model", "unknown")
        self.tag = head.get("tag", self.name)
        self.ep_size = _as_int(head.get("ep_size"), len(dumps))
        self.world_size = _as_int(head.get("world_size"), len(dumps))
        self.topk = _as_int(head.get("topk"), -1)
        self.redundant = _as_int(head.get("redundant_experts"), 0)
        declared_global = _as_int(head.get("global_experts"), -1)
        declared_local = _as_int(head.get("local_experts"), -1)

        # What the rows are index by. A run that used both cannot be read as either;
        # it is labelled and read as the legacy local layout, because that at least
        # produces the same numbers it always did -- the warning says the meanings
        # are now uncertain rather than letting a global row pass as a local one.
        scopes = {d.scope for d in dumps}
        self.scope = next(iter(scopes)) if len(scopes) == 1 else "mixed"
        if len(scopes) > 1:
            self.warnings.append(
                "同一个目录里的 dump 写明了不同的计数口径（" + "、".join(sorted(scopes)) + "）；"
                "行宽的含义不确定，按「本地专家槽位」解读——这份运行的数字不可信，请分开分析"
            )

        # MoonEP's two-view dump: the primary rows plus a compute row per (step,
        # layer). It is not a "mixed" dump -- both views span the model's experts and
        # each has one meaning -- so only the primary part decides how a row is read.
        header_view = "compute" in self.scope
        row_view = any(_SCOPE_COMPUTE in d.row_scopes for d in dumps)
        self.has_compute = header_view or row_view
        if header_view != row_view:
            self.warnings.append(
                "计数口径的头与行不一致：头部写的是 counts_scope="
                f"{self.scope}，而 dump 里{'有' if row_view else '没有'} scope=compute 的行；"
                "以行上写的为准（多出来的那一套按「实际计算」解读）"
            )
        self.primary = self.scope.split("+", 1)[0] if "+" in self.scope else self.scope
        # Every "is this a global dump" decision reads this, so the two-view dumps do
        # not silently take the local path.
        self.global_scope = self.primary == "global"

        widths = {d.row_width for d in dumps if d.row_width > 0}
        if len(widths) > 1:
            self.warnings.append(
                f"各 rank 的 dump 行宽不一致 {sorted(widths)}；按最宽的一行定布局，"
                "较窄的 rank 会按补零来读"
            )
        self.row_width = max(widths) if widths else 0
        for dump in dumps[1:]:
            if dump.meta.get("model") != head.get("model"):
                self.warnings.append(
                    f"rank {dump.rank} 跑的是模型 {dump.meta.get('model')!r}，而 rank "
                    f"{dumps[0].rank} 跑的是 {head.get('model')!r}；这些 dump 来自不同的作业"
                )
                break

        self.n_ranks = len(self.rank_ids)
        self.n_steps = len(self.steps)
        self.n_layers = len(self.layer_ids)

        # The map's own length is authoritative for the global count; the header
        # is the fallback, and ep_size * local covers a header that lost it.
        map_len = max(
            (len(layer["expert_map"]) for d in dumps for layer in d.layers.values() if layer.get("expert_map")),
            default=0,
        )
        if self.global_scope:
            # A global row is counted over the model's expert space itself, so the
            # width is the expert count and the header only cross-checks it.
            self.global_experts = self.row_width
            self.local_experts = declared_local if declared_local > 0 else -1
            if declared_global > 0 and declared_global != self.global_experts:
                self.warnings.append(
                    f"头部写的是 global_experts={declared_global}，而全局口径的行宽是 "
                    f"{self.global_experts}；以行宽为准——它是打点真正数过的那一维"
                )
        else:
            self.local_experts = self.row_width
            self.global_experts = map_len or declared_global
            if self.global_experts <= 0:
                self.global_experts = max(self.ep_size, 1) * max(self.local_experts, 1)
                self.warnings.append(
                    "头部没有全局专家数、也没有 expert_map；按 "
                    f"ep_size * local_experts = {self.global_experts} 假定"
                )
        if self.local_experts <= 0 and self.ep_size > 0 and self.global_experts > 0:
            self.local_experts = max(1, self.global_experts // self.ep_size)

        # load[r, t, l] = token-expert pairs rank r handled in layer l at step t
        self.load = np.full((self.n_ranks, self.n_steps, self.n_layers), np.nan, dtype=np.float64)
        # Share of the single busiest local expert in each (rank, step, layer) row.
        # MoonEP's capacity padding lands in one slot, so this is what tells a
        # padded dump apart from a real hotspot or a broken join.
        self.top_share = np.full_like(self.load, np.nan)
        self.rank_expert = np.zeros((self.n_ranks, self.global_experts), dtype=np.float64)
        self.rank_expert_seen = np.zeros(self.n_ranks, dtype=bool)
        self.layer_expert = np.zeros((self.n_layers, self.global_experts), dtype=np.float64)
        self.step_expert = np.zeros((self.n_steps, self.global_experts), dtype=np.float64)
        self.method_of_step = [""] * self.n_steps
        self.map_rows = 0
        self.identity_rows = 0
        self.whole_rows = 0
        self.truncated_rows = 0
        self.methods_seen: set[str] = set()
        # Pairs each rank counted in its own rows, before anything is attributed to
        # anyone: the denominator of the pad-accounting table.
        self.sent_by_rank = np.zeros(self.n_ranks, dtype=np.float64)

        # -- the actual-work view, from MoonEP's ``scope=compute`` rows ---------- #
        # The same dispatch read the other way: the rows each rank ran, attributed
        # to the rank that wrote them rather than to the expert's holder. Only a
        # two-view dump has these, and they are the view a rank-level ratio should
        # be read on -- relocation is precisely what moves work between ranks.
        self.compute_seen = np.zeros(self.n_ranks, dtype=bool)
        self.load_compute = np.full((self.n_ranks, self.n_steps, self.n_layers), np.nan, dtype=np.float64)
        self.compute_rank_expert = np.zeros((self.n_ranks, self.global_experts), dtype=np.float64)
        self.compute_layer_expert = np.zeros((self.n_layers, self.global_experts), dtype=np.float64)
        self.compute_step_expert = np.zeros((self.n_steps, self.global_experts), dtype=np.float64)

        # -- the padding ledger, from the ``scope=pad`` rows -------------------- #
        # What MoonEP padded the dispatch with. The routing rows had it subtracted
        # (so they are real routing) and the compute rows still carry it (on the
        # receiving side it is not separable), so it is kept here as its own
        # quantity: 总量 = 真实 + pad is then answerable at every level, which is
        # what turns "these two ranks differ" into "by how much real work".
        #
        # Held in the routing's own [T, L, E] shape because that is what the owner
        # attribution consumes; ``pad_pairs_by_rank`` is the injector's own view,
        # which is a different question (who wrote the pads, not who owns them).
        self.pad_routing = np.zeros((self.n_steps, self.n_layers, self.global_experts), dtype=np.int32)
        self.pad_pairs_by_rank = np.zeros(self.n_ranks, dtype=np.float64)
        self.pad_rows = 0
        # Filled from ``pad_routing`` after the fold for every scope; only a global
        # run can go on to attribute them to a rank (see _attribute_global).
        self.pad_layer_expert = np.zeros((self.n_layers, self.global_experts), dtype=np.float64)
        self.pad_step_expert = np.zeros((self.n_steps, self.global_experts), dtype=np.float64)
        self.load_pad = np.zeros((self.n_ranks, self.n_steps, self.n_layers), dtype=np.float64)
        self.rank_pad = np.zeros(self.n_ranks, dtype=np.float64)

        self.unowned_slots = 0
        self.partial_steps = 0
        # Whether the dump carries the exit-summary line the recorder writes at the
        # end of the file. Without it the counters that live only there -- incomplete
        # steps and subtracted pads -- are unknown rather than zero.
        self.tail_missing = False
        if self.global_scope:
            self._prepare_ownership()
            # The summed routing of every (step, layer). A global row is one rank's
            # sending side, so it is the *sum* over ranks that is the run's routing;
            # holding it costs less than the text it was parsed from and lets the
            # attribution run once, on the finished vector.
            self.routing = np.zeros((self.n_steps, self.n_layers, self.global_experts), dtype=np.int32)
        self._build_holders()

        for dump in dumps:
            rank_row = self.rank_ix[dump.rank]
            for index, (step, layer, method, counts) in enumerate(
                zip(dump.steps, dump.layer_ids, dump.methods, dump.counts)
            ):
                si, li = self.step_ix[step], self.layer_ix[layer]
                kind = dump.row_kind(index)
                if kind in (_SCOPE_COMPUTE, _SCOPE_PAD):
                    # Both of these span the model's whole expert space, so they are
                    # held to the same width as a global row is.
                    if len(counts) != self.global_experts:
                        self.truncated_rows += 1
                        continue
                    if kind == _SCOPE_COMPUTE:
                        self.compute_seen[rank_row] = True
                        self._fold_compute_row(rank_row, si, li, counts)
                    else:
                        self._fold_pad_row(rank_row, si, li, counts)
                    if not self.method_of_step[si]:
                        self.method_of_step[si] = method
                    self.methods_seen.add(method)
                    continue
                if self.global_scope and len(counts) != self.global_experts:
                    # A global row is the model's whole expert space; any other width
                    # means two sides disagreed about that space, and reading it as a
                    # short row would silently rename experts.
                    self.truncated_rows += 1
                    continue
                if not self.global_scope and len(counts) > self.local_experts:
                    # Never quietly drop load: a longer row means the layout is wider than
                    # the widest declared dump, and truncating it would understate a rank.
                    self.truncated_rows += 1
                    continue
                self.rank_expert_seen[rank_row] = True
                if not self.method_of_step[si]:
                    self.method_of_step[si] = method
                self.methods_seen.add(method)
                if self.global_scope:
                    self._fold_global_row(rank_row, si, li, counts)
                    continue
                width = min(len(counts), self.local_experts)
                self.load[rank_row, si, li] = sum(counts[:width])
                self.sent_by_rank[rank_row] += self.load[rank_row, si, li]
                if width and self.load[rank_row, si, li] > 0:
                    self.top_share[rank_row, si, li] = max(counts[:width]) / self.load[rank_row, si, li]
                self._fold_row(dump, rank_row, si, li, counts)

        # The padding ledger at expert granularity, which needs no attribution at
        # all: a pad row names its expert by global id, whatever scope the dump uses.
        self.pad_layer_expert = self.pad_routing.astype(np.float64).sum(axis=0)  # [L, E]
        self.pad_step_expert = self.pad_routing.astype(np.float64).sum(axis=1)  # [T, E]

        if self.has_compute and not self.compute_seen.any():
            # The header promised a second view and none was written. Reporting the
            # view as present and empty would read as "no rank computed anything".
            self.has_compute = False
            self.warnings.append(
                "头部写的是两套口径（counts_scope=" + self.scope + "），但 dump 里没有一行 scope=compute；"
                "实际计算口径按缺失处理，只出路由归属口径"
            )
        if self.has_compute and not self.compute_seen.all():
            # The work view is complete only if every rank wrote it; a rank that did
            # not is missing work, not idle, and zero would say the wrong thing.
            idle = [self.rank_ids[i] for i in np.flatnonzero(~self.compute_seen)]
            self.load_compute[~self.compute_seen, :, :] = np.nan
            self.compute_rank_expert[~self.compute_seen] = 0.0
            self.warnings.append(
                "rank " + "、".join(str(r) for r in idle) + " 没有任何 scope=compute 行；"
                "实际计算口径里这几个 rank 显示为缺失（缺失不等于没干活）"
            )

        if self.global_scope:
            self._attribute_global()
            if not self.rank_expert_seen.all():
                # Every rank's row is part of the sum that makes a step's routing
                # whole, so a rank that never wrote one leaves a hole no other rank
                # can fill.
                idle = [self.rank_ids[i] for i in np.flatnonzero(~self.rank_expert_seen)]
                self.load[~self.rank_expert_seen, :, :] = np.nan
                self.rank_expert[~self.rank_expert_seen] = 0.0
                self.warnings.append(
                    "rank " + "、".join(str(r) for r in idle) + " 没有任何 dump 行；全局口径必须把各 rank 的"
                    "行相加才是完整路由，缺了行的部分既不是 0 也推不回来——这几个 rank 在图表里显示为缺失"
                )

    def _prepare_ownership(self) -> None:
        """Work out which rank holds each logical expert, for a global-scope run.

        Global rows do not say where their experts live: the routing space is the
        logical one, and a rank's own row counts the tokens *it* sent, not the work
        *it* got. Ownership is therefore read from ``expert_map``, and the load of a
        replicated expert is credited to the lowest-numbered rank holding it rather
        than split by a rule nobody recorded. That keeps ``sum_r load == pairs``
        exact and makes the ratio an *upper bound*: real sharing can only flatten it.
        """
        self.owner = np.full((self.n_layers, self.global_experts), -1, dtype=np.int64)
        self.owner_masks: list[np.ndarray] = []  # one [L, E] mask per rank
        held = [np.zeros((self.n_layers, self.global_experts), dtype=bool) for _ in range(self.n_ranks)]
        for dump in self.dumps:
            rank_row = self.rank_ix[dump.rank]
            for li, layer_id in enumerate(self.layer_ids):
                mask, source = _owned_mask(
                    dump.layers.get(layer_id), self.local_experts, self.global_experts, dump.ep_rank
                )
                if source == _SRC_EXPERT_MAP:
                    self.map_rows += 1
                elif source == _SRC_WHOLE:
                    self.whole_rows += 1
                else:
                    self.identity_rows += 1
                held[rank_row][li] |= mask
                claim = self.owner[li]
                claim[mask & (claim < 0)] = rank_row
        if self.identity_rows:
            self.warnings.append(
                f"有 {self.identity_rows} 个 (rank, 层) 没有可用的 expert_map，"
                f"按连续布局（{_SRC_IDENTITY}）推断归属；当每个 rank 持有等长且有序的一段时成立"
            )
        self.unowned_slots = int(np.sum(self.owner < 0))
        if self.unowned_slots:
            self.warnings.append(
                f"有 {self.unowned_slots} 个 (层, 专家) 没有任何 rank 声明持有；"
                "落在这些专家上的 token-专家对不计入任何 rank 的负载，各 rank 之和会小于路由总量"
            )
        self.owner_masks = held

    def _build_holders(self) -> None:
        """Which ranks hold each logical expert, for the report's expert table.

        Read from the same ``expert_map`` in both scopes, so the table can show a
        replicated expert as held by several ranks even though the local-scope
        attribution cannot split its load between them.
        """
        self.expert_holders: list[list[int]] = [[] for _ in range(self.global_experts)]
        for dump in self.dumps:
            rank_row = self.rank_ix[dump.rank]
            for layer_id in self.layer_ids:
                mask, _ = _owned_mask(
                    dump.layers.get(layer_id), self.local_experts, self.global_experts, dump.ep_rank
                )
                for expert in np.flatnonzero(mask):
                    holders = self.expert_holders[int(expert)]
                    if rank_row not in holders:
                        holders.append(rank_row)
        self.duplicated_experts = sum(1 for holders in self.expert_holders if len(holders) > 1)

    def _fold_global_row(self, rank_row: int, si: int, li: int, counts: list[int]) -> None:
        """Add one global-scope row to the step's routing.

        Nothing is attributed here. A global row is one rank's sending side, and
        what decides a rank's load is the vector those rows sum to -- not which rank
        contributed which part of it -- so the attribution happens once, on the
        finished sum, in ``_attribute_global``.
        """
        self.routing[si, li] += np.asarray(counts, dtype=np.int32)
        self.sent_by_rank[rank_row] += float(sum(counts))

    def _fold_compute_row(self, rank_row: int, si: int, li: int, counts: list[int]) -> None:
        """Fold one ``scope=compute`` row: the work this rank actually ran.

        A compute row is the writer's own receiving side, index by global expert
        id, so -- unlike a routing row -- it is *already* attributed: no summing
        over ranks and no ownership lookup. A row the plan relocated to this rank
        therefore counts against this rank, which is the whole difference between
        the two views.

        It keeps MoonEP's capacity pads: on this side a pad row sits in the same
        slot as a real row of the same expert and may be computed by any rank that
        holds that expert, so it cannot be told apart here. ``_fold_pad_row``
        carries the same pads as their own row, which is what lets the report split
        the total into real work and padding where that split is knowable.
        """
        vector = np.asarray(counts, dtype=np.float64)
        self.load_compute[rank_row, si, li] = float(vector.sum())
        self.compute_rank_expert[rank_row] += vector
        self.compute_layer_expert[li] += vector
        self.compute_step_expert[si] += vector

    def _fold_pad_row(self, rank_row: int, si: int, li: int, counts: list[int]) -> None:
        """Fold one ``scope=pad`` row: the pads this rank wrote into the dispatch.

        The pad block was subtracted from the routing rows, so it is not in the
        routing view; the compute rows still carry it, so it is in the work view.
        Neither can show it as its own quantity, and it is neither real traffic nor
        free -- the rows are still matmul'd -- so it is recorded separately.

        The writer is the rank that *injected* the block, and MoonEP aims a rank's
        pads at its own first home expert, so this row is also exactly where the
        load would land under owner attribution. Attribution to a computing rank is
        the one thing it cannot give (see :meth:`_finalize_ledger`).
        """
        vector = np.asarray(counts, dtype=np.int32)
        self.pad_pairs_by_rank[rank_row] += float(vector.sum())
        self.pad_routing[si, li] += vector
        self.pad_rows += 1

    def _attribute_global(self) -> None:
        """Credit the run's routing to the ranks holding the experts it landed on.

        An expert's load is its whole column of ``routing``; a rank's load is that
        column restricted to the experts the rank holds. Under replication an expert
        held by two ranks is credited to the first of them -- the share each replica
        actually served is not recorded anywhere, and inventing a split would make
        the rank totals an estimate. Crediting one holder keeps them exact and makes
        the ratio an *upper bound*: real sharing can only flatten it.
        """
        self.layer_expert = self.routing.astype(np.float64).sum(axis=0)  # [L, E]
        self.step_expert = self.routing.astype(np.float64).sum(axis=1)  # [T, E]
        for rank_row, mask in enumerate(self.owner_masks):
            self.load[rank_row] = np.einsum("tle,le->tl", self.routing, mask.astype(np.float64))
            self.rank_expert[rank_row] = (self.layer_expert * mask).sum(axis=0)
        # The same attribution for the padding, so the routing view can show
        # 总路由负载 = 真实路由 + pad per rank, both exact. A rank's pads are aimed at
        # its own first home expert, which it also owns, so this mostly returns the
        # block to whoever wrote it -- but it is derived, not assumed: the rule is
        # the same ownership rule as the routing's.
        for rank_row, mask in enumerate(self.owner_masks):
            self.load_pad[rank_row] = np.einsum("tle,le->tl", self.pad_routing, mask.astype(np.float64))
            self.rank_pad[rank_row] = float((self.pad_layer_expert * mask).sum())
        # Busiest single expert within each rank's slice, as a share of that rank's
        # row. Chunked over steps: the intermediate is T x L x E wide, which is the
        # same order as the routing itself and not worth holding all at once.
        for rank_row, mask in enumerate(self.owner_masks):
            for lo in range(0, self.n_steps, 256):
                chunk = self.routing[lo : lo + 256]
                peak = np.where(mask[None], chunk, 0).max(axis=2)
                load = self.load[rank_row, lo : lo + 256]
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.top_share[rank_row, lo : lo + 256] = np.where(load > 0, peak / np.maximum(load, 1.0), np.nan)

    def _fold_row(self, dump: RankData, rank_row: int, si: int, li: int, counts: list[int]) -> None:
        """Fold one (rank, step, layer) row into the global-expert accumulators."""
        loc2glob, source = _loc2glob(
            dump.layers.get(self.layer_ids[li]), self.local_experts, self.global_experts, dump.ep_rank
        )
        if source == _SRC_EXPERT_MAP:
            self.map_rows += 1
        elif source == _SRC_WHOLE:
            self.whole_rows += 1
        else:
            self.identity_rows += 1
        vector = np.zeros(self.local_experts, dtype=np.float64)
        width = min(len(counts), self.local_experts)
        vector[:width] = counts[:width]
        valid = (loc2glob >= 0) & (loc2glob < self.global_experts)
        if not valid.any():
            return
        np.add.at(self.rank_expert[rank_row], loc2glob[valid], vector[valid])
        np.add.at(self.layer_expert[li], loc2glob[valid], vector[valid])
        np.add.at(self.step_expert[si], loc2glob[valid], vector[valid])

    # -- derived quantities -------------------------------------------------- #
    def _finalize(self, declared_global: int) -> None:
        if self.truncated_rows:
            what = (
                "行的专家数与这份 dump 声明的全局专家数不一致"
                if self.global_scope
                else "行的本地专家数比最宽的 dump 还多"
            )
            self.warnings.append(
                f"{self.truncated_rows} {what}，已整行丢弃（而不是截断）；"
                "这些 (rank, step, layer) 会显示为缺失单元"
            )
        # The recorder counts a step it had to close with fewer layers than it had
        # already seen: a partial forward would otherwise be written as real load
        # with silent zeros for the layers that did not run.
        partial = max((_as_int(d.meta.get("partial_steps"), 0) for d in self.dumps), default=0)
        self.partial_steps = partial
        if partial:
            self.warnings.append(
                f"打点侧报告有 {partial} 步在结束时不完整（少于已知的 {self.n_layers} 层）；"
                "这些步里没跑到的层被记成了 0，会低估负载"
            )
        self._finalize_pad()
        if not self.global_scope:
            # A global-scope run counts its ownership once per (rank, layer) in
            # _prepare_ownership and has already said what it had to assume.
            placed = self.identity_rows + self.map_rows + self.whole_rows
            if self.identity_rows and placed > self.identity_rows:
                self.warnings.append(
                    f"{placed} 行中有 {self.identity_rows} 行没有可用的 expert_map，退回到"
                    f"连续布局（{_SRC_IDENTITY}）；这些行给出的全局专家 id 只是按位置猜的"
                )
            elif self.identity_rows:
                self.warnings.append(
                    "dump 里没有 expert_map，因此全局专家 id 按连续布局假定"
                    f"（{_SRC_IDENTITY}）——当每个 rank 持有等长且有序的一段时成立"
                )
            if declared_global > 0 and declared_global != self.global_experts:
                self.warnings.append(
                    f"头部写的是 global_experts={declared_global}，但 expert_map 有 "
                    f"{self.global_experts} 项；以 map 为准"
                )

        present = ~np.isnan(self.load)
        self.present = present
        pack = _par_pack(self.load, present, self.n_steps, self.n_layers, self.n_ranks, self.topk)
        self.par_layer = pack["par_layer"]
        self.par_by_step = pack["par_by_step"]
        self.par_agg = pack["par_agg"]
        self.par_whole = pack["par_whole"]
        self.rank_step_load = pack["rank_step_load"]
        self.step_pairs = pack["step_pairs"]
        self.tokens_per_step = pack["tokens_per_step"]
        self.per_rank_mean = pack["per_rank_mean"]
        self.per_rank_max = pack["per_rank_max"]
        self.rank_total = pack["rank_total"]

        # The same three ratios over the work each rank actually ran -- the series
        # the report leads with, because "who is busy" is what a rank-level ratio is
        # for and relocation is exactly what changes it. Absent on a dump that only
        # recorded routing.
        self.present_compute = ~np.isnan(self.load_compute) if self.has_compute else None
        self.compute = (
            _par_pack(self.load_compute, self.present_compute, self.n_steps, self.n_layers, self.n_ranks, self.topk)
            if self.has_compute
            else None
        )
        bench = self._read_bench()
        self.checks = self._self_check()
        self.bench = self._bench_check(bench)
        self.perf = self._perf(bench)
        self.notes = self._notes()
        # After _notes(), which assigns rather than extends: the ledger's own
        # observations belong to the same list.
        self._finalize_ledger()
        if self.bench and self.bench["verdict"] != "exact" and not self.pad_suspect:
            # A shortfall this side of the dump: the recorder buffers the steps and
            # writes them a window at a time, so a process that never ran its exit
            # hook (a forked worker leaves through os._exit) takes the in-flight
            # window with it. Telling the reader the size of that window is the most
            # useful thing that can be said about the number.
            self.warnings.append(
                f"dump 推出的 token 总数与 bench 的请求口径差 {self.bench['delta']:.0f}——"
                f"落盘是每 {self.flush_every} 步一次，进程走 os._exit（fork 起的 worker 不执行 "
                "atexit 钩子）时，退出时还在窗口里的那几步不会写进来；差额的量级因此不超过"
                f"「窗口步数 × 每步 token 数」，但具体丢了几步看不出来。请用 "
                "VLLM_WORKER_MULTIPROC_METHOD=spawn（moe_par_bench.py 默认）或 --flush-every 1 重跑"
            )

    def _finalize_pad(self) -> None:
        """Account for MoonEP's capacity padding: what came out, and what it cost.

        MoonEP pads every step's MoE input up to a pooled capacity and routes the
        pad rows to one of its own experts. The recorder subtracts them where they
        were counted -- in the dispatcher's own histogram, in global expert space --
        so a dump that says ``pad_pairs_subtracted`` needs no fingerprinting. A dump
        written before that change still carries the padding and cannot be
        reconciled offline (the pad block is only knowable to a few percent), so it
        is identified by its shape and flagged instead: MoonEP's own pad rows sit in
        the same slot as real rows of the same expert, which is why no subtraction
        against the receiving side could ever be exact.
        """
        # Each rank subtracted only its own pads, so the run total is the sum.
        self.pad_subtracted = sum(_as_int(d.meta.get("pad_pairs_subtracted"), 0) for d in self.dumps)
        self.pad_misplaced = sum(_as_int(d.meta.get("pad_notes_misplaced"), 0) for d in self.dumps)
        self.pad_negative = sum(_as_int(d.meta.get("negative_cells_clamped"), 0) for d in self.dumps)
        self.pad_expert = next(
            (_as_int(d.meta.get("pad_expert_global"), -1) for d in self.dumps if "pad_expert_global" in d.meta), -1
        )
        # The exit summary line. It is not cosmetic: ``partial_steps`` and the pad
        # ledger exist nowhere else, so a dump without it cannot say whether the
        # run's tail is missing or whether MoonEP's pads were subtracted -- "0"
        # there would be an invented number, not an observation.
        self.tail_missing = not any("partial_steps" in d.meta for d in self.dumps)
        # Steps per host copy, as the dump recorded it: the size of the window a
        # process without an exit hook can lose.
        self.flush_every = max((_as_int(d.meta.get("flush_every"), 0) for d in self.dumps), default=0) or 16
        busy = self.top_share[np.isfinite(self.top_share)]
        share = float(np.median(busy)) if busy.size else float("nan")
        moonep = any("MoonEP" in method for method in self.methods_seen)
        # Only a local-scope MoonEP dump can be padded: a global one subtracted the
        # pads before writing, and its rows are the sending side either way.
        self.pad_suspect = bool(
            moonep and not self.global_scope and not self.pad_subtracted and np.isfinite(share) and share > 0.5
        )
        self.pad_by_rank = [
            [d.rank, _as_int(d.meta.get("pad_pairs_subtracted"), 0), float(self.sent_by_rank[self.rank_ix[d.rank]])]
            for d in self.dumps
        ]
        if self.tail_missing:
            self.warnings.append(
                "这份 dump 没有退出汇总行（文件末尾的 "
                "`# meta partial_steps=… pad_pairs_subtracted=…`），因此「有多少步不完整」和"
                "「扣掉了多少 pad」这两项不可知——不能当成 0。退出汇总行由打点侧在每次落盘后重写，"
                "缺它说明这份 dump 来自更早版本的打点侧，或者那次运行的 worker 一个批次都没落盘就被结束"
                "（用 fork 起 worker 时进程走 os._exit，跳过 atexit 钩子；`moe_par_bench.py` 因此默认用 "
                "spawn）。请以对账自查为准判断数据完整性，必要时重跑"
            )
        if self.pad_suspect:
            self.warnings.append(
                f"这份 dump 里 MoonEP 的容量 pad 还留在计数里：每行最忙的那个本地专家平均占 {share:.0%}，"
                "而且没有任何 pad_pairs_subtracted 记录。MoonEP 每步会把输入 pad 到池化容量"
                "（至少 128 个 token），pad 出来的行全部路由到本 rank 的第一个本地专家——它们不携带"
                "真实 token，却和真实路由一起被计数，于是每格的 (token, 专家) 对数被撑到 capacity × topk。"
                "这份 dump 的绝对负载与峰均比（PAR）因此偏高，专家分布图上那根柱子是 pad 不是热点；"
                "请用带 pad 修正的打点重跑（见 docs/moe_par.md 的「MoonEP 的容量 pad」）"
            )
        if self.pad_misplaced:
            self.warnings.append(
                f"打点侧有 {self.pad_misplaced} 次 MoonEP 的 pad 注记指向了直方图之外的专家，"
                "这些 pad 仍留在计数里；说明派发侧与打点侧对「全局专家数」的理解不一致，"
                "这份 dump 的绝对负载偏高"
            )
        if self.pad_negative:
            self.warnings.append(
                f"打点侧有 {self.pad_negative} 个格子算出负数并被夹到 0（不可能出现：pad 是在"
                "它被计入的那个专家空间里扣掉的）；说明那几层两侧对不上，这些行可疑"
            )
        self.pad = {
            "moonep": moonep,
            "scope": self.scope,
            "subtracted": self.pad_subtracted,
            "misplaced": self.pad_misplaced,
            "negative": self.pad_negative,
            "expert": self.pad_expert,
            "by_rank": self.pad_by_rank,
            # The same pads as the rows that carry them: exact totals, and the
            # injector's own split (which is a different question from the owner's
            # attribution above -- see Run._fold_pad_row).
            "rows": int(self.pad_rows),
            "pairs": float(self.pad_layer_expert.sum()),
            "injected_by_rank": [float(v) for v in self.pad_pairs_by_rank],
            "owner_by_rank": [float(v) for v in self.rank_pad],
            "top_share": None if not np.isfinite(share) else round(share, 4),
            "suspect": self.pad_suspect,
            "tail_missing": self.tail_missing,
        }

    def _finalize_ledger(self) -> None:
        """Lay the three quantities of a MoonEP dispatch side by side.

        A dispatch moves three ledgers, and the two views see different ones:

        ``真实路由``
            The (token, expert) pairs the gate chose. Exact in the routing view --
            the sending-side histogram, with the pads subtracted where they were
            counted, so this number is real traffic and nothing else.
        ``pad``
            The rows MoonEP wrote to fill the dispatch up to its pooled capacity.
            Not real traffic, but not free either: they are matmul'd like any other
            row. Exact, from the dispatcher's own pad block.
        ``丢弃``
            Pairs the capacity envelope dropped before they reached a slot. Only
            knowable as a residual -- nothing else can remove a pair from the path,
            so ``发送 = 真实路由 + pad`` and ``计算 = 发送 - 丢弃``.

        At expert granularity all three are exact in both views, and per rank they
        are exact in the routing view (pad is attributed by the same ownership rule
        as the routing, so 总路由负载 = 真实 + pad holds per rank). The one number
        the data cannot give is the rank dimension of the compute view: once a pad
        row is in a slot it is a row like any other and the plan may hand it to any
        rank holding that expert, so no rank can say how many pads it ran.
        ``pad_est`` therefore credits each rank ``rows / sent_e`` of the padding of
        every expert ``e`` it ran -- the pads of the expert spread evenly over the
        rows that expert was sent -- and ``real_est`` is the computed total minus
        that. Summing it back up returns only the pads that survived the drops, so
        the estimate can under-count but never invent padding; the uncertainty is
        bounded by the pad total, which is small next to it whenever padding is
        working as intended (a few percent of the rows).

        Everything the dump itself says about these numbers is cross-checked here
        and reported, because a ledger that quietly disagrees with the rows it came
        from is worse than no ledger.
        """
        route_total = float(_nansum(self.rank_total, axis=0))
        pad_total = float(self.pad_layer_expert.sum())
        sent_total = route_total + pad_total
        compute_total = float(_nansum(self.compute["rank_total"], axis=0)) if self.compute else float("nan")
        dropped = sent_total - compute_total if np.isfinite(compute_total) else float("nan")

        per_rank = []
        for rank_row, rank_id in enumerate(self.rank_ids):
            route = self.rank_total[rank_row]
            entry = {
                "rank": int(rank_id),
                "route": None if not np.isfinite(route) else float(route),
                "pad": float(self.rank_pad[rank_row]),
                "sent": None if not np.isfinite(route) else float(route) + float(self.rank_pad[rank_row]),
                "compute": None,
                "pad_est": None,
                "real_est": None,
            }
            per_rank.append(entry)
        if self.compute is not None:
            sent_by_expert = self.layer_expert.sum(axis=0) + self.pad_layer_expert.sum(axis=0)  # [E]
            pad_by_expert = self.pad_layer_expert.sum(axis=0)  # [E]
            # The rank that ran a row is credited the padding that row could have
            # carried: `pad_e / sent_e` per row of expert e. Summed over the ranks
            # this returns `pad_e * computed_e / sent_e` -- every pad that was sent
            # and survived the drops -- so a rank can never be credited padding that
            # was not there, and a dropped pad is credited to no one.
            with np.errstate(invalid="ignore", divide="ignore"):
                pad_per_row = np.where(
                    sent_by_expert > 0, pad_by_expert / np.maximum(sent_by_expert, 1e-12), 0.0
                )
            pad_est = self.compute_rank_expert @ pad_per_row  # [R]
            for rank_row, entry in enumerate(per_rank):
                total = self.compute["rank_total"][rank_row]
                if not np.isfinite(total):
                    continue
                entry["compute"] = float(total)
                entry["pad_est"] = float(pad_est[rank_row])
                entry["real_est"] = float(total - pad_est[rank_row])

        # What the dump's own exit summary claims, against what its rows say. The
        # counters only exist in dumps written by the two-view recorder, so their
        # absence is not evidence of a problem -- it is an older dump.
        self.tail_compute_rows = sum(_as_int(d.meta.get("compute_rows_total"), 0) for d in self.dumps)
        self.tail_compute_pairs = sum(_as_int(d.meta.get("compute_pairs_total"), 0) for d in self.dumps)
        self.tail_pad_rows = sum(_as_int(d.meta.get("pad_rows_total"), 0) for d in self.dumps)
        rows_read = int(np.sum(~np.isnan(self.load_compute))) if self.compute is not None else 0
        pairs_read = float(self.compute_layer_expert.sum()) if self.compute is not None else float("nan")

        if self.compute is not None and np.isfinite(dropped) and dropped < -0.5:
            self.warnings.append(
                f"实际计算量比发送量还多 {-dropped:.0f} 对 (token, 专家)：容量包络只能丢行、不能凭空多出来，"
                "所以要么同一对被两个 rank 各算了一遍（副本被重复计算），要么两套打点对不上——"
                "这份 dump 的「实际计算」偏高"
            )
        if self.tail_compute_rows and self.tail_compute_rows != rows_read:
            self.warnings.append(
                f"打点侧记了自己写过 {self.tail_compute_rows} 行「实际计算」行，dump 里只读到 {rows_read} 行"
                "（同一 (step, 层) 的第二行会覆盖第一行，窗口没落盘也会少行）；这份 dump 的计算口径不完整"
            )
        if self.tail_compute_pairs and np.isfinite(pairs_read) and abs(self.tail_compute_pairs - pairs_read) > 0.5:
            self.warnings.append(
                f"打点侧记的实际计算总量是 {self.tail_compute_pairs} 对，dump 里的计算行加起来是 "
                f"{pairs_read:.0f} 对；两者应当相等，差额说明有行没被读进来"
            )
        if self.pad_rows and self.tail_pad_rows and self.tail_pad_rows != self.pad_rows:
            self.warnings.append(
                f"打点侧记了 {self.tail_pad_rows} 行 pad，dump 里读到 {self.pad_rows} 行；"
                "pad 账本不完整，pad 与真实路由的拆分因此偏小"
            )
        if self.pad_rows and abs(pad_total - self.pad_subtracted) > 0.5:
            self.warnings.append(
                f"pad 行加起来是 {pad_total:.0f} 对，而退出汇总行说扣掉了 {self.pad_subtracted} 对；"
                "两者应当相等（扣掉的每一对都单独写成一行），差额说明 pad 账本与路由行对不上"
            )
        if not self.pad_rows and self.pad_subtracted > 0.5:
            # Old recorder: the summary counts the pads it subtracted, but they were
            # never attributed to an expert, so the ledger's ``pad_total`` is 0 only
            # because nothing placed them -- not because the dispatch carried none.
            self.warnings.append(
                f"退出汇总行说扣掉了 {self.pad_subtracted} 对 pad，但这份 dump 的 pad 没有落到具体专家上"
                "（旧版打点只记总数，不写「pad 行」）；所以「真实路由」是准的，「发送量 = 真实路由 + pad」"
                "里的 pad 这一项在这里不可拆分，不要把它当成 0"
            )

        self.ledger = {
            "route_total": route_total,
            "pad_total": pad_total,
            "sent_total": sent_total,
            "compute_total": None if not np.isfinite(compute_total) else compute_total,
            "dropped": None if not np.isfinite(dropped) else dropped,
            "dropped_fraction": (dropped / sent_total) if np.isfinite(dropped) and sent_total > 0 else None,
            "per_rank": per_rank,
            "tail_compute_rows": self.tail_compute_rows,
            "tail_compute_pairs": self.tail_compute_pairs,
            "tail_pad_rows": self.tail_pad_rows,
            "compute_rows_read": rows_read,
            "compute_pairs_read": None if not np.isfinite(pairs_read) else pairs_read,
            "pad_rows_read": int(self.pad_rows),
        }
        if self.compute is not None and self.ledger["dropped"] is not None and abs(self.ledger["dropped"]) > 0.5:
            where = "丢弃" if self.ledger["dropped"] > 0 else "重复计算"
            self.notes.append(
                f"发送 {sent_total:.0f} 对 = 真实路由 {route_total:.0f} + pad {pad_total:.0f}，"
                f"实际计算 {compute_total:.0f} 对，差 {abs(self.ledger['dropped']):.0f} 对（{where}，"
                f"占发送的 {abs(self.ledger['dropped_fraction']):.1%}）"
            )
        if self.ledger["per_rank"] and any(entry["pad_est"] for entry in self.ledger["per_rank"]):
            self.notes.append(
                "逐 rank 的「实际计算」含 pad，而 pad 落到哪个 rank 上是无法从 dump 里读出来的（pad 行与真实"
                "行同槽、且可能被预取到别的 rank）；pad_est 是按「该 expert 各行里由本 rank 计算的比例」摊分"
                "出来的估计，误差不超过 pad 总量"
            )

    def _self_check(self) -> dict:
        """Cross-rank consistency: a step's pairs-per-layer must not depend on the layer.

        The stitch is the only place the ranks meet, so this is the one check
        that can catch a broken join. Every MoE layer of a forward sees the same
        tokens, so ``sum_r load[r, t, l]`` has to be flat in ``l``.
        """
        per_layer_total = np.where(self.present, self.load, np.nan).sum(axis=0)  # [T, L]
        full = np.sum(self.present, axis=0) == self.n_ranks  # [T, L]
        checked = 0
        worst = 0.0
        for si in range(self.n_steps):
            values = per_layer_total[si][full[si]]
            if values.size < 2:
                continue
            checked += 1
            median = float(np.median(values))
            if median > 0:
                worst = max(worst, float(np.max(np.abs(values - median)) / median))
        return {
            "steps_checked": checked,
            "max_layer_spread": worst,
            "missing_cells": int(np.sum(~self.present)),
            # With one rank there is nothing to reconcile: the sum over ranks is that
            # rank's own load, which is flat by construction. Say so instead of
            # reporting a pass the check cannot earn.
            "applicable": bool(self.n_ranks > 1),
            "flat": bool(checked and worst < 1e-6),
        }

    def _read_bench(self) -> dict | None:
        """The bench.json sitting next to the dumps, if the driver left one there."""
        path = os.path.join(self.dir, "bench.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def _bench_check(self, bench: dict | None) -> dict | None:
        """Reconcile the dump's pair total against the bench's own token count.

        The only check here whose number comes from *outside* the dump. Every
        forward sends each of its tokens to topk experts in every MoE layer, so
        the pairs recorded across all ranks and layers, divided by topk * layers,
        must equal the number of tokens the model really ran. ``bench.json``
        counts requests rather than forwards, so the two differ by exactly one
        token per request -- the token a request ends on is sampled but never fed
        back through the model.

        Returns None when there is no bench.json to compare against, which is the
        normal case for a dump that was moved or produced by another driver.
        """
        if bench is None:
            return None
        requests = bench.get("requests") or []
        timing = bench.get("timing") or {}
        if requests:
            expected = float(sum(r.get("prompt_tokens", 0) + r.get("output_tokens", 0) - 1 for r in requests))
            source = "requests"
        elif timing:
            n_prompts = _as_int((bench.get("config") or {}).get("num_prompts"), 0)
            expected = float(timing.get("prompt_tokens", 0) + timing.get("output_tokens", 0) - n_prompts)
            source = "timing"
        else:
            return None
        if expected <= 0:
            return None
        actual = float(_nansum(self.tokens_per_step, axis=0))
        delta = actual - expected
        return {
            "source": source,
            "n_requests": len(requests),
            "expected": expected,
            "actual": actual,
            "delta": delta,
            "relative": delta / expected,
            # Exact is the normal outcome. A shortfall of a few percent is the tail
            # step the recorder drops on exit; anything larger is a real disagreement.
            "verdict": "exact" if abs(delta) < 0.5 else ("close" if abs(delta) <= 0.02 * expected else "off"),
        }

    def _perf(self, bench: dict | None) -> dict | None:
        """TTFT / TPOT / throughput, per request and aggregated, from the bench.

        These are the bench's own measurements, not something the load dump can
        imply: ``bench.json`` carries ``timing`` (wall clock around ``generate``)
        and, from the version that records them, per-request ``latency`` numbers
        read off ``RequestOutput.metrics``.

        The numbers are quoted from whichever source the bench wrote, in this
        order: the bench's own ``perf`` summary, then a recomputation from the
        per-request latencies, then throughput alone from ``timing``. An older
        bench.json -- written before latency was recorded -- lands in the last
        case, and the report says so rather than showing an empty panel.

        Returns None when there is no bench.json at all.
        """
        if bench is None:
            return None
        timing = bench.get("timing") or {}
        requests = bench.get("requests") or []
        if not timing and not requests:
            return None

        summary = bench.get("perf") or {}
        reported = summary.get("quantiles") or {}
        latency_fields = [r["latency"] for r in requests if isinstance(r.get("latency"), dict)]

        quantiles: dict[str, dict] = {}
        for name, label in _PERF_NAMES.items():
            entry = reported.get(name)
            if not entry:
                values = [f[name] for f in latency_fields if f.get(name) is not None]
                entry = _quantiles(values) if values else None
            if entry:
                quantiles[name] = {**entry, "label": label}

        # Totals come from timing when it is there: it counts what the engine really
        # ran, which is the same quantity the throughput denominators use.
        prompt_tokens = _as_int(timing.get("prompt_tokens"), 0) or sum(
            _as_int(r.get("prompt_tokens"), 0) for r in requests
        )
        output_tokens = _as_int(timing.get("output_tokens"), 0) or sum(
            _as_int(r.get("output_tokens"), 0) for r in requests
        )
        generate_s = float(timing.get("generate_s") or 0.0)
        throughput = summary.get("throughput") or {}
        if not throughput and generate_s > 0:
            throughput = {
                "input_tok_s": round(prompt_tokens / generate_s, 2),
                "output_tok_s": round(output_tokens / generate_s, 2),
                "total_tok_s": round((prompt_tokens + output_tokens) / generate_s, 2),
                "req_s": round(len(requests) / generate_s, 3),
            }

        return {
            "source": summary.get("source") if latency_fields else None,
            "measured": len(latency_fields),
            "n_requests": len(requests),
            "generate_s": round(generate_s, 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "quantiles": quantiles,
            "throughput": throughput,
            # The per-request rows behind the charts, kept whole so the report can
            # answer "which request was slow" and not just "how slow on average".
            "requests": [
                {
                    "idx": _as_int(r.get("idx"), i),
                    "prompt_tokens": _as_int(r.get("prompt_tokens"), 0),
                    "output_tokens": _as_int(r.get("output_tokens"), 0),
                    "finish_reason": r.get("finish_reason") or "",
                    **(r.get("latency") if isinstance(r.get("latency"), dict) else {}),
                }
                for i, r in enumerate(requests)
            ],
        }

    def _notes(self) -> list[str]:
        notes = []
        if self.n_ranks < 2:
            notes.append(
                "这个 dump 里只有一个 rank，没有可度量的离散度，PAR 不是 1 而是未定义"
                "——负载图表仍然有效"
            )
        finite = self.tokens_per_step[np.isfinite(self.tokens_per_step)]
        if self.topk > 0 and finite.size:
            notes.append(
                f"每步 token 数 {finite.min():.0f}–{finite.max():.0f}（topk {self.topk}）"
            )
        if self.duplicated_experts:
            notes.append(
                f"有 {self.duplicated_experts} 个逻辑专家同时被多个 rank 持有"
                f"（头部记 redundant_experts={self.redundant}）；专家负载按逻辑专家记，副本算同一个专家；"
                "rank 负载记在持有它的那个 rank 上，副本之间的真实分摊没有任何记录、也不去猜"
                "——所以本运行的 PAR 是上界，真实分摊只会让它更平"
            )
        elif self.redundant:
            notes.append(
                f"头部记了 {self.redundant} 个冗余专家，但没有一个逻辑专家同时出现在两个 rank 的"
                "expert_map 里；按整体表解读为无副本"
            )
        if self.methods_seen:
            readable = sorted(_METHOD_LABELS.get(m, m.removesuffix("CommImpl")) for m in self.methods_seen)
            notes.append("实际在链路上的通信方式：" + "、".join(readable))
        if self.pad_subtracted:
            where = f"，落在逻辑专家 {self.pad_expert}" if self.pad_expert >= 0 else ""
            notes.append(
                f"打点侧已从计数里扣除 {self.pad_subtracted} 个 MoonEP 容量 pad pair{where}；"
                "这份 dump 记的是真实 (token, 专家) 对"
            )
        if self.perf:
            if not self.perf["quantiles"]:
                notes.append(
                    "bench.json 里没有逐请求时延数据——这一份是旧版 bench 脚本产出的（或起服务时关了"
                    " metrics 统计），因此只有吞吐可比；要 TTFT/TPOT 就用当前版本的 bench 脚本重跑一次"
                )
            elif "queue_ms" in self.perf["quantiles"]:
                notes.append(
                    "TTFT 从请求入队那一刻开始计时，所以它含排队时间（本次排队均值 "
                    f"{self.perf['quantiles']['queue_ms']['mean']:.1f} ms）；离线一次性提交整批 prompt 时，"
                    "排在后面的请求 TTFT 会被前面的请求抬高，看 P90 而不是均值更能反映这一点"
                )
        return notes

    # -- payload ------------------------------------------------------------- #
    def to_json(self) -> dict:
        def clean(array: np.ndarray):
            """Non-finite -> None, so the report can tell 'missing' from 'zero'."""
            return np.where(np.isfinite(array), array, None).tolist()

        return {
            "name": self.name,
            "dir": self.dir,
            "model": self.model,
            "tag": self.tag,
            "ep_size": self.ep_size,
            "world_size": self.world_size,
            "scope": self.scope,
            # What the primary (routing) rows are index by, and whether the dump
            # carries MoonEP's second view of the same dispatch.
            "primary": self.primary,
            "has_compute": bool(self.has_compute),
            "global_experts": int(self.global_experts),
            "local_experts": int(self.local_experts),
            "row_width": int(self.row_width),
            "topk": int(self.topk),
            "redundant": int(self.redundant),
            "duplicated_experts": int(self.duplicated_experts),
            "unowned_slots": int(self.unowned_slots),
            "partial_steps": int(self.partial_steps),
            "flush_every": int(self.flush_every),
            "expert_holders": [list(h) for h in self.expert_holders],
            "rank_ids": [int(r) for r in self.rank_ids],
            "steps": [int(s) for s in self.steps],
            "n_steps": self.n_steps,
            "n_layers": self.n_layers,
            "n_ranks": self.n_ranks,
            "load": clean(self.load),  # R x T x L
            "rank_expert": clean(self.rank_expert),  # R x G
            "rank_expert_seen": [bool(v) for v in self.rank_expert_seen],
            "layer_expert": clean(self.layer_expert),  # L x G
            "step_expert": clean(self.step_expert),  # T x G
            "par_layer": clean(self.par_layer),  # T x L
            "par_by_step": clean(self.par_by_step),  # T
            "par_agg": clean(self.par_agg),  # T
            "par_whole": None if not np.isfinite(self.par_whole) else float(self.par_whole),
            "rank_total": clean(self.rank_total),  # R, summed over every step and layer
            "rank_step_load": clean(self.rank_step_load),  # R x T
            "per_rank_mean": clean(self.per_rank_mean),  # R
            "per_rank_max": clean(self.per_rank_max),  # R
            "tokens_per_step": clean(self.tokens_per_step),  # T
            "methods_per_step": [_METHOD_LABELS.get(m, m.removesuffix("CommImpl")) for m in self.method_of_step],
            # The second view: the same dispatch, attributed to the rank that ran
            # each row. Same shape as the primary fields, so the report can put the
            # two series side by side -- and leads with this one.
            "compute": None
            if self.compute is None
            else {
                "load": clean(self.compute["load"]),  # R x T x L, by computing rank
                "rank_step_load": clean(self.compute["rank_step_load"]),  # R x T
                "par_layer": clean(self.compute["par_layer"]),  # T x L
                "par_by_step": clean(self.compute["par_by_step"]),  # T
                "par_agg": clean(self.compute["par_agg"]),  # T
                "par_whole": None if not np.isfinite(self.compute["par_whole"]) else float(self.compute["par_whole"]),
                "rank_total": clean(self.compute["rank_total"]),  # R, whole run
                "tokens_per_step": clean(self.compute["tokens_per_step"]),  # T, pads included
                "per_rank_mean": clean(self.compute["per_rank_mean"]),  # R
                "per_rank_max": clean(self.compute["per_rank_max"]),  # R
                "rank_expert": clean(self.compute_rank_expert),  # R x E
                "layer_expert": clean(self.compute_layer_expert),  # L x E
                "step_expert": clean(self.compute_step_expert),  # T x E
                "seen": [bool(v) for v in self.compute_seen],
            },
            # The padding as its own ledger, at expert granularity (exact in every
            # scope) and per rank in the routing view (by the same ownership rule as
            # the routing). The rank dimension of the compute view is the one place
            # it cannot be exact -- see Run._finalize_ledger.
            "layer_expert_pad": clean(self.pad_layer_expert),  # L x E
            "step_expert_pad": clean(self.pad_step_expert),  # T x E
            "rank_pad": clean(self.rank_pad),  # R, owner-attributed
            "ledger": self.ledger,
            "checks": self.checks,
            "pad": self.pad,
            "bench": self.bench,
            "perf": self.perf,
            "notes": self.notes,
            "warnings": self.warnings,
        }

    def finalize(self) -> "Run":
        self._finalize(_as_int(self.dumps[0].meta.get("global_experts"), -1))
        return self


def build_payload(runs: list[Run]) -> dict:
    warnings: list[str] = []
    models = {run.model for run in runs}
    # ``primary``, not ``scope``: a two-view dump is comparable to a one-view one
    # (they share the routing view), and calling those two different shapes would
    # warn about a difference that does not exist.
    shapes = {(run.primary, run.global_experts, run.local_experts, run.topk) for run in runs}
    if len(models) > 1:
        warnings.append(
            "这些运行来自不同模型（" + "、".join(sorted(models)) + "）；比值仍然可比，"
            "但专家级图表不可比"
        )
    if len(shapes) > 1:
        warnings.append(
            "各运行在 (口径, global_experts, local_experts, topk) 上不一致 = "
            + "、".join(str(s) for s in sorted(shapes))
        )
    for run in runs:
        warnings.extend(f"{run.name}: {w}" for w in run.warnings)
    return {
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [run.to_json() for run in runs],
        "warnings": warnings,
    }


def report_js_problems(html: str) -> list[str]:
    """Parse the report's own JavaScript, if a JS engine is around to do it.

    The report is one HTML string with its own script: a stray brace in the template
    leaves every panel blank in the browser while the analyzer still exits 0 and
    writes the file, and nothing upstream notices because the HTML stays well-formed.
    So each non-payload ``<script>`` block is handed to ``node --check``. No node on
    PATH means no opinion -- an empty list is "nothing found", not "verified".
    """
    node = shutil.which("node")
    if node is None:
        return []
    problems = []
    blocks = [
        m.group(2)
        for m in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S)
        if 'id="payload"' not in m.group(1)
    ]
    for index, block in enumerate(blocks, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
            handle.write(block)
            name = handle.name
        try:
            proc = subprocess.run([node, "--check", name], capture_output=True, text=True)
        finally:
            os.unlink(name)
        if proc.returncode != 0:
            detail = next((line for line in proc.stderr.splitlines() if line.strip()), "?")
            problems.append(f"报告的 JS 第 {index} 块解析不过（浏览器里会白屏）：{detail}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze per-rank MoE load dumps and write a self-contained HTML report."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="[NAME=]DIR",
        help="a dump directory, optionally named; repeat once per communication method",
    )
    parser.add_argument("-o", "--out", default="moe_par_report.html", help="output HTML path")
    parser.add_argument("--json", metavar="PATH", help="also write the computed payload as JSON")
    args = parser.parse_args()

    runs: list[Run] = []
    for spec in args.run:
        name, _, directory = spec.rpartition("=")
        if not name:
            name, directory = os.path.basename(os.path.normpath(spec)), spec
        runs.append(Run(name, directory).finalize())

    for run in runs:
        check = run.checks

        def _stat(values: np.ndarray, how: str) -> str:
            kept = values[np.isfinite(values)]
            if kept.size == 0:
                return "n/a"  # e.g. PAR with a single rank: no spread to measure
            return f"{getattr(kept, how)():.3f}"

        whole = "n/a" if not np.isfinite(run.par_whole) else f"{run.par_whole:.3f}"
        pad = ""
        if run.pad["subtracted"]:
            pad = f" | MoonEP pads subtracted {run.pad['subtracted']} (global ids)"
        elif run.pad["suspect"]:
            share = run.pad["top_share"] or 0.0
            pad = f" | MoonEP PADS PRESENT (top expert {share:.0%} of every row, not subtracted)"
        elif run.pad["tail_missing"]:
            pad = " | no exit-summary line: pads subtracted and partial steps UNKNOWN"
        scope = f" | counts {run.scope}" + (
            " (logical experts)" if run.primary == "global" else " (local slots)"
        )
        print(
            f"[moe-par] {run.name}: {run.n_ranks} ranks x {run.n_steps} steps x {run.n_layers} layers{scope} | "
            f"cross-rank join "
            f"{'flat' if check['flat'] else 'NOT FLAT'} (spread {check['max_layer_spread']:.2e}){pad}",
            file=sys.stderr,
        )
        # 两套口径：实际计算是主口径（屏障等的是它），路由归属作参照。没有第二套时
        # 只打一行，语义与以前完全一致。
        if run.compute is None:
            print(
                f"[moe-par] {run.name} (路由归属): PAR per-layer mean "
                f"{_stat(run.par_by_step, 'mean')} max {_stat(run.par_by_step, 'max')} | "
                f"PAR per-step (whole step) mean {_stat(run.par_agg, 'mean')} | "
                f"PAR whole run {whole}",
                file=sys.stderr,
            )
        else:
            compute_whole = (
                "n/a" if not np.isfinite(run.compute["par_whole"]) else f"{run.compute['par_whole']:.3f}"
            )
            ledger = run.ledger
            dropped = ledger["dropped"] or 0.0
            share = ledger["pad_total"] / max(1.0, ledger["sent_total"])
            split = f"真实 {ledger['route_total']:.0f} + pad {ledger['pad_total']:.0f}"
            if abs(dropped) > 0.5:
                split += f" - 丢弃 {dropped:.0f}"
            print(
                f"[moe-par] {run.name} (实际计算·主): PAR per-layer mean "
                f"{_stat(run.compute['par_by_step'], 'mean')} "
                f"max {_stat(run.compute['par_by_step'], 'max')} | "
                f"PAR per-step (whole step) mean {_stat(run.compute['par_agg'], 'mean')} | "
                f"PAR whole run {compute_whole} | 计算量 {ledger['compute_total']:.0f} = {split} "
                f"(pad 占发送量 {share:.1%})",
                file=sys.stderr,
            )
            print(
                f"[moe-par] {run.name} (路由归属·参照): PAR per-layer mean "
                f"{_stat(run.par_by_step, 'mean')} max {_stat(run.par_by_step, 'max')} | "
                f"PAR per-step (whole step) mean {_stat(run.par_agg, 'mean')} | "
                f"PAR whole run {whole}  [被搬到别处算的行仍记属主，故 rank 维度上是上界]",
                file=sys.stderr,
            )
        for warning in run.warnings:
            print(f"[moe-par]   warning: {warning}", file=sys.stderr)

    html = _TEMPLATE.replace(
        "__PAYLOAD__",
        json.dumps(build_payload(runs), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"[moe-par] wrote {args.out} ({len(html) / 1024:.0f} KiB)", file=sys.stderr)
    for problem in report_js_problems(html):
        print(f"[moe-par]   warning: {problem}", file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(build_payload(runs), handle)
        print(f"[moe-par] wrote {args.json}", file=sys.stderr)
    return 0


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoE 负载与峰均比报告</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f3f0;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7973;
    --border: #dedcd6; --grid: #e8e6e0;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
    --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
    --status-warning: #eda100; --status-critical: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19; --surface-2: #232322;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
      --border: #3a3a37; --grid: #2e2e2b;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
      --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
    --border: #3a3a37; --grid: #2e2e2b;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    background: var(--surface-1); color: var(--text-primary); -webkit-font-smoothing: antialiased;
  }
  .viz-root { max-width: 1460px; margin: 0 auto; padding: 28px 24px 80px; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; letter-spacing: -.01em; }
  h2 { font-size: 17px; font-weight: 620; margin: 44px 0 2px; letter-spacing: -.005em; }
  h3 { font-size: 13.5px; font-weight: 620; margin: 0 0 3px; }
  p.sub { color: var(--text-secondary); margin: 0; }
  .caption { color: var(--text-muted); font-size: 12.5px; margin: 3px 0 12px; max-width: 96ch; }
  /* 口径徽标：这一份 dump 的负载是按什么数出来的。它是整份报告的前提，
     所以跟着运行标题走，颜色只用于区分「可信 / 有前提 / 不可信」。 */
  .badge { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 600;
           border: 1px solid var(--border); border-left: 3px solid #1baf7a; border-radius: 7px;
           padding: 5px 11px; margin: 2px 0 10px; background: var(--surface-2); }
  .badge.warn { border-left-color: var(--status-warning); }
  .badge.crit { border-left-color: var(--status-critical); }
  .badge i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; background: #1baf7a; }
  .badge.warn i { background: var(--status-warning); }
  .badge.crit i { background: var(--status-critical); }
  .badge small { font-weight: 400; color: var(--text-muted); }
  .sem table { font-size: 12.5px; }
  .sem td { text-align: left; vertical-align: top; line-height: 1.55; padding-right: 14px; }
  .sem th { white-space: nowrap; }
  .sem td:first-child { white-space: nowrap; }
  .sem h4 { font-size: 12.5px; margin: 16px 0 4px; color: var(--text-secondary); }
  .bar { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; margin: 16px 0 6px; }
  button, select {
    font: inherit; font-size: 12.5px; color: var(--text-primary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 7px; padding: 5px 11px; cursor: pointer;
  }
  button[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1);
                                border-color: var(--text-primary); }
  select { padding: 5px 7px; }
  .spacer { flex: 1; }
  .warn { display: flex; gap: 9px; align-items: flex-start; border: 1px solid var(--border);
          border-left: 3px solid var(--status-warning); border-radius: 7px; padding: 9px 12px;
          margin: 10px 0; font-size: 13px; background: var(--surface-2); }
  .warn.crit { border-left-color: var(--status-critical); }
  .warn.ok { border-left-color: #1baf7a; }
  .card { border: 1px solid var(--border); border-radius: 9px; padding: 11px 13px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(172px, 1fr)); gap: 10px; margin: 14px 0 6px; }
  .card .k { font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .045em; }
  .card .v { font-size: 24px; font-weight: 640; letter-spacing: -.02em; margin-top: 3px;
             font-variant-numeric: tabular-nums; }
  .card .u { font-size: 12px; color: var(--text-muted); font-weight: 400; margin-left: 3px; }
  .card .n { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
  .panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 10px; min-width: 0; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  text { font-family: inherit; }
  .tick { font-size: 10.5px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
  .axis-title { font-size: 11px; fill: var(--text-secondary); }
  .dlabel { font-size: 10.5px; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 8px 0 2px; font-size: 12px;
            color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .tip { position: fixed; z-index: 40; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; font-size: 12px;
    line-height: 1.45; box-shadow: 0 6px 22px rgba(0,0,0,.16); color: var(--text-primary);
    max-width: 340px; font-variant-numeric: tabular-nums; }
  .tip .sm { color: var(--text-muted); }
  details { margin: 10px 0 2px; }
  summary { cursor: pointer; font-size: 12.5px; color: var(--text-secondary); }
  /* 报告前段的两长块（术语表 / 怎么读）默认收起：标题本身就是开关。 */
  details.fold { margin: 8px 0 2px; border-top: 1px solid var(--border); padding-top: 9px; }
  details.fold > summary { font-size: 13px; font-weight: 600; color: var(--text-primary); }
  details.fold > summary::marker { color: var(--text-muted); }
  details.fold > .gloss { margin-top: 6px; }
  .foldbar { margin: 4px 0 0; }
  .foldbar button { font-size: 12px; padding: 4px 10px; }
  /* 结论区：三个 PAR 口径。它是报告里最该先看的一块，所以数字放大、加高亮边。 */
  .panel.hero { border-left: 3px solid var(--series-1); margin: 14px 0 20px; }
  .hero .scopes { display: grid; grid-template-columns: repeat(auto-fit, minmax(212px, 1fr));
                  gap: 10px; margin: 12px 0 2px; }
  .scope { border: 1px solid var(--border); border-top: 3px solid var(--border);
           border-radius: 9px; padding: 9px 11px 10px; }
  .scope .sk { font-size: 12.5px; font-weight: 620; }
  .scope .sv { font-size: 31px; font-weight: 660; letter-spacing: -.02em; line-height: 1.15;
               margin-top: 2px; font-variant-numeric: tabular-nums; }
  .scope .sn { font-size: 11.5px; color: var(--text-muted); line-height: 1.5; margin-top: 3px; }
  .scope .sbar { height: 5px; border-radius: 3px; background: var(--grid); margin: 7px 0 6px;
                 overflow: hidden; }
  .scope .sbar i { display: block; height: 100%; border-radius: 3px; }
  .hero table { font-size: 12.5px; }
  .hero table td:last-child { font-size: 13.5px; font-weight: 620; white-space: nowrap; }
  table { border-collapse: collapse; font-size: 12px; margin-top: 8px; font-variant-numeric: tabular-nums; }
  th, td { border-bottom: 1px solid var(--border); padding: 3px 12px 3px 0; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--text-secondary); font-weight: 600; }
  /* 两行表头的汇总表：第一行是「指标（单位）」分组，居中、字号略小；第二行才是各统计量。
     列多时整表可以横向滚动，不撑破面板。 */
  .gt thead tr:first-child th { text-align: center; font-size: 11.5px; }
  .gt thead tr:first-child th:first-child { text-align: left; }
  .tw { overflow-x: auto; }
  footer { margin-top: 48px; color: var(--text-muted); font-size: 12px; max-width: 110ch; }
  code { font-size: 12px; padding: 1px 5px; border-radius: 4px; white-space: nowrap;
         background: var(--surface-2, rgba(127,127,127,.12)); }
  .gloss { max-width: 96ch; }
  .gloss h4 { font-size: 12.5px; margin: 16px 0 4px; color: var(--text-secondary); }
  .gloss p, .gloss li { margin: 4px 0; font-size: 12.5px; line-height: 1.65; }
  .gloss ul { margin: 4px 0 4px 20px; padding: 0; }
  .gloss table { margin-top: 6px; }
  .gloss td { text-align: left; vertical-align: top; line-height: 1.5; padding-right: 16px; }
  .gloss th { white-space: nowrap; }
  .gloss td:first-child { white-space: nowrap; }
</style>
</head>
<body>
<div class="viz-root" id="root">
  <h1>MoE 负载分布与峰均比</h1>
  <p class="sub" id="subtitle"></p>
  <div id="warnings"></div>
  <div class="bar">
    <span class="axis-title">运行</span>
    <span id="runpick"></span>
    <span class="spacer"></span>
    <button id="theme" type="button">深色</button>
  </div>
  <div id="report"></div>
  <footer>PAR = 各 rank 之间的 峰值 / 均值，取值 [1, rank 数]；1 表示完全均衡。负载以 token-专家对计数，
  一个 token 被路由给 <i>k</i> 个专家就贡献 <i>k</i>；某个 rank 的负载是它本地专家上的求和。</footer>
</div>
<div class="tip" id="tip" hidden></div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const NS = "http://www.w3.org/2000/svg";
const DATA = JSON.parse(document.getElementById("payload").textContent);
const RUNS = DATA.runs;
const SERIES = ["--series-1","--series-2","--series-3","--series-4",
                "--series-5","--series-6","--series-7","--series-8"];
const TIP = document.getElementById("tip");
const ROOT = document.getElementById("root");

/* Sequential ramp and diverging poles, per mode. The dark sequential ramp is the
   same single blue hue reversed, so "more" still reads as "more visible" against
   a dark surface while staying monotonic in lightness. */
const RAMP = {
  light: { seq: ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5",
                 "#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"],
           lo: "#2a78d6", mid: "#f0efec", hi: "#e34948", empty: "#f4f3f0" },
  dark:  { seq: ["#0d366b","#104281","#184f95","#1c5cab","#256abf","#2a78d6","#3987e5",
                 "#5598e7","#6da7ec","#86b6ef","#9ec5f4","#b7d3f6","#cde2fb"],
           lo: "#3987e5", mid: "#383835", hi: "#e66767", empty: "#232322" }
};
const darkNow = () => document.documentElement.getAttribute("data-theme") === "dark"
  || (!document.documentElement.hasAttribute("data-theme")
      && window.matchMedia("(prefers-color-scheme: dark)").matches);
const ramp = () => RAMP[darkNow() ? "dark" : "light"];

/* ---------------------------------------------------------------- helpers */
function el(tag, attrs, kids) {
  const node = document.createElementNS(NS, tag);
  for (const k in (attrs || {})) if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
  for (const kid of (kids || [])) node.appendChild(document.createTextNode(kid));
  return node;
}
function div(cls, text) {
  const node = document.createElement("div");
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
function heading(level, text) {
  const node = document.createElement(level);
  node.textContent = text;
  return node;
}
function svgRoot(w, h, label) {
  const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label || "chart" });
  return s;
}
function css(name) { return getComputedStyle(ROOT).getPropertyValue(name).trim(); }
function seriesColor(i) { return css(SERIES[i % SERIES.length]); }
function mixHex(a, b, t) {
  const A = [1,3,5].map(i => parseInt(a.slice(i, i+2), 16));
  const B = [1,3,5].map(i => parseInt(b.slice(i, i+2), 16));
  return "#" + A.map((v, i) => Math.round(v + (B[i]-v)*t).toString(16).padStart(2, "0")).join("");
}
function seqColor(t) {
  const s = ramp().seq;
  // isFinite(null) is true, so the null check has to come first: a missing cell
  // must stay visibly missing, not clamp to the lightest step of the ramp.
  if (t === null || t === undefined || !isFinite(t)) return ramp().empty;
  t = Math.max(0, Math.min(1, t));
  const x = t * (s.length - 1), i = Math.min(s.length - 2, Math.floor(x));
  return mixHex(s[i], s[i+1], x - i);
}
function divColor(v, lo, hi, mid) {
  const r = ramp();
  if (v === null || v === undefined || !isFinite(v)) return r.empty;
  if (v >= mid) return mixHex(r.mid, r.hi, hi > mid ? Math.min(1, (v-mid)/(hi-mid)) : 0);
  return mixHex(r.mid, r.lo, mid > lo ? Math.min(1, (mid-v)/(mid-lo)) : 0);
}
function finite(values) { return values.filter(v => v !== null && v !== undefined && isFinite(v)); }
function minMax(values) {
  const xs = finite(values);
  return xs.length ? [Math.min(...xs), Math.max(...xs)] : [NaN, NaN];
}
function ticksOf(lo, hi, n) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (hi <= lo) hi = lo + 1;
  const raw = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = mag * (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10);
  const out = [];
  for (let v = Math.ceil(lo/step)*step; v <= hi + step*1e-6; v += step) out.push(+v.toPrecision(12));
  return out;
}
function fmt(v, d) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  if (d !== undefined) return v.toFixed(d);
  const a = Math.abs(v);
  if (a >= 1e9) return (v/1e9).toFixed(2) + "B";
  if (a >= 1e6) return (v/1e6).toFixed(2) + "M";
  if (a >= 100) return Math.round(v).toLocaleString();
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}
/* 计数（请求数、token 数）整数展示：fmt 的默认两位小数会把 24 写成 24.00。 */
function count(v) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return Math.round(v).toLocaleString();
}
function mean(values) {
  const xs = finite(values);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}
function quantiles(values) {
  const xs = finite(values).sort((a, b) => a - b);
  if (!xs.length) return null;
  const q = p => {
    const k = (xs.length - 1) * p, i = Math.floor(k);
    return i === k ? xs[i] : xs[i] + (xs[i+1]-xs[i])*(k-i);
  };
  return { lo: xs[0], q1: q(.25), med: q(.5), q3: q(.75), hi: xs[xs.length-1], n: xs.length };
}
function showTip(evt, html) {
  TIP.innerHTML = html;
  TIP.hidden = false;
  const pad = 14, w = TIP.offsetWidth, h = TIP.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  TIP.style.left = Math.max(4, x) + "px";
  TIP.style.top = Math.max(4, y) + "px";
}
const hideTip = () => { TIP.hidden = true; };
function hoverable(node, html) {
  node.addEventListener("pointerenter", e => showTip(e, html));
  node.addEventListener("pointermove", e => showTip(e, html));
  node.addEventListener("pointerleave", hideTip);
}
function table(headers, rows) {
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of headers) hr.appendChild(heading("th", h));
  thead.appendChild(hr);
  t.appendChild(thead);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const c of row) tr.appendChild(heading("td", String(c)));
    body.appendChild(tr);
  }
  t.appendChild(body);
  return t;
}
function collapsible(summary, node, open) {
  const d = document.createElement("details");
  if (open) d.open = true;
  d.appendChild(heading("summary", summary));
  d.appendChild(node);
  return d;
}
/* 大块长文（术语表 / 怎么读）用带样式的折叠块：标题即开关，原生 <details>，
   不依赖 JS 也能点开。展开状态记在 FOLD_STATE 里，这样切深色/浅色触发整页重绘后
   仍保持用户刚才的选择。 */
const FOLD_STATE = {};
function foldSection(key, summary, node, openByDefault) {
  const d = collapsible(summary, node, key in FOLD_STATE ? FOLD_STATE[key] : !!openByDefault);
  d.className = "fold";
  d.addEventListener("toggle", () => { FOLD_STATE[key] = d.open; });
  return d;
}
function foldAll(open) {
  document.querySelectorAll("details.fold").forEach(d => { d.open = open; });
}
function legendOf(items) {
  const host = div("legend");
  for (const [label, color] of items) {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = color;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(label));
    host.appendChild(span);
  }
  return host;
}

/* ------------------------------------------------------------------ charts */
/* Vertical bars, one series, so no legend box -- the title names it. The dashed
   reference line carries the mean, which is the denominator of every ratio here. */
function barChart(host, o) {
  const W = o.width || 900, H = o.height || 260;
  const m = { t: 16, r: 16, b: o.rotate ? 62 : 46, l: 64 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const [lo, hi] = minMax(o.values.concat(o.refLine === undefined || o.refLine === null ? [] : [o.refLine]));
  const y0 = Math.min(0, isFinite(lo) ? lo : 0);
  const y1 = (isFinite(hi) ? hi : 1) * 1.08 || 1;
  const sy = v => m.t + ih - (v - y0) / (y1 - y0) * ih;
  const svg = svgRoot(W, H, o.yLabel + " 对 " + o.xLabel);
  for (const t of ticksOf(y0, y1, 5)) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t),
      stroke: "var(--grid)", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: m.l - 8, y: sy(t) + 3.5, "text-anchor": "end", class: "tick" }, [fmt(t)]));
  }
  svg.appendChild(el("text", { x: 12, y: m.t + ih/2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 12 ${m.t + ih/2})` }, [o.yLabel]));
  const step = iw / Math.max(1, o.labels.length);
  const bw = Math.max(2, Math.min(34, step - (step > 12 ? 8 : 1)));
  const colorAt = i => o.colors ? o.colors[i] : (o.color || seriesColor(0));
  o.labels.forEach((label, i) => {
    const v = o.values[i];
    const x = m.l + step * (i + 0.5);
    if (v === null || !isFinite(v)) return;
    const rect = el("rect", { x: x - bw/2, y: sy(v), width: bw, height: Math.max(0.5, sy(y0) - sy(v)),
      rx: bw > 4 ? 3 : 1, style: `fill:${colorAt(i)}` });
    hoverable(rect, `<b>${o.tipLabel || o.xLabel} ${label}</b><br>${o.yLabel}: <b>${fmt(v)}</b>` +
      (o.tipExtra ? o.tipExtra(i) : ""));
    svg.appendChild(rect);
  });
  if (o.refLine !== undefined && o.refLine !== null && isFinite(o.refLine)) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(o.refLine), y2: sy(o.refLine),
      stroke: "var(--text-secondary)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: m.l + iw, y: sy(o.refLine) - 6, "text-anchor": "end", class: "dlabel" },
      [`均值 ${fmt(o.refLine)}`]));
  }
  const every = Math.ceil(o.labels.length / 28);
  o.labels.forEach((label, i) => {
    if (i % every && i !== o.labels.length - 1) return;
    const x = m.l + step * (i + 0.5);
    if (o.rotate) {
      svg.appendChild(el("text", { x, y: m.t + ih + 8, class: "tick", "text-anchor": "end",
        transform: `rotate(-60 ${x} ${m.t + ih + 8})` }, [label]));
    } else {
      svg.appendChild(el("text", { x, y: m.t + ih + 15, class: "tick", "text-anchor": "middle" }, [label]));
    }
  });
  svg.appendChild(el("text", { x: m.l + iw/2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, [o.xLabel]));
  host.appendChild(svg);
}

/* Multi-series lines with a crosshair. The tooltip reads the nearest point of
   every series to the hovered x, so it stays correct whether the series share an
   x grid (steps) or not (tokens per step). */
function lineChart(host, o) {
  const W = o.width || 900, H = o.height || 300;
  const m = { t: 16, r: o.rightPad || 78, b: 46, l: 64 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = o.x;
  let xLo = xs[0], xHi = xs[xs.length - 1];
  if (!(xHi > xLo)) {   // every sample at one x (e.g. a fixed bench): open the axis
    xLo -= Math.max(1, Math.abs(xLo) * 0.1);
    xHi += Math.max(1, Math.abs(xHi) * 0.1);
  }
  const refY = (o.refLines || []).map(r => r.y);
  // A series arrives either as [x, y] pairs (a scatter, e.g. PAR against tokens) or
  // as a flat list of y values aligned to o.x (a metric over steps). Normalise to
  // pairs once, dropping anything not finite: isFinite(null) is true in JS, so a
  // missing sample has to be excluded by hand, and an all-missing series (PAR under
  // a single rank) must still leave a finite axis behind.
  const series = o.series.map(s => {
    const pairs = (s.points.length && Array.isArray(s.points[0]))
      ? s.points
      : s.points.map((v, i) => [o.x[i], v]);
    return { name: s.name, points: pairs.filter(p => p && Number.isFinite(p[0]) && Number.isFinite(p[1])) };
  });
  const [dlo, dhi] = minMax(series.flatMap(s => s.points.map(p => p[1])).concat(refY));
  const lo = isFinite(dlo) ? dlo : 0, hi = isFinite(dhi) ? dhi : 1;
  const pad = (hi - lo) * 0.1 || 1;
  const y0 = Math.min(lo - pad, o.yMin === undefined ? Infinity : o.yMin);
  const y1 = hi + pad;
  const sx = v => m.l + (v - xLo) / Math.max(1e-9, xHi - xLo) * iw;
  const sy = v => m.t + ih - (v - y0) / (y1 - y0) * ih;
  const svg = svgRoot(W, H, o.yLabel + " 随 " + o.xLabel);
  for (const t of ticksOf(y0, y1, 5)) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t),
      stroke: "var(--grid)", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: m.l - 8, y: sy(t) + 3.5, "text-anchor": "end", class: "tick" }, [fmt(t)]));
  }
  svg.appendChild(el("text", { x: 12, y: m.t + ih/2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 12 ${m.t + ih/2})` }, [o.yLabel]));
  for (const r of (o.refLines || [])) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(r.y), y2: sy(r.y),
      stroke: "var(--text-secondary)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: m.l + 4, y: sy(r.y) - 6, class: "dlabel" }, [r.label]));
  }
  // Ticks come from the domain, never the other way round: a tick list used as
  // the domain would start above the smallest sample and push it off the axis.
  for (const t of ticksOf(xLo, xHi, 6)) {
    svg.appendChild(el("text", { x: sx(t), y: m.t + ih + 15, "text-anchor": "middle", class: "tick" }, [fmt(t)]));
  }
  svg.appendChild(el("text", { x: m.l + iw/2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, [o.xLabel]));
  const drawn = [];
  series.forEach((s, i) => {
    const color = seriesColor(i);
    const pts = s.points;
    if (!pts.length) return;
    svg.appendChild(el("path", { fill: "none", style: `stroke:${color}`, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      d: pts.map((p, k) => `${k ? "L" : "M"}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`).join(" ") }));
    if (o.markers !== false && pts.length <= 60) {
      for (const p of pts) svg.appendChild(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.6,
        style: `fill:${color}`, stroke: "var(--surface-1)", "stroke-width": 2 }));
    }
    drawn.push({ s, color, pts });
  });
  if (!drawn.length) {
    // Bare axes look like a broken chart. Say why there is nothing to draw.
    svg.appendChild(el("text", { x: m.l + iw/2, y: m.t + ih/2, "text-anchor": "middle", class: "dlabel" },
      [o.emptyNote || "无数据"]));
  }
  if (drawn.length <= 8) {   // direct labels beside the last point, in series order
    const placed = [];
    for (const { s, color, pts } of drawn) {
      const last = pts[pts.length - 1];
      let y = sy(last[1]) + 3.5;
      while (placed.some(p => Math.abs(p - y) < 12)) y += 12;
      placed.push(y);
      svg.appendChild(el("text", { x: sx(last[0]) + 7, y, class: "dlabel", style: `fill:${color}` }, [s.name]));
    }
  }
  const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: "var(--text-muted)", "stroke-width": 1, opacity: 0 });
  svg.appendChild(cross);
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  hit.addEventListener("pointermove", evt => {
    const box = svg.getBoundingClientRect();
    const svgX = (evt.clientX - box.left) / box.width * W;
    const atX = xLo + (svgX - m.l) / iw * (xHi - xLo);
    const rows = drawn.map(({ s, color, pts }) => {
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - atX) < Math.abs(best[0] - atX)) best = p;
      return { s, color, p: best, d: Math.abs(best[0] - atX) };
    });
    if (!rows.length) return;   // every series was all-missing: nothing to read off
    rows.sort((a, b) => a.d - b.d);
    cross.setAttribute("x1", sx(rows[0].p[0]));
    cross.setAttribute("x2", sx(rows[0].p[0]));
    cross.setAttribute("opacity", 1);
    const body = rows.slice().sort((a, b) => b.p[1] - a.p[1]).map(({ s, color, p }) =>
      `<div><span style="display:inline-block;width:9px;height:9px;border-radius:2px;` +
      `background:${color};margin-right:6px"></span>` +
      `${s.name}: <b>${fmt(p[1], o.tipDigits)}</b></div>`).join("");
    showTip(evt, `<b>${o.xLabel} ${fmt(rows[0].p[0], o.xIntegers ? 0 : undefined)}</b>${body}`);
  });
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
  svg.appendChild(hit);
  host.appendChild(svg);
  if (series.length >= 2) host.appendChild(legendOf(series.map((s, i) => [s.name, seriesColor(i)])));
}

/* Cell grid: sequential by default, categorical when o.colorFor is given. */
function heatmap(host, o) {
  const W = o.width || 1100, H = o.height || 320;
  // A sequential heatmap carries its colour scale in the right margin; a
  // categorical one only needs room for the row labels it does not have there.
  const m = { t: 16, r: o.rightPad || (o.colorFor ? 24 : 156), b: 42, l: 62 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const rows = o.values.length, cols = rows ? o.values[0].length : 0;
  const svg = svgRoot(W, H, o.rowLabel + " 对 " + o.colLabel);
  if (!rows || !cols) { host.appendChild(svg); return; }
  const cw = iw / cols, ch = ih / rows;
  const hi = o.vMax !== undefined ? o.vMax : Math.max(1e-9, ...finite(o.values.flat()));
  const gap = cw > 5 && ch > 5 ? 1.5 : 0;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const v = o.values[r][c];
    const fill = o.colorFor ? o.colorFor(v, r, c) : seqColor(v / hi);
    const rect = el("rect", { x: m.l + c*cw, y: m.t + r*ch, width: Math.max(0.5, cw - gap),
      height: Math.max(0.5, ch - gap), rx: gap ? 1.5 : 0, style: `fill:${fill}` });
    hoverable(rect, o.tip(r, c, v));
    svg.appendChild(rect);
  }
  const rowEvery = Math.ceil(rows / 24), colEvery = Math.ceil(cols / 16);
  for (let r = 0; r < rows; r++) {
    if (r % rowEvery && r !== rows - 1) continue;
    svg.appendChild(el("text", { x: m.l - 7, y: m.t + r*ch + ch/2 + 3.5, "text-anchor": "end",
      class: "tick" }, [o.rowLabels[r]]));
  }
  for (let c = 0; c < cols; c++) {
    if (c % colEvery && c !== cols - 1) continue;
    svg.appendChild(el("text", { x: m.l + c*cw + cw/2, y: m.t + ih + 15, "text-anchor": "middle",
      class: "tick" }, [o.colLabels[c]]));
  }
  svg.appendChild(el("text", { x: 12, y: m.t + ih/2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 12 ${m.t + ih/2})` }, [o.rowLabel]));
  svg.appendChild(el("text", { x: m.l + iw/2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, [o.colLabel]));
  if (!o.colorFor) {  // sequential legend: one hue, light -> dark
    const steps = 40, bw = 120 / steps, bx = m.l + iw + 16, by = m.t;
    for (let i = 0; i < steps; i++) svg.appendChild(el("rect", { x: bx + i*bw, y: by, width: bw + 0.5, height: 9,
      style: `fill:${seqColor(i / (steps - 1))}` }));
    svg.appendChild(el("text", { x: bx, y: by + 23, class: "tick" }, ["0"]));
    svg.appendChild(el("text", { x: bx + 120, y: by + 23, "text-anchor": "end", class: "tick" }, [fmt(hi)]));
    svg.appendChild(el("text", { x: bx, y: by + 38, class: "tick" }, ["token-专家对"]));
  }
  host.appendChild(svg);
}

/* Horizontal-running boxes over steps: median, quartiles, whiskers, extremes. */
function boxChart(host, o) {
  const W = o.width || 640, H = o.height || 210;
  const m = { t: 16, r: 16, b: 52, l: 64 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const [lo, hi] = minMax(o.boxes.flatMap(b => b ? [b.lo, b.hi] : []).concat((o.refLines || []).map(r => r.y)));
  const y0 = Math.min(isFinite(lo) ? lo : 1, o.yMin === undefined ? Infinity : o.yMin);
  let y1 = (isFinite(hi) ? hi : 1) + ((isFinite(hi) ? hi : 1) - y0) * 0.08;
  // No boxes and a yMin that the reference line already pins (PAR >= 1 with no
  // measurements) leaves y1 === y0 and every coordinate 0/0. Open the top only:
  // yMin is a floor the caller asked for, so the axis must not dip below it.
  if (!(y1 > y0)) y1 = y0 + Math.max(0.1, Math.abs(y0) * 0.1);
  const sy = v => m.t + ih - (v - y0)/(y1 - y0)*ih;
  const svg = svgRoot(W, H, o.yLabel);
  for (const t of ticksOf(y0, y1, 4)) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t),
      stroke: "var(--grid)", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: m.l - 8, y: sy(t) + 3.5, "text-anchor": "end", class: "tick" }, [fmt(t)]));
  }
  for (const r of (o.refLines || [])) {
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: sy(r.y), y2: sy(r.y),
      stroke: "var(--text-secondary)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: m.l + 4, y: sy(r.y) - 6, class: "dlabel" }, [r.label]));
  }
  const step = iw / Math.max(1, o.boxes.length);
  const bw = Math.max(10, Math.min(40, step - 22));
  let drawn = false;
  o.boxes.forEach((b, i) => {
    const x = m.l + step * (i + 0.5);
    svg.appendChild(el("text", { x, y: m.t + ih + 10, "text-anchor": "end", class: "tick",
      transform: `rotate(-32 ${x} ${m.t + ih + 10})` }, [o.labels[i]]));
    if (!b) return;
    drawn = true;
    const color = o.colorFor ? o.colorFor(i) : seriesColor(0);
    const g = el("g", {});
    g.appendChild(el("line", { x1: x, x2: x, y1: sy(b.lo), y2: sy(b.hi), style: `stroke:${color}`,
      "stroke-width": 1.5, opacity: .6 }));
    g.appendChild(el("rect", { x: x - bw/2, y: sy(b.q3), width: bw, height: Math.max(1, sy(b.q1) - sy(b.q3)),
      rx: 3, style: `fill:${color}`, "fill-opacity": .85 }));
    g.appendChild(el("line", { x1: x - bw/2, x2: x + bw/2, y1: sy(b.med), y2: sy(b.med),
      stroke: "var(--surface-1)", "stroke-width": 2 }));
    hoverable(g, `<b>${o.labels[i]}</b><br>中位 <b>${fmt(b.med, 3)}</b>` +
      `<br>q1 ${fmt(b.q1, 3)} &middot; q3 ${fmt(b.q3, 3)}` +
      `<br>最小 ${fmt(b.lo, 3)} &middot; 最大 ${fmt(b.hi, 3)}<br><span class="sm">n = ${b.n} 步</span>`);
    svg.appendChild(g);
  });
  svg.appendChild(el("text", { x: 12, y: m.t + ih/2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 12 ${m.t + ih/2})` }, [o.yLabel]));
  if (!drawn) {
    // Labels alone under an axis read as a rendering failure; say what is missing.
    svg.appendChild(el("text", { x: m.l + iw/2, y: m.t + ih/2, "text-anchor": "middle", class: "dlabel" },
      [o.emptyNote || "无数据"]));
  }
  host.appendChild(svg);
}

/* --------------------------------------------------------------- rendering */
let current = 0;
let layerPick = 0;

function kpiCard(key, value, note, hint) {
  const c = div("card");
  if (hint) c.title = hint;
  c.appendChild(div("k", key));
  c.appendChild(div("v", value));
  if (note) c.appendChild(div("n", note));
  return c;
}

/* 口径说明。维度从当前运行的 # meta 里取，所以换模型后数字自动跟着变。 */
function renderGlossary() {
  const r0 = RUNS[current];
  const L = r0.n_layers, K = r0.topk, R = r0.n_ranks, G = r0.global_experts;
  const p = panel("怎么读这份报告 · 术语表",
    "两块长文默认收起，需要时点标题展开（展开状态会保留）。"
    + "每个词、每个数字的口径都在这里：KPI 卡片给结论，正文的图回答「分布长什么样」。"
    + "鼠标停在上面的卡片上也有简述，图和表的每一处都能悬停看精确值。");
  const bar = div("foldbar");
  for (const [label, open] of [["展开全部", true], ["折叠全部", false]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", () => foldAll(open));
    bar.appendChild(b);
  }
  p.appendChild(bar);
  const terms = div("gloss"), howto = div("gloss");
  terms.innerHTML =
    "<table><thead><tr><th>术语</th><th>说明</th></tr></thead><tbody>" +
    "<tr><td><b>rank</b></td><td>一个跑着模型副本的进程（这里 1 卡 = 1 rank）。" +
    "EP 下每个 rank 只持有<b>一部分</b>专家，所以 rank 之间才会有负载差。</td></tr>" +
    "<tr><td><b>MoE 层</b>（moe_layers）</td><td>模型里用了「专家混合」的那几层（本模型 " + L + " 层）。" +
    "每层里一个 token 只走 topk 个专家，而不是走整个 FFN。</td></tr>" +
    "<tr><td><b>专家</b>（expert / global_experts）</td><td>MoE 层里的一个 FFN 分支。" +
    "本模型共 " + G + " 个，报告里一律指<b>逻辑专家</b>（路由空间里的那个 id）；" +
    "某个逻辑专家被复制成多份放在不同 rank 上时，它们仍算<b>同一个专家</b>。</td></tr>" +
    "<tr><td><b>本地专家</b>（local_experts）</td><td>坐在<b>本 rank 上</b>的专家数 = 逻辑专家 ÷ rank 数" +
    "（本例 " + G + " ÷ " + R + " = " + r0.local_experts + "）。" +
    "只有它们要花本 rank 的算力。<b>注意区分「专家是谁」和「谁在算」</b>：" +
    "负载按<b>逻辑专家</b>记（副本算同一个专家），再按「谁持有它」落到各 rank。</td></tr>" +
    "<tr><td><b>冗余专家</b>（redundant）</td><td>EPLB 为削峰而<b>复制</b>出来的专家副本，" +
    "让一个热门专家能被两个 rank 同时扛。为 0 时，「谁持有哪个专家」就是一个划分。" +
    "有副本时报告仍按<b>逻辑专家</b>记负载、并把副本的份额算在其中一个持有者头上" +
    "（因为副本各自接了多少 token 在发送侧看不见），所以此时的 <b>PAR 是上界</b>——" +
    "真实分摊只会让各 rank 更平。</td></tr>" +
    "<tr><td><b>EP</b> / ep_size</td><td>Expert Parallel：把 " + G + " 个专家<b>摊到各 rank</b> 上，" +
    "而不是每个 rank 都放全量。这份报告关心的不平衡是 EP 带来的。</td></tr>" +
    "<tr><td><b>放置</b>（placement）</td><td>「哪个专家坐在哪个 rank」。" +
    "这是 EPLB 唯一能改的东西：token 流向决定倾斜有多大，放置决定这份倾斜落在谁头上。</td></tr>" +
    "<tr><td><b>token</b></td><td>模型处理的最小单位（一个词片）。一个 token 经过<b>每一层</b>都会被路由一次。</td></tr>" +
    "<tr><td><b>topk</b></td><td>每个 token 在每层被送给几个专家（本模型 " + K + "）。</td></tr>" +
    "<tr><td><b>token-专家对</b></td><td>token × 被它选中的专家 = 一次路由。" +
    "<b>这就是本报告的负载单位</b>，与 EPLB 的 moe_load 同口径，<b>不含 padding</b>。</td></tr>" +
    "<tr><td><b>专家负载</b></td><td>某个<b>逻辑专家</b>收到的 token-专家对数。" +
    "副本算同一个专家。</td></tr>" +
    "<tr><td><b>rank 负载</b></td><td>把该 rank 身上的 token-专家对数加起来。" +
    "<b>算谁头上分两套口径</b>：<b>实际计算</b>（主）= 这个 rank 真正算过的行；" +
    "<b>路由归属</b>（参照）= 落到它所持专家上的行。" +
    "EP 下每个 rank 只算自己那部分 token，但 token 会被投递到别的 rank 的专家上——" +
    "落到谁的专家就记谁，<b>不是「这个 rank 发出了多少」</b>；" +
    "而专家被预取到别的 rank 上算时，两套又不一样。见下面两条。</td></tr>" +
    "<tr><td><b>预取槽</b>（prefetch slot）</td><td>MoonEP 允许 rank 在自己家里留几个槽位、" +
    "把<b>别的 rank 的专家</b>预取过来算（dump 里的 <code>experts_to_copy</code>）。" +
    "于是「专家 A 的行」可能在 rank1 上被算，而 expert_map 里 A 的属主仍是 rank0——" +
    "这正是要分开两套口径的原因。</td></tr>" +
    "<tr><td><b>路由归属口径</b></td><td>每一对记在<b>专家属主</b>头上：谁持有这个逻辑专家，" +
    "负载就算谁的；被搬到别的 rank 上算的行<b>仍记属主</b>。" +
    "它回答「放置好不好」——是不做搬迁时的分布，衡量 EPLB 的依据。" +
    "有副本或预取槽时，它在 rank 维度上是<b>上界</b>（真实分摊只会更平）。</td></tr>" +
    "<tr><td><b>实际计算口径</b></td><td>每一对记在<b>算它的那个 rank</b> 头上（接收侧计数）。" +
    "被搬到 rank1 上算的行记 rank1、其余记属主。它回答「屏障在等谁」——性能由它决定。" +
    "<b>它含 pad</b>（pad 也是被算的行），不含被容量包络丢弃的对。" +
    "报告的 rank 级结论以它为主。</td></tr>" +
    "<tr><td><b>pad</b> / padding</td><td>MoonEP 每步把 MoE 输入补齐到池化容量（至少 128 个 token）" +
    "时塞进去的零行，不携带真实 token。<b>它不计入路由量</b>：打点侧在产生它的那个 rank 上、" +
    "按逻辑专家 id 直接从计数里扣掉并单独记下来，汇总值写在 dump 的退出头部" +
    "（<code>pad_pairs_subtracted</code>）。<b>但它计入计算量</b>——零行同样进 matmul、同样占算力。</td></tr>" +
    "<tr><td><b>发送量 / 计算量</b></td><td><code>发送量 = 真实路由 + pad</code>（rank 之间真正递送的行）；" +
    "<code>计算量 = 发送量 − 丢弃</code>（真正进 matmul 的行）。两者都是 token-专家对。" +
    "完整拆分见「计算量分层账本」。</td></tr>" +
    "<tr><td><b>pad 计算量</b></td><td>pad 行贡献的那部分计算量。它不是真实流量，但也不是零成本——" +
    "专家维度能精确拆开（pad 行单独落盘、按逻辑专家 id 记）；" +
    "rank 维度只能按行数比例估（pad 进了槽位就和真实行长得一样，还可能被搬走）。</td></tr>" +
    "<tr><td><b>丢弃</b>（dropped）</td><td>容量包络没能收下的对（超出专家 cap 配额的那部分）。" +
    "只能作为残差得到：<code>丢弃 = 发送量 − 计算量</code>。" +
    "它是「计算量小于发送量」的唯一来源——计算量不会凭空多出来。</td></tr>" +
    "<tr><td><b>步</b>（step）</td><td>一次前向（一次 model execute）。" +
    "<b>一步 = 模型把所有 MoE 层各跑一遍</b>，所以一步产生 " + L + " 行。自回归时一步出一个新 token。</td></tr>" +
    "<tr><td><b>prefill</b> / <b>decode</b></td><td>prefill = 把 prompt 一次性吃进去、" +
    "算出第一个 token（一步可以有上千 token）；decode = 之后每步新增一个 token" +
    "（一步的 token 数 = 还在跑的请求数）。</td></tr>" +
    "<tr><td><b>--batch-size</b></td><td>同时最多跑几个请求 → 决定 <b>decode 一步有多少 token</b>。</td></tr>" +
    "<tr><td><b>--max-num-batched-tokens</b></td><td>一步最多吃多少 token" +
    "（prefill 会被切成这个大小的块）→ 决定 <b>prefill 步有多大</b>。</td></tr>" +
    "<tr><td><b>PAR</b> / 峰均比</td><td>peak-to-average ratio = 最忙的 rank ÷ 平均的 rank。" +
    "<b>1 = 完全均衡</b>，上界 = rank 数。</td></tr>" +
    "<tr><td><b>均衡度</b></td><td>1 ÷ PAR，可读作「最忙那张卡的有效利用率上限」。</td></tr>" +
    "<tr><td><b>EPLB</b></td><td>Expert Parallel Load Balancer：按层把专家在 rank 之间搬动或复制，" +
    "专门削单层尖峰。它优化的正是<b>逐层 PAR</b>。</td></tr>" +
    "<tr><td><b>通信方式</b>（method）</td><td>一步的 token 怎么在 rank 之间递送。" +
    "报告按步记录实际走了哪条路；走哪条<b>由这一步的 token 数决定</b>。" +
    "它只是承载方式，<b>不改变路由结果、也不改变 PAR</b>。" +
    "本仓库会出现：MoonEP (SHMEM)、MC2、all-gather、all-to-all。</td></tr>" +
    "<tr><td>MoonEP (SHMEM)</td><td>本仓库的 EP 通信实现，走共享内存。" +
    "引擎侧开关是 <code>additional_config.enable_moonep</code>（脚本里一般是 <code>--moonep</code>）；" +
    "<b>开着也不等于每步都走它</b>——只有这一步的 token 数不超过 SHMEM 容量时才选它，超了就退回 " +
    "MC2 / all-to-all。哪些步走了它，看「通信方式」那一列。" +
    "（环境变量 <code>MOONEP=1</code> 是启动脚本的约定标记，代码里没有读它的地方，" +
    "它和 <code>enable_moonep</code> 配对出现。）</td></tr>" +
    "<tr><td>MC2</td><td>把通信融进矩阵乘的融合算子（Matmul + Communication）。" +
    "token 数在它容量之内时走它。</td></tr>" +
    "<tr><td>all-gather / all-to-all</td><td>常规集合通信路径：all-gather 把各 rank 的 token 收全，" +
    "all-to-all 按专家投递。token 数超出 MC2 容量时走它们。</td></tr>" +
    "<tr><td><b>dump</b></td><td>打点写出的日志文件，每 rank 一个。" +
    "一行 = 一个（步，层）的计数，就是本报告的全部原始数据。</td></tr>" +
    "<tr><td><b>profile 步</b> / dummy 步</td><td>启动时为了量显存峰值而跑的假前向，没有真实请求。" +
    "打点处已自动滤掉，不在数据里。</td></tr>" +
    "<tr><td><b>中位数 / 均值 / P90</b></td><td>把一串数从小到大排：中位数 = 正中间那个（P50）；" +
    "P90 = 90% 位置那个（只有 10% 的数比它大）；P99 同理。这里的「一串数」对负载口径是<b>各步</b>，" +
    "对性能口径是<b>各请求</b>。" +
    "<b>均值会被极端值拉高，中位数不会</b>，所以报告里两个都给。</td></tr>" +
    "<tr><td><b>TTFT</b>（首 token 时延）</td><td>从请求进入引擎到<b>吐出第一个 token</b> 的时间。" +
    "<b>它含排队时间</b>：离线一次性提交整批 prompt，排在前面的请求先被调度，" +
    "后面的还没开始算就得先等，所以 P90 往往被排队抬高。单位 ms，越小越好。</td></tr>" +
    "<tr><td><b>TPOT</b>（每输出 token 时延）</td><td>吐出第一个 token 之后，" +
    "每多吐一个 token 平均要多久：<code>(最后 token 时刻 − 首 token 时刻) ÷ (输出 token 数 − 1)</code>。" +
    "分母减 1 是因为第一个 token 是 prefill 吐出来的，它的耗时已经算在 TTFT 里了，不重复计。" +
    "它衡量的是 <b>decode 阶段一步一个 token 的速度</b>。单位 ms。</td></tr>" +
    "<tr><td><b>端到端时延</b>（e2e）</td><td>一个请求从进引擎到吐完最后一个 token 的总时长，" +
    "报告按 <code>TTFT + decode 段</code> 相加得到，<b>不是</b>拿两个时刻直接相减——" +
    "请求到达时刻用的是墙上时钟，引擎内部那些时刻用的是另一个单调时钟，两者不能相减。</td></tr>" +
    "<tr><td><b>排队 / prefill 段 / decode 段</b></td><td>把 TTFT 拆开看：" +
    "<b>排队</b> = 引擎收下请求到真正调度它；<b>prefill 段</b> = 调度到吐出第一个 token（读 prompt）；" +
    "<b>decode 段</b> = 首 token 到末 token（逐个吐字）。三者相加 = 端到端时延。" +
    "引擎没打这三个时刻时，报告就不显示它们。</td></tr>" +
    "<tr><td><b>吞吐</b>（tok/s、req/s）</td><td>整段测试里<b>每秒</b>处理掉多少输入 token / " +
    "输出 token / 请求数：总量 ÷ 生成耗时。它是<b>整体</b>指标，看不出单个请求快慢，" +
    "要和逐请求的 TTFT/TPOT 一起读——吞吐高但 P90 差，说明是被少数慢请求拖着的。</td></tr>" +
    "<tr><td><b>请求序号</b></td><td>这一批 prompt 的提交顺序（0 开始）。" +
    "离线跑时所有 prompt 同时提交，序号越大通常排得越靠后、TTFT 越高。</td></tr>" +
    "<tr><td><b>bench.json</b></td><td>跑测试的脚本自己写下的记录（请求数、token 数、耗时、" +
    "逐请求时延）。<b>性能面板的数字全部来自它</b>，不是从负载 dump 推出来的；" +
    "负载 dump 里没有也不能有任何时间信息。目录里没有它就只显示负载部分。</td></tr>" +
    "<tr><td><b>热力图</b></td><td>一格一个数，<b>颜色越深数越大</b>。" +
    "用来一屏看「哪一行（rank）累、哪一列（专家）热」。</td></tr>" +
    "<tr><td><b>箱线图</b> / 四分位</td><td>把一串数排序后，" +
    "箱体底部 = 25% 位置（Q1）、中间线 = 50% 位置（中位数）、顶部 = 75% 位置（Q3）；" +
    "须线伸到观测到的最大/最小值。箱子越短说明这一步接一步地越稳定。</td></tr>" +
    "<tr><td><b>峰值 / 均值</b></td><td>就是 PAR 的算式本身，见上面「PAR / 峰均比」一行；" +
    "图上的 y 轴写这个是为了说明横轴之外的那一维是什么。</td></tr>" +
    "<tr><td><b>expert_map</b> / log2phy</td><td>dump 头部记的专家映射表（每 rank 一层一份）：" +
    "第 i 项 = <b>逻辑</b>第 i 号专家在这块卡上的本地槽位（−1 = 不在这块卡上）。" +
    "两个方向都用得上：本地口径靠它把「本地槽位上的计数」还原成「逻辑专家号」；" +
    "全局口径靠它反过来回答「<b>谁持有这个逻辑专家</b>」，把全局路由归属到各 rank。" +
    "一个逻辑专家在这张表里出现两次（两个 rank 都 ≥ 0）就是副本。</td></tr>" +
    "</tbody></table>";
  howto.innerHTML =
    "<h4>报告地图：想看什么，看哪张图</h4>" +
    "<ul>" +
    "<li><b>口径是什么 / 数据可不可信</b> → 页面最上的「口径与统计 · 先读这里」（T1 口径表 + T2 自查）</li>" +
    "<li><b>token 在 rank 上的分布</b> → 「每个 rank 的负载」+「各 rank 负载随步变化」</li>" +
    "<li><b>token 在专家上的分布</b> → 「rank × 逻辑专家」+「层 × 逻辑专家」+" +
    "「专家负载剖面：单层 / 单步」</li>" +
    "<li><b>专家在 rank 上的分布</b> → 「每个专家住在哪个 rank」</li>" +
    "<li><b>峰均比</b> → 「PAR 随 step」+「各运行的峰均比」+「整个服务过程（全程累计）」</li>" +
    (r0.has_compute
      ? "<li><b>两套口径差在哪 / 总量怎么拆成实际 + pad</b> → " +
        "「口径与统计」的 T0 +「计算量分层账本」（总账、按 rank、按专家三层）</li>"
      : "") +
    "<li><b>一步有多大 / 采集对不对</b> → 「PAR 与每步 token 数的关系」+ 页面顶部的对账横幅</li>" +
    "<li><b>性能：TTFT / TPOT / 吞吐</b> → 「性能：TTFT / TPOT / 吞吐」" +
    "（逐请求的图和表 + 整体的吞吐，以及各运行的对比）</li>" +
    "</ul>" +

    "<h4>dump 里的一行是什么</h4>" +
    "<p>一行 = <b>一步</b>里<b>某一个 MoE 层</b>、<b>某一个 rank</b> 的一串计数。" +
    "<b>一步 = 一次前向</b>（所有 MoE 层各跑一遍），所以一步会产生 " + L + " 行。" +
    "启动期的 profile 步与 dummy 步已被打点处自动过滤，不在数据里。</p>" +
    "<p>行里数列的含义由 dump 头部第 " +
    "<code>counts_scope</code> 决定，报告会自动认：</p>" +
    "<table><thead><tr><th>counts_scope</th><th>一行数的是什么</th><th>谁来用</th></tr></thead><tbody>" +
    "<tr><td><code>global</code></td>" +
    "<td>该 rank 在这一层 dispatch <b>之前</b>、按<b>逻辑专家 id</b> 数的路由直方图，" +
    "宽度 = 全部逻辑专家（" + G + "）。pad 在产生它的 rank 上已按全局 id 扣掉</td>" +
    "<td>MoonEP：这一行是发送侧的真实路由，各 rank 按 (步, 层) 相加即全局路由</td></tr>" +
    "<tr><td><code>global+compute</code></td>" +
    "<td>同一份 dump 里还有一类 <code>scope=compute</code> 的行（还有稀疏的 " +
    "<code>scope=pad</code> 行）：该 rank 在这一层<b>真正算了</b>多少行，" +
    "按<b>逻辑专家 id</b> 排列（本地专家段 + 预取槽段按槽里的专家号摊开），宽同样 = " +
    G + "。<b>含 pad</b></td>" +
    "<td>MoonEP（本报告的主口径）：这就是「谁算了谁背」的那套数；" +
    "主行（<code>scope=global</code>）仍是路由归属那套</td></tr>" +
    "<tr><td><code>local</code></td>" +
    "<td>该 rank 的<b>本地槽位</b>各收到多少（宽 = 本 rank 的专家数），" +
    "靠 expert_map 还原成逻辑专家</td>" +
    "<td>MC2 / all-gather / all-to-all 等，以及旧的 MoonEP dump</td></tr>" +
    "</tbody></table>" +

    "<h4>负载口径：数的是什么，不数什么</h4>" +
    "<p>报告里每个数都按下面这套口径。前三句是所有运行通用的；" +
    "后面两句只在 dump 带 <code>compute</code> 行（MoonEP 两套口径）时才有意义。</p>" +
    "<ul>" +
    "<li><b>专家负载 = 逻辑专家的负载。</b>路由空间里的专家 id 才是「专家」；" +
    "一个专家被复制到多个 rank 时，副本算同一个专家，负载记在原专家名下。" +
    "两套口径下这个数<b>完全相同</b>。</li>" +
    "<li><b>rank 负载按「谁持有这个专家的行」记（路由归属口径，参照）。</b>" +
    "EP 下每个 rank 只算自己那部分 token，但 token 会被投递到别的 rank 的专家上；" +
    "落到谁的专家就记谁。所以它<b>不是</b>「这个 rank 发出了多少」——" +
    "后者会漏掉别的 rank 投过来的部分，实测少算约一半。" +
    "另外，被预取槽搬到别处算的行也仍记属主，所以它是不做搬迁时的分布。</li>" +
    "<li><b>rank 负载也可以按「谁算的这一行」记（实际计算口径，主）。</b>" +
    "接收侧的数就是各 rank 真正算过的行，谁写的算谁的：专家 A 的属主是 rank0，" +
    "但它的 25 行在 rank1 上算，那 25 行记 rank1。<b>性能与屏障只认这一套</b>，" +
    "所以报告的 rank 级结论以它为主。</li>" +
    "<li><b>pad 不计入路由量，但计入计算量。</b>MoonEP 为对齐池化容量塞进来的零行" +
    "（(capacity − T) × topk 个对）由打点侧在产生它的 rank 上扣掉，报告只如实转述扣了多少。 " +
    "扣减位置是<b>逻辑专家空间</b>里的某个 id——pad 行全部路由到本 rank 的某个本地专家，" +
    "在本地槽位空间里扣永远扣不准（这也是旧 dump 对不上账的原因）。" +
    "但 pad 行是真的行：它一样进 matmul、一样占算力，所以在「实际计算」那一套里它<b>保留</b>，" +
    "并单独写成 <code>scope=pad</code> 的行，让报告能把计算量拆成「实际 + pad」。</li>" +
    "</ul>" +
    "<p>为什么能这么数：每个 rank 在 dispatch 之前手里就有完整的路由结果" +
    "（gate 选完专家之后才投递），按逻辑专家数一遍是免费的、<b>不需要任何额外通信</b>；" +
    "接收侧则是投递完成之后的计数，既有 pad、又可能因为配额被搬动或丢弃，" +
    "算「该收」不行、算「实收」正好——两套合起来才是完整的一张账：" +
    "<code>发送量 = 真实路由 + pad</code>，<code>计算量 = 发送量 − 丢弃</code>。" +
    "专家维度这三项逐项精确；rank 维度上「实际计算」的总量精确、" +
    "它内部「多少是 pad」只能按行数比例估（pad 进了槽位就和真实行一样了）。</p>" +
    "<p><b>冗余/预取槽的边界</b>：真的有副本时，副本各自接了多少 token 只体现在接收侧，" +
    "发送侧看不见。路由归属那一套此时把负载记在其中一个持有者头上，不去猜分摊——" +
    "<b>所以是它在 rank 维度上是上界</b>（真实分摊只会更平）；" +
    "实际计算那一套本来就是接收侧的实测，不受这个限制。" +
    "每处相关结论都会标注是哪一套。</p>" +

    "<h4>token-专家对：负载的计量单位</h4>" +
    "<p>一个 token 被路由给 topk 个专家，就产生 <b>topk</b> 个 token-专家对（本模型 topk = " + K + "）。" +
    "每步对数 = token 数 × topk × 层数，逐层对数 = token 数 × topk（与层无关）。" +
    "这里计的是真实路由量，<b>不含 padding</b>，与 EPLB 的 moe_load 同口径。</p>" +

    "<h4>每步 token 数</h4>" +
    "<p>公式：<code>Σ(所有 rank, 所有层) 对数 ÷ (topk × MoE 层数)</code>，" +
    "本次即 <code>Σ对数 ÷ (" + K + " × " + L + ")</code>。" +
    "<b>为什么要除以层数</b>：每个 token 都会经过<b>所有</b> MoE 层，每层都产生 " + K + " 个对，" +
    "只除以 topk 会把 token 数放大 " + L + " 倍。</p>" +
    "<p>这个数的上界就是 <code>--max-num-batched-tokens</code>，纯 decode 步则等于 <code>--batch-size</code>；" +
    "两者一起能反过来验证采集是否正常。</p>" +

    "<h4>三个口径，回答三个不同的问题</h4>" +
    "<p>报告里的 PAR 都是<b>峰均比</b>：<code>最忙的 rank ÷ 平均的 rank</code>。" +
    "三者的区别只在于「把哪些负载先加起来」——在<b>层</b>这个维度聚合，还是连<b>时间</b>也一起聚合。" +
    (r0.has_compute
      ? "这三个口径<b>两套负载各算一遍</b>（实际计算为主、路由归属为参照），" +
        "用的是同一个算式，所以两套的差别只可能来自数据：" +
        "看两套的差就知道搬迁把哪些行挪到了谁头上。"
      : "") + "</p>" +
    "<table><thead><tr><th>口径</th><th>怎么算</th><th>回答什么问题</th></tr></thead><tbody>" +
    "<tr><td><b>逐层 PAR</b></td>" +
    "<td>对<b>每一个 (步, 层)</b> 各算一次 <code>最忙÷平均</code>，" +
    "再对该步内全部 " + L + " 层取平均 → <b>每步得到 1 个数</b></td>" +
    "<td>单个层内部有没有热点。这是 <b>EPLB 自己优化的那个比值</b></td></tr>" +
    "<tr><td><b>整步 PAR</b></td>" +
    "<td><b>只在一步之内</b>把 " + L + " 层<b>加总</b>，再算 <code>峰值 ÷ 均值</code> → " +
    "<b>每步得到 1 个数</b></td>" +
    "<td>这一步整体有多卡 —— 最慢的 rank 在同步屏障前要等多久</td></tr>" +
    "<tr><td><b>全程 PAR</b></td>" +
    "<td>把<b>全部步、全部层都加总</b>，再算 <code>峰值 ÷ 均值</code> → " +
    "<b>整个运行只有 1 个数</b></td>" +
    "<td>整段服务里谁一共做得多。看总账，用于评估放置与规划</td></tr>" +
    "</tbody></table>" +
    "<p><b>注意「整步」只在层维度聚合，不跨步。</b>KPI 卡片上的「整步 PAR」是那串逐步值的<b>中位数</b>" +
    "（下方另外给最大值）；报告不会把整步 PAR 再跨步平均成一个数——那正是「全程 PAR」，两者会混掉。</p>" +
    "<p>三者都落在 <code>[1, rank 数]</code>（本次 " + R + " 个 rank，故上界为 " + R + "），" +
    "<b>1 = 完全均衡</b>。</p>" +

    "<h4>均衡度</h4>" +
    "<p><code>均衡度 = 1 ÷ PAR</code>，即「平均负载 ÷ 峰值负载」，" +
    "可读作<b>最忙那张卡的有效利用率上限</b>：0.877 意味着最忙的卡比理想均衡多做约 14% 的工。</p>" +

    "<h4>为什么口径越大，数字越小</h4>" +
    "<p>逐层 → 整步 → 全程，数值依次下降。原因是各层倾斜的<b>方向并不一致</b>：" +
    "第 3 层可能 rank0 忙、第 17 层可能 rank1 忙，跨层累加时互相抵消；步与步之间同理。" +
    "所以 <b>逐层 PAR 高 ≠ 整体卡</b>。</p>" +
    "<p>该看哪个：<b>判断性能看整步 PAR</b>（屏障是逐步发生的）；" +
    "<b>看 EPLB 收益看逐层 PAR</b>（它要削的是单层尖峰）；" +
    "<b>全程 PAR 只说明活干完时两边是否扯平</b>，它预测不了任何一次等待。</p>" +

    "<h4>PAR 与通信方式无关</h4>" +
    "<p>无论 MC2 还是 all-gather 承载 token，gate 都挑同一批专家。" +
    "PAR 是<b>路由的性质，不是集合通信的性质</b>——这张表里 method 只是记录实际走了哪条路。" +
    "跨运行比较 PAR 时请保证<b>每步 token 数相同</b>，否则比的其实是 batch 组成。" +
    "同一次运行里通信方式可能逐 step 切换（它由 token 数决定），报告按 step 分别记录。</p>" +

    "<h4>这批数据的规模</h4>" +
    "<p>本次：<b>" + R + " 个 rank</b> × <b>" + L + " 个 MoE 层</b>，" +
    "<b>" + G + " 个逻辑专家</b>（每 rank " + r0.local_experts + " 个本地槽位" +
    (r0.redundant ? "，头部记冗余 " + r0.redundant + " 个" : "，头部记冗余 0 个") +
    (r0.duplicated_experts ? "；按 expert_map 数，有 " + r0.duplicated_experts + " 个专家被多个 rank 持有"
                           : "；按 expert_map 数，每个专家只有一个持有者") + "）。" +
    "计数口径为 <code>" + r0.scope + "</code>。" +
    (r0.has_compute
      ? "这份 dump 带两套口径：<b>路由归属</b>的逻辑专家对数和 = 每步 token 数 × topk × 层数（不含 pad），" +
        "各 rank 归属之和 = 同一个总数；<b>实际计算</b>的总量 = 路由归属 + pad − 丢弃，" +
        "各 rank 计算量之和 = 同一个总数（每一行恰好被一个 rank 算）。" +
        "两套的专家维度完全相同。"
      : "<b>逻辑专家</b>对数和 = 每步 token 数 × topk × 层数（不含 pad）；" +
        "<b>各 rank</b> 负载之和 = 同一个总数（每个对恰好归属一个持有者）。") + "</p>";
  p.appendChild(foldSection("terms", "术语表：报告里出现的每个词（"
    + (terms.innerHTML.match(/<tr><td>/g) || []).length + " 条）", terms));
  p.appendChild(foldSection("howto", "怎么读这份报告：报告地图、口径、常见疑问（"
    + (howto.innerHTML.match(/<h4>/g) || []).length + " 节）", howto));
  return p;
}

function renderHeader() {
  document.getElementById("subtitle").textContent =
    `${RUNS.length} 个运行 · ${RUNS.map(r => r.name).join(", ")} · ` +
    `${RUNS[0].model} · 生成于 ${DATA.generated}`;
  const box = document.getElementById("warnings");
  box.innerHTML = "";
  const esc = s => s.replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
  if (DATA.warnings.length) {
    const w = div("warn crit");
    w.innerHTML = "<span>▲</span><span><b>" + DATA.warnings.length +
      ` 条注意事项。</b> ` + DATA.warnings.map(esc).join("<br>") + "</span>";
    box.appendChild(w);
  }
  /* 口径横幅：先讲清这份数据数的是什么，再讲自查结论。 */
  const scoped = RUNS.filter(r => r.primary === "global");
  const mixed = RUNS.filter(r => r.scope === "mixed");
  if (scoped.length) {
    const w = div("warn ok");
    w.innerHTML = "<span>✓</span><span><b>负载口径（" +
      scoped.map(r => esc(r.name)).join("、") + "）：全局发送侧的真实路由，pad 已扣。</b> " +
      "每个 rank 在 dispatch 之前按<b>逻辑专家 id</b> 数一遍路由（宽 " +
      scoped[0].global_experts + " = 全部逻辑专家），分析器按 (步, 层) 把各 rank 加起来，" +
      "就是这一步的真实路由；再按 expert_map 的「谁持有这个专家」把负载归属到各 rank。" +
      "<b>pad 行不计入</b>：它由 dispatcher 在产生它的那个 rank 上、按逻辑专家 id 扣除，" +
      "不是事后估算。" +
      (scoped.some(r => r.has_compute)
        ? "这份 dump 还多记了一套「<b>实际计算</b>」的行（接收侧，含 pad），" +
          "两套数不一样——见下面那条横幅与「口径与统计」。"
        : "") +
      "每一步的完整口径见下面的「口径与统计」。若日志头部写着 counts_scope=global，" +
      "就是这套口径。</span>";
    box.appendChild(w);
  }
  /* 两套口径横幅：这是 MoonEP 打点最容易被误读的地方，单独讲清楚。 */
  const twoview = RUNS.filter(r => r.has_compute);
  if (twoview.length) {
    const names = twoview.map(r => esc(r.name)).join("、");
    const w = div("warn ok");
    w.innerHTML = "<span>✓</span><span><b>两套负载口径（" + names +
      "）：这份 dump 同时记了「路由归属」与「实际计算」。</b> " +
      "同一个 dispatch 被数了两遍：<b>路由归属</b>按「谁持有这个专家」把每一对记在他的属主头上；" +
      "<b>实际计算</b>按「谁算的这一行」记在算它的 rank 头上。" +
      "专家维度两套完全相同（都按逻辑专家记）；rank 维度不同——" +
      "被预取槽搬到别的 rank 上算的那些行，路由归属记属主、实际计算记干活的那个 rank。" +
      "本报告的 rank 级结论<b>以「实际计算」为主、以「路由归属」为参照</b>：前者是屏障真正等的东西，" +
      "后者是不做搬迁时的分布（衡量放置与 EPLB 用）。pad 的归属见「计算量分层账本」。</span>";
    box.appendChild(w);
  }
  if (mixed.length) {
    const w = div("warn crit");
    w.innerHTML = "<span>▲</span><span><b>口径不一致：" + mixed.map(r => esc(r.name)).join("、") +
      "。</b>同一份数据里既有「本地槽位」的行、又有「逻辑专家」的行，" +
      "同一个列号在两种行上指的不是同一个专家，任何求和都无意义。<b>请分开分析</b>，" +
      "或删掉其中一种再重跑。</span>";
    box.appendChild(w);
  }
  const bad = RUNS.filter(r => r.checks.applicable && !r.checks.flat);
  const okr = RUNS.filter(r => r.checks.applicable && r.checks.flat);
  const na = RUNS.filter(r => !r.checks.applicable);
  const padBad = bad.filter(r => r.pad && r.pad.suspect);
  const joinBad = bad.filter(r => !(r.pad && r.pad.suspect));
  if (joinBad.length) {
    const w = div("warn crit");
    w.innerHTML = "<span>▲</span><span><b>逐层自查不通过：" +
      joinBad.map(r => `${esc(r.name)}（层间最大偏差 ${r.checks.max_layer_spread.toExponential(1)}）`).join("、") +
      "。同一步里，各 rank 加起来在每个 MoE 层上计入的 token-专家对数必须相等；" +
      "不相等说明各 rank 的步没有对齐（拼装错了），专家级图表不可信。</span>";
    box.appendChild(w);
  }
  if (padBad.length) {
    const w = div("warn crit");
    w.innerHTML = "<span>▲</span><span><b>逐层自查不通过——成因是 MoonEP 的容量 pad，不是各 rank 的步没对齐：" +
      padBad.map(r => esc(r.name) + "（层间最大偏差 " + r.checks.max_layer_spread.toExponential(1) +
        (r.pad.top_share != null ? "，最忙的那个本地专家占每行的中位数 " + (r.pad.top_share * 100).toFixed(0) + "%" : "") +
        "）").join("、") +
      "。MoonEP 每一步把 MoE 的输入 pad 到池化容量（至少 128 个 token），pad 出来的行全部路由到本 rank 的第一个本地专家，" +
      "这些行不携带真实 token，却和真实路由一起被计进了 (token, 专家) 对数。pad 与层无关、各 rank 也一样多，" +
      "所以各层之和只剩真实路由的那点差别（几个百分点）——偏差小并不代表数据干净：" +
      "<b>绝对负载、峰均比（PAR）和专家分布都被 pad 抬高了</b>，分布图上那根突出的柱子是 pad 而不是热点。" +
      "用带 pad 修正的打点重跑即可（见 docs/moe_par.md 的「MoonEP 的容量 pad」）。</span>";
    box.appendChild(w);
  }
  if (okr.length) {
    const w = div("warn ok");
    w.innerHTML = "<span>✓</span><span><b>逐层自查通过：" +
      okr.map(r => `${esc(r.name)}（${r.checks.steps_checked} 步）`).join("、") +
      "。每一步里，各 rank 加起来在每个 MoE 层上的 token-专家对数都一样（相对偏差 &lt; 1e-6）" +
      "——每个 token 都过所有 MoE 层，所以这个等式本该成立，它验证的是各 rank 的步确实对齐、" +
      "拼装没错位。它<b>不能</b>证明总量没被漏记，那由下面的 bench 对账负责。</span>";
    box.appendChild(w);
  }
  if (na.length) {
    const w = div("warn");
    w.innerHTML = "<span>·</span><span>逐层自查不适用：" +
      na.map(r => esc(r.name)).join("、") + " 只有 1 个 rank，没有可比的两个副本，这项自查无从谈起。</span>";
    box.appendChild(w);
  }
  RUNS.forEach(r => {
    if (!r.bench) return;
    const b = r.bench;
    const isOff = b.verdict === "off";
    const pad = r.pad || {};
    const w = div(isOff ? "warn crit" : (b.verdict === "exact" ? "warn ok" : "warn"));
    const mark = isOff ? "▲" : (b.verdict === "exact" ? "✓" : "·");
    const verdict = b.verdict === "exact"
      ? "完全对上"
      : (isOff ? "对不上" : "略有出入");
    w.innerHTML = `<span>${mark}</span><span><b>与 bench.json 对账（${esc(r.name)}）：${verdict}。</b> ` +
      `dump 推出的 token 总数 <code>${fmt(b.actual, 0)}</code>，` +
      `bench 记的请求口径 <code>${fmt(b.expected, 0)}</code>` +
      "（prompt + output − 每个请求 1 个末尾 token；末尾那个 token 只被采样、不再进模型），" +
      `差 <code>${fmt(b.delta, 0)}</code>。` +
      (isOff
        ? (pad.suspect
          ? "这个差就是 MoonEP 的容量 pad，不是漏记的前向：每个 rank 每步至少把输入 pad 到 128 个 token，" +
            "其中大部分只是 pad、不是真实 token，而它们同样被计进了 (token, 专家) 对数，" +
            "于是 dump 推出的 token 总数约等于 Σ(每步池化容量) × rank 数。pad 的规模随步的 batch 而变、" +
            "事后只能估到几个百分点，所以这份 dump 的绝对负载与峰均比（PAR）不可信；" +
            "用带 pad 修正的打点重跑后，这里应当精确对上。"
          : "这个差值太大，说明有前向没被记录（或 step 计数错位），token 数、负载绝对值都不可信。")
        : (b.verdict === "exact"
          ? "这是唯一一个取自 dump 之外的数——它证明每一次前向都被记到了。"
          : "差额来自退出时没能落盘的那几步：打点侧每 <code>" + (r.flush_every || 16) +
            "</code> 步做一次 host 拷贝并写盘，进程若没跑退出钩子（<code>fork</code> 起的 worker 走 " +
            "<code>os._exit</code>，跳过 atexit），窗口里那几步就丢了。差额 ≤ 窗口步数 × 每步 token 数，" +
            "但具体丢了几步看不出来——用 <code>VLLM_WORKER_MULTIPROC_METHOD=spawn</code>" +
            "（<code>moe_par_bench.py</code> 默认）或 <code>--flush-every 1</code> 重跑即可对上。"))
      + (pad.subtracted
        ? `打点侧已扣除 <code>${fmt(pad.subtracted, 0)}</code> 个 MoonEP 容量 pad pair` +
          (pad.expert >= 0
            ? `（按逻辑专家 id <code>${pad.expert}</code> 在产生它的那个 rank 上扣）`
            : "") + "。"
        : "")
      + "</span>";
    box.appendChild(w);
  });
  const missing = RUNS.reduce((a, r) => a + r.checks.missing_cells, 0);
  if (missing) {
    const w = div("warn");
    w.innerHTML = `<span>▲</span><span>有 ${missing} 个 (rank, step, layer) 单元在 dump 中缺失；` +
      "这些单元不计入均值，而不是当作 0。</span>";
    box.appendChild(w);
  }
}

function renderRunPicker() {
  const host = document.getElementById("runpick");
  host.innerHTML = "";
  RUNS.forEach((r, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = r.name;
    b.setAttribute("aria-pressed", String(i === current));
    b.addEventListener("click", () => { current = i; layerPick = 0; renderAll(); });
    host.appendChild(b);
  });
}

function renderKPIs() {
  const host = div("cards");
  for (const r of RUNS) {
    /* 两套口径的 run：实际计算在前（主），路由归属在后（参照）。两者用的是同一套
       PAR 算式，只换了负载序列，所以卡片除了标题完全对称，差别只在数据。 */
    const lead = r.has_compute ? r.compute : r;
    const views = r.has_compute
      ? [["实际计算", "（主）", lead, r.pad],
         ["路由归属", "（参照）", r, r.pad]]
      : [["", "", r, r.pad]];
    for (const [tag, rankMark, view, pad] of views) {
      const perLayer = quantiles(view.par_by_step), agg = quantiles(view.par_agg);
      const tagS = tag ? " · " + tag + rankMark : "";
      const padNote = pad && pad.subtracted
        ? (tag === "实际计算" ? "含 " + fmt(pad.subtracted, 0) + " 个 pad 对（pad 也进 matmul）。"
                             : "不含 pad（pad 由属主扣掉、另记）。")
        : "";
      /* 计算量的 PAR 恒等于 1：容量对称 + pad 补齐，这个 1 与负载倾斜无关。卡片上是头条数字，
         不加这句话就会被读成「均衡良好」。 */
      const flatNote = (tag === "实际计算" && perLayer && perLayer.med < 1.001)
        ? "<b>" + fmt(perLayer.med, 3) + " 是 pad 补出来的构造结果，不是均衡的证据</b>：" +
          "池化容量对所有 rank 对称，pad 把每个 rank 补齐到同一个数，真实路由的差额被抹平——" +
          "真实倾斜看同一位置的「路由归属」卡片（逐层 PAR " +
          fmt(quantiles(r.par_by_step) && quantiles(r.par_by_step).med, 3) + "）。"
        : "";
      /* 卡片上那个 1.000 太容易被读成「均衡良好」，所以不只在 tooltip 里说，note 这一行也要挂个标记。 */
      const flag = (tag === "实际计算" && perLayer && perLayer.med < 1.001)
        ? " · 由 pad 补齐（非均衡结果）" : "";
      host.appendChild(kpiCard(`${r.name}${tagS} · 逐层 PAR`, fmt(perLayer && perLayer.med, 3),
        (perLayer ? `最大 ${fmt(perLayer.hi, 3)} · 均衡度 ${(1/perLayer.med).toFixed(3)}` : "无数据") + flag,
        (tag === "实际计算"
          ? "主口径：谁算的行记谁（预取槽搬过来的行记在干活的 rank 头上）。"
          : tag === "路由归属"
            ? "参照口径：按「谁持有这个专家」记，被搬走算的行仍记在属主头上，比实际更倾斜。"
            : "") +
        "跨步的中位数。逐层算 最忙rank÷平均rank，再对该步内所有层取平均。" +
        "衡量单个层内部有没有热点，是 EPLB 优化的那个比值。1 = 完全均衡，上界 = rank 数。" +
        "均衡度 = 1÷PAR，可读作最忙那张卡的利用率上限。" + padNote + flatNote));
      host.appendChild(kpiCard(`${r.name}${tagS} · 整步 PAR`, fmt(agg && agg.med, 3),
        agg ? `最大 ${fmt(agg.hi, 3)}` : "无数据",
        (tag === "实际计算" ? "主口径：屏障真正等的就是它。" : "") +
        (tag === "路由归属" ? "参照口径：不做搬迁时的分布。" : "") +
        "跨步的中位数。只在一步内部把 " + r.n_layers + " 层加总，再算 峰值÷均值——" +
        "不是把整个测试加起来。各层不均衡方向不一致会互相抵消，所以它通常明显小于逐层 PAR。" +
        "这个数决定这一步最慢的 rank 要在同步屏障前等多久。"));
      host.appendChild(kpiCard(`${r.name}${tagS} · 全程 PAR`, fmt(view.par_whole, 3),
        `${r.n_steps} 步的总账`,
        "整个服务过程一个数：把每个 rank 在全部步、全部层上的负载加起来，再算 峰值÷均值。" +
        "它回答「总共谁做得多」，比逐步的整步 PAR 更小，因为步与步之间也会互相抵消。" +
        "性能由逐步口径决定，这个数只用于评估放置与规划。"));
    }
    /* 分层账本一行账：总量 = 实际计算 + pad（− 丢弃）。两套口径的 run 才有。 */
    const led = r.ledger;
    if (r.has_compute && led && led.compute_total != null) {
      const dropped = led.dropped || 0;
      host.appendChild(kpiCard(`${r.name} · 每步 token`, fmt(mean(r.tokens_per_step), 0),
        `${r.n_steps} 步 · topk ${r.topk} · 真实 token（不含 pad）`,
        "Σ(所有 rank, 所有层) 对数 ÷ (topk × MoE 层数)。每个 token 经过所有 MoE 层，" +
        "每层都产生 topk 个对，所以要除以层数。上界是 --max-num-batched-tokens，" +
        "纯 decode 步则等于 --batch-size。这里是<b>真实</b> token：pad 行已扣掉。"));
      host.appendChild(kpiCard(`${r.name} · 计算量分层`, fmt(led.compute_total, 0),
        `真实 ${fmt(led.route_total, 0)} + pad ${fmt(led.pad_total, 0)}` +
        (Math.abs(dropped) > 0.5 ? ` − 丢弃 ${fmt(dropped, 0)}` : ""),
        "整段服务的计算量（token-专家对）：<b>总量 = 真实路由 + pad − 丢弃</b>。" +
        "pad 是容量对齐注入的零行，不携带 token 但同样要算；丢弃是容量包络没能收下的对。" +
        "专家维度这三项逐专家精确，rank 维度见「计算量分层账本」。"));
    } else {
      host.appendChild(kpiCard(`${r.name} · 每步 token`, fmt(mean(r.tokens_per_step), 0),
        `${r.n_steps} 步 · topk ${r.topk}`,
        "Σ(所有 rank, 所有层) 对数 ÷ (topk × MoE 层数)。每个 token 经过所有 MoE 层，" +
        "每层都产生 topk 个对，所以要除以层数。上界是 --max-num-batched-tokens，" +
        "纯 decode 步则等于 --batch-size。"));
    }
  }
  const r0 = RUNS[current];
  host.appendChild(kpiCard(`${r0.name} · rank × 专家`, `${r0.n_ranks} × ${r0.global_experts}`,
    `每 rank ${r0.local_experts} 个槽位 · ${r0.n_layers} 个 MoE 层 · 计数口径 ${r0.scope}`,
    "热力图 F5 的形状：一行一个 rank，一列一个逻辑专家，" +
    "格子 = 归到该 rank 的该专家 token-专家对数。" +
    (r0.has_compute
      ? "两套口径都画：实际计算（主，谁算谁背）与路由归属（参照，按属主）。"
      : "（副本只记在一个持有者头上）。") +
    "行和 = 该 rank 的负载。" +
    (!r0.has_compute && r0.duplicated_experts ? "本运行有副本：PAR 是上界。"
      : (!r0.has_compute ? "本运行无副本：每个逻辑专家恰好一个持有者，归属是精确的。" : ""))));
  return host;
}

function panel(title, caption) {
  const p = div("panel");
  p.appendChild(heading("h3", title));
  if (caption) p.appendChild(div("caption", caption));
  return p;
}

function renderPerRank(r) {
  const two = !!r.has_compute;
  const lead = two ? r.compute : r;  // 主口径：实际计算
  const p = panel(two ? "每个 rank 的负载（跨步均值 · 实际计算，主口径）"
                      : "每个 rank 的负载（跨步均值）",
    (two
      ? "各 rank <b>真正算过</b>的 token-专家对（含 pad），跨步取平均——" +
        "谁算的行记谁：被预取槽搬到别的 rank 上算的行记在干活的那个 rank 头上。" +
        "这条序列的峰值与其自身均值之比，就是整步 PAR（主）。表里同时给出路由归属口径作参照。"
      : "各 rank 承载的 token-专家对总数，跨步取平均。这条序列的峰值与其自身均值之比，" +
        "就是整步 PAR。"));
  barChart(p, { labels: r.rank_ids.map(String), values: lead.per_rank_mean, refLine: mean(lead.per_rank_mean),
    color: seriesColor(0), xLabel: "rank", yLabel: "token-专家对", tipLabel: "rank", height: 250,
    tipExtra: i => `<br><span class="sm">峰值步 ${fmt(lead.per_rank_max[i])} · ` +
      `${(lead.per_rank_mean[i] / Math.max(1e-9, mean(lead.per_rank_mean)) / r.n_ranks * 100)
        .toFixed(1)}% 占全程</span>` +
      (two ? `<br><span class="sm">路由归属口径 ${fmt(r.per_rank_mean[i])}（` +
        `${r.per_rank_mean[i] - lead.per_rank_mean[i] >= 0 ? "+" : "−"}` +
        `${fmt(Math.abs(r.per_rank_mean[i] - lead.per_rank_mean[i]))} 净差）</span>` : "") });
  /* 每 rank 的负载是「落在它所持专家上的对数」：token 可能来自任何一个 rank，
     落到谁的专家就记谁。所持专家数直接给出分母，方便看「活多是不是因为专家多」。 */
  const held = r.rank_ids.map(() => 0);
  for (const holders of (r.expert_holders || []))
    for (const j of holders) if (j >= 0 && j < held.length) held[j]++;
  const total = lead.per_rank_mean.reduce((a, b) => a + (b || 0), 0);
  const busiest = Math.max(...finite(lead.per_rank_mean), 1e-9);
  const head = two
    ? ["rank", "所持逻辑专家", "实际计算：步均值", "实际计算：峰值",
       "路由归属：步均值", "净差（计算 − 归属）", "占全程（%）", "相对最忙（×）"]
    : ["rank", "所持逻辑专家", "步均值（token-专家对/步）", "峰值负载（token-专家对）",
       "占全程（%）", "相对最忙（×）"];
  p.appendChild(collapsible("表", table(head,
    r.rank_ids.map((id, i) => {
      const cells = two
        ? [id, held[i], "<b>" + fmt(lead.per_rank_mean[i]) + "</b>", fmt(lead.per_rank_max[i]),
           fmt(r.per_rank_mean[i]),
           (r.per_rank_mean[i] - lead.per_rank_mean[i] >= 0 ? "+" : "−") +
             fmt(Math.abs(r.per_rank_mean[i] - lead.per_rank_mean[i])),
           (100 * lead.per_rank_mean[i] / Math.max(1e-9, total)).toFixed(2) + "%",
           (lead.per_rank_mean[i] / busiest).toFixed(3)]
        : [id, held[i], fmt(lead.per_rank_mean[i]), fmt(lead.per_rank_max[i]),
           (100 * lead.per_rank_mean[i] / Math.max(1e-9, total)).toFixed(2) + "%",
           (lead.per_rank_mean[i] / busiest).toFixed(3)];
      return cells;
    }))));
  p.appendChild(div("caption",
    "「峰值负载」= 该 rank 在最忙那一步的对数（不是步号），它 ÷ 步均值就是该 rank 的逐层 PAR 上界。" +
    "「相对最忙」= 该 rank 的步均值 ÷ 最忙 rank 的步均值：1.000 表示它就是瓶颈，" +
    "0.900 表示它比瓶颈少做 10%（这部分就是指望不上的算力）。" +
    (two
      ? " 两套口径的差别就是搬迁：<b>净差为正</b>说明这个 rank 净多算了别人的行（替人打工）、" +
        "为负说明净搬出去。pad 只出现在实际计算那一列（它也要算），归属那一列不含 pad。"
      : "") +
    (!two && r.duplicated_experts
      ? " 有副本时，副本实际分担走的量在发送侧看不见，所以各 rank 的真实差距只会比这里更小。"
      : "")));
  return p;
}

function renderOverSteps(r) {
  const p = panel("各 rank 负载随步变化",
    "每条线是一个 rank 在该步的总量。最高线与最低线之间的差距，" +
    "就是同步屏障为之付出代价的不均衡量。");
  lineChart(p, { x: r.steps, xLabel: "步", yLabel: "token-专家对", height: 300,
    series: r.rank_ids.map((id, i) => ({ name: "rank " + id, points: r.rank_step_load[i] })) });
  return p;
}

function renderParOverSteps() {
  const host = div("grid2");
  const panels = [
    ["逐层 PAR（EPLB 的定义）", "par_by_step",
     "最忙 rank 的负载除以平均 rank 的负载，再对该步内各层取平均。" +
     "这是 EPLB 优化的那个比值。"],
    ["整步 PAR（一步内所有层加总）", "par_agg",
     "每个 rank 整步的总量，峰值除以均值。这是最慢的那个 rank 实际等待的比值。"]
  ];
  for (const [title, key, caption] of panels) {
    const p = panel(title, caption);
    const maxSteps = Math.max(...RUNS.map(r => r.n_steps));
    const x = Array.from({ length: maxSteps }, (_, i) => i);
    lineChart(p, { x, xLabel: RUNS.every(r => r.n_steps === RUNS[0].n_steps) ? "步" : "步序号",
      yLabel: "峰值 / 均值", height: 290, yMin: 1, markers: false, tipDigits: 3,
      emptyNote: "PAR 至少需要两个 rank（本 dump 只有一个）",
      refLines: [{ y: 1, label: "完全均衡" }],
      series: RUNS.map(r => ({ name: r.name, points: r[key] })) });
    host.appendChild(p);
  }
  return host;
}

function renderParVsTokens() {
  const p = panel("PAR 与每步 token 数的关系",
    "决定一步有多倾斜的是路由（gate），而不是集合通信：无论哪个 all-to-all 承载，" +
    "gate 都挑同一批专家。不同运行之间的差别在于一步里落了多少 token，" +
    "所以真正的差异会体现在这里。");
  const series = RUNS.slice(0, 8).map(r => ({
    name: r.name,
    points: r.tokens_per_step.map((t, i) => [t, r.par_agg[i]])
      .filter(pt => Number.isFinite(pt[0]) && Number.isFinite(pt[1])).sort((a, b) => a[0] - b[0])
  }));
  const span = minMax(series.flatMap(s => s.points.map(pt => pt[0])));
  lineChart(p, { x: isFinite(span[0]) ? span : [0, 1], xLabel: "每步 token 数", yLabel: "整步 PAR",
    height: 300, yMin: 1, markers: true, xIntegers: true, tipDigits: 3,
    refLines: [{ y: 1, label: "完全均衡" }], series });
  if (RUNS.length > 8)
    p.appendChild(div("caption", "只画了前 8 个运行；完整列表见下方表格。"));
  return p;
}

/* 表格，但允许单元格里带标记（<b>/<code>）。单元格内容全部由本文件生成，
   数字都经过 fmt()，没有来自 dump 的自由文本。 */
function richTable(headers, rows) {
  const t = document.createElement("table");
  t.innerHTML = "<thead><tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
  return t;
}

/* 带分组的汇总表：第一行表头是「指标（单位）」，横跨该指标下的各统计量列。
   列数因此可以到十几个而不至于读不出哪一列属于哪个指标——比在每列头上
   重复写一遍单位更清楚。外面套一层可横向滚动的容器。 */
function groupedTable(firstCol, groups, stats, rows) {
  const t = document.createElement("table");
  t.className = "gt";
  t.innerHTML =
    "<thead><tr><th rowspan=\"2\">" + firstCol + "</th>" +
      groups.map(g => `<th colspan="${stats.length}">${g}</th>`).join("") +
    "</tr><tr>" + groups.map(() => stats.map(s => `<th>${s}</th>`).join("")).join("") +
    "</tr></thead><tbody>" +
    rows.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") +
    "</tbody>";
  return scrollWrap(t);
}

function scrollWrap(node) {
  const w = div("tw");
  w.appendChild(node);
  return w;
}

/* 口径徽标：这一份 dump 的计数是按什么口径数的。它决定了后面每张图怎么读，
   所以标题下面第一眼就该看到。 */
function scopeBadge(r) {
  const pad = r.pad || {};
  const node = div("badge");
  if (r.primary === "global") {
    node.innerHTML = "<i></i><span>口径：全局发送侧（真实路由）" +
      (pad.tail_missing
        ? " · pad 已按逻辑专家 id 扣除（dump 无退出汇总行，扣了多少不可知）"
        : (pad.subtracted ? " · pad 已扣 " + fmt(pad.subtracted, 0) + " 对" : " · 本运行无 pad")) +
      (r.has_compute ? " · 另有「实际计算」一套" : "") +
      "</span><small>行宽 " + r.row_width + " = 全部逻辑专家" +
      (r.has_compute ? " · 两套口径，rank 级以「实际计算」为主" : "") + "</small>";
  } else if (r.scope === "mixed") {
    node.className = "badge crit";
    node.innerHTML = "<i></i><span>口径：混合（不可信）</span>" +
      "<small>同一份数据里既有本地槽位、又有逻辑专家的行，含义不确定</small>";
  } else if (pad.suspect) {
    node.className = "badge crit";
    node.innerHTML = "<i></i><span>口径：本地槽位（接收侧）· 含 pad，不可信</span>" +
      "<small>这份 dump 早于 pad 修正，负载被 pad 抬高</small>";
  } else {
    node.className = "badge" + (pad.misplaced ? " warn" : "");
    node.innerHTML = "<i></i><span>口径：本地槽位（各 method 上报的真实计数）" +
      (pad.subtracted ? " · pad 已扣" : "") + "</span><small>行宽 " + r.row_width + " = 本 rank 的专家</small>";
  }
  return node;
}

/* 层间偏差：0 是常态（说明各 rank 的步确实对齐），所以别写成 0.0e+0。 */
function spreadText(c) {
  if (!c.applicable) return "不适用（只有一个 rank）";
  const v = Number(c.max_layer_spread);
  if (!isFinite(v)) return "–";
  if (v === 0) return "0（各层完全一致）";
  if (v < 1e-4) return v.toExponential(1);
  return (100 * v).toFixed(2) + "%";
}

/* 口径与统计：报告里每个数字的定义、算法和自查结果。放在最前面，因为
   下面所有的图都按它画；用户要的是「先解释清楚，再看数」。 */
function renderSemantics(r) {
  const pad = r.pad || {};
  const G = r.global_experts, L = r.n_layers, K = r.topk, R = r.n_ranks, T = r.n_steps;
  const c = r.checks || {};
  const global = r.primary === "global";
  const two = !!r.has_compute;
  const p = panel("口径与统计 · 先读这里",
    "三张表：负载数的是什么、每个数怎么算、这份数据自查过什么。下面的图全部按这里的口径画。");
  p.className = "panel sem";
  p.appendChild(scopeBadge(r));

  /* 两套口径：这份 dump 的核心区别，放在 T1 之前单独讲清楚。 */
  if (two) {
    p.appendChild(heading("h3", "T0 · 两套口径：路由归属 vs 实际计算"));
    p.appendChild(div("caption",
      "MoonEP 把专家的行<b>搬到别的 rank 上算</b>（预取槽），于是「这个专家的负载算谁的」有两种问法，" +
      "两种都对、答案不同。这份 dump 两套都记了，报告也两套都画：" +
      "<b>rank 级结论以「实际计算」为主</b>（屏障等的是它），<b>「路由归属」作为参照</b>" +
      "（它是不做搬迁时的分布，衡量放置与 EPLB 用）。专家维度的两套数完全相同。"));
    p.appendChild(richTable(
      ["口径", "rank 负载怎么记", "回答什么问题", "边界"],
      [["<b>路由归属</b>（参照）",
        "每一对记在该专家<b>属主</b>名下：谁持有这个逻辑专家，负载就算谁的。" +
        "例：专家 A 的属主是 rank0，它的 100 对路由里 25 对是在 rank1 上算的，" +
        "<b>这 100 对全记 rank0</b>。",
        "放置好不好、EPLB 该往哪搬——它是不做搬迁时的分布。",
        "被搬到别的 rank 上算的行也仍记在属主头上，所以它<b>不等于</b>任何一张卡真正算的量；" +
        "有副本时它是上界（副本间怎么分在发送侧看不见）。"],
       ["<b>实际计算</b>（主）",
        "每一对记在<b>算它的那个 rank</b> 头上（接收侧计数，谁写这一行算谁的）。" +
        "上例那 25 对记 rank1、另外 75 对记 rank0。",
        "屏障真正等的是谁——性能由这条序列决定。" + "它含 pad 行（pad 也是被算的行），" +
        "且不含被容量包络丢弃的对。",
        "它把「专家的倾斜」和「放置的倾斜」混在一起（搬迁既可能削峰、也可能把行搬到本就忙的卡上），" +
        "所以它<b>不是</b>放置质量的度量，两者要一起看。"]]));
    p.appendChild(div("caption",
      "<b>两套之间唯一的恒等式</b>：专家维度完全相同（都按逻辑专家记，Σ对数 = 全部真实路由）；" +
      "rank 维度上 <code>实际计算 − (路由归属真实 + 路由归属pad) = 净搬入 − 自己名下被丢弃</code>，" +
      "也就是「这个 rank 净多算了多少行」。两套的差就是搬迁与丢弃，没有第三种来源。" +
      "完整的账见下面的「计算量分层账本」。"));
    /* 这里必须点名「实际计算的 PAR 恒等于 1.000 是 pad 构造出来的」：容量在所有 rank 上对称，
       pad 把每个 rank 补齐到同一个数，于是计算量的 PAR 与负载倾斜无关。不说明这一点，
       读者会把 1.000 读成「平衡良好」——正好把结论读反。 */
    const led0 = r.ledger || {};
    const perRankLayer = (led0.compute_total != null && T > 0 && L > 0 && R > 0)
      ? led0.compute_total / (T * L * R) : null;
    if (perRankLayer != null && led0.compute_total) {
      const cmpL = quantiles(r.compute.par_by_step), runL = quantiles(r.par_by_step);
      const padShare = 100 * (led0.pad_total || 0) / led0.compute_total;
      const realShare = 100 * (led0.route_total || 0) / led0.compute_total;
      p.appendChild(div("caption",
        "<b>读这两套数最容易踩的坑：实际计算那一套的 PAR 恒等于 " + fmt(r.compute.par_whole, 3) +
        "，那是 pad 补出来的构造结果，不是「负载均衡」的证据。</b>" +
        "池化容量对所有 rank 对称（每个 rank 都按同一个容量 pad），pad 把每个 rank 补齐到同一个数：" +
        "本次每个 rank 每层恰好算 <b>" + fmt(perRankLayer, 0) + " 对</b>（容量 × topk），" +
        "所以逐层、整步、全程的 PAR 全是 " + fmt(cmpL ? cmpL.med : null, 3) + "。" +
        "真实路由的差额全被 pad 抹平了——本次 pad 占计算量 <b>" + padShare.toFixed(1) + "%</b>，" +
        "真实路由只占 " + realShare.toFixed(1) + "%，而负载差异恰恰只在那 " + realShare.toFixed(1) + "% 里，" +
        "它只体现在「路由归属」那一套（逐层 PAR 均值 " + fmt(mean(r.par_by_step), 3) +
        (runL ? "、最大 " + fmt(runL.hi, 3) : "") + "，全程 " + fmt(r.par_whole, 3) + "）。" +
        "<b>两套一比就看出 pad 与搬迁各改写掉了多少</b>：pad 把 PAR 从 " + fmt(mean(r.par_by_step), 3) +
        " 拉到 " + fmt(mean(r.compute.par_by_step), 3) + "，剩下的差额才是搬迁与丢弃（逐 rank 见 T2 的净差）。" +
        "要做放置优化、EPLB，看「路由归属」那一套；要回答「屏障在等谁」，看「实际计算」那一套。"));
    }
  }

  const srcExpert = global
    ? "每个 rank 每层 dispatch <b>之前</b>的路由直方图（按逻辑专家 id 计数，宽 " + G +
      "），各 rank 按（步, 层）相加 = 这一步的真实路由"
    : "各 method 上报的本地计数（宽 " + r.row_width + " = 本 rank 的本地槽位），再按 expert_map 还原成逻辑专家";
  p.appendChild(heading("h3", "T1 · 负载口径"));
  p.appendChild(richTable(
    ["量", "定义", "本运行的数据来源", "不含"],
    [["<b>专家负载</b>",
      "该<b>逻辑专家</b>收到的 token-专家对数。一个 token 在每个 MoE 层被路由 topk(" + K +
      ") 次，一对 = 一次专家计算；<b>副本算同一个专家</b>。" +
      (two ? "两套口径下这个数完全相同。" : ""),
      srcExpert, "pad 行"],
     ["<b>rank 负载</b>",
      two
        ? "<b>两套</b>：<b>实际计算</b>（主）= 该 rank 真正算了的行（含 pad）；" +
          "<b>路由归属</b>（参照）= 落到该 rank 所持专家上的真实路由（不含 pad）。" +
          "搬走的行两套的记法不同，见 T0。"
        : "<b>实际落到该 rank 所持专家上</b>的对数——token 来自所有 rank，" +
          "落到谁的专家就记谁；不是「这个 rank 发出了多少」。",
      two
        ? "一套来自发送侧直方图按 expert_map 的归属；另一套来自各 rank 收到的行（scope=compute）"
        : (global ? "上面的全局路由按 expert_map 的「谁持有这个专家」归属到各 rank"
                  : "各 rank 报的计数就是它本地专家收到的量，直接取用"),
      two ? "两套各自见 T0" : "pad 行"],
     ["<b>pad 对数</b>",
      "MoonEP 每步把 MoE 输入 pad 到池化容量（≥128 token）时注入的零行：" +
      "(capacity − T) × topk。它们不携带真实 token，<b>但确实进 matmul</b>——" +
      "所以它是「计算量」的一部分、不是「路由量」的一部分。",
      global ? ("打点侧在<b>产生它的 rank</b> 上、按逻辑专家 id 从计数里直接扣除" +
                (pad.subtracted ? "（本运行已扣 " + fmt(pad.subtracted, 0) + " 对）" : "") +
                "，并单独写成 scope=pad 的行，所以按专家扣得准、按 rank 只算得到属主")
             : ("这份 dump 没有发送侧直方图可用；" +
                (pad.subtracted ? "打点侧已扣 " + fmt(pad.subtracted, 0) + " 对" : "旧 dump 的 pad 无法离线还原")),
      "—— 它不含真实 token，但计入「实际计算」"]]));

  const dup = r.duplicated_experts || 0;
  const spread = spreadText(c);
  const boundary = div("caption");
  if (pad.suspect)
    boundary.innerHTML +=
      "<b>先看这一条</b>：这份 dump 的计数里还混着 MoonEP 的容量 pad（每行最忙的本地专家" +
      (pad.top_share != null ? "平均占 " + (pad.top_share * 100).toFixed(0) + "%" : "占了大头") +
      "，而 dump 头部没有任何扣除记录），所以下面的负载绝对值与 PAR 都被 pad 抬高了。" +
      "以下口径描述的是<b>这套统计该怎么数</b>；要拿到可用的数，请用带 pad 修正的打点重跑。<br>";
  boundary.innerHTML += dup
    ? ("<b>冗余 / 副本的边界（本运行适用）</b>：按 expert_map 统计，有 <b>" + dup +
       " 个逻辑专家同时被多个 rank 持有</b>（头部记 redundant_experts=" + r.redundant + "）。" +
       "专家负载按逻辑专家记，副本算同一个专家；但<b>副本之间各自处理了多少</b>" +
       (two ? "只有「实际计算」那一套看得见，" +
              "「路由归属」那一套看不见（发送侧只有路由结果），它把整个专家的负载记在一个持有者头上。"
            : "没有任何记录，报告把 rank 负载记在持有它的那个 rank 上，不去猜分摊。") +
       (two
         ? "<b>所以：rank 级以「实际计算」为准，它是谁算了谁背的实测值；" +
           "「路由归属」在（且仅在）这一维上是上界——真实分摊只会让各个 rank 更平。</b>"
         : "因此这份运行的 <b>PAR 是上界</b>——真实分摊只会让各个 rank 更平。"))
    : ("<b>冗余 / 副本的边界（本运行不适用，但要说明）</b>：本运行头部 redundant_experts=" + r.redundant +
       "，expert_map 里<b>没有一个逻辑专家同时被两个 rank 持有</b>，所以「谁持有哪个专家」是一个划分：" +
       "某个逻辑专家的负载整个记在它的持有者头上，各 rank 之和 = 全部真实路由，不多不少。" +
       (two
         ? "预取槽仍然会把行搬到别的 rank 上算，那部分差异由「实际计算」那一套精确给出（见 T0）。"
         : "如果换成有副本的放置（MoonEP 的预取槽 / EPLB 冗余），副本之间的分摊在发送侧看不见，" +
           "报告会退到同样的归属口径并标注 PAR 是上界。"));
  p.appendChild(boundary);

  p.appendChild(heading("h3", "T2 · 统计与自查"));
  const flatCell = !c.applicable
    ? "只有 1 个 rank，无从度量离散度"
    : (c.flat ? "通过" : "<b>不通过</b>");
  const benchCell = !r.bench
    ? "目录里没有 bench.json"
    : (r.bench.verdict === "exact" ? "对上" : (r.bench.verdict === "off" ? "<b>对不上</b>" : "略有出入"));
  const padCell = pad.tail_missing
    ? "<b>不可知</b>（dump 里没有退出汇总行）"
    : (pad.subtracted
        ? "已扣 " + fmt(pad.subtracted, 0) + " 对"
        : (pad.suspect ? "<b>未扣除</b>" : (pad.misplaced ? "<b>有落空的扣除</b>" : "本运行无 pad")));
  const ownCell = dup || r.unowned_slots
    ? (dup + " 个专家被多 rank 持有 / " + (r.unowned_slots || 0) + " 个 (层, 专家) 无人持有")
    : "每个逻辑专家恰好一个持有者";
  p.appendChild(richTable(
    ["项目", "怎么算", "本次结果", "判定"],
    [["① 层间平坦",
      "同一步里 <code>Σ_r rank负载[r,t,l]</code> 与层 l 无关——每个 token 都过所有 MoE 层，所以这个和必须平坦",
      "最大相对偏差 " + spread + "（" + (c.steps_checked || 0) + " 步）",
      flatCell],
     ["② bench 对账",
      "<code>Σ对数 ÷ (topk × 层数)</code> 与 <code>Σ(prompt + output − 1)</code> 比" +
      (r.bench ? "：" + fmt(r.bench.actual, 0) + " vs " + fmt(r.bench.expected, 0) : ""),
      r.bench ? ("差 " + fmt(r.bench.delta, 0) + " 对（token-专家对）") : "–",
      benchCell],
     ["③ pad 记账",
      "打点侧在产生 pad 的那个 rank 上扣掉 (capacity − T) × topk，每批落盘后把累计值重写进 dump 末尾的汇总行" +
      "（<code>pad_pairs_subtracted</code>）；分析器不做事后估算",
      padCell + (!pad.tail_missing && pad.expert >= 0 ? "（逻辑专家 " + pad.expert + "）" : ""),
      pad.tail_missing
        ? "<b>不可知</b>"
        : (pad.subtracted && !pad.misplaced ? "通过" : (pad.suspect ? "<b>需重跑</b>" : "通过"))],
     ["④ 归属完备",
      "每个逻辑专家恰好归属一个 rank（副本时取编号最小的持有者）",
      ownCell,
      dup || r.unowned_slots ? "<b>是上界</b>" : "通过"],
     ["⑤ 步的完整性",
      "打点侧在 dump 末尾的汇总行里记下「少于已知层数就结束」的步数（<code>partial_steps</code>）；" +
      "这样的步会把没跑到的层记成 0，进而低估负载",
      (pad.tail_missing ? "<b>不可知</b>（没有退出汇总行）"
        : (r.partial_steps ? "<b>" + r.partial_steps + " 步不完整</b>" : "0 步")) +
      (r.checks.missing_cells ? " · " + r.checks.missing_cells + " 个单元缺失" : ""),
      pad.tail_missing ? "<b>不可知</b>" : (r.partial_steps ? "<b>偏低</b>" : "通过")]]));

  if (pad.subtracted || pad.misplaced || (pad.by_rank || []).some(row => row[1])) {
    p.appendChild(heading("h3", "T6 · pad 记账（每 rank）"));
    const rows = (pad.by_rank || []).map(([rank, pads, sent]) => {
      const counted = sent + pads;
      return [rank, fmt(pads, 0), counted > 0 ? (100 * pads / counted).toFixed(2) + "%" : "–"];
    });
    const totalPads = (pad.by_rank || []).reduce((a, row) => a + row[1], 0);
    const totalSent = (pad.by_rank || []).reduce((a, row) => a + row[2], 0);
    if (rows.length > 1)
      rows.push(["<b>合计</b>", "<b>" + fmt(totalPads, 0) + "</b>",
        (totalSent + totalPads) > 0 ? "<b>" + (100 * totalPads / (totalSent + totalPads)).toFixed(2) + "%</b>" : "–"]);
    p.appendChild(richTable(["rank", "扣除的 pad 对数（token-专家对）", "占该 rank 计数（%）"], rows));
    p.appendChild(div("caption",
      "分母 = 该 rank 自己数过的对（已扣 pad 的真实计数 + 扣掉的 pad）。" +
      "pad 与真实 token 同量级时最容易看出：占比越高，说明这份数据里有多少是空转。"));
  }

  p.appendChild(div("caption",
    "每步对数 = token 数 × topk × 层数，逐层对数 = token 数 × topk（与层无关）——" +
    "所以上表的 ①② 是同一件事的两种对账：① 查各 rank 有没有对齐，② 查总量有没有漏记。"));
  return p;
}

/* 三个 PAR 口径：报告的结论区。它比正文的分布图更该先看，所以数字放大、
   加高亮边，并且放在 KPI 卡片正下方。
   三个口径是同一个负载量的三种加总层次（层 / 步 / 全程），所以用同一条蓝色
   斜升的三阶，而不是三个互不相干的类别色。 */
function renderParScopes(r) {
  const seq = ramp().seq;
  const two = !!r.has_compute;
  /* 两套口径用同一个算式（见 _par_pack），所以这里除了数据没有任何差别：
     先把两套并排画出来，再用一张表把三个口径逐个对照。 */
  const series = [
    { key: "逐层 PAR", pick: v => [mean(v.par_by_step), quantiles(v.par_by_step)], color: seq[11],
      note: v => `逐 (步, 层) 各算一次，再对该步内 ${r.n_layers} 层取平均` +
                 `（中位 ${fmt(v && v.med, 3)} · 最大 ${fmt(v && v.hi, 3)}）`,
      why: "EPLB 要削的就是它：单个层内部的热点。" },
    { key: "整步 PAR", pick: v => [mean(v.par_agg), quantiles(v.par_agg)], color: seq[8],
      note: v => `一步内 ${r.n_layers} 层加总后算 峰值÷均值` +
                 `（中位 ${fmt(v && v.med, 3)} · 最大 ${fmt(v && v.hi, 3)}）`,
      why: "决定这一步最慢的 rank 要在同步屏障前等多久。" },
    { key: "全程 PAR", pick: v => [v.par_whole, null], color: seq[5],
      note: () => `全部 ${r.n_steps} 步、${r.n_layers} 层加总成一个总账`,
      why: "层间、步间方向相反的倾斜都已互相抵消，所以它最小。" },
  ];
  const views = two
    ? [["实际计算（主）", r.compute, "谁算的行记谁：预取槽搬过来的行记在干活的 rank 头上，pad 也在里面。"],
       ["路由归属（参照）", r, "按属主记：被搬走算的行仍记在属主头上，不含 pad。放置与 EPLB 看的分布。"]]
    : [["", r, ""]];
  const p = div("panel hero");
  p.appendChild(heading("h3", "三个 PAR 口径 · 结论先看这里"));
  p.appendChild(div("caption",
    `${r.name}：同一个负载量的三种加总口径，数值依次变小是正常的——口径越大，` +
    "层与层、步与步之间方向相反的倾斜抵消得越充分。" +
    "PAR = 最忙 rank ÷ 平均 rank，是<b>比值、无单位</b>（1.000 表示完全均衡，上界 = rank 数）；" +
    "均衡度 = 1/PAR，同为比值，可读作最忙那张卡的负载占满时的利用率。" +
    "负载本身（各 rank 的对数）单位是 token-专家对。" +
    (two
      ? "<b>这份 dump 有两套负载口径，全部并列给出：rank 级以「实际计算」为主、" +
        "「路由归属」为参照（见上方 T0）。</b>"
      : "") +
    (RUNS.length > 1 ? "切换上方的运行按钮，这一块跟着换。" : "")));
  for (const [label, view, why] of views) {
    if (label) {
      const head = heading("h4", label);
      head.style.marginTop = "4px";
      p.appendChild(head);
      p.appendChild(div("caption", why));
    }
    const base = Math.max(1e-9, ...series.map(s => {
      const v = s.pick(view)[0];
      return isFinite(v) ? v : 0;
    }));
    const grid = div("scopes");
    for (const s of series) {
      const [value, q] = s.pick(view);
      const cell = div("scope");
      cell.style.borderTopColor = s.color;
      cell.appendChild(div("sk", s.key));
      cell.appendChild(div("sv", fmt(value, 3)));
      const bar = div("sbar");
      const fill = document.createElement("i");
      fill.style.width = Math.max(2, 100 * (isFinite(value) ? value : 0) / base).toFixed(1) + "%";
      fill.style.background = s.color;
      bar.appendChild(fill);
      cell.appendChild(bar);
      cell.appendChild(div("sn", s.note(q)));
      cell.appendChild(div("sn", s.why));
      grid.appendChild(cell);
    }
    p.appendChild(grid);
  }
  const perLayerMean = mean(r.par_by_step), aggMean = mean(r.par_agg);
  const bal = v => (isFinite(v) && v > 0 ? (1 / v).toFixed(3) : "–");
  const cmp = view => [
    [mean(view.par_by_step), quantiles(view.par_by_step)],
    [mean(view.par_agg), quantiles(view.par_agg)],
    [view.par_whole, null],
  ];
  const ref = two ? cmp(r) : null, own = cmp(two ? r.compute : r);
  const rows = [];
  series.forEach((s, i) => {
    const row = [s.key,
      s.key === "逐层 PAR"
        ? `每个 (步, 层) 各算一次 最忙 rank÷平均 rank，再对该步内 ${r.n_layers} 层取平均`
        : (s.key === "整步 PAR" ? `每一步内 ${r.n_layers} 层加总后，再算 峰值÷均值`
                                : `全部 ${r.n_steps} 步加总后，再算 峰值÷均值`)];
    if (two) {
      row.push(`<b>${fmt(own[i][0], 3)}</b>（中位 ${fmt(own[i][1] && own[i][1].med, 3)}）`);
      row.push(`${fmt(ref[i][0], 3)}（中位 ${fmt(ref[i][1] && ref[i][1].med, 3)}）`);
      const d = own[i][0] - ref[i][0];
      row.push(isFinite(d)
        ? (Math.abs(d) < 0.0005 ? "≈ 0（无搬迁）"
           : `${d > 0 ? "+" : ""}${d.toFixed(3)}（${d < 0 ? "搬迁削平了" : "搬迁把行搬到了更忙的 rank"}）`)
        : "–");
    } else {
      row.push(`${fmt(own[i][0], 3)}（${r.n_steps} 步的均值，中位 ${fmt(own[i][1] && own[i][1].med, 3)}）`);
    }
    row.push(bal(own[i][0]));
    rows.push(row);
  });
  p.appendChild(table(
    two ? ["口径", "怎么算（负载口径见上方 T0）", "实际计算（主）", "路由归属（参照）", "差（主 − 参照）", "均衡度 = 1/PAR"]
        : ["口径", "怎么算（负载口径见上方 T1）", "本运行", "均衡度 = 1/PAR"],
    rows));
  p.appendChild(div("caption",
    two
      ? "两套数用的是同一个算式，只换了负载序列（见「口径与统计」）。" +
        "<b>差为负 = 搬迁把行从忙的 rank 挪给了闲的，实际计算口径更平</b>；" +
        "差为正 = 有 rank 被搬进了超出它份额的行，那正是屏障要多等的地方。" +
        "「路由归属」那一列不含 pad，实际计算含 pad（pad 也进 matmul），" +
        "所以两者之间还有一个 pad 量级的差——精确的拆分见下面的「计算量分层账本」。"
      : `上表的分子分母都按 T1 的口径数：专家按逻辑专家、rank 按实际落到它所持专家上的对数、不含 pad。` +
        (r.primary === "global" ? "" : " 注意这份 dump 的计数取自各 method 上报的接收侧本地计数。") +
        ` 均衡度 = 1 ÷ PAR，可读作最忙那张卡的有效利用率上限。`));
  const tail = div("caption");
  tail.innerHTML =
    "该看哪个：<b>判断性能看整步 PAR</b>（屏障是逐步发生的）；" +
    "<b>看 EPLB 收益看逐层 PAR</b>（它要削的是单层尖峰）；" +
    "<b>全程 PAR 只说明活干完时两边是否扯平</b>——它最小，但预测不了任何一次等待。" +
    (r.n_ranks < 2 ? " 本次只有 1 个 rank，PAR 无定义。" : "");
  p.appendChild(tail);
  return p;
}

/* 计算量分层账本：把「一个 dispatch 上流动的三个量」按层拆开。
   用户要的是「总量 = 实际计算 + pad」这一层区分，rank 和专家两个维度都要有，
   所以三张表：总账、按 rank、按专家。 */
function renderLedger(r) {
  if (!r.has_compute) return null;
  const led = r.ledger || {};
  if (led.compute_total == null) return null;
  const pad = r.pad || {};
  const L = r.n_layers, G = r.global_experts;
  const p = panel("计算量分层账本 · 总量 = 实际计算 + pad",
    "一个 dispatch 上流动着三个量：<b>真实路由</b>（gate 选中的对）、" +
    "<b>pad</b>（为对齐池化容量塞进去的零行）、<b>丢弃</b>（容量包络没能收下的对）。" +
    "两句话就能串起来：<code>发送量 = 真实路由 + pad</code>、" +
    "<code>计算量 = 发送量 − 丢弃</code>。" +
    "<b>pad 算不算计算量？算。</b>它不携带真实 token，但它和真实行一样进 matmul、" +
    "一样占 rank 的算力，所以它计入「计算量」而不计入「路由量」——" +
    "这正是下面这张账要分开的东西。");
  const sum = a => a.reduce((x, y) => x + y, 0);
  const routeT = led.route_total, padT = led.pad_total, sentT = led.sent_total;
  const compT = led.compute_total, dropT = led.dropped || 0;
  const pct = v => (sentT + compT) > 0 ? (100 * v / Math.max(sentT, compT)).toFixed(2) + "%" : "–";

  p.appendChild(heading("h3", "T1 · 总账（整段服务）"));
  p.appendChild(richTable(
    ["量", "怎么来的", "对数（token-专家对）", "占发送量（%）"],
    [["<b>真实路由</b>", "gate 选中的 (token, 专家) 对；发送侧直方图已把 pad 扣掉，所以它是纯真实流量。",
      fmt(routeT, 0), (100 * routeT / Math.max(1e-9, sentT)).toFixed(2) + "%"],
     ["<b>pad 计算量</b>",
      "MoonEP 为对齐池化容量注入的零行：每 rank 每步 (capacity − T) × topk。" +
      "零行也进 matmul，所以它是计算量的一部分。",
      fmt(padT, 0), (100 * padT / Math.max(1e-9, sentT)).toFixed(2) + "%"],
     ["<b>发送量 = 真实 + pad</b>", "两个 rank 之间真正被递送的行。",
      "<b>" + fmt(sentT, 0) + "</b>", "100.00%"],
     ["<b>丢弃</b>",
      "容量包络下没能进任何槽位的对（排在最热专家后面、超出配额的那部分）。" +
      "只有残差可知：发送量 − 计算量。",
      (dropT >= 0 ? "− " : "+ ") + fmt(Math.abs(dropT), 0), pct(dropT)],
     ["<b>实际计算量 = 发送 − 丢弃</b>",
      "各 rank 真正算过的行。它 = 真实路由 + pad − 丢弃，是屏障为之排队的总工作量。",
      "<b>" + fmt(compT, 0) + "</b>", pct(compT)]]));

  p.appendChild(heading("h3", "T2 · 按 rank：路由归属 vs 实际计算"));
  const byRank = (led.per_rank || []).slice().sort((a, b) => a.rank - b.rank);
  const rows = byRank.map(e => {
    const sent = e.sent, comp = e.compute;
    const net = (sent != null && comp != null) ? comp - sent : null;
    return [e.rank,
      fmt(e.route, 0), fmt(e.pad, 0), sent != null ? "<b>" + fmt(sent, 0) + "</b>" : "–",
      comp != null ? "<b>" + fmt(comp, 0) + "</b>" : "–",
      fmt(e.pad_est, 0), fmt(e.real_est, 0),
      net == null ? "–"
        : (Math.abs(net) < 0.5 ? "≈ 0" : (net > 0 ? "+" : "−") + fmt(Math.abs(net), 0))];
  });
  if (rows.length > 1) {
    const t = k => sum(byRank.map(e => (e[k] == null ? 0 : e[k])));
    rows.push(["<b>合计</b>", "<b>" + fmt(t("route"), 0) + "</b>", "<b>" + fmt(t("pad"), 0) + "</b>",
      "<b>" + fmt(t("sent"), 0) + "</b>", "<b>" + fmt(t("compute"), 0) + "</b>",
      "<b>" + fmt(t("pad_est"), 0) + "</b>", "<b>" + fmt(t("real_est"), 0) + "</b>",
      (t("compute") - t("sent") > 0 ? "+" : "−") + fmt(Math.abs(t("compute") - t("sent")), 0)]);
  }
  p.appendChild(scrollWrap(table(
    ["rank", "路由归属：真实", "路由归属：pad", "归属合计",
      "实际计算：计算量", "实际计算：其中 pad（估）", "实际计算：其中真实（估）", "净差"],
    rows)));
  p.appendChild(div("caption",
    "<b>路由归属那三列是精确的</b>：真实按专家属主归属，pad 也按同一个规则归属" +
    "（pad 行被 dispatcher 记在产生它的 rank 上，而它总是落在这个 rank 自己的专家上），" +
    "所以「归属合计 = 真实 + pad」逐 rank 成立。" +
    "<b>实际计算里「计算量」一列也是精确的</b>——它就是该 rank 写下的那一行行数之和。" +
    "<b>唯一只能估的是它的成分拆解</b>：pad 行一旦进了槽位就和真实行长得一模一样，" +
    "而且可能被搬到别的 rank 上，所以「其中 pad」按 <code>该rank算的该专家行数 ÷ 该专家发送量</code> " +
    "把每个专家的 pad 摊开，剩下的算真实；它的误差上限就是 pad 总量本身（几 %）。" +
    "<b>净差 = 实际计算 − 归属合计</b>：正数表示净搬到这个 rank 上算的行多（它替别人打了工）、" +
    "负数表示净搬出去（或被丢弃）。" +
    (led.dropped != null && Math.abs(led.dropped) > 0.5
      ? " 本次丢弃 " + fmt(led.dropped, 0) + " 对（占发送量 " +
        (100 * led.dropped / Math.max(1e-9, sentT)).toFixed(2) + "%），" +
        "它只出现在「发送了却没算」的属主那一行，所以净差的负数里也含它。"
      : " 本次没有丢弃，净差就纯粹是搬迁的净额。")));

  p.appendChild(heading("h3", "T3 · 按专家（三列全部精确）"));
  /* 专家维度是唯一可以逐项对上的地方：pad 行是按逻辑专家 id 扣的、也是按逻辑专家 id 记的，
     所以「计算 = 真实 + pad − 丢弃」对每个专家都成立，不需要任何估计。 */
  const routed = [], pads = [], computed = [];
  for (let g = 0; g < G; g++) {
    let a = 0, b = 0, c = 0;
    for (let l = 0; l < L; l++) {
      a += r.layer_expert[l][g] || 0;
      b += (r.layer_expert_pad && r.layer_expert_pad[l]) ? (r.layer_expert_pad[l][g] || 0) : 0;
      c += (r.compute.layer_expert && r.compute.layer_expert[l]) ? (r.compute.layer_expert[l][g] || 0) : 0;
    }
    routed.push(a); pads.push(b); computed.push(c);
  }
  const order = routed.map((v, g) => [computed[g], g]).sort((x, y) => y[0] - x[0]);
  const expertRows = order.map(([_, g]) => {
    const drop = routed[g] + pads[g] - computed[g];
    return [g, fmt(routed[g], 0), fmt(pads[g], 0), "<b>" + fmt(computed[g], 0) + "</b>",
      Math.abs(drop) < 0.5 ? "0" : (drop > 0 ? fmt(drop, 0) : "− " + fmt(-drop, 0)),
      computed[g] > 0 ? (100 * pads[g] / computed[g]).toFixed(1) + "%" : "–"];
  });
  p.appendChild(scrollWrap(table(
    ["逻辑专家", "真实路由", "pad", "计算量 = 真实 + pad − 丢弃", "丢弃", "pad 占计算量（%）"],
    expertRows)));
  const hot = order.slice(0, 3).map(([_, g]) => g);
  p.appendChild(div("caption",
    "按计算量从大到小排（可滚动）。这三列全部精确：pad 是按<b>逻辑专家 id</b> 扣的、" +
    "也是按逻辑专家 id 另写成行的，所以每个专家都能逐项对上。" +
    (hot.length ? "本次最忙的三个专家是 " + hot.join("、") + "；" : "") +
    "看这一列就能分清「负载差异是路由倾斜造成的」还是「pad 撑起来的」——" +
    "pad 占比高的那一行，它多出来的量本来就不携带任何 token。"));
  return p;
}

/* 全程口径：把整段测试加起来看总账。它回答「总共谁做得多」，
   而不是「谁在哪一步等」——后者是逐步口径，性能由那个决定。 */
function renderWholeRun(r) {
  const p = panel("整个服务过程（全程累计）",
    `把全部 ${r.n_steps} 步加起来看总账：每个 rank 全程一共扛了多少、最忙的比平均多多少。` +
    "这是评估专家放置、容量规划的指标；每一步实际要等多久，看上面的逐步口径。");
  const total = r.rank_total.reduce((a, b) => a + (b || 0), 0);
  const meanTotal = total / Math.max(1, r.n_ranks);
  barChart(p, {
    labels: r.rank_ids.map(String), values: r.rank_total, refLine: meanTotal,
    color: seriesColor(0), xLabel: "rank", yLabel: "token-专家对（全程累计）",
    tipLabel: "rank", height: 250,
    tipExtra: i => `<br><span class="sm">占全程 ` +
      `${(100 * r.rank_total[i] / Math.max(1e-9, total)).toFixed(2)}% · 平均 ${fmt(meanTotal)}</span>`,
  });
  p.appendChild(div("caption",
    "三个 PAR 口径（逐层 / 整步 / 全程）的对照表在页面顶部的" +
    "「三个 PAR 口径 · 结论先看这里」一块里——这块只讲总账。"));
  return p;
}

function renderRankExpert(r) {
  const p = panel(`token 去了哪里：rank × 逻辑专家（${r.name}）`,
    "本次运行的每一个 token-专家对，按「哪个 rank 算的」和「哪个专家收的」摆开。" +
    "一行的合计就是该 rank 的工作量；一列的分布说明某个专家的流量是" +
    "集中还是分摊。");
  const values = r.rank_expert.map((row, i) => r.rank_expert_seen[i] ? row : row.map(() => null));
  const rowTotal = r.rank_expert.map(row => row.reduce((a, b) => a + b, 0));
  heatmap(p, {
    values, rowLabels: r.rank_ids.map(id => "rank " + id),
    colLabels: Array.from({ length: r.global_experts }, (_, i) => String(i)),
    rowLabel: "rank", colLabel: "逻辑专家 id",
    height: Math.max(150, 110 + r.n_ranks * 22),
    tip: (row, col, v) =>
      `<b>rank ${r.rank_ids[row]} · 专家 ${col}</b><br>token-专家对：<b>${fmt(v)}</b><br>` +
      `<span class="sm">占该 rank 全程的 ${(100 * v / Math.max(1e-9, rowTotal[row])).toFixed(2)}%</span>`
  });
  return p;
}

function renderLayerExpert(r) {
  const grand = r.layer_expert.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0);
  const p = panel(`层 × 逻辑专家（${r.name}）`,
    "同样的流量按层拆开。某层出现一条深色带，说明该层的不均衡主导了逐层 PAR；" +
    "某层在这里是平的，就说明它对此毫无贡献。");
  heatmap(p, {
    values: r.layer_expert,
    rowLabels: Array.from({ length: r.n_layers }, (_, i) => String(i)),
    colLabels: Array.from({ length: r.global_experts }, (_, i) => String(i)),
    rowLabel: "层", colLabel: "逻辑专家 id",
    height: Math.max(240, 130 + r.n_layers * 5),
    tip: (row, col, v) => `<b>层 ${row} · 专家 ${col}</b><br>token-专家对：<b>${fmt(v)}</b><br>` +
      `<span class="sm">占全部 token-专家对的 ${(100 * v / Math.max(1e-9, grand)).toFixed(3)}%</span>`
  });
  return p;
}

/* The two drill-downs below carry the diverging scale: the midpoint is the mean,
   so "warm" is literally "above average" -- which is what PAR measures. */
function renderExpertProfile(r) {
  const p = panel(`专家负载剖面：单层（${r.name}）`);
  const capLead = div("caption");
  capLead.innerHTML =
    "柱高 = 该<b>逻辑专家</b>在这一层收到的 token-专家对数（跨全部步求和，不含 pad）。" +
    "判定列按「相对该层均值」分档：≥1.5× 记热点、≤0.5× 记偏冷、其余正常——" +
    "这是阅读用的分档，不是阈值告警。副本算同一个专家，持有 rank 一栏把副本都列出。";
  p.appendChild(capLead);
  const bar = div("bar");
  bar.appendChild(div("axis-title", "层"));
  const sel = document.createElement("select");
  for (let i = 0; i < r.n_layers; i++) {
    const opt = document.createElement("option");
    opt.value = String(i); opt.textContent = "层 " + i;
    sel.appendChild(opt);
  }
  sel.value = String(Math.min(layerPick, r.n_layers - 1));
  bar.appendChild(sel);
  p.appendChild(bar);
  const cap = div("caption");
  p.appendChild(cap);
  const plot = div();
  p.appendChild(plot);
  const tableHost = div();
  p.appendChild(tableHost);

  function draw() {
    layerPick = Number(sel.value);
    const row = r.layer_expert[layerPick];
    const m = mean(row);
    const [lo, hi] = minMax(row);
    const order = row.map((v, i) => i).sort((a, b) => row[b] - row[a]);
    plot.innerHTML = ""; tableHost.innerHTML = "";
    cap.textContent =
      `第 ${layerPick} 层的逻辑专家负载，跨全部 ${r.n_steps} 步求和。高于该层均值的柱子` +
      "偏暖、低于均值的偏冷；与均值的距离正是峰均比所度量的东西。";
    barChart(plot, {
      labels: order.map(String), values: order.map(i => row[i]), width: 1100, height: 300, rotate: true,
      xLabel: "逻辑专家 id（按负载排序）", yLabel: "token-专家对", tipLabel: "专家", refLine: m,
      colors: order.map(i => divColor(row[i], lo, hi, m)),
      tipExtra: i => `<br><span class="sm">${(row[Number(i)] / Math.max(1e-9, m)).toFixed(2)}× 该层均值` +
        holdersOf(Number(i)) + `</span>`
    });
    /* 专家负载按「逻辑专家」记：副本算同一个专家。这里把持有它的 rank 都列出来，
       读者才能判断这个热点是不是已经摊在多个 rank 上。 */
    tableHost.appendChild(collapsible("表（前 40）", table(
      ["专家", "持有 rank", "对数（token-专家对）", "占该层（%）", "相对均值（×）", "判定"],
      order.slice(0, 40).map(i => {
        const rel = row[i] / Math.max(1e-9, m);
        return [i, holdersOf(i).replace(/^ · /, "") || "—", fmt(row[i]),
          (100 * row[i] / Math.max(1e-9, row.reduce((a, b) => a + (b || 0), 0))).toFixed(2) + "%",
          rel.toFixed(3) + "×", rel >= 1.5 ? "热点" : (rel <= 0.5 ? "偏冷" : "正常")];
      }))));
  }

  function holdersOf(g) {
    const hs = (r.expert_holders || [])[g] || [];
    return hs.length ? " · 持有 rank " + hs.map(j => r.rank_ids[j]).join("/") +
      (hs.length > 1 ? "（副本 " + hs.length + " 份）" : "") : " · 无持有者";
  }
  sel.addEventListener("change", draw);
  draw();
  return p;
}

function renderStepProfile(r) {
  const p = panel(`专家负载剖面：单步（${r.name}）`, "");
  const bar = div("bar");
  bar.appendChild(div("axis-title", "步"));
  const sel = document.createElement("select");
  r.steps.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = String(i); opt.textContent = "步 " + s;
    sel.appendChild(opt);
  });
  sel.value = String(r.n_steps - 1);
  bar.appendChild(sel);
  p.appendChild(bar);
  const cap = div("caption");
  p.appendChild(cap);
  const plot = div();
  p.appendChild(plot);

  function draw() {
    const i = Number(sel.value);
    const row = r.step_expert[i];
    const m = mean(row);
    const [lo, hi] = minMax(row);
    const order = row.map((v, k) => k).sort((a, b) => row[b] - row[a]);
    plot.innerHTML = "";
    cap.textContent = `第 ${r.steps[i]} 步的逻辑专家负载 — ${fmt(r.tokens_per_step[i], 0)} token，` +
      `通信方式 ${r.methods_per_step[i] || "?"}。各柱合计为 ` +
      `${fmt(row.reduce((a, b) => a + b, 0))} 对 = token 数 × topk × ${r.n_layers} 层。`;
    barChart(plot, {
      labels: order.map(String), values: order.map(k => row[k]), width: 1100, height: 300, rotate: true,
      xLabel: "逻辑专家 id（按负载排序）", yLabel: "token-专家对", tipLabel: "专家", refLine: m,
      colors: order.map(k => divColor(row[k], lo, hi, m)),
      tipExtra: k => `<br><span class="sm">${(row[Number(k)] / Math.max(1e-9, m)).toFixed(2)}× 该步均值</span>`
    });
  }
  sel.addEventListener("change", draw);
  draw();
  return p;
}

function renderPlacement() {
  const r = RUNS[current];
  const p = panel("每个专家住在哪个 rank（由负载推断）",
    "一格 = 一个（层，逻辑专家），颜色是「处理了该专家最多流量的那个 rank」。" +
    "没有冗余专家时这是一个划分——每个专家恰好落在一个 rank 上——这正是 EPLB " +
    "会去搬动的形状。它预测均衡性，而不度量均衡性。");
  p.appendChild(div("caption", `展示的是当前选中的运行（${r.name}）。同一个模型下的各次运行共用` +
    "同一套放置，所以看一次就够。"));
  const bar = div("bar");
  bar.appendChild(div("axis-title", "层"));
  const sel = document.createElement("select");
  const optAll = document.createElement("option");
  optAll.value = "all"; optAll.textContent = "全部层";
  sel.appendChild(optAll);
  for (let i = 0; i < r.n_layers; i++) {
    const opt = document.createElement("option");
    opt.value = String(i); opt.textContent = "层 " + i;
    sel.appendChild(opt);
  }
  bar.appendChild(sel);
  p.appendChild(bar);
  const plot = div();
  p.appendChild(plot);
  const note = div("caption");
  p.appendChild(note);

  function draw() {
    const single = sel.value !== "all";
    const rows = single ? [Number(sel.value)] : Array.from({ length: r.n_layers }, (_, i) => i);
    plot.innerHTML = "";
    const values = rows.map(l => Array.from({ length: r.global_experts }, (_, g) => {
      let best = null, bestV = 0, total = 0;
      for (let i = 0; i < r.n_ranks; i++) {
        const v = r.layer_expert[l][g] !== null ? r.rank_expert[i][g] : 0;
        total += v;
        if (v > bestV) { bestV = v; best = i; }
      }
      return total > 0 ? best : null;
    }));
    note.textContent = single
      ? `第 ${rows[0]} 层：颜色是处理该专家最多流量的那个 rank。` +
        "没有冗余时落败者恒为 0，所以它就是归属者。"
      : "全部层。只要没有专家被搬动，每一层都会落到同一行，" +
        "这正是预期的基线。";
    heatmap(plot, {
      values, rowLabels: rows.map(String),
      colLabels: Array.from({ length: r.global_experts }, (_, i) => String(i)),
      rowLabel: "层", colLabel: "逻辑专家 id", height: Math.max(150, 100 + rows.length * 9),
      colorFor: v => v === null ? ramp().empty : seriesColor(v % 8),
      tip: (row, col, v) => v === null
        ? `<b>层 ${rows[row]} · 专家 ${col}</b><br><span class="sm">无流量记录</span>`
        : `<b>层 ${rows[row]} · 专家 ${col}</b><br>最忙 rank：<b>rank ${r.rank_ids[v]}</b><br>` +
          `<span class="sm">由负载读出，不是声明的放置</span>`
    });
    plot.appendChild(legendOf(r.rank_ids.map((id, i) => ["rank " + id, seriesColor(i)])));
  }
  sel.addEventListener("change", draw);
  draw();
  return p;
}

/* 性能：逐请求 + 整体。数字由 bench 脚本在跑测试时量下（引擎自己的时钟），
   负载 dump 里没有任何时间信息，所以这里画的全是 bench.json 带来的东西。
   返回 null 表示这批运行根本没有 bench.json，调用方跳过整块。 */
function renderPerf() {
  const runs = RUNS.filter(r => r.perf);
  if (!runs.length) return null;
  const cur = RUNS[current];
  const perf = cur.perf;
  const has = perf && Object.keys(perf.quantiles).length > 0;
  const p = panel("性能：TTFT / TPOT / 吞吐",
    "负载回答「谁在等」，这里回答「等了多久」。数字来自 bench.json，由引擎自己的时钟量出，" +
    "不是从计数推出来的。吞吐是整体口径，TTFT / TPOT 是逐请求口径，" +
    "两者要一起读：吞吐高但 P90 差，说明整体是被少数慢请求拖着的。");

  if (!has) {
    // 说清楚缺的是哪一层：整个 bench.json 不在，还是它没有逐请求时延。别画空图。
    p.appendChild(div("caption", perf
      ? "这份 bench.json 里没有逐请求时延（旧版 bench 脚本，或起服务时关了 metrics 统计），"
        + "只有整体吞吐可用；要 TTFT / TPOT 就用当前版本的 bench 脚本重跑一次。"
      : `当前运行 ${cur.name} 的目录下没有 bench.json，逐请求时延无从谈起；`
        + "下面只列其他运行（如果它们有）。"));
  }

  // --- 当前运行的磁贴：整体吞吐 + 逐请求分位 ------------------------------- //
  const q = has ? perf.quantiles : {};
  const tp = perf ? perf.throughput : {};
  const cards = div("cards");
  if (perf) {
    cards.appendChild(kpiCard(`${cur.name} · 输出吞吐`, tp.output_tok_s === undefined ? "–" : fmt(tp.output_tok_s, 1),
      `${count(perf.output_tokens)} 输出 token / ${fmt(perf.generate_s, 1)} s`,
      "整体口径：这一段测试平均每秒吐出多少输出 token。它是把全部请求的总量除以生成耗时，" +
      "看不出单个请求快慢。"));
    cards.appendChild(kpiCard(`${cur.name} · 输入吞吐`, tp.input_tok_s === undefined ? "–" : fmt(tp.input_tok_s, 1),
      `${count(perf.prompt_tokens)} 输入 token`,
      "整体口径：每秒吃掉多少 prompt token。它和输出吞吐之差反映 prefill 与 decode 的比重。" +
      "纯 prefill 的负荷看这个数，纯 decode 的负荷看输出吞吐。"));
    cards.appendChild(kpiCard(`${cur.name} · 请求速率`, tp.req_s === undefined ? "–" : fmt(tp.req_s, 2),
      `${count(perf.n_requests)} 个请求`,
      "整体口径：每秒完成多少个请求。它被最长的那几个请求拖着，" +
      "所以个位数和吞吐一起看才能判断整批是不是被少数慢请求压住。"));
  }
  if (has) {
    cards.appendChild(kpiCard(`${cur.name} · TTFT P50`, fmt(q.ttft_ms && q.ttft_ms.p50, 1) + " ms",
      q.ttft_ms ? `P90 ${fmt(q.ttft_ms.p90, 1)} · P99 ${fmt(q.ttft_ms.p99, 1)} · ` +
        `均值 ${fmt(q.ttft_ms.mean, 1)} · 最大 ${fmt(q.ttft_ms.max, 1)}` : "无数据",
      "首 token 时延的中位数。它含排队时间：离线整批提交时，排后面的请求还没开始算就得先等。" +
      "P90 / P99 / 最大 一起看排队把尾巴抬了多少，均值则被中间那批请求主导——" +
      "两者差距越大，说明越少数的请求在拖后腿。"));
    cards.appendChild(kpiCard(`${cur.name} · TPOT P50`, q.tpot_ms ? fmt(q.tpot_ms.p50, 2) + " ms" : "–",
      q.tpot_ms ? `P90 ${fmt(q.tpot_ms.p90, 2)} · P99 ${fmt(q.tpot_ms.p99, 2)} · ` +
        `均值 ${fmt(q.tpot_ms.mean, 2)} · 共 ${q.tpot_ms.n} 个请求` : "无数据",
      "每输出 token 时延的中位数，只算 decode 段，与 TTFT 无关。" +
      "它直接反映 decode 一步一个 token 要多久——MoE 通信方式的好坏主要体现在这个数上。" +
      "TPOT 各分位通常挨得很近，一旦 P99 明显高出，多半是某几步的 batch 组成变了。"));
    cards.appendChild(kpiCard(`${cur.name} · 端到端 P50`, q.e2e_ms ? fmt(q.e2e_ms.p50, 0) + " ms" : "–",
      q.e2e_ms ? `P90 ${fmt(q.e2e_ms.p90, 0)} · P99 ${fmt(q.e2e_ms.p99, 0)} · ` +
        `均值 ${fmt(q.e2e_ms.mean, 0)}` : "无数据",
      "一个请求从进引擎到吐完最后一个字的总时长 = TTFT + decode 段。" +
      "它受输出长度影响极大（本批每请求 " + count(mean(perf.requests.map(r => r.output_tokens))) +
      " 个 token），所以跨运行比较时 TPOT 比它更干净。"));
  }
  if (perf) p.appendChild(cards);

  // --- 各运行对比：吞吐表 + 时延汇总表 + 时延分布 -------------------------- //
  {
    const trows = runs.map(r => {
      const rr = r.perf;
      return [r.name, count(rr.n_requests),
        fmt(rr.throughput.input_tok_s, 1), fmt(rr.throughput.output_tok_s, 1),
        fmt(rr.throughput.req_s, 2), fmt(rr.generate_s, 1)];
    });
    p.appendChild(collapsible("表：各运行整体对比（吞吐）", table(
      ["运行", "请求数（个）", "输入吞吐（tok/s）", "输出吞吐（tok/s）",
       "请求速率（req/s）", "生成耗时（s）"], trows), true));
    /* 时延汇总：三个量 × 四个统计量。P99 与均值此前只在逐请求明细里能自己算，
       现在直接给出。单位写在分组表头里，不在每个数字后面重复。 */
    const LAT = [["ttft_ms", 1], ["tpot_ms", 2], ["e2e_ms", 0]];
    const lrows = runs.map(r => {
      const rq = r.perf.quantiles;
      return [r.name, ...LAT.flatMap(([key, digits]) => {
        const s = rq[key];
        return [fmt(s && s.p50, digits), fmt(s && s.p90, digits),
          fmt(s && s.p99, digits), fmt(s && s.mean, digits)];
      })];
    });
    p.appendChild(collapsible("表：各运行时延汇总（ms）", groupedTable(
      "运行", ["TTFT (ms)", "TPOT (ms)", "端到端 (ms)"], ["P50", "P90", "P99", "均值"], lrows), true));
    p.appendChild(div("caption",
      "换了通信方式只改通信，路由与 batch 组成不变，所以这两张表就是性能对比的主表。" +
      "相比吞吐，<b>TPOT</b> 最能分离出通信方式的影响：它只含 decode 段，" +
      "而这正是 MoE all-to-all / MC2 / MoonEP 每一步都要走的那一段。" +
      "P50 → P99 的抬升幅度看的是尾延迟：离线整批提交时，排后面的请求要等前面算完，" +
      "所以 TTFT 的尾巴天然比 TPOT 长。每个请求的原始值在下面的箱线图和逐请求明细表里。" +
      "注意各运行的每步 token 数与输出长度必须一致，否则这几列不可比。"));
    if (runs.some(r => r.perf.quantiles.ttft_ms)) {
      const grid = div("grid2");
      for (const [key, title] of [["ttft_ms", "TTFT 的逐请求分布"], ["tpot_ms", "TPOT 的逐请求分布"]]) {
        const cell = div();
        cell.appendChild(heading("h3", title));
        boxChart(cell, {
          labels: runs.map(r => r.name),
          boxes: runs.map(r => quantiles(r.perf.requests.map(x => x[key]))),
          colorFor: i => seriesColor(i), yLabel: key === "ttft_ms" ? "TTFT (ms)" : "TPOT (ms)",
          height: 210, width: 640, emptyNote: "这个运行没有逐请求时延数据" });
        grid.appendChild(cell);
      }
      p.appendChild(grid);
      p.appendChild(div("caption",
        "箱体 = 四分位，须线 = 观测到的最小 / 最大值。" +
        "TTFT 的箱子通常比 TPOT 长得多：TTFT 里混着排队和 prefill（长 prompt 更贵），" +
        "TPOT 只含 decode，所以它更短、更稳，也更能代表通信方式本身。"));
    }
  }

  // --- 当前运行的逐请求曲线 ---------------------------------------------- //
  if (has && perf.requests.length) {
    const idx = perf.requests.map(r => r.idx);
    const ttft = perf.requests.map(r => (r.ttft_ms === undefined ? null : r.ttft_ms));
    const tpot = perf.requests.map(r => (r.tpot_ms === undefined ? null : r.tpot_ms));
    const h3 = heading("h3", "逐请求 TTFT 与 TPOT");
    p.appendChild(h3);
    lineChart(p, { x: idx, xLabel: "请求序号（提交顺序）", yLabel: "毫秒", height: 300,
      xIntegers: true, tipDigits: 2, markers: perf.requests.length <= 60,
      refLines: q.ttft_ms ? [{ y: q.ttft_ms.p50, label: "TTFT P50" }] : [],
      series: [{ name: "TTFT (ms)", points: ttft }, { name: "TPOT (ms)", points: tpot }] });
    p.appendChild(div("caption",
      "横轴是提交顺序，所以一条向上翘的 TTFT 曲线就是排队——" +
      "离线整批提交时后发的请求要等前面的算完。TPOT 一般是平的：" +
      "它只跟 decode 步里落了多少 token 有关，与排队几乎无关。" +
      "两个序列单位都是毫秒，所以可以放在同一根轴上；TPOT 通常在几十毫秒量级，" +
      "被 TTFT 压扁是正常的，读它的绝对值请用上面的磁贴或表。"));
    p.appendChild(collapsible("表：逐请求明细", table(
      ["序号", "输入 token（个）", "输出 token（个）", "TTFT（ms）", "TPOT（ms）",
       "端到端（ms）", "排队（ms）", "prefill（ms）", "decode（ms）"],
      perf.requests.map(r => [r.idx, count(r.prompt_tokens), count(r.output_tokens),
        fmt(r.ttft_ms, 1), fmt(r.tpot_ms, 2), fmt(r.e2e_ms, 0),
        fmt(r.queue_ms, 1), fmt(r.prefill_ms, 1), fmt(r.decode_ms, 0)]))));
  }
  return p;
}

function renderComparison() {
  const p = panel("各运行的峰均比",
    "各运行 PAR 随步的分布：左边是逐层，右边是聚合。" +
    "箱体是中位数与四分位；须线到观测到的最小值和最大值。");
  const grid = div("grid2");
  for (const [key, title] of [["par_by_step", "逐层 PAR"], ["par_agg", "整步 PAR"]]) {
    const cell = div();
    cell.appendChild(heading("h3", title));
    boxChart(cell, { labels: RUNS.map(r => r.name), boxes: RUNS.map(r => quantiles(r[key])),
      colorFor: i => seriesColor(i), yLabel: "峰值 / 均值", height: 210, yMin: 1, width: 640,
      emptyNote: "PAR 至少需要两个 rank", refLines: [{ y: 1, label: "完全均衡" }] });
    grid.appendChild(cell);
  }
  p.appendChild(grid);
  p.appendChild(collapsible("表", table(
    ["运行", "步数", "逐层 PAR 均值（×）", "中位（×）", "最大（×）",
      "整步 PAR 均值（×）", "最大（×）", "每步 token（个）"],
    RUNS.map(r => {
      const a = quantiles(r.par_by_step), b = quantiles(r.par_agg);
      return [r.name, r.n_steps, fmt(a && mean(r.par_by_step), 3), fmt(a && a.med, 3), fmt(a && a.hi, 3),
        fmt(b && mean(r.par_agg), 3), fmt(b && b.hi, 3), fmt(mean(r.tokens_per_step), 0)];
    })), true));
  p.appendChild(div("caption",
    "PAR 是「最忙 rank ÷ 平均 rank」，均衡度 = 1/PAR，两者都是比值、<b>没有单位</b>，" +
    "1.000× 表示完全均衡、上界是 rank 数；表里「每步 token」的单位是个（一个 token 算一个）。" +
    "PAR 是路由的性质，不是集合通信的性质——在每步 token 数相同时，" +
    "各运行应当彼此接近。若差距明显大于此，问题出在 batch 组成上，" +
    "而不是通信方式上。"));
  return p;
}

function renderTables() {
  const p = panel("数据", "图表所画的全部内容，完整列出。" +
    "单位：写着「token-专家对」的列都指同一件事——某个 token 在某一层被路由到某个专家一次" +
    "（逐步表与每 rank 表的各 rank 列、逐层表的「对数」列都是它）；PAR 与占比是比值，无单位。");
  for (const r of RUNS) {
    const rows = [];
    for (let si = 0; si < r.n_steps; si++) {
      rows.push([r.steps[si], r.methods_per_step[si] || "?", fmt(r.tokens_per_step[si], 0),
        fmt(r.par_by_step[si], 4), fmt(r.par_agg[si], 4),
        ...r.rank_step_load.map(col => fmt(col[si]))]);
    }
    p.appendChild(collapsible(`${r.name} — 逐步`,
      table(["步", "通信方式", "token（个）", "逐层 PAR（×）", "整步 PAR（×）",
        ...r.rank_ids.map(id => "rank " + id)], rows)));
  }
  for (const r of RUNS) {
    const totals = r.rank_expert.map(row => row.reduce((a, b) => a + b, 0));
    const grand = totals.reduce((a, b) => a + b, 0);
    p.appendChild(collapsible(`${r.name} — 每 rank`, table(
      ["rank", "对数（token-专家对）", "占比（%）", "步均值（token-专家对/步）", "峰值负载（token-专家对）"],
      r.rank_ids.map((id, i) => [id, fmt(totals[i]), (100 * totals[i] / Math.max(1e-9, grand)).toFixed(2) + "%",
        fmt(r.per_rank_mean[i]), fmt(r.per_rank_max[i])]))));
  }
  for (const r of RUNS) {
    const rows = [];
    for (let l = 0; l < r.n_layers; l++) {
      const row = r.layer_expert[l];
      const total = row.reduce((a, b) => a + b, 0);
      const top = row.map((v, g) => [v, g]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      rows.push([l, fmt(total), fmt(mean(r.par_layer.map(col => col[l])), 4),
        top.map(([v, g]) => `${g} (${(100 * v / Math.max(1e-9, total)).toFixed(1)}%)`).join(", ")]);
    }
    p.appendChild(collapsible(`${r.name} — 逐层`,
      table(["层", "对数（token-专家对）", "逐层 PAR 均值（×）", "最热专家（占比）"], rows)));
  }
  return p;
}

function renderRun(r) {
  const host = div();
  host.appendChild(heading("h2", `运行：${r.name} — ${r.model}`));
  host.appendChild(scopeBadge(r));
  host.appendChild(div("caption",
    `${r.n_ranks} 个 rank × ${r.n_layers} 个 MoE 层 × ${r.n_steps} 步 · topk ${r.topk} · ` +
    `${r.global_experts} 个逻辑专家` +
    (r.local_experts > 0 ? `，每 rank 持有 ${r.local_experts} 个` : "") +
    (r.redundant ? `，冗余 ${r.redundant} 个` : "") +
    (r.duplicated_experts ? `，${r.duplicated_experts} 个专家被多 rank 持有` : "") +
    (r.notes.length ? " · " + r.notes.join(" · ") : "")));
  const grid = div("grid2");
  grid.appendChild(renderPerRank(r));
  grid.appendChild(renderOverSteps(r));
  host.appendChild(grid);
  host.appendChild(renderParOverSteps());
  host.appendChild(renderWholeRun(r));
  if (r.n_ranks > 1) host.appendChild(renderRankExpert(r));
  host.appendChild(renderLayerExpert(r));
  const drill = div("grid2");
  drill.appendChild(renderExpertProfile(r));
  drill.appendChild(renderStepProfile(r));
  host.appendChild(drill);
  host.appendChild(renderPlacement());
  return host;
}

function renderAll() {
  renderRunPicker();
  const host = document.getElementById("report");
  host.innerHTML = "";
  /* 口径与统计放最前：后面每一张图都按它画，先读口径再读结论。 */
  host.appendChild(renderSemantics(RUNS[current]));
  host.appendChild(renderKPIs());
  host.appendChild(renderParScopes(RUNS[current]));
  const ledger = renderLedger(RUNS[current]);
  if (ledger) host.appendChild(ledger);
  host.appendChild(renderGlossary());
  host.appendChild(renderRun(RUNS[current]));
  const perf = renderPerf();
  if (perf) host.appendChild(perf);
  if (RUNS.length > 1) host.appendChild(renderParVsTokens());
  host.appendChild(renderComparison());
  host.appendChild(renderTables());
}

document.getElementById("theme").addEventListener("click", () => {
  const next = darkNow() ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  document.getElementById("theme").textContent = next === "dark" ? "浅色" : "深色";
  hideTip();
  renderAll();
});
window.addEventListener("resize", hideTip);
if (RUNS.length) {
  document.getElementById("theme").textContent = darkNow() ? "浅色" : "深色";
  renderHeader();
  renderAll();
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
