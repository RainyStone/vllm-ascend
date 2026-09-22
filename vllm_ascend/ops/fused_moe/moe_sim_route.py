# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Synthetic expert-selection injection, for routing-skew performance experiments.

The real router cannot be constrained -- that is the point of the experiment this
module serves. ``VLLM_ASCEND_MOE_SIM_ROUTE`` hands ``select_experts`` a per-expert
target distribution and the router's output is replaced wholesale, so a sweep over
expert-selection skew becomes an *input* knob while rank-level compute and comm cost
stay the *measured* quantity.

Contract (see ``experts_selector.select_experts``): return ``(topk_weights, topk_ids)``,
both shaped ``[T, K]`` on the input device, ``topk_ids`` int32 in **logical** expert
space ``[0, E)``. The logical->physical ``log2phy`` remap happens downstream in
``moe_comm_method.fused_experts``, so ids must NOT be pre-mapped here. The weights are
a flat ``1/K``: they scale the combine's arithmetic, never its communication pattern,
and a synthetic run has no meaningful gate values to carry.

Three properties the caller depends on:

* **Distinct per row.** A real ``torch.topk`` never returns the same expert twice for
  one token, so no expert can take more than ``T`` routes. That bounds every target
  entry at ``100/K`` percent and is enforced, not merely documented -- it is also why
  a real run's measured expert skew (42x of mean at E=256) already sits at the 1/K
  ceiling and cannot be pushed further.
* **Exact totals.** ``sum(counts) == T * K`` always, via largest-remainder apportionment.
  Rounding each entry independently would silently drop routes.
* **Layer-fixed.** The result is cached on ``T``, so all MoE layers of one forward and
  every same-length step share one allocation. ``select_experts`` sees no step counter;
  keying on ``T`` sidesteps needing one and makes a run bit-reproducible.

The startup dummy forward (``in_profile_run``) is skipped entirely. That forward exists
only to size the KV cache, and letting it see a skewed allocation would give each leg a
different cache size and destroy cross-leg comparability.
"""

from __future__ import annotations

import math
import os

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

_PERCENT_TOTAL = 100.0
_SUM_TOLERANCE = 1e-6
# Absolute slack, in percentage points, on the per-expert cap. See _validate.
_CAP_SLACK = 0.01

# Cache: (T, device, dtype) -> (topk_weights, topk_ids, alloc_id). Keyed on the shape/device
# the callers actually use; a hit means this forward shares the previous allocation. The
# alloc id travels with the tensors so the reuse line can name the table being reused --
# the module-global counter names the newest table, which after a T switch belongs to a
# different token count.
# Bounded: served traffic produces a new T constantly (every prefill chunk frontier,
# every decode batch-size change), and an entry costs ~36*T bytes per rank, so an
# unbounded cache would grow into the hundreds of MB across a long run. Evicting the
# oldest only costs a recompute, never correctness.
_CACHE_MAX = 64
_CACHE: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, int]] = {}
_PROFILE: list[float] | None = None
_PROFILE_MISS = False
_VALIDATED_FOR: tuple[int, int] | None = None
_SEED = 0
_TABLE = True
_EVERY = 43

_LAST_T: int | None = None
_SINCE_FULL = 0
_ALLOC_ID = 0
_EP_WORLD: int | None = None
_MIX_WARNED = False


def active() -> bool:
    """Whether injection is configured. Reads the env lazily, not at import.

    A served engine exports this before ``vllm serve`` and the workers inherit it via
    spawn, so an import-time read would work too -- but a lazy read also survives the
    offline ``moe_par_bench.py`` path, which sets env programmatically.
    """
    return bool(envs.VLLM_ASCEND_MOE_SIM_ROUTE.strip())


def _parse_spec(spec: str) -> list[float] | dict[int, float]:
    """Parse the profile. Three accepted spellings.

    ``@/path/to/file``  -- the file holds the list (whitespace/comma/newline separated,
                           ``#`` starts a comment). Preferred for E=256.
    ``0:50,7:30``       -- sparse: unlisted experts split whatever is left of 100 evenly.
    ``50,30,1,1,...``   -- the full list, one entry per expert.
    """
    if spec.startswith("@"):
        with open(spec[1:].strip()) as handle:
            spec = "\n".join(line.split("#", 1)[0] for line in handle)

    if ":" in spec:
        entries: dict[int, float] = {}
        for part in spec.replace("\n", ",").split(","):
            part = part.strip()
            if not part:
                continue
            index_text, _, pct_text = part.partition(":")
            entries[int(index_text.strip())] = float(pct_text.strip())
        if not entries:
            raise ValueError("VLLM_ASCEND_MOE_SIM_ROUTE: sparse spec listed no experts")
        return entries

    values = [float(part) for part in spec.replace("\n", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("VLLM_ASCEND_MOE_SIM_ROUTE: empty profile")
    return values


def _materialize(parsed: list[float] | dict[int, float], num_experts: int) -> list[float]:
    """Expand a sparse spec against the model's expert count."""
    if not isinstance(parsed, dict):
        return parsed
    if num_experts <= 0:
        raise ValueError(
            "VLLM_ASCEND_MOE_SIM_ROUTE: a sparse 'idx:pct' spec needs the model's expert "
            "count, which is not known this early; use a full list or @file instead"
        )
    out_of_range = [e for e in parsed if e < 0 or e >= num_experts]
    if out_of_range:
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: sparse spec names expert {min(out_of_range)} "
            f"but the model has {num_experts} experts"
        )
    profile = [0.0] * num_experts
    for expert, share in parsed.items():
        profile[expert] = share
    rest = _PERCENT_TOTAL - math.fsum(parsed.values())
    unlisted = [e for e in range(num_experts) if e not in parsed]
    if rest < -_SUM_TOLERANCE:
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: sparse spec allocates more than 100% "
            f"(over by {-rest:.6f})"
        )
    if unlisted and rest > 0:
        share = rest / len(unlisted)
        for expert in unlisted:
            profile[expert] = share
    return profile


def _load_profile(num_experts: int, top_k: int) -> list[float] | None:
    """Parsed, materialised and validated profile; None when injection is off."""
    global _PROFILE, _PROFILE_MISS, _VALIDATED_FOR, _SEED, _TABLE, _EVERY
    if _PROFILE is not None:
        # The expert count is not known on the very first call in every caller shape;
        # re-check once it shows up so a mismatched profile cannot slip through.
        if (num_experts, top_k) != _VALIDATED_FOR and num_experts > 0:
            _validate(_PROFILE, num_experts, top_k)
            _VALIDATED_FOR = (num_experts, top_k)
        return _PROFILE
    if _PROFILE_MISS:
        return None
    spec = envs.VLLM_ASCEND_MOE_SIM_ROUTE.strip()
    if not spec:
        _PROFILE_MISS = True
        return None
    _SEED = envs.VLLM_ASCEND_MOE_SIM_ROUTE_SEED
    _TABLE = envs.VLLM_ASCEND_MOE_SIM_ROUTE_TABLE
    # <= 0 silences the heartbeat entirely (a profiling window wants a quiet log).
    # Note this is the opposite of "every call", which is what max(1, ...) would give.
    _every = envs.VLLM_ASCEND_MOE_SIM_ROUTE_EVERY
    _EVERY = _every if _every > 0 else None
    profile = _materialize(_parse_spec(spec), num_experts)
    _validate(profile, num_experts, top_k)
    _PROFILE = profile
    _VALIDATED_FOR = (num_experts, top_k)
    return _PROFILE


def _validate(profile: list[float], num_experts: int, top_k: int) -> None:
    if num_experts > 0 and len(profile) != num_experts:
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: profile has {len(profile)} entries but the "
            f"model has {num_experts} experts"
        )
    if any(value < 0 for value in profile):
        bad = next(i for i, value in enumerate(profile) if value < 0)
        raise ValueError(f"VLLM_ASCEND_MOE_SIM_ROUTE: negative share at expert {bad}: {profile[bad]}")
    total = math.fsum(profile)
    if abs(total - _PERCENT_TOTAL) > _SUM_TOLERANCE:
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: shares sum to {total:.6f}, expected {_PERCENT_TOTAL:g}"
        )
    cap = _PERCENT_TOTAL / top_k
    worst = max(profile)
    # Slack: the binding constraint is integral (count[e] <= T, enforced in _apportion),
    # so this float check is a diagnostic on the profile's intent and must not reject a
    # share that merely rounds to the cap. 16.6667 is what a user writes for 100/6.
    if worst > cap + _CAP_SLACK:
        bad = profile.index(worst)
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: expert {bad} takes {worst:.4f}% but top_k={top_k} "
            f"caps any expert at {cap:.4f}% -- one token's {top_k} experts are distinct, "
            f"so no expert can take more than {top_k}-th of the routes"
        )


def _apportion(profile: list[float], slots: int, tokens: int) -> list[int]:
    """Largest-remainder apportionment: exact total, each entry within 1 of ideal."""
    ideal = [value / _PERCENT_TOTAL * slots for value in profile]
    counts = [int(math.floor(value)) for value in ideal]
    remainder = slots - sum(counts)
    if remainder:
        order = sorted(range(len(profile)), key=lambda e: (-(ideal[e] - counts[e]), e))
        for expert in order[:remainder]:
            counts[expert] += 1

    # A remainder bump can push an entry one past the per-token cap; move the excess to
    # whichever experts still have room, largest fractional deficit first.
    for _ in range(len(profile)):
        if max(counts) <= tokens:
            break
        donor = counts.index(max(counts))
        room = [e for e in range(len(counts)) if counts[e] < tokens]
        if not room:
            raise ValueError(
                "VLLM_ASCEND_MOE_SIM_ROUTE: cannot place every route without repeating an "
                f"expert for a token (T={tokens})"
            )
        receiver = min(room, key=lambda e: (counts[e] - ideal[e], e))
        counts[donor] -= 1
        counts[receiver] += 1
    return counts


def _assign(counts: list[int], tokens: int, top_k: int, seed: int) -> torch.Tensor:
    """Build a ``[T, K]`` id table with exact column counts and distinct rows.

    Greedy degree-constrained realisation, hottest expert first, always onto the
    currently least-loaded rows. Every expert lands on a row at most once (rows stay
    distinct by construction, so no repair pass is needed) and column sums are the
    target counts by construction. Row sums balance to ``K`` whenever the target is
    realisable, which the ``max(p) <= 100/K`` check guarantees.

    Rows are kept in one bucket per load level, so "the least-loaded rows" is a pointer
    walk instead of a re-scan: the cost is ``O(T*K)`` overall rather than a top-k over
    all T rows per expert. That matters because this runs inside ``select_experts`` --
    on the forward's critical path -- and at T=16384 / E=256 the re-scan version cost
    ~68ms per new token count (~20ms here). Within a bucket the order comes from one
    seeded ``randperm``, which is where the seed-dependent variation lives.
    """
    table = torch.full((tokens, top_k), -1, dtype=torch.int64)
    row_load = torch.zeros(tokens, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    # buckets[l] = rows carrying exactly l experts; levels above top_k never exist.
    buckets: list[list[int]] = [torch.randperm(tokens, generator=generator).tolist()]
    buckets.extend([] for _ in range(top_k))
    cursor = [0] * (top_k + 1)

    for expert in sorted(range(len(counts)), key=lambda e: (-counts[e], e)):
        need = counts[expert]
        while need > 0:
            level = next((level for level in range(top_k) if cursor[level] < len(buckets[level])), None)
            if level is None:
                raise ValueError(
                    "VLLM_ASCEND_MOE_SIM_ROUTE: allocation did not balance -- profile is "
                    f"not realisable at T={tokens}, K={top_k}: every row already holds "
                    f"{top_k} experts but {sum(counts)} routes were asked for"
                )
            take = min(need, len(buckets[level]) - cursor[level])
            chosen = buckets[level][cursor[level] : cursor[level] + take]
            cursor[level] += take
            rows = torch.tensor(chosen, dtype=torch.int64)
            table[rows, (table[rows] >= 0).sum(dim=1)] = expert
            row_load[rows] += 1
            buckets[level + 1].extend(chosen)  # those rows just gained one expert
            need -= take

    if int(row_load.min()) != top_k or int(table.min()) < 0:
        raise ValueError(
            "VLLM_ASCEND_MOE_SIM_ROUTE: allocation did not balance -- profile is not "
            f"realisable at T={tokens}, K={top_k} (row loads {int(row_load.min())}.."
            f"{int(row_load.max())})"
        )
    return table


def _ep_world(num_experts: int) -> int | None:
    """EP world size, or None when the group is not up yet."""
    global _EP_WORLD
    if _EP_WORLD is not None:
        return _EP_WORLD
    try:
        from vllm.distributed import get_ep_group

        _EP_WORLD = int(get_ep_group().world_size)
    except Exception:
        return None
    return _EP_WORLD


def _rank_id() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "-1"))


def _rank_line(counts: list[int]) -> list[str]:
    """Predicted rank-level home rows -- the routed distribution's effect on compute.

    Assumes an identity logical->physical map (no EPLB) and contiguous expert sharding
    in blocks of ``E/R``. With EPLB active the remap downstream invalidates this; the
    moe_par dump stays authoritative there.
    """
    world = _ep_world(len(counts))
    if not world or world < 1:
        return []
    if len(counts) % world:
        # Contiguous sharding needs E/R to be a whole number. Say so rather than
        # omitting the line, which reads as "the injection did not run".
        return [
            f"[sim-route r{_rank_id()}]  home-rank rows 不可预测：E={len(counts)} 不被 R={world} "
            f"整除，分片不是 E/R 整块（真实 EP 布局不应走到这里，请核对 EP size）"
        ]
    per_rank = len(counts) // world
    rows = [0] * world
    for expert, count in enumerate(counts):
        rows[expert // per_rank] += count
    mean = sum(rows) / world
    spread = max(rows) / mean if mean else 0.0
    return [
        f"[sim-route r{_rank_id()}]  home-rank rows = {rows}  "
        f"max/mean = {spread:.2f}  (epn={per_rank}, R={world})"
    ]


def _log_table(tokens: int, top_k: int, counts: list[int], ideal: list[float], profile: list[float]) -> None:
    rank = _rank_id()
    slots = tokens * top_k
    deviation = max(abs(counts[e] - ideal[e]) for e in range(len(counts)))
    lines = [
        f"[sim-route r{rank}] T={tokens} K={top_k} slots={slots} seed={_SEED} "
        f"E={len(counts)} alloc#{_ALLOC_ID}",
        f"[sim-route r{rank}]  e     req%    req_cnt  act_cnt  dev",
    ]
    for expert, (share, want, got) in enumerate(zip(profile, ideal, counts)):
        lines.append(f"[sim-route r{rank}]  {expert:>4d} {share:>9.3f} {want:>10.1f} {got:>8d} {got - want:>+5.0f}")
    lines.append(
        f"[sim-route r{rank}]  max|dev|={deviation:.2f}  SUM_act={sum(counts)}  SUM_slots={slots}"
    )
    lines.extend(_rank_line(counts))
    print("\n".join(lines), flush=True)


def _log_short(tokens: int, top_k: int, alloc_id: int, reason: str) -> None:
    """``alloc_id`` is the allocation being reused, not the newest one built."""
    print(
        f"[sim-route r{_rank_id()}] T={tokens} K={top_k} reuse alloc#{alloc_id} ({reason})",
        flush=True,
    )


def maybe_select_experts(
    hidden_states: torch.Tensor,
    top_k: int,
    num_experts: int,
    mix_placement: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Synthetic ``(topk_weights, topk_ids)``, or None to fall through to the real router."""
    global _LAST_T, _SINCE_FULL, _ALLOC_ID, _MIX_WARNED

    if not active():
        return None

    # ``mix_placement`` appends shared-expert columns after routing; synthesising those
    # is out of scope, so that path keeps the real router. Only w8a8_dynamic passes it.
    if mix_placement:
        if not _MIX_WARNED:
            _MIX_WARNED = True
            logger.warning(
                "VLLM_ASCEND_MOE_SIM_ROUTE is set but mix_placement is on; that path needs "
                "the shared-expert columns, so the real router is used and the injection "
                "is inactive for this layer."
            )
        return None

    # The startup dummy forward only sizes the KV cache. Injecting there would give each
    # leg a different cache size and break cross-leg comparison; force_load_balance keeps
    # it balanced, which is exactly what we want to leave alone.
    try:
        if bool(_EXTRA_CTX.in_profile_run):
            return None
    except Exception:
        pass

    profile = _load_profile(num_experts, top_k)
    if profile is None:
        return None

    tokens = int(hidden_states.shape[0])
    if tokens < top_k:
        raise ValueError(
            f"VLLM_ASCEND_MOE_SIM_ROUTE: T={tokens} < top_k={top_k}; a token cannot take "
            f"{top_k} distinct experts out of only {tokens} slots"
        )

    key = (tokens, str(hidden_states.device), hidden_states.dtype)
    cached = _CACHE.pop(key, None)
    if cached is not None:
        _CACHE[key] = cached  # re-insert at the end: this is the most recently used
        weights, ids, alloc_id = cached
        if _LAST_T != tokens:
            _LAST_T = tokens
            _SINCE_FULL = 0
            _log_short(tokens, top_k, alloc_id, "same as previous")
        elif _EVERY is not None:
            _SINCE_FULL += 1
            if _SINCE_FULL >= _EVERY:
                _SINCE_FULL = 0
                _log_short(tokens, top_k, alloc_id, f"heartbeat, every {_EVERY} calls")
        return weights, ids

    slots = tokens * top_k
    counts = _apportion(profile, slots, tokens)
    table = _assign(counts, tokens, top_k, _SEED)

    _ALLOC_ID += 1
    _LAST_T = tokens
    _SINCE_FULL = 0
    if _TABLE:
        _log_table(tokens, top_k, counts, [v / _PERCENT_TOTAL * slots for v in profile], profile)
    else:
        print("\n".join([f"[sim-route r{_rank_id()}] T={tokens} slots={slots}"] + _rank_line(counts)), flush=True)

    ids = table.to(device=hidden_states.device, dtype=torch.int32)
    weights = torch.full(
        (tokens, top_k),
        1.0 / top_k,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if len(_CACHE) >= _CACHE_MAX:
        # dicts keep insertion order and hits re-insert, so the front is the LRU entry.
        # That keeps the steady-state decode T resident no matter how much churn the
        # prefill frontier produces around it.
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (weights, ids, _ALLOC_ID)
    return weights, ids
