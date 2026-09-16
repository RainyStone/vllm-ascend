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
"""Tests for the per-rank MoE load dump and its offline analysis.

Covers the two halves together on purpose: the recorder writes a dump and the
analyzer reads it back, so a change to either format breaks a test here rather
than a report weeks later. Everything runs on CPU -- the recorder touches device
memory only to buffer the counts, and a CPU tensor exercises the same code.
"""

import importlib.util
import os
import re
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe import moe_par_stats

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "scripts")


def _load_analyzer():
    """scripts/ is not a package; load the analysis module straight from its path."""
    path = os.path.join(SCRIPTS_DIR, "analyze_moe_par.py")
    spec = importlib.util.spec_from_file_location("analyze_moe_par", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_layer(instance_id, local_experts, global_experts, ep_rank, ep_size, topk=8, expert_map=None):
    """The subset of an AscendFusedMoE that the recorder and the meta header read."""
    if expert_map is None:
        # A contiguous partition: global id -> local slot, -1 for experts on other ranks.
        first = ep_rank * local_experts
        expert_map = [g - first if first <= g < first + local_experts else -1 for g in range(global_experts)]
    return types.SimpleNamespace(
        moe_instance_id=instance_id,
        moe_config=types.SimpleNamespace(experts_per_token=topk),
        local_num_experts=local_experts,
        global_num_experts=global_experts,
        global_redundant_expert_num=0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        log2phy=None,
        _expert_map=torch.tensor(expert_map, dtype=torch.int32),
    )


class MoeParStatsTestCase(TestBase):
    """Reset the recorder's module-level state so tests do not leak into each other."""

    flush_every = 2

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_analyzer()

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp(prefix="moe-par-ut-")
        self._patch(moe_par_stats, "_ENABLED", True)
        self._patch(moe_par_stats, "_DIR", self.tmpdir)
        self._patch(moe_par_stats, "_TAG", "ut")
        self._patch(moe_par_stats, "_FLUSH_EVERY", self.flush_every)
        # Outside a distributed run the recorder falls back to $RANK, which would put
        # the dump under an unexpected name.
        self._patch(moe_par_stats, "_rank_id", lambda: 0)
        self._reset()

    def _patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _reset(self):
        moe_par_stats._ord_of.clear()
        moe_par_stats._layers.clear()
        moe_par_stats._valid.clear()
        moe_par_stats._step_seen.clear()
        moe_par_stats._methods.clear()
        moe_par_stats._warned.clear()
        moe_par_stats._buf = None
        moe_par_stats._slot = 0
        moe_par_stats._step_base = 0
        moe_par_stats._meta_written = False
        moe_par_stats._current_method = "?"
        if moe_par_stats._out is not None:
            moe_par_stats._out.close()
        moe_par_stats._out = None

    def _dump_path(self, rank=0):
        return os.path.join(self.tmpdir, f"moe_par_ut_rank{rank}.log")

    def _read_dump(self):
        path = self._dump_path()
        if not os.path.exists(path):
            return []
        with open(path) as handle:
            return handle.read().splitlines()

    def _sample(self, layer, counts, method="MoonEPCommImpl"):
        """One layer's worth of counts, as the model would hand them over (count mode)."""
        moe_par_stats.record(layer, torch.tensor(counts, dtype=torch.int32), 1, method)


class TestToCounts(MoeParStatsTestCase):
    def test_count_mode_passes_through(self):
        flat = torch.tensor([3, 0, 5, 2], dtype=torch.int32)
        self.assertEqual(moe_par_stats._to_counts(flat, 1).tolist(), [3, 0, 5, 2])

    def test_cumulative_mode_is_differenced(self):
        # group_list_type != 1 carries a running total per expert; the dump wants counts.
        cumulative = torch.tensor([3, 3, 8, 10], dtype=torch.int32)
        self.assertEqual(moe_par_stats._to_counts(cumulative, 0).tolist(), [3, 0, 5, 2])

    def test_cumulative_mode_keeps_the_first_expert(self):
        cumulative = torch.tensor([7, 7], dtype=torch.int32)
        self.assertEqual(moe_par_stats._to_counts(cumulative, 2).tolist(), [7, 0])


class TestRecorder(MoeParStatsTestCase):
    def setUp(self):
        super().setUp()
        self.l0 = _fake_layer(11, 2, 4, ep_rank=0, ep_size=2)
        self.l1 = _fake_layer(12, 2, 4, ep_rank=0, ep_size=2)

    def test_disabled_writes_nothing(self):
        with patch.object(moe_par_stats, "_ENABLED", False):
            self._sample(self.l0, [1, 2])
            self._sample(self.l1, [2, 1])
            self._sample(self.l0, [1, 2])
            self._sample(self.l1, [2, 1])
        self.assertEqual(self._read_dump(), [])

    def test_step_boundary_is_a_repeated_layer(self):
        # Two layers per step, flush every two steps: the first flush lands when the
        # third step opens, because only then is the second step known to be whole.
        self._sample(self.l0, [1, 2])
        self._sample(self.l1, [2, 1])
        self._sample(self.l0, [3, 0])
        self.assertEqual(self._read_dump(), [], "a step is only complete once its layers repeat")

        self._sample(self.l1, [0, 3])
        self._sample(self.l0, [4, 4])  # opens step 2, completing steps 0 and 1
        lines = self._read_dump()
        body = [ln for ln in lines if ln.startswith("step=")]
        self.assertEqual(len(body), 4, "two steps x two layers")
        self.assertIn("step=0 layer=0 method=MoonEPCommImpl counts=[1,2]", body)
        self.assertIn("step=0 layer=1 method=MoonEPCommImpl counts=[2,1]", body)
        self.assertIn("step=1 layer=0 method=MoonEPCommImpl counts=[3,0]", body)
        self.assertIn("step=1 layer=1 method=MoonEPCommImpl counts=[0,3]", body)

    def test_layer_order_comes_from_first_sight_not_call_order(self):
        self._sample(self.l1, [5, 5])
        self._sample(self.l0, [6, 6])
        self._sample(self.l1, [7, 7])
        self._sample(self.l0, [8, 8])
        self._sample(self.l1, [9, 9])
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertIn("step=0 layer=0 method=MoonEPCommImpl counts=[5,5]", body)
        self.assertIn("step=0 layer=1 method=MoonEPCommImpl counts=[6,6]", body)

    def test_shutdown_closes_a_complete_step_but_not_a_partial_one(self):
        self._sample(self.l0, [1, 1])
        self._sample(self.l1, [1, 1])
        self._sample(self.l0, [2, 2])  # step 1 opens; step 0 still buffered
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        # Step 0 was complete and is written; step 1 saw only one of two layers, and a
        # half-counted step would understate the load and skew the ratio.
        self.assertEqual(len(body), 2, "only the complete step is written")
        self.assertTrue(all(ln.startswith("step=0 ") for ln in body))

    def test_shutdown_keeps_a_complete_final_step(self):
        self._sample(self.l0, [1, 1])
        self._sample(self.l1, [1, 1])
        self._sample(self.l0, [2, 2])
        self._sample(self.l1, [2, 2])
        self._sample(self.l0, [3, 3])  # opens step 2 with one layer
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertEqual(len(body), 4, "steps 0 and 1 both closed on exit")
        self.assertTrue(any(ln.startswith("step=1 layer=1 ") for ln in body))

    def test_profile_run_is_filtered(self):
        context = types.SimpleNamespace(in_profile_run=True)
        with patch.object(moe_par_stats, "_EXTRA_CTX", context):
            self._sample(self.l0, [1, 2])
            self._sample(self.l1, [2, 1])
            self._sample(self.l0, [1, 2])
            self._sample(self.l1, [2, 1])
        self.assertEqual(self._read_dump(), [], "profile and dummy forwards must not reach the dump")

    def test_none_counts_are_ignored(self):
        self._sample(self.l0, [1, 2])
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")
        self.assertEqual([ln for ln in self._read_dump() if ln.startswith("step=")], [])

    def test_a_layer_of_another_width_is_skipped_once(self):
        wide = _fake_layer(13, 4, 4, ep_rank=0, ep_size=1)
        self._sample(self.l0, [1, 2])          # buffer is sized 2 wide here
        self._sample(wide, [1, 1, 1, 1])       # 4 wide: cannot join this buffer
        self._sample(self.l1, [1, 2])
        self._sample(self.l0, [1, 2])
        self.assertTrue(any(k.startswith("width:") for k in moe_par_stats._warned))
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(all(len(re.search(r"counts=\[([^\]]*)\]", ln).group(1).split(",")) == 2 for ln in body))

    def test_meta_header_carries_every_dimension(self):
        self._sample(self.l0, [1, 2])
        self._sample(self.l1, [2, 1])
        self._sample(self.l0, [1, 2])  # opens step 1, so step 0 is whole
        moe_par_stats._shutdown()      # flush it without waiting for the window to fill
        meta = next(ln for ln in self._read_dump() if ln.startswith("# meta"))
        fields = dict(kv.split("=", 1) for kv in meta[len("# meta "):].split())
        self.assertEqual(fields["tag"], "ut")
        self.assertEqual(fields["moe_layers"], "2")
        self.assertEqual(fields["topk"], "8")
        self.assertEqual(fields["global_experts"], "4")
        self.assertEqual(fields["local_experts"], "2")
        self.assertEqual(fields["ep_rank"], "0")
        self.assertEqual(fields["ep_size"], "2")
        layers = [ln for ln in self._read_dump() if ln.startswith("# layer")]
        self.assertEqual(len(layers), 2)
        self.assertIn("instance_id=11", layers[0])
        self.assertIn("log2phy=None", layers[0])
        self.assertIn("expert_map=[0,1,-1,-1]", layers[0])

    def test_meta_survives_a_property_that_raises(self):
        # Swappable models expose different surface area; a raising property must not
        # cost us the whole header.
        class Rude:
            moe_instance_id = 99
            moe_config = types.SimpleNamespace(experts_per_token=4)
            local_num_experts = 2
            global_num_experts = 4
            global_redundant_expert_num = 0
            log2phy = None
            _expert_map = None

            @property
            def ep_rank(self):
                raise RuntimeError("not available before the model is built")

            @property
            def ep_size(self):
                raise RuntimeError("not available before the model is built")

        rude = Rude()
        self._sample(rude, [1, 2])
        self._sample(self.l1, [2, 1])
        self._sample(rude, [1, 2])
        moe_par_stats._shutdown()
        meta = next(ln for ln in self._read_dump() if ln.startswith("# meta"))
        self.assertIn("ep_rank=-1", meta)
        self.assertIn("ep_size=-1", meta)
        self.assertIn("topk=4", meta)


class TestAnalyzer(TestBase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_analyzer()

    def test_parse_int_list_strips_brackets_and_tolerates_none(self):
        self.assertEqual(self.mod._parse_int_list("[0,1,-1]"), [0, 1, -1])
        self.assertIsNone(self.mod._parse_int_list("None"))
        self.assertIsNone(self.mod._parse_int_list(""))
        self.assertIsNone(self.mod._parse_int_list("nonsense"))

    def test_loc2glob_inverts_the_expert_map(self):
        # expert_map is indexed by global id and holds the local slot (-1 if absent),
        # so the analyzer has to invert it to place a local slot back on the global grid.
        layer = {"expert_map": [-1, -1, 0, 1]}
        mapping, source = self.mod._loc2glob(layer, local_experts=2, global_experts=4, ep_rank=1)
        self.assertEqual(list(mapping), [2, 3])
        self.assertEqual(source, self.mod._SRC_EXPERT_MAP)

    def test_loc2glob_falls_back_when_the_map_is_absent(self):
        mapping, source = self.mod._loc2glob({}, local_experts=2, global_experts=8, ep_rank=2)
        self.assertEqual(list(mapping), [4, 5], "a contiguous partition is the only sane reading")
        self.assertEqual(source, self.mod._SRC_IDENTITY)

    def test_loc2glob_rejects_a_map_that_does_not_cover_every_slot(self):
        # Two global ids pointing at the same slot would silently drop an expert.
        mapping, source = self.mod._loc2glob({"expert_map": [0, 0, -1, -1]}, 2, 4, 0)
        self.assertEqual(list(mapping), [0, 1])
        self.assertEqual(source, self.mod._SRC_IDENTITY)

    def test_loc2glob_is_exact_when_every_expert_is_local(self):
        # ep_size=1 (or one rank holding the whole table): the slot *is* the global id,
        # so this is not the contiguous guess and must not be reported as one.
        mapping, source = self.mod._loc2glob({}, local_experts=8, global_experts=8, ep_rank=0)
        self.assertEqual(list(mapping), list(range(8)))
        self.assertEqual(source, self.mod._SRC_WHOLE)

    def test_a_whole_table_dump_raises_no_placement_warning(self):
        # The all-gather leg runs ep_size=1 and carries no expert_map; warning there
        # would cry wolf on the one configuration that needs no inference at all.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "moe_par_whole_rank0.log")
            with open(path, "w") as fh:
                fh.write(
                    f"# meta model=/models/m tag=whole rank=0 world_size=1 ep_rank=0 ep_size=1 "
                    f"moe_layers=2 topk=2 global_experts=8 local_experts=8 redundant_experts=0\n"
                )
                for layer in range(2):
                    fh.write(f"# layer layer_ord={layer} instance_id={layer + 2} log2phy=None expert_map=None\n")
                for step in range(3):
                    for layer in range(2):
                        fh.write(f"step={step} layer={layer} method=AllGatherImpl counts=[1,1,1,1,1,1,1,1]\n")
            run = self.mod.Run("whole", tmp)
            self.assertEqual([w for w in run.warnings if "contiguous" in w], [])


class TestRecorderToAnalyzerRoundTrip(MoeParStatsTestCase):
    """Record four ranks in-process, then let the analyzer read them back."""

    n_ranks = 4
    global_experts = 8
    local_experts = 2
    topk = 2
    n_layers = 3
    n_steps = 5
    tokens = 64

    def _write_all_ranks(self):
        rng = np.random.default_rng(20260915)
        pairs = self.tokens * self.topk
        layers = [
            _fake_layer(100 + i, self.local_experts, self.global_experts, ep_rank=0, ep_size=self.n_ranks,
                        topk=self.topk)
            for i in range(self.n_layers)
        ]
        # One global row per (step, layer) keeps every layer's total identical, which is
        # exactly the property the analyzer's cross-rank self-check asserts.
        rows = {}
        for step in range(self.n_steps):
            for li in range(self.n_layers):
                weights = rng.dirichlet(np.full(self.global_experts, 0.3))
                row = np.floor(weights * pairs).astype(np.int64)
                np.add.at(row, rng.choice(self.global_experts, size=pairs - row.sum(), replace=True), 1)
                rows[(step, li)] = row

        for rank in range(self.n_ranks):
            self._reset()
            with patch.object(moe_par_stats, "_rank_id", lambda r=rank: r):
                for layer in layers:
                    # Give every rank its own expert_map so the local->global join is not
                    # assumed to be a contiguous partition.
                    layer._expert_map = torch.tensor(
                        [g - rank * self.local_experts
                         if rank * self.local_experts <= g < (rank + 1) * self.local_experts else -1
                         for g in range(self.global_experts)], dtype=torch.int32)
                for step in range(self.n_steps):
                    for li, layer in enumerate(layers):
                        offset = rank * self.local_experts
                        counts = rows[(step, li)][offset:offset + self.local_experts].tolist()
                        self._sample(layer, counts, method="MC2CommImpl" if rank % 2 else "MoonEPCommImpl")
                moe_par_stats._shutdown()

    def test_round_trip_reconstructs_the_global_load(self):
        self._write_all_ranks()
        dumps = self.mod.find_dumps(self.tmpdir)
        self.assertEqual(len(dumps), self.n_ranks)

        run = self.mod.Run("ut", self.tmpdir).finalize()
        self.assertEqual(run.n_ranks, self.n_ranks)
        self.assertEqual(run.n_steps, self.n_steps)
        self.assertEqual(run.n_layers, self.n_layers)

        check = run.checks
        self.assertTrue(check["flat"], f"cross-rank join should reconcile: {check}")
        self.assertEqual(check["missing_cells"], 0)

        # Every token-expert pair of the step must be accounted for by some rank:
        # layers x tokens x topk, and no rank may have lost its share.
        totals = run.rank_step_load
        self.assertTrue(np.isfinite(totals).all())
        np.testing.assert_allclose(totals.sum(axis=0), self.n_layers * self.tokens * self.topk)

        # PAR is only defined across ranks and must land in (1, n_ranks]. The upper
        # bound is what makes the join meaningful -- a run whose ranks were never
        # told apart would sit at exactly 1.
        for name, values in (("per-layer", run.par_by_step), ("aggregate", run.par_agg)):
            finite = values[np.isfinite(values)]
            self.assertEqual(finite.size, self.n_steps, f"{name} PAR should be defined for every step")
            self.assertGreaterEqual(finite.min(), 1.0 - 1e-9)
            self.assertLessEqual(finite.max(), self.n_ranks + 1e-9)
            self.assertGreater(finite.max(), 1.0 + 1e-6, f"{name} PAR should show the routing's real skew")

    def test_a_single_rank_leaves_par_undefined(self):
        # Peak over mean with one rank is 1/1, which would read as "perfectly balanced"
        # when there is in fact no spread to measure.
        rng = np.random.default_rng(7)
        row = np.zeros(self.global_experts, dtype=np.int64)
        np.add.at(row, rng.choice(self.global_experts, size=self.tokens * self.topk, replace=True), 1)
        self._reset()
        with patch.object(moe_par_stats, "_rank_id", lambda: 0):
            layer = _fake_layer(1, self.global_experts, self.global_experts, ep_rank=0, ep_size=1, topk=self.topk)
            for _ in range(3):
                self._sample(layer, row.tolist(), method="MC2CommImpl")
            moe_par_stats._shutdown()

        run = self.mod.Run("solo", self.tmpdir).finalize()
        self.assertEqual(run.n_ranks, 1)
        self.assertTrue(np.isnan(run.par_by_step).all())
        self.assertTrue(np.isnan(run.par_agg).all())
        self.assertTrue(any("只有一个 rank" in w for w in run.notes), run.notes)
        # The load data itself is still there; only the ratio is undefined.
        self.assertTrue(np.isfinite(run.rank_step_load).all())


if __name__ == "__main__":
    unittest.main()
