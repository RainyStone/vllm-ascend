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

MOE_PAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "scripts", "moe_par")


def _load_analyzer():
    """scripts/ is not a package; load the analysis module straight from its path."""
    path = os.path.join(MOE_PAR_DIR, "analyze_moe_par.py")
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
        moe_par_stats._route_note = None
        moe_par_stats._pad_pairs_subtracted = 0
        moe_par_stats._pad_notes_used = 0
        moe_par_stats._pad_notes_misplaced = 0
        moe_par_stats._pad_expert_used = None
        moe_par_stats._negative_cells = 0
        moe_par_stats._scopes.clear()
        moe_par_stats._tail_at = None
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


class TestRouteNote(MoeParStatsTestCase):
    """MoonEP announces each dispatch's sending-side histogram, index by *global*
    (logical) expert id, together with the capacity pad it wrote into it.

    That histogram is the exact reading of the dispatch -- the receiving side carries
    the pads and misses the rows the dispatch dropped -- so it replaces the method's
    own counts, and the dump it lands in is a global-scope one.
    """

    def setUp(self):
        super().setUp()
        self.l0 = _fake_layer(21, 4, 8, ep_rank=0, ep_size=2)
        self.l1 = _fake_layer(22, 4, 8, ep_rank=0, ep_size=2)

    def _note(self, histogram, pad_pairs=0, pad_index=-1):
        moe_par_stats.note_route_counts(torch.tensor(histogram, dtype=torch.int32), pad_pairs, pad_index)

    def _dispatch(self, layer, histogram, pad_pairs=0, pad_index=-1):
        """One MoonEP layer: the dispatcher announces, then the comm method reports
        its (receive-side) counts, which the recorder must ignore in favour of the
        histogram."""
        self._note(histogram, pad_pairs, pad_index)
        moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")

    def _two_steps(self, first, later=None):
        """Five dispatches of two layers: the fifth closes the second step and flushes."""
        later = first if later is None else later
        for layer, hist in ((self.l0, first), (self.l1, first), (self.l0, later), (self.l1, later), (self.l0, later)):
            self._dispatch(layer, hist)

    def test_the_histogram_wins_over_the_received_counts(self):
        # (capacity - T) * topk = 6 pad pairs on global expert 2, which this rank owns.
        self._dispatch(self.l0, [1, 0, 10, 0, 0, 0, 0, 0], pad_pairs=6, pad_index=2)
        self._dispatch(self.l1, [0, 3, 0, 0, 0, 0, 0, 0])
        self._dispatch(self.l0, [2, 2, 2, 2, 0, 0, 0, 0])
        self._dispatch(self.l1, [1, 1, 1, 1, 0, 0, 0, 0])
        self._dispatch(self.l0, [5, 0, 0, 0, 0, 0, 0, 0])
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(body, "the dump should hold the two completed steps")
        self.assertIn("step=0 layer=0 method=MoonEPCommImpl counts=[1,0,4,0,0,0,0,0]", body)
        self.assertIn("step=0 layer=1 method=MoonEPCommImpl counts=[0,3,0,0,0,0,0,0]", body)

    def test_the_dump_is_global_scope_and_the_header_says_so(self):
        self._two_steps([4, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats._shutdown()
        meta = next(ln for ln in self._read_dump() if ln.startswith("# meta "))
        fields = dict(kv.split("=", 1) for kv in meta[len("# meta "):].split())
        self.assertEqual(fields["counts_scope"], "global")
        # The row spans the model's experts, not this rank's four slots.
        body = next(ln for ln in self._read_dump() if ln.startswith("step="))
        self.assertEqual(len(re.search(r"counts=\[([^\]]*)\]", body).group(1).split(",")), 8)

    def test_a_note_is_consumed_once(self):
        # The note is taken by the very next record(), even one that ends up using the
        # method's own counts, so a stale note can never be applied to another layer.
        a = _fake_layer(31, 8, 8, ep_rank=0, ep_size=1)
        b = _fake_layer(32, 8, 8, ep_rank=0, ep_size=1)
        fives = torch.full((8,), 5, dtype=torch.int32)
        self._note([3, 1, 0, 0, 0, 0, 0, 0], pad_pairs=2, pad_index=0)
        moe_par_stats.record(a, None, 1, "MoonEPCommImpl")
        moe_par_stats.record(b, fives, 1, "MC2CommImpl")
        moe_par_stats.record(a, fives, 1, "MC2CommImpl")  # closes step 0
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertIn("step=0 layer=0 method=MoonEPCommImpl counts=[1,1,0,0,0,0,0,0]", body)
        self.assertTrue(any("counts=[5,5,5,5,5,5,5,5]" in ln for ln in body), body)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 2)
        self.assertEqual(moe_par_stats._pad_notes_used, 1)

    def test_no_note_leaves_the_received_counts_alone(self):
        # Every method but MoonEP reports its own counts, index by local slot.
        for layer, counts in ((self.l0, [10, 2, 3, 1]), (self.l1, [10, 2, 3, 1]), (self.l0, [10, 2, 3, 1]),
                              (self.l1, [10, 2, 3, 1]), (self.l0, [10, 2, 3, 1])):
            self._sample(layer, counts, method="MC2CommImpl")
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(body, "the dump should hold the two completed steps")
        self.assertTrue(all("counts=[10,2,3,1]" in ln for ln in body), body)
        meta = next(ln for ln in self._read_dump() if ln.startswith("# meta "))
        self.assertIn("counts_scope=local", meta)

    def test_zero_pairs_still_replaces_the_counts(self):
        # A step whose token count already fills the pooled capacity: no pad written,
        # but the histogram is still the better reading of the dispatch.
        self._note([7, 7, 7, 0, 0, 0, 0, 0], pad_pairs=0, pad_index=-1)
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")
        self._note([7, 7, 7, 0, 0, 0, 0, 0], pad_pairs=0, pad_index=-1)
        moe_par_stats.record(self.l1, None, 1, "MoonEPCommImpl")
        self._note([7, 7, 7, 0, 0, 0, 0, 0], pad_pairs=0, pad_index=-1)
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")  # closes step 0
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any("counts=[7,7,7,0,0,0,0,0]" in ln for ln in body), body)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 0)
        self.assertEqual(moe_par_stats._pad_notes_used, 3)

    def test_the_dispatchers_histogram_is_not_written_through(self):
        # The dispatcher's ``tpe`` is int32 and contiguous, so a plain ``to(torch.int32)``
        # returns the *same* tensor and the pad subtraction would land on the engine's
        # copy. The plan does not read it back today; the recorder must not be the thing
        # that finds out whether that stays true.
        hist = torch.tensor([1, 0, 10, 0, 0, 0, 0, 0], dtype=torch.int32)
        moe_par_stats.note_route_counts(hist, 6, 2)
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")
        self.assertEqual(int(hist[2]), 10, "the dispatcher's own histogram must be left as it was")
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 6)
        for layer in (self.l1, self.l0):  # the repeat closes step 0
            self._dispatch(layer, [0, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any("counts=[1,0,4,0,0,0,0,0]" in ln for ln in body), body)

    def test_a_pad_index_outside_the_histogram_is_reported(self):
        # The dispatcher names the pad expert in the histogram's own space; an index
        # past the end means the two sides disagree about the expert count.
        self._note([1, 2, 3, 4, 0, 0, 0, 0], pad_pairs=6, pad_index=99)
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")
        self._note([0, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats.record(self.l1, None, 1, "MoonEPCommImpl")
        self._note([0, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")  # closes step 0
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any("counts=[1,2,3,4,0,0,0,0]" in ln for ln in body), body)
        self.assertEqual(moe_par_stats._pad_notes_misplaced, 1)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 0)
        self.assertIn("pad_notes_misplaced=1", self._read_dump()[-1])

    def test_negative_cells_are_clamped_and_counted(self):
        # Should be impossible: the pad is subtracted in the space it was counted in.
        # A negative cell means the two sides disagreed, and the dump says so.
        self._note([3, 0, 0, 0, 0, 0, 0, 0], pad_pairs=7, pad_index=0)
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")
        self._note([1, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats.record(self.l1, None, 1, "MoonEPCommImpl")
        self._note([1, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats.record(self.l0, None, 1, "MoonEPCommImpl")  # closes step 0
        moe_par_stats._shutdown()
        dump = self._read_dump()
        body = [ln for ln in dump if ln.startswith("step=")]
        self.assertTrue(any("counts=[0,0,0,0,0,0,0,0]" in ln for ln in body), body)
        self.assertEqual(moe_par_stats._negative_cells, 1)
        self.assertIn("negative_cells_clamped=1", dump[-1])

    def test_scope_is_mixed_when_both_paths_are_used(self):
        # One width cannot mean two things: a dump that mixed the scopes must not be
        # read as either, so it is labelled and the analysis warns.
        whole = _fake_layer(41, 8, 8, ep_rank=0, ep_size=1)
        self._note([1, 1, 0, 0, 0, 0, 0, 0], pad_pairs=1, pad_index=0)
        moe_par_stats.record(whole, None, 1, "MoonEPCommImpl")
        self._sample(whole, [1, 1, 0, 0, 0, 0, 0, 0], method="MC2CommImpl")
        self._sample(whole, [1, 1, 0, 0, 0, 0, 0, 0], method="MC2CommImpl")
        moe_par_stats._shutdown()
        meta = next(ln for ln in self._read_dump() if ln.startswith("# meta "))
        self.assertIn("counts_scope=mixed", meta)

    def test_the_exit_header_reports_what_was_subtracted(self):
        self._two_steps([10, 0, 0, 0, 0, 0, 0, 0])
        self._note([3, 0, 9, 0, 0, 0, 0, 0], pad_pairs=6, pad_index=2)
        moe_par_stats.record(self.l1, None, 1, "MoonEPCommImpl")  # pad 6, held in the window
        moe_par_stats._shutdown()
        header = [ln for ln in self._read_dump() if ln.startswith("# meta partial_steps=")]
        self.assertEqual(len(header), 1, header)
        self.assertIn("pad_pairs_subtracted=6", header[0])
        self.assertIn("pad_expert_global=2", header[0])
        self.assertNotIn("pad_pairs_skipped", header[0])


class TestSummaryLineSurvivesWithoutAnExitHook(MoeParStatsTestCase):
    """The ledger must be readable even if the process never runs ``atexit``.

    A forked worker ends in ``os._exit()``, which skips the exit handlers, so a
    dump whose ledger only appeared there lost the pad accounting entirely and
    reported "no pad" for a run that had subtracted millions of pairs. The line is
    now refreshed at every flush instead of only at exit.
    """

    def setUp(self):
        super().setUp()
        self.l0 = _fake_layer(23, 4, 8, ep_rank=0, ep_size=2)
        self.l1 = _fake_layer(24, 4, 8, ep_rank=0, ep_size=2)

    def _tails(self):
        return [ln for ln in self._read_dump() if ln.startswith("# meta partial_steps=")]

    def _two_steps(self, histogram, pad_pairs=0, pad_index=-1):
        """Five dispatches of two layers: the fifth closes the second step and flushes.

        The pad rides on the first layer of each closed step, once per step, which is
        what the dispatcher does -- every step pads on its own.
        """
        for li, layer in enumerate((self.l0, self.l1, self.l0, self.l1, self.l0)):
            moe_par_stats.note_route_counts(
                torch.tensor(histogram, dtype=torch.int32), pad_pairs if li in (0, 2) else 0, pad_index
            )
            moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")

    def test_a_flush_alone_writes_the_summary(self):
        self._two_steps([8, 2, 0, 0, 0, 0, 0, 0], pad_pairs=3, pad_index=0)
        dump = self._read_dump()
        self.assertTrue(dump, "the window filled, so a flush happened")
        self.assertTrue(dump[-1].startswith("# meta partial_steps="), "the summary ends the file")
        self.assertEqual(len(self._tails()), 1, self._tails())
        self.assertIn(f"pad_pairs_subtracted={moe_par_stats._pad_pairs_subtracted}", self._tails()[0])
        self.assertIn("pad_pairs_subtracted=6", self._tails()[0])

    def test_the_summary_is_refreshed_not_appended(self):
        # Two flushes: the pad total grew, and there is still exactly one ledger line,
        # holding the new value -- not two lines with the stale one first.
        self._two_steps([8, 2, 0, 0, 0, 0, 0, 0], pad_pairs=3, pad_index=0)
        first = self._tails()
        self.assertEqual(len(first), 1, first)
        self._two_steps([4, 0, 0, 0, 0, 0, 0, 0], pad_pairs=5, pad_index=0)
        second = self._tails()
        self.assertEqual(len(second), 1, second)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 16)
        self.assertIn("pad_pairs_subtracted=16", second[0])
        # The rows written before the second flush survive the rewrite.
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any(ln.startswith("step=1 ") for ln in body), body)
        self.assertTrue(any(ln.startswith("step=3 ") for ln in body), body)

    def test_no_rows_are_lost_by_the_rewrite(self):
        self._two_steps([1, 1, 0, 0, 0, 0, 0, 0])
        self._two_steps([1, 1, 0, 0, 0, 0, 0, 0])
        steps = sorted({int(re.match(r"step=(\d+)", ln).group(1)) for ln in self._read_dump() if ln.startswith("step=")})
        self.assertEqual(steps, [0, 1, 2, 3], steps)


class TestAnalyzer(TestBase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_analyzer()

    def test_owned_mask_reads_the_map_as_a_holding(self):
        # expert_map is indexed by global id and holds the local slot, so "not -1"
        # is "this rank holds it" -- the reading that survives replication.
        mask, source = self.mod._owned_mask({"expert_map": [-1, -1, 0, 1]}, 2, 4, ep_rank=1)
        self.assertEqual(list(mask), [False, False, True, True])
        self.assertEqual(source, self.mod._SRC_EXPERT_MAP)

    def test_owned_mask_reads_replication_as_two_holders(self):
        # The same logical expert in two ranks' maps: a copy, not a contradiction.
        mask, _ = self.mod._owned_mask({"expert_map": [0, -1, 0, -1]}, 2, 4, ep_rank=0)
        self.assertEqual(list(mask), [True, False, True, False])

    def test_owned_mask_falls_back_to_a_contiguous_partition(self):
        mask, source = self.mod._owned_mask({}, 2, 8, ep_rank=2)
        self.assertEqual(list(mask), [False, False, False, False, True, True, False, False])
        self.assertEqual(source, self.mod._SRC_IDENTITY)

    def test_owned_mask_covers_everything_when_the_table_is_whole(self):
        mask, source = self.mod._owned_mask({}, 8, 8, ep_rank=0)
        self.assertTrue(mask.all())
        self.assertEqual(source, self.mod._SRC_WHOLE)

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


class TestGlobalScopeRoundTrip(MoeParStatsTestCase):
    """The MoonEP path end to end: each rank dumps the dispatcher's *sending-side*
    histogram over the model's whole expert space, and the analysis sums the ranks of
    a (step, layer) into the run's real routing before crediting it to the ranks that
    hold those experts."""

    n_ranks = 4
    global_experts = 8
    local_experts = 2
    topk = 2
    n_layers = 3
    n_steps = 5
    tokens = 64
    pads = 6  # (capacity - T) * topk, as the dispatcher would report it

    def _histogram_blocks(self):
        """The step's real routing, split into per-rank sending sides.

        The split stands in for ``topk_c``: it is not how the dispatcher computes
        anything (the routes are per token, not per rank), but it reproduces what the
        recorder gets -- one row per rank per layer that only sums to the step's
        routing when the ranks are added up.
        """
        rng = np.random.default_rng(31337)
        pairs = self.tokens * self.topk
        parts = {}
        for step in range(self.n_steps):
            for li in range(self.n_layers):
                weights = rng.dirichlet(np.full(self.global_experts, 0.3))
                real = np.floor(weights * pairs).astype(np.int64)
                np.add.at(real, rng.choice(self.global_experts, size=pairs - real.sum(), replace=True), 1)
                shares = rng.dirichlet(np.full(self.n_ranks, 1.0))
                split = np.floor(real * shares[:, None]).astype(np.int64)
                split[0] += real - split.sum(axis=0)
                for rank in range(self.n_ranks):
                    parts[(rank, step, li)] = split[rank]
        return parts

    def _write_all_ranks(self):
        parts = self._histogram_blocks()
        layers = [
            _fake_layer(200 + i, self.local_experts, self.global_experts, ep_rank=0, ep_size=self.n_ranks, topk=self.topk)
            for i in range(self.n_layers)
        ]
        for rank in range(self.n_ranks):
            self._reset()
            with patch.object(moe_par_stats, "_rank_id", lambda r=rank: r):
                for layer in layers:
                    layer._expert_map = torch.tensor(
                        [g - rank * self.local_experts
                         if rank * self.local_experts <= g < (rank + 1) * self.local_experts else -1
                         for g in range(self.global_experts)], dtype=torch.int32)
                for step in range(self.n_steps):
                    for li, layer in enumerate(layers):
                        sent = parts[(rank, step, li)]
                        pad_expert = rank * self.local_experts
                        padded = sent.copy()
                        padded[pad_expert] += self.pads
                        moe_par_stats.note_route_counts(
                            torch.tensor(padded, dtype=torch.int32), self.pads, pad_expert)
                        moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")
                moe_par_stats._shutdown()

    def test_round_trip_reconstructs_the_routing_and_attributes_it(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()

        self.assertEqual(run.scope, "global")
        self.assertEqual(run.row_width, self.global_experts, "a global row spans every expert")
        self.assertEqual(run.local_experts, self.local_experts, "read off the header, not the row")
        self.assertEqual(run.duplicated_experts, 0)
        self.assertEqual(run.unowned_slots, 0)

        check = run.checks
        self.assertTrue(check["flat"], f"cross-rank join should reconcile: {check}")
        self.assertEqual(check["missing_cells"], 0)

        # The join is exact and the pads are gone: every pair of the step is on
        # some rank's experts, and none of it is padding.
        totals = run.rank_step_load
        np.testing.assert_allclose(totals.sum(axis=0), self.n_layers * self.tokens * self.topk)
        self.assertEqual(run.pad_subtracted, self.n_ranks * self.n_steps * self.n_layers * self.pads)
        self.assertEqual(run.pad_misplaced, 0)
        # The experts' own totals are the whole reconstruction: with the pads out,
        # nothing was counted twice and nothing was lost.
        np.testing.assert_allclose(run.layer_expert.sum(axis=1), self.tokens * self.topk * self.n_steps)
        # Attribution is a partition here (no replication): summing the ranks back
        # up must return the routing itself, expert for expert.
        np.testing.assert_allclose(run.rank_expert.sum(axis=0), run.layer_expert.sum(axis=0))
        self.assertAlmostEqual(
            float(run.rank_expert.sum()), self.n_layers * self.tokens * self.topk * self.n_steps
        )
        # A global dump subtracted its pads, so nothing about it should look padded.
        self.assertFalse(run.pad_suspect)

        # Each rank's load is the slice of the global routing that its experts saw,
        # and every rank sees both its own tokens and the others'.
        self.assertTrue((run.rank_total > 0).all())
        self.assertTrue(np.isfinite(run.par_agg).all())
        self.assertGreater(run.par_agg.max(), 1.0 + 1e-6, "the routing's skew should show up")

    def test_the_scope_reaches_the_report_payload(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        payload = self.mod.build_payload([run])["runs"][0]
        self.assertEqual(payload["scope"], "global")
        self.assertEqual(len(payload["expert_holders"]), self.global_experts)
        self.assertEqual(payload["pad"]["subtracted"], run.pad_subtracted)
        self.assertEqual(len(payload["pad"]["by_rank"]), self.n_ranks)
        self.assertEqual(sum(row[1] for row in payload["pad"]["by_rank"]), run.pad_subtracted)

    def test_a_missing_pad_index_is_reported_not_guessed(self):
        # The pad expert is named in the histogram's own space; an index the row does
        # not have means the two sides disagree, and those pads stay in the counts.
        layer = _fake_layer(300, self.local_experts, self.global_experts, ep_rank=0, ep_size=self.n_ranks)
        for step in range(3):
            for li in range(2):
                moe_par_stats.note_route_counts(torch.zeros(self.global_experts, dtype=torch.int32), 6, 99)
                moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")
        moe_par_stats._shutdown()
        run = self.mod.Run("broken", self.tmpdir).finalize()
        self.assertEqual(run.pad_subtracted, 0)
        self.assertEqual(run.pad_misplaced, 6)
        self.assertTrue(any("pad" in w for w in run.warnings), run.warnings)

    def test_a_dump_without_the_exit_summary_says_unknown_not_zero(self):
        # A forked worker never runs the exit hook, and before the summary line was
        # rewritten at every flush that cost the whole pad ledger. Reading its absence
        # as "no pad" would report a subtracted run as an unsubtracted one, so the
        # analysis must say it does not know.
        self._write_all_ranks()
        for path in self.mod.find_dumps(self.tmpdir):
            kept = [ln for ln in open(path).read().splitlines() if not ln.startswith("# meta partial_steps=")]
            with open(path, "w") as handle:
                handle.write("\n".join(kept) + "\n")

        run = self.mod.Run("moonep", self.tmpdir).finalize()
        self.assertTrue(run.tail_missing)
        self.assertTrue(run.pad["tail_missing"])
        self.assertEqual(run.pad_subtracted, 0, "the ledger is gone from the dump")
        self.assertTrue(any("退出汇总行" in w for w in run.warnings), run.warnings)
        # The report reads the flag off the payload, so it has to be in there.
        payload = self.mod.build_payload([run])["runs"][0]
        self.assertTrue(payload["pad"]["tail_missing"])


if __name__ == "__main__":
    unittest.main()
