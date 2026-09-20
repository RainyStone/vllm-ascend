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

import contextlib
import importlib.util
import io
import os
import re
import shutil
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
        moe_par_stats._partial_steps = 0
        moe_par_stats._current_method = "?"
        moe_par_stats._route_note = None
        moe_par_stats._route_pairs_total = 0
        moe_par_stats._pad_pairs_subtracted = 0
        moe_par_stats._pad_notes_used = 0
        moe_par_stats._pad_notes_misplaced = 0
        moe_par_stats._pad_expert_used = None
        moe_par_stats._negative_cells = 0
        moe_par_stats._compute_note = None
        moe_par_stats._buf_raw = None
        moe_par_stats._raw_geo = None
        moe_par_stats._compute_ords.clear()
        moe_par_stats._compute_rows_total = 0
        moe_par_stats._compute_pairs_total = 0
        moe_par_stats._pad_rows.clear()
        moe_par_stats._pad_rows_total = 0
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
        # The pad is announced here but its row lives in the window, so it is not in
        # the ledger yet: the total follows the rows that got written.
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 0)
        for layer in (self.l1, self.l0):  # the repeat closes step 0
            self._dispatch(layer, [0, 0, 0, 0, 0, 0, 0, 0])
        moe_par_stats._shutdown()
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any("counts=[1,0,4,0,0,0,0,0]" in ln for ln in body), body)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 6)

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
        # Eleven of the sixteen announced pairs: the third pad of this pair of steps is
        # still in the window, and the ledger counts the rows the dump holds (the
        # invariant itself is pinned in TestPadLedgerFollowsTheRows).
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 11)
        self.assertIn("pad_pairs_subtracted=11", second[0])
        # The rows written before the second flush survive the rewrite.
        body = [ln for ln in self._read_dump() if ln.startswith("step=")]
        self.assertTrue(any(ln.startswith("step=1 ") for ln in body), body)
        self.assertTrue(any(ln.startswith("step=3 ") for ln in body), body)

    def test_no_rows_are_lost_by_the_rewrite(self):
        self._two_steps([1, 1, 0, 0, 0, 0, 0, 0])
        self._two_steps([1, 1, 0, 0, 0, 0, 0, 0])
        steps = sorted({int(re.match(r"step=(\d+)", ln).group(1)) for ln in self._read_dump() if ln.startswith("step=")})
        self.assertEqual(steps, [0, 1, 2, 3], steps)


class TestPadLedgerFollowsTheRows(MoeParStatsTestCase):
    """``pad_pairs_subtracted`` counts the pad rows the dump holds, not the notes.

    It used to be accrued when the dispatcher announced the pad -- one layer ahead of
    the flush that writes it -- so the total ran ahead of the ``scope=pad`` rows by the
    layer still in flight and the analysis cross-check flagged a dump that was exact.
    """

    flush_every = 1

    def setUp(self):
        super().setUp()
        self.l0 = _fake_layer(51, 4, 8, ep_rank=0, ep_size=2)
        self.l1 = _fake_layer(52, 4, 8, ep_rank=0, ep_size=2)

    def _dispatch(self, layer, pad_pairs=0, pad_index=-1):
        moe_par_stats.note_route_counts(
            torch.tensor([9, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int32), pad_pairs, pad_index
        )
        moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")

    def _pad_pairs_written(self):
        total = 0
        for line in self._read_dump():
            if "scope=pad " in line:
                total += sum(int(v) for v in re.search(r"counts=\[([^\]]*)\]", line).group(1).split(","))
        return total

    def _meta_total(self):
        tails = [ln for ln in self._read_dump() if ln.startswith("# meta partial_steps=")]
        if not tails:
            return None
        return int(re.search(r"pad_pairs_subtracted=(\d+)", tails[-1]).group(1))

    def test_a_pad_is_not_counted_before_its_row_is_written(self):
        self._dispatch(self.l0, pad_pairs=5, pad_index=0)
        self.assertEqual(self._read_dump(), [], "the window has not been flushed")
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 0)
        self._dispatch(self.l0, pad_pairs=0)  # the repeat closes the step and flushes
        self.assertEqual(self._pad_pairs_written(), 5)
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 5)
        self.assertEqual(self._meta_total(), 5)

    def test_the_total_never_runs_ahead_of_the_rows(self):
        for i, pads in enumerate((4, 5, 6, 3, 2, 1)):
            self._dispatch((self.l0, self.l1)[i % 2], pad_pairs=pads, pad_index=0)
            written = self._pad_pairs_written()
            self.assertEqual(moe_par_stats._pad_pairs_subtracted, written, f"after dispatch {i}")
            meta = self._meta_total()
            if meta is not None:
                self.assertEqual(meta, written, f"the # meta line disagrees after dispatch {i}")
        moe_par_stats._shutdown()
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, self._pad_pairs_written())
        self.assertEqual(self._pad_pairs_written(), sum((4, 5, 6, 3, 2, 1)))


class TestLedgerLine(MoeParStatsTestCase):
    """``MOONEP_BALANCE_STATS=1`` prints the ledger the dump itself holds.

    A served run is force-killed, so its workers never reach an exit hook: this line
    is the only ledger left in the log. It is printed from ``_flush`` out of the same
    totals the ``# meta`` line carries, so the log cannot disagree with the dump.
    """

    flush_every = 1

    def setUp(self):
        super().setUp()
        self._patch(moe_par_stats, "_BALANCE_ON", True)
        self._patch(moe_par_stats, "_BALANCE_EVERY", 1)
        self.l0 = _fake_layer(61, 2, 8, ep_rank=0, ep_size=4)
        self.l1 = _fake_layer(62, 2, 8, ep_rank=0, ep_size=4)

    def _dispatch(self, layer):
        """One layer: 13 real pairs sent, 4 pad pairs, 17 (token, expert) pairs run.

        The rank keeps experts 0 and 1, and prefetches expert 2 into a slot, so its
        compute row is ``[9, 2, 6]`` home-first -- the pads land on its own expert 0,
        which is what the dispatcher's home-first padding does.
        """
        moe_par_stats.note_route_counts(torch.tensor([9, 2, 6, 0, 0, 0, 0, 0], dtype=torch.int32), 4, 0)
        moe_par_stats.note_compute_load(
            torch.tensor([9, 2, 6], dtype=torch.int32), 0, 2, torch.tensor([2], dtype=torch.int64)
        )
        moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")

    def _steps(self, count):
        """``count`` complete steps of two layers; the third dispatch closes a step."""
        for _ in range(count):
            self._dispatch(self.l0)
            self._dispatch(self.l1)

    def _ledger(self, buf):
        return [ln for ln in buf.getvalue().splitlines() if ln.startswith("[moe-par-ledger")]

    def _fields(self, line):
        """The line's ``key=value`` pairs, ignoring the ``(N/layer)`` annotations."""
        return {m.group(1): int(m.group(2)) for m in re.finditer(r"([a-z_]+)=(\d+)", line)}

    def test_the_line_carries_the_dumps_own_totals(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._steps(2)
            moe_par_stats._shutdown()
        lines = self._ledger(buf)
        self.assertEqual(len(lines), 2, lines)
        # One step's worth: 2 layers x (13 rows, 4 pads, 17 compute).
        self.assertEqual(self._fields(lines[0]), {"steps": 1, "rows": 26, "pads": 8, "compute": 34})
        self.assertEqual(self._fields(lines[1]), {"steps": 2, "rows": 52, "pads": 16, "compute": 68})
        self.assertNotIn("dropped=", lines[1], "every pair that was sent was computed")
        # And the same totals are the ones the dump ends with.
        tail = next(ln for ln in self._read_dump() if ln.startswith("# meta partial_steps="))
        self.assertIn("pad_pairs_subtracted=16", tail)
        self.assertIn("compute_rows_total=4 compute_pairs_total=68", tail)
        rows = sum(
            sum(int(v) for v in re.search(r"counts=\[([^\]]*)\]", ln).group(1).split(","))
            for ln in self._read_dump()
            # The primary rows: the routing left after the pad block came out.
            if ln.startswith("step=") and "scope=compute" not in ln and "scope=pad" not in ln
        )
        self.assertEqual(rows, 52)
        self.assertEqual(moe_par_stats._route_pairs_total, rows)

    def test_the_interval_is_layers_and_not_steps(self):
        # MOONEP_BALANCE_STATS_EVERY is documented in layers, because that is the unit
        # the operator sees in the model; 4 layers of a 2-layer step is every 2 steps.
        with patch.object(moe_par_stats, "_BALANCE_EVERY", 4):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._steps(2)
                moe_par_stats._shutdown()
        lines = self._ledger(buf)
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(self._fields(lines[0])["steps"], 2)

    def test_nothing_is_printed_unless_asked(self):
        for name, value in (("_BALANCE_ON", False), ("_BALANCE_EVERY", 0)):
            self._reset()
            buf = io.StringIO()
            with patch.object(moe_par_stats, name, value), contextlib.redirect_stdout(buf):
                self._steps(2)
                moe_par_stats._shutdown()
            self.assertEqual(self._ledger(buf), [], f"{name}={value} must stay quiet")


class TestAnalyzer(TestBase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_analyzer()

    def test_held_mask_reads_the_map_as_a_holding(self):
        # expert_map is indexed by global id and holds the local slot, so "not -1"
        # is "this rank holds it" -- the reading that survives replication.
        mask, source = self.mod._held_mask({"expert_map": [-1, -1, 0, 1]}, 2, 4, ep_rank=1)
        self.assertEqual(list(mask), [False, False, True, True])
        self.assertEqual(source, self.mod._SRC_EXPERT_MAP)

    def test_held_mask_reads_replication_as_two_holders(self):
        # The same logical expert in two ranks' maps: a copy, not a contradiction.
        mask, _ = self.mod._held_mask({"expert_map": [0, -1, 0, -1]}, 2, 4, ep_rank=0)
        self.assertEqual(list(mask), [True, False, True, False])

    def test_held_mask_falls_back_to_a_contiguous_partition(self):
        mask, source = self.mod._held_mask({}, 2, 8, ep_rank=2)
        self.assertEqual(list(mask), [False, False, False, False, True, True, False, False])
        self.assertEqual(source, self.mod._SRC_IDENTITY)

    def test_held_mask_covers_everything_when_the_table_is_whole(self):
        mask, source = self.mod._held_mask({}, 8, 8, ep_rank=0)
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
        totals = run.compute["real"]["rank_step_load"]
        self.assertTrue(np.isfinite(totals).all())
        np.testing.assert_allclose(totals.sum(axis=0), self.n_layers * self.tokens * self.topk)

        # A local-scope dump is the degenerate case of the three ledgers: nothing on
        # this path pads, so 真实 and 合计 are the same array and pad is a zeroed
        # copy. The ledger the report switches between must still exist, or every
        # panel would need a second code path for the non-MoonEP legs.
        np.testing.assert_allclose(run.compute["total"]["rank_step_load"], totals)
        self.assertTrue((run.compute["pad"]["rank_step_load"] == 0).all())
        identities = run.checks["identities"]
        self.assertEqual(identities["real_error"], 0.0, identities)
        self.assertEqual(identities["pad_injected"], 0.0, identities)
        self.assertFalse(identities["capacity_pinned"], "no capacity envelope without MoonEP")

        # PAR is only defined across ranks and must land in (1, n_ranks]. The upper
        # bound is what makes the join meaningful -- a run whose ranks were never
        # told apart would sit at exactly 1.
        for name, values in (("per-layer", run.compute["real"]["par_by_step"]),
                             ("aggregate", run.compute["real"]["par_agg"])):
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
        self.assertTrue(np.isnan(run.compute["real"]["par_by_step"]).all())
        self.assertTrue(np.isnan(run.compute["real"]["par_agg"]).all())
        self.assertTrue(any("只有一个 rank" in w for w in run.notes), run.notes)
        # The load data itself is still there; only the ratio is undefined.
        self.assertTrue(np.isfinite(run.compute["real"]["rank_step_load"]).all())


class TestGlobalScopeRoundTrip(MoeParStatsTestCase):
    """The MoonEP path end to end, in its one-view form: each rank dumps the
    dispatcher's *sending-side* histogram over the model's whole expert space, and the
    analysis sums the ranks of a (step, layer) into the run's real routing. What it
    does not get from this dump is who ran those rows -- that is the second view, and
    a dump without it says so rather than reconstructing it from placement."""

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

    def test_the_sending_side_is_the_routing(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()

        self.assertEqual(run.scope, "global")
        self.assertEqual(run.row_width, self.global_experts, "a global row spans every expert")
        self.assertEqual(run.local_experts, self.local_experts, "read off the header, not the row")
        self.assertEqual(run.duplicated_experts, 0)

        # This dump has the routing view and nothing else, and the *expert* dimension
        # is what that view answers: the sending side of every rank, summed, is the
        # step's routing -- no placement, no attribution, no rank axis needed.
        check = run.checks
        self.assertFalse(check["applicable"], check)
        self.assertFalse(check["identities"]["applicable"], check["identities"])
        # Every (rank, step, layer) cell is unread rather than zero: the rank dimension
        # is what would fill them, and this dump does not carry it.
        self.assertEqual(check["missing_cells"], self.n_ranks * self.n_steps * self.n_layers)
        self.assertTrue(np.isnan(run.load_real).all())
        self.assertIsNone(run.ledger["compute_total"])

        # The experts' own totals are the whole reconstruction: with the pads out,
        # nothing was counted twice and nothing was lost. The expert dimension is
        # the one this view is for -- it needs no placement and no attribution.
        np.testing.assert_allclose(run.layer_expert.sum(axis=1), self.tokens * self.topk * self.n_steps)
        np.testing.assert_allclose(run.expert_total.sum(), self.n_layers * self.tokens * self.topk * self.n_steps)
        self.assertEqual(run.pad_subtracted, self.n_ranks * self.n_steps * self.n_layers * self.pads)
        self.assertEqual(run.pad_misplaced, 0)
        # The pads the injectors reported land on the injector's own first expert,
        # which is where the split rule will look for them.
        self.assertEqual(run.pad_experts, [rank * self.local_experts for rank in range(self.n_ranks)])
        self.assertEqual(float(run.expert_pad_total.sum()), run.pad_subtracted)
        self.assertTrue((run.layer_expert[:, run.pad_experts] >= 0).all())
        # A global dump subtracted its pads, so nothing about it should look padded.
        self.assertFalse(run.pad_suspect)

    def test_a_routing_only_dump_leaves_the_rank_dimension_missing(self):
        # Sending-side rows say what was routed, not who ran it. Reconstructing the
        # rank dimension from `expert_map` would be the owner attribution this view
        # replaced, so the analyzer reports it as missing and says why -- a rank
        # column of zeros would read as a measurement.
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()

        self.assertFalse(run.has_compute)
        self.assertTrue(np.isnan(run.load_real).all(), "no compute rows means no rank dimension")
        self.assertTrue(np.isnan(run.compute["real"]["rank_total"]).all())
        self.assertTrue(any("没有 scope=compute 行" in w for w in run.warnings), run.warnings)
        # The expert dimension is unaffected: it never needed the rank axis.
        self.assertGreater(float(run.expert_total.sum()), 0.0)
        # The identity check needs a rank axis to sum over, and says so instead of
        # reporting `0 != 路由总量` as a failure it cannot have measured.
        self.assertFalse(run.checks["identities"]["applicable"], run.checks["identities"])

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


class TestComputeNote(MoeParStatsTestCase):
    """The receive-side row: the rows a rank ran, as opposed to the ones routed to it.

    MoonEP hands the recorder its own counts, laid out as ``[home experts | prefetch
    slot groups]``. The row has to come out in global expert coordinates, or the two
    accounting views could not be compared expert for expert.
    """

    global_experts = 8
    local_experts = 2

    def setUp(self):
        super().setUp()
        self.layer = _fake_layer(400, self.local_experts, self.global_experts, ep_rank=0, ep_size=4)

    def _note(self, counts, home_first=0, slots=None):
        moe_par_stats.note_compute_load(
            torch.tensor(counts, dtype=torch.int32),
            home_first,
            self.local_experts,
            None if slots is None else torch.tensor(slots, dtype=torch.int64),
        )

    def _step(self, hist, compute, pads=0, pad_expert=-1, slots=None, home_first=0, received=None):
        """One layer the way MoonEP drives it: what it sent, and what it ran."""
        moe_par_stats.note_route_counts(torch.tensor(hist, dtype=torch.int32), pads, pad_expert)
        self._note(compute, home_first, slots)
        moe_par_stats.record(
            self.layer,
            torch.tensor(received or hist[: self.local_experts], dtype=torch.int32),
            1,
            "MoonEPCommImpl",
        )

    def _rows(self, scope):
        return [ln for ln in self._read_dump() if f"scope={scope} " in ln]

    def _values(self, line):
        return [int(v) for v in line.split("counts=[", 1)[1].rstrip("]").split(",")]

    def test_the_row_is_what_this_rank_ran_in_global_coordinates(self):
        self._step([4, 4, 4, 4, 0, 0, 0, 0], [4, 4])
        self._step([4, 4, 4, 4, 0, 0, 0, 0], [4, 4])
        moe_par_stats._shutdown()

        rows = self._rows("compute")
        self.assertEqual(len(rows), 2, "one compute row per layer the rank ran")
        self.assertEqual(self._values(rows[0]), [4, 4, 0, 0, 0, 0, 0, 0])

    def test_a_slot_group_lands_on_the_expert_it_carries(self):
        # Slot b holds whichever expert the plan pulled in -- a global id, not a
        # position -- and an empty slot (-1) carries nothing.
        self._step([0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 9, 5, 0], slots=[6, 3, -1])
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("compute")[0]), [2, 1, 0, 5, 0, 0, 9, 0])

    def test_the_pads_are_not_subtracted_from_the_rows_that_ran(self):
        # The routing row is real traffic, so its pads come out; the compute row is
        # the rows the matmul saw, and a pad row is one of them.
        self._step([15, 8, 0, 0, 0, 0, 0, 0], [15, 8], pads=3, pad_expert=0)
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("global")[0]), [12, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self._values(self._rows("compute")[0]), [15, 8, 0, 0, 0, 0, 0, 0])
        # And the pads are their own row, so the split is exact in the dump itself.
        self.assertEqual(self._values(self._rows("pad")[0]), [3, 0, 0, 0, 0, 0, 0, 0])

    def test_the_note_is_consumed_once(self):
        hist = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
        moe_par_stats.note_route_counts(hist, 0, -1)
        self._note([5, 5])
        moe_par_stats.record(self.layer, None, 1, "MoonEPCommImpl")
        # The next step announces its routing again but not its work, and a note is
        # per record: nothing is carried over to a layer that did not announce.
        moe_par_stats.note_route_counts(hist, 0, -1)
        moe_par_stats.record(self.layer, None, 1, "MoonEPCommImpl")
        moe_par_stats._shutdown()
        rows = self._rows("compute")
        self.assertEqual(len(rows), 1, "one announcement, one row")
        self.assertEqual(self._values(rows[0]), [5, 5, 0, 0, 0, 0, 0, 0])
        self.assertEqual(len(self._rows("global")), 2, "the routing rows are both there")

    def test_a_note_is_snapshotted_not_held(self):
        # The caller goes on to hand the same tensor to the grouped matmul as its
        # group_list; a kernel rewriting it in place must not change what was recorded.
        counts = torch.tensor([3, 2], dtype=torch.int32)
        moe_par_stats.note_route_counts(torch.tensor([3, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int32), 0, -1)
        moe_par_stats.note_compute_load(counts, 0, self.local_experts)
        counts.fill_(99)
        moe_par_stats.record(self.layer, None, 1, "MoonEPCommImpl")
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("compute")[0]), [3, 2, 0, 0, 0, 0, 0, 0])

    def test_a_home_block_that_does_not_fit_is_reported_not_guessed(self):
        self._step([0] * 8, [5, 5], home_first=7)
        moe_par_stats._shutdown()
        self.assertEqual(self._rows("compute"), [])
        # The routing row is unaffected, and the problem is said out loud once.
        self.assertEqual(len(self._rows("global")), 1)
        self.assertIn("compute-geometry", moe_par_stats._warned)

    def test_a_profile_run_records_no_compute_row(self):
        context = types.SimpleNamespace(in_profile_run=True)
        with patch.object(moe_par_stats, "_EXTRA_CTX", context):
            self._step([1, 1, 0, 0, 0, 0, 0, 0], [1, 1])
        self.assertEqual(self._read_dump(), [])

    def test_the_header_names_the_pair_of_views_and_the_exit_line_counts_them(self):
        self._step([1, 1, 0, 0, 0, 0, 0, 0], [3, 4])
        self._step([1, 1, 0, 0, 0, 0, 0, 0], [8, 6], pads=3, pad_expert=1)
        moe_par_stats._shutdown()
        lines = self._read_dump()
        header = [ln for ln in lines if ln.startswith("# meta")][0]
        self.assertIn("counts_scope=global+compute", header)
        self.assertIn("scope=global", self._rows("global")[0], "the primary rows are named when the dump is not")
        tail = [ln for ln in lines if ln.startswith("# meta partial_steps=")][0]
        self.assertIn("compute_rows_total=2", tail)
        self.assertIn("compute_pairs_total=21", tail)
        # The pad counters sit in the same line, next to the compute ones.
        self.assertIn("pad_pairs_subtracted=3", tail)
        self.assertIn("pad_rows_total=1", tail)

    def test_the_counts_of_a_rank_that_ran_only_slots_still_add_up(self):
        # The rank holds nothing of its home experts this step -- all of its work came
        # in through the slots. That is a legitimate row and must not look empty.
        self._step([0] * 8, [0, 0, 6, 6, 0], slots=[5, 7, -1], pads=2, pad_expert=1)
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("compute")[0]), [0, 0, 0, 0, 0, 6, 0, 6])

    def test_two_slots_holding_one_expert_accumulate(self):
        # A replicated expert can sit in two slots at once. Both groups' rows were
        # computed, so the fold has to add them; an assignment would silently report
        # only the second group and understate that expert's work.
        self._step([0] * 8, [2, 1, 9, 5, 0], slots=[6, 6, -1])
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("compute")[0]), [2, 1, 0, 0, 0, 0, 14, 0])

    def test_the_slot_mapping_is_copied_not_held(self):
        # The mapping is the dispatch's own plan tensor, and the next layer's plan
        # rewrites it. The row is staged raw now and folded at the flush, so the copy
        # has to happen while the layer is still the one being recorded.
        mapping = torch.tensor([6], dtype=torch.int64)
        moe_par_stats.note_route_counts(torch.tensor([0] * 8, dtype=torch.int32), 0, -1)
        moe_par_stats.note_compute_load(
            torch.tensor([2, 1, 9], dtype=torch.int32), 0, self.local_experts, mapping
        )
        moe_par_stats.record(self.layer, None, 1, "MoonEPCommImpl")
        mapping.fill_(3)
        moe_par_stats._shutdown()
        self.assertEqual(self._values(self._rows("compute")[0]), [2, 1, 0, 0, 0, 0, 9, 0])

    def test_a_slot_mapping_that_does_not_match_is_refused(self):
        # Two slot groups, one mapping entry: scattering with it would drop a group's
        # rows on the floor, so the row goes and the disagreement is said out loud.
        self._step([0] * 8, [2, 1, 9, 5], slots=[6])
        moe_par_stats._shutdown()
        self.assertEqual(self._rows("compute"), [])
        self.assertEqual(len(self._rows("global")), 1, "the routing row is unaffected")
        self.assertIn("compute-slots", moe_par_stats._warned)

    def test_a_layout_that_moves_is_refused(self):
        # The width of a compute row is fixed by the first one recorded: every row of
        # a dump is read as one expert space, so a layer whose plan has a different
        # number of slot groups cannot join it.
        self._step([0] * 8, [2, 1, 9], slots=[6])
        self._step([0] * 8, [2, 1, 9, 5], slots=[6, 3])
        moe_par_stats._shutdown()
        rows = self._rows("compute")
        self.assertEqual(len(rows), 1, "only the layout the buffer was built for")
        self.assertEqual(self._values(rows[0]), [2, 1, 0, 0, 0, 0, 9, 0])
        self.assertIn("compute-layout", moe_par_stats._warned)
        self.assertEqual(len(self._rows("global")), 2, "both routing rows are there")

    def test_a_multi_step_window_takes_each_steps_pads_off_its_own_row(self):
        # The pads are subtracted at the flush, on the whole window at once, so each
        # one has to find its own (step, layer) rather than the newest.
        self._step([15, 8, 0, 0, 0, 0, 0, 0], [15, 8], pads=3, pad_expert=0)
        self._step([9, 4, 0, 0, 0, 0, 0, 0], [9, 4], pads=4, pad_expert=1)
        moe_par_stats._shutdown()
        rows = [self._values(ln) for ln in self._rows("global")]
        self.assertEqual(rows[0], [12, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(rows[1], [9, 0, 0, 0, 0, 0, 0, 0])
        # Each step's own row, and the pads sum to what the exit line claims.
        self.assertEqual(moe_par_stats._pad_pairs_subtracted, 7)
        self.assertEqual(moe_par_stats._pad_expert_used, 1)


class TestThreeLedgerRoundTrip(MoeParStatsTestCase):
    """A MoonEP dispatch whose plan relocated rows and dropped some: one load, three
    ledgers, and the split has to hold all of them together.

    The construction is fixed by hand, which is the point: every number asserted below
    is arithmetic on this docstring's own setup rather than a re-derivation by the code
    under test. Per (step, layer), over ``n_ranks = 4`` ranks holding 2 experts each:

    - 128 pairs are routed, 44 of them to expert 2 -- rank 1's expert, the hotspot;
    - 20 of expert 2's rows were pulled into prefetch slots held by ranks 0, 2 and 3
      (7, 7 and 6 rows), so three ranks run rows whose expert rank 1 owns;
    - 5 rows of expert 4 (rank 2's) fell to the capacity envelope and were never run;
    - every rank pads its own first expert (0, 2, 4, 6) with 6 pairs per layer.

    合计 (= 真实 + pad, what each rank ran) is therefore 37/42/32/36, and 真实 (the
    routed rows, credited to whoever ran them) is 24/56/24/24: pad is what separates
    the two ledgers, and the relocation is what makes the expert axis -- 44 rows of
    expert 2, run by four different ranks -- differ from the rank axis.
    """

    n_ranks = 4
    global_experts = 8
    local_experts = 2
    topk = 2
    n_layers = 3
    n_steps = 5
    tokens = 64
    pads = 6
    routed = (12, 12, 44, 12, 12, 12, 12, 12)
    moved = {0: 7, 2: 7, 3: 6}  # rank -> rows of the hot expert pulled into its slot
    drop = 5  # pairs the capacity envelope rejected
    hot_expert = 2  # rank 1's: the rank that owns it is not the one that runs all of it
    dropped_expert = 4  # rank 2's

    @property
    def step_layers(self):
        return self.n_steps * self.n_layers

    def _receive(self, rank):
        """The counts this rank runs: its home block, minus what left, plus its slots."""
        home = rank * self.local_experts
        counts = [self.routed[home] + self.pads, self.routed[home + 1]]
        if home == self.hot_expert:
            # The hot expert's owner runs whatever the plan did not relocate.
            counts[0] -= sum(self.moved.values())
        if home == self.dropped_expert:
            counts[0] -= self.drop
        slots = []
        if rank in self.moved:
            counts.append(self.moved[rank])
            slots.append(self.hot_expert)
        if rank == self.n_ranks - 1:
            counts.append(0)  # the plan left this slot empty
            slots.append(-1)
        return counts, (slots or None)

    def _write_all_ranks(self):
        layers = [
            _fake_layer(500 + i, self.local_experts, self.global_experts, ep_rank=0, ep_size=self.n_ranks,
                        topk=self.topk)
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
                    for layer in layers:
                        home = rank * self.local_experts
                        # The sending side: `topk_c` splits a step's routing over the
                        # ranks, so the sum over the ranks is the step's routing again.
                        split = np.tile(np.array(self.routed, dtype=np.int64) // self.n_ranks,
                                        (self.n_ranks, 1))
                        split[0] += np.array(self.routed) - split.sum(axis=0)
                        padded = split[rank].copy()
                        padded[home] += self.pads
                        counts, slots = self._receive(rank)
                        moe_par_stats.note_route_counts(torch.tensor(padded, dtype=torch.int32), self.pads, home)
                        moe_par_stats.note_compute_load(
                            torch.tensor(counts, dtype=torch.int32),
                            home,
                            self.local_experts,
                            None if slots is None else torch.tensor(slots, dtype=torch.int64),
                        )
                        moe_par_stats.record(layer, None, 1, "MoonEPCommImpl")
                moe_par_stats._shutdown()

    def test_the_three_ledgers_add_up_to_the_rows_each_rank_ran(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()

        self.assertEqual(run.scope, "global+compute")
        self.assertEqual(run.primary, "global")
        self.assertTrue(run.has_compute)
        self.assertTrue(run.checks["flat"], f"the join must still reconcile: {run.checks}")
        self.assertEqual(run.checks["missing_cells"], 0)

        sl = self.step_layers
        # 合计: exactly the rows each rank wrote, straight off its compute row.
        np.testing.assert_allclose(run.compute["total"]["rank_total"], [37 * sl, 42 * sl, 32 * sl, 36 * sl])
        # 真实: the routed rows that rank ran. Rank 1's own expert was its hotspot, so
        # it carries the whole 44 even though 20 of those rows were run elsewhere --
        # that is the one agreed exception of the split, not an accident.
        np.testing.assert_allclose(run.compute["real"]["rank_total"], [24 * sl, 56 * sl, 24 * sl, 24 * sl])
        # pad 行 = 合计 - 真实, per rank, and it is allowed to be negative where a rank
        # ran *fewer* rows of the pad expert than the routing it is credited with
        # (rank 1: 30 rows of expert 2 in its own block, 44 credited to it).
        np.testing.assert_allclose(run.compute["pad"]["rank_total"],
                                   [13 * sl, -14 * sl, 8 * sl, 12 * sl])
        # 合计 = 真实 + pad, cell by cell and in total: the three ledgers are one load
        # split in two, not three measurements.
        np.testing.assert_allclose(run.load_real + run.load_pad, run.load_total, equal_nan=True)
        for ledger in ("real", "pad", "total"):
            pack = run.compute[ledger]
            # The whole-run figure is the per-step table summed over the steps -- the
            # report shows both, so they have to be the same reduction.
            np.testing.assert_allclose(pack["rank_step_load"].sum(axis=1), pack["rank_total"],
                                       rtol=0, atol=1e-6)

    def test_the_expert_axis_needs_no_attribution(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        sl = self.step_layers

        # Expert totals are the routing itself (pads already subtracted by the
        # recorder) -- placement, relocation and drops cannot move them.
        for expert in range(self.global_experts):
            self.assertAlmostEqual(run.layer_expert[:, expert].sum(), self.routed[expert] * sl,
                                   msg=f"expert {expert} routing")
            self.assertAlmostEqual(run.pad_layer_expert[:, expert].sum(),
                                   self.pads * sl if expert % 2 == 0 else 0, msg=f"expert {expert} pads")
        self.assertAlmostEqual(float(run.expert_total.sum()), 128 * sl)
        self.assertAlmostEqual(float(run.expert_pad_total.sum()), self.n_ranks * self.pads * sl)
        self.assertEqual(run.pad_experts, [0, 2, 4, 6], "the injectors' own first experts")

    def test_the_pad_experts_real_rows_go_to_their_first_holder(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        sl = self.step_layers
        real = run.rank_expert_real

        # Expert 2 belongs to rank 1, so the whole 44 lands there even though ranks 0,
        # 2 and 3 ran 7, 7 and 6 of those rows -- the agreed exception, and the reason
        # its column can read negative.
        self.assertAlmostEqual(real[1, 2], 44 * sl)
        self.assertEqual(float(real[[0, 2, 3], 2].sum()), 0.0)
        # The relocation is what comes out negative: rank 1 ran 30 rows of its own
        # expert but is credited with all 44. Cells here are (rank, step, layer), so
        # the one rank shows up once per cell, and the pairs are the 14 rows a layer.
        self.assertEqual(run.pad_negative_cells, sl, "one rank, in every (step, layer) cell")
        self.assertAlmostEqual(run.pad_negative_pairs, 14 * sl)
        # The expert-level split and the rank-level split are the same load read two
        # ways, so summing one over the experts has to land on the other. Everywhere
        # but the pad experts the two are the same row, which is what makes this exact.
        np.testing.assert_allclose(run.rank_expert_real.sum(axis=1), run.load_real.sum(axis=(1, 2)))
        np.testing.assert_allclose(run.rank_expert_pad.sum(axis=1), run.load_pad.sum(axis=(1, 2)))
        # And they add back up to what was computed, cell by cell.
        np.testing.assert_allclose(run.rank_expert_real + run.rank_expert_pad,
                                   run.compute_rank_expert)

    def test_the_identities_survive_a_dropped_row(self):
        # A dropped row belongs to 真实行 or pad 行 and the dump does not say which, so
        # neither identity can be checked to zero on its own -- but the pair is exact:
        # their sum is the computed total, therefore `① + ② == −丢弃`, both <= 0.
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        idn = run.checks["identities"]
        sl = self.step_layers

        self.assertTrue(idn["applicable"])
        self.assertAlmostEqual(idn["route_total"], 128 * sl)
        self.assertAlmostEqual(idn["pad_injected"], self.n_ranks * self.pads * sl)
        self.assertAlmostEqual(idn["compute_total"], 147 * sl)
        self.assertAlmostEqual(idn["drop_total"], self.drop * sl)
        # The 5 dropped rows were expert 4's -- a pad expert -- so ① is still exact and
        # ② carries the whole drop.
        self.assertEqual(idn["real_error"], 0.0)
        self.assertAlmostEqual(idn["pad_error"], -self.drop * sl)
        self.assertAlmostEqual(idn["error_sum"], -idn["drop_total"])
        self.assertTrue(idn["drop_accounted"])
        # The check reads every (rank, step, layer) cell, so this fixture is *not*
        # capacity-pinned: its ranks run 37/42/32/36 rows a cell, which is what makes
        # the rank ledger interesting here. A real MoonEP leg has the same number in
        # every cell, and only then does the report call the 合计 PAR 1.000 a
        # construction result rather than a measurement.
        self.assertFalse(idn["capacity_pinned"])
        self.assertEqual(idn["cell_total_min"], 32)
        self.assertEqual(idn["cell_total_max"], 42)

    def test_the_rank_ledger_keeps_the_sending_side_as_a_reference(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        ledger = run.ledger
        sl = self.step_layers

        self.assertAlmostEqual(ledger["route_total"], 128 * sl)
        self.assertAlmostEqual(ledger["pad_total"], self.n_ranks * self.pads * sl)
        self.assertAlmostEqual(ledger["sent_total"], ledger["route_total"] + ledger["pad_total"])
        self.assertAlmostEqual(ledger["compute_total"], 147 * sl)
        self.assertAlmostEqual(ledger["real_total"], 128 * sl, "no real row was dropped")
        self.assertAlmostEqual(ledger["compute_pad_total"], ledger["pad_total"] - self.drop * sl)

        by_rank = {entry["rank"]: entry for entry in ledger["per_rank"]}
        # The sending side is the neutral reference, and here it is the boring one:
        # every rank's own tokens routed 32 pairs a layer, so the skew the rank ledger
        # shows comes entirely from the dispatch, not from the input. The injected
        # pads are their own column -- each rank pads its own first expert by 6, and
        # that is a different quantity from what it sent.
        np.testing.assert_allclose([by_rank[r]["sent"] for r in range(4)], [32 * sl] * 4)
        np.testing.assert_allclose([by_rank[r]["injected"] for r in range(4)], [self.pads * sl] * 4)
        self.assertAlmostEqual(float(np.sum([by_rank[r]["sent"] for r in range(4)])), ledger["route_total"])
        np.testing.assert_allclose([by_rank[r]["real"] for r in range(4)], [24 * sl, 56 * sl, 24 * sl, 24 * sl])
        np.testing.assert_allclose([by_rank[r]["total"] for r in range(4)], [37 * sl, 42 * sl, 32 * sl, 36 * sl])
        self.assertAlmostEqual(by_rank[1]["real_share"], 56 / 42, places=6)
        self.assertTrue(all("pad_est" not in entry for entry in ledger["per_rank"]),
                        "the pro-rata estimate went away with the owner view")

    def test_the_payload_carries_the_three_ledgers(self):
        self._write_all_ranks()
        run = self.mod.Run("moonep", self.tmpdir).finalize()
        payload = self.mod.build_payload([run])["runs"][0]

        self.assertEqual(payload["scope"], "global+compute")
        self.assertTrue(payload["has_compute"])
        # Every key the report's panels read has to be here for each ledger, or the
        # page renders blank exactly when someone switches books.
        for ledger in ("real", "pad", "total"):
            view = payload["compute"][ledger]
            for key in ("par_layer", "par_by_step", "par_agg", "par_whole", "rank_total",
                        "rank_step_load", "per_rank_mean", "per_rank_max", "tokens_per_step",
                        "step_pairs", "load"):
                self.assertIn(key, view, f"compute.{ledger}.{key}")
            self.assertEqual(len(view["rank_total"]), self.n_ranks)
            self.assertEqual(np.shape(view["par_layer"]), (self.n_steps, self.n_layers))
        self.assertEqual(np.shape(payload["compute"]["rank_expert"]), (self.n_ranks, self.global_experts))
        self.assertEqual(len(payload["expert_total"]), self.global_experts)
        self.assertEqual(len(payload["expert_pad_total"]), self.global_experts)
        self.assertEqual(payload["split"]["pad_experts"], [0, 2, 4, 6])
        self.assertEqual(payload["checks"]["identities"]["drop_total"], self.drop * self.step_layers)
        # The pad ledger reaches the report on its own, so 合计 can be split there.
        self.assertAlmostEqual(payload["ledger"]["compute_total"], 147 * self.step_layers)
        self.assertAlmostEqual(payload["ledger"]["pad_total"], run.pad_subtracted)
        self.assertEqual(payload["pad"]["rows"], self.n_ranks * self.step_layers)
        self.assertAlmostEqual(payload["pad"]["pairs"], run.pad_subtracted)



class TestReportJavaScript(TestBase):
    """The report is one HTML string carrying its own script, and nothing parses it.

    A stray brace in the template leaves every panel blank in the browser while the
    analyzer still exits 0 and writes the file -- the HTML itself stays well-formed,
    so no other check in this suite can see it. Hence a test rather than a comment.
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_analyzer()

    @unittest.skipUnless(shutil.which("node"), "node 不在 PATH 上；报告本身不依赖它，跳过这一项")
    def test_the_template_parses(self):
        self.assertEqual(self.mod.report_js_problems(self.mod._TEMPLATE), [])

    @unittest.skipUnless(shutil.which("node"), "node 不在 PATH 上；报告本身不依赖它，跳过这一项")
    def test_a_stray_brace_in_the_template_is_caught(self):
        broken = self.mod._TEMPLATE.replace("function renderAll() {", "function renderAll() { {")
        problems = self.mod.report_js_problems(broken)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("白屏", problems[0])


if __name__ == "__main__":
    unittest.main()
