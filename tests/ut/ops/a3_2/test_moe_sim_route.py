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
import ast
from types import SimpleNamespace

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.ops.fused_moe import experts_selector, moe_sim_route

NUM_EXPERTS = 256
TOP_K = 6
CAP = 100.0 / TOP_K

# A cap-respecting skewed profile: 16% to expert 0, 10% each to experts 1..5, the
# remaining 34% spread over the other 250. Illustrates the shape a real hotspot takes.
SKEWED = [16.0] + [10.0] * 5 + [(100.0 - 16.0 - 50.0) / (NUM_EXPERTS - 6)] * (NUM_EXPERTS - 6)
UNIFORM = [100.0 / NUM_EXPERTS] * NUM_EXPERTS


@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    """The module caches the parsed profile and the per-T tables process-wide.

    The per-expert table is off by default here; only the test that asserts on it
    turns it back on, so the other cases do not each print 259 lines.
    """
    monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_TABLE", "0")
    moe_sim_route._PROFILE = None
    moe_sim_route._PROFILE_MISS = False
    moe_sim_route._VALIDATED_FOR = None
    moe_sim_route._CACHE.clear()
    moe_sim_route._LAST_T = None
    moe_sim_route._SINCE_FULL = 0
    moe_sim_route._ALLOC_ID = 0
    moe_sim_route._EP_WORLD = None
    moe_sim_route._MIX_WARNED = False
    moe_sim_route._SMALL_T_WARNED = False
    yield


def make_hidden(tokens: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.zeros(tokens, 16, dtype=dtype)


class TestParseSpec(PytestBase):
    def test_full_list(self):
        assert moe_sim_route._parse_spec("50,30,20") == [50.0, 30.0, 20.0]

    def test_full_list_tolerates_newlines_and_blanks(self):
        assert moe_sim_route._parse_spec(" 50 ,\n30,\n\n20 \n") == [50.0, 30.0, 20.0]

    def test_sparse_returns_mapping(self):
        assert moe_sim_route._parse_spec("0:50,7:30") == {0: 50.0, 7: 30.0}

    def test_at_file_with_comments_and_newlines(self, tmp_path):
        spec_file = tmp_path / "skew.txt"
        spec_file.write_text("# hotspot profile\n50,30,20\n")
        assert moe_sim_route._parse_spec(f"@{spec_file}") == [50.0, 30.0, 20.0]

    def test_at_file_sparse(self, tmp_path):
        spec_file = tmp_path / "skew.txt"
        spec_file.write_text("0:16   # the hotspot\n7:10\n")
        assert moe_sim_route._parse_spec(f"@{spec_file}") == {0: 16.0, 7: 10.0}


class TestMaterialize(PytestBase):
    def test_sparse_expands_and_splits_remainder(self):
        profile = moe_sim_route._materialize({0: 16.0, 7: 10.0}, NUM_EXPERTS)
        assert len(profile) == NUM_EXPERTS
        assert profile[0] == pytest.approx(16.0)
        assert profile[7] == pytest.approx(10.0)
        assert profile[1] == pytest.approx(74.0 / 254)  # 74% over the 254 unlisted
        assert sum(profile) == pytest.approx(100.0)

    def test_full_list_passes_through(self):
        assert moe_sim_route._materialize(SKEWED, NUM_EXPERTS) is SKEWED

    def test_sparse_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="256 experts"):
            moe_sim_route._materialize({300: 50.0}, NUM_EXPERTS)

    def test_sparse_over_100_rejected(self):
        with pytest.raises(ValueError, match="more than 100%"):
            moe_sim_route._materialize({0: 60.0, 1: 50.0}, NUM_EXPERTS)

    def test_sparse_without_expert_count_rejected(self):
        with pytest.raises(ValueError, match="expert count"):
            moe_sim_route._materialize({0: 50.0}, -1)


class TestValidate(PytestBase):
    def test_accepts_valid_profiles(self):
        moe_sim_route._validate(SKEWED, NUM_EXPERTS, TOP_K)
        moe_sim_route._validate(UNIFORM, NUM_EXPERTS, TOP_K)

    def test_accepts_expert_at_exactly_the_cap(self):
        at_cap = [CAP] + [(100.0 - CAP) / (NUM_EXPERTS - 1)] * (NUM_EXPERTS - 1)
        moe_sim_route._validate(at_cap, NUM_EXPERTS, TOP_K)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="255 entries"):
            moe_sim_route._validate([1.0] * 255, NUM_EXPERTS, TOP_K)

    def test_negative_share(self):
        with pytest.raises(ValueError, match="negative share at expert 2"):
            moe_sim_route._validate([50.0, 51.0, -1.0] + [0.0] * (NUM_EXPERTS - 3), NUM_EXPERTS, TOP_K)

    def test_sum_must_be_100(self):
        with pytest.raises(ValueError, match="sum to"):
            moe_sim_route._validate([1.0] * NUM_EXPERTS, NUM_EXPERTS, TOP_K)

    def test_share_above_one_over_k_rejected(self):
        # The point of the cap: one token's top_k experts are distinct, so no expert can
        # take more than 1/top_k of the routes. 50% to a single expert is unreachable.
        over = [CAP + 0.5] + [(100.0 - CAP - 0.5) / (NUM_EXPERTS - 1)] * (NUM_EXPERTS - 1)
        with pytest.raises(ValueError, match="caps any expert at"):
            moe_sim_route._validate(over, NUM_EXPERTS, TOP_K)


class TestApportion(PytestBase):
    @pytest.mark.parametrize("profile", [SKEWED, UNIFORM], ids=["skewed", "uniform"])
    @pytest.mark.parametrize("tokens", [1, 2, 7, 128, 512, 1024, 16384])
    def test_exact_total_within_one_of_ideal(self, profile, tokens):
        slots = tokens * TOP_K
        counts = moe_sim_route._apportion(profile, slots, tokens)
        assert sum(counts) == slots
        ideal = [value / 100.0 * slots for value in profile]
        assert max(abs(counts[e] - ideal[e]) for e in range(NUM_EXPERTS)) < 1.0 + 1e-9
        # No expert may be routed to more tokens than exist.
        assert max(counts) <= tokens


class TestAssign(PytestBase):
    @pytest.mark.parametrize("profile", [SKEWED, UNIFORM], ids=["skewed", "uniform"])
    @pytest.mark.parametrize("tokens", [1, 7, 128, 512, 2048])
    def test_counts_exact_rows_distinct_and_in_range(self, profile, tokens):
        counts = moe_sim_route._apportion(profile, tokens * TOP_K, tokens)
        table = moe_sim_route._assign(counts, tokens, TOP_K, seed=0)
        assert table.shape == (tokens, TOP_K)
        assert int(table.min()) >= 0 and int(table.max()) < NUM_EXPERTS
        # Column sums reproduce the apportioned targets exactly.
        assert table.reshape(-1).bincount(minlength=NUM_EXPERTS).tolist() == counts
        # A real topk never repeats an expert inside one row.
        assert not bool((table.sort(dim=1).values.diff(dim=1) == 0).any())

    def test_same_seed_is_bit_identical(self):
        counts = moe_sim_route._apportion(SKEWED, 512 * TOP_K, 512)
        assert bool((moe_sim_route._assign(counts, 512, TOP_K, 0) == moe_sim_route._assign(counts, 512, TOP_K, 0)).all())

    def test_different_seed_differs_but_keeps_counts(self):
        counts = moe_sim_route._apportion(SKEWED, 512 * TOP_K, 512)
        zero = moe_sim_route._assign(counts, 512, TOP_K, 0)
        one = moe_sim_route._assign(counts, 512, TOP_K, 1)
        assert not bool((zero == one).all())
        assert one.reshape(-1).bincount(minlength=NUM_EXPERTS).tolist() == counts


class TestMaybeSelectExperts(PytestBase):
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_MOE_SIM_ROUTE", raising=False)
        assert moe_sim_route.maybe_select_experts(make_hidden(8), TOP_K, NUM_EXPERTS, False) is None

    def test_returns_none_for_mix_placement(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        assert moe_sim_route.maybe_select_experts(make_hidden(8), TOP_K, NUM_EXPERTS, True) is None

    def test_returns_none_during_profile_run(self, monkeypatch):
        # The startup dummy forward only sizes the KV cache; letting it see the skew
        # would give each leg a different cache size and break cross-leg comparison.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setattr(moe_sim_route, "_EXTRA_CTX", SimpleNamespace(in_profile_run=True))
        assert moe_sim_route.maybe_select_experts(make_hidden(8), TOP_K, NUM_EXPERTS, False) is None

    def test_one_token_step_still_injects(self, monkeypatch):
        # The DP sync dummy batch runs at T=1 (NPUWorker.execute_dummy_batch) and must keep
        # the injection: with it suspended the step reaches the real DSv4 hash gate, which
        # faults the AI core at this shape. One token still fills a full row of K distinct
        # experts, so the contract holds.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        weights, ids = moe_sim_route.maybe_select_experts(make_hidden(1), TOP_K, NUM_EXPERTS, False)
        assert ids.shape == (1, TOP_K) and weights.shape == (1, TOP_K)
        assert len(set(ids[0].tolist())) == TOP_K

    def test_contract_shape_dtype_and_counts(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_SEED", "0")
        tokens = 128
        weights, ids = moe_sim_route.maybe_select_experts(make_hidden(tokens), TOP_K, NUM_EXPERTS, False)
        assert ids.shape == (tokens, TOP_K) and weights.shape == (tokens, TOP_K)
        assert ids.dtype == torch.int32
        assert weights.dtype == torch.bfloat16
        assert int(ids.min()) >= 0 and int(ids.max()) < NUM_EXPERTS
        counts = ids.reshape(-1).bincount(minlength=NUM_EXPERTS).tolist()
        assert sum(counts) == tokens * TOP_K
        assert counts[0] == pytest.approx(0.16 * tokens * TOP_K, abs=1)
        assert counts[7] == pytest.approx(0.10 * tokens * TOP_K, abs=1)

    def test_weights_are_flat_one_over_k(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        weights, _ = moe_sim_route.maybe_select_experts(make_hidden(8), TOP_K, NUM_EXPERTS, False)
        assert bool((weights == weights[0, 0]).all())
        assert float(weights[0, 0]) == pytest.approx(1.0 / TOP_K, abs=1e-3)

    def test_cached_per_token_count(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        first = moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)[1]
        # Same T -> the same tensor, so every layer of one forward agrees.
        assert moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)[1] is first
        other = moe_sim_route.maybe_select_experts(make_hidden(32), TOP_K, NUM_EXPERTS, False)[1]
        assert other is not first
        assert other.shape[0] == 32
        assert len(moe_sim_route._CACHE) == 2

    def test_cache_is_bounded_and_keeps_the_hot_entry(self, monkeypatch):
        # Served traffic churns T constantly while one steady-state T is used every step.
        # The cache must not grow without bound, and the per-step entry must survive the
        # churn -- that is the difference between LRU and FIFO, and the steady-state T is
        # exactly the one FIFO would throw away.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        hot = moe_sim_route.maybe_select_experts(make_hidden(16), TOP_K, NUM_EXPERTS, False)[1]
        for tokens in range(32, 32 + moe_sim_route._CACHE_MAX * 2):
            moe_sim_route.maybe_select_experts(make_hidden(tokens), TOP_K, NUM_EXPERTS, False)
            moe_sim_route.maybe_select_experts(make_hidden(16), TOP_K, NUM_EXPERTS, False)
        assert len(moe_sim_route._CACHE) <= moe_sim_route._CACHE_MAX
        # Identity, not equality: a recompute is deterministic and would compare equal,
        # so only `is` proves the entry survived rather than being rebuilt.
        assert moe_sim_route.maybe_select_experts(make_hidden(16), TOP_K, NUM_EXPERTS, False)[1] is hot

    def test_every_zero_silences_the_heartbeat(self, monkeypatch, capsys):
        # envs.py documents 0 as "disables the heartbeat"; the opposite (every call)
        # is what max(1, ...) would have produced, so pin the intended reading.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "0")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        capsys.readouterr()
        for _ in range(50):
            moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        assert capsys.readouterr().out == ""

    def test_tokens_below_top_k_injects_at_the_coarsest_granularity(self, monkeypatch):
        # Not a hard error: a row *has* top_k slots, so one token fills a full row of
        # distinct experts, and the only real constraint is count[e] <= T. Steps this
        # small do happen (the DP-padded batch is dp_size * max_tokens_across_dp), and
        # killing the engine there ends the run instead of showing up in the dump.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        tokens = TOP_K - 1
        weights, ids = moe_sim_route.maybe_select_experts(make_hidden(tokens), TOP_K, NUM_EXPERTS, False)
        assert ids.shape == (tokens, TOP_K) and weights.shape == (tokens, TOP_K)
        counts = ids.reshape(-1).bincount(minlength=NUM_EXPERTS).tolist()
        assert sum(counts) == tokens * TOP_K
        assert max(counts) <= tokens
        assert counts[0] == pytest.approx(0.16 * tokens * TOP_K, abs=1)
        assert all(len(set(row)) == TOP_K for row in ids.tolist())

    def test_one_token_still_fills_a_row(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        _, ids = moe_sim_route.maybe_select_experts(make_hidden(1), TOP_K, NUM_EXPERTS, False)
        assert ids.shape == (1, TOP_K)
        assert len(set(ids[0].tolist())) == TOP_K

    def test_tokens_below_top_k_warns_once(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        warned: list[tuple] = []
        monkeypatch.setattr(moe_sim_route.logger, "warning", lambda *args, **kwargs: warned.append(args))
        for tokens in (TOP_K - 1, 1, TOP_K - 2):
            moe_sim_route.maybe_select_experts(make_hidden(tokens), TOP_K, NUM_EXPERTS, False)
        assert len(warned) == 1
        # At or above top_k the allocation is no longer coarse, so nothing is reported.
        moe_sim_route.maybe_select_experts(make_hidden(TOP_K), TOP_K, NUM_EXPERTS, False)
        assert len(warned) == 1

    def test_full_list_length_mismatch_is_loud(self, monkeypatch):
        # A sparse spec always expands to the model's expert count, so only a full list
        # can disagree; that must fail loudly rather than silently mis-index.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "50,50")
        with pytest.raises(ValueError, match="64 experts"):
            moe_sim_route.maybe_select_experts(make_hidden(8), TOP_K, 64, False)

    def test_table_prints_every_expert(self, monkeypatch, capsys):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_TABLE", "1")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        tokens = 128
        moe_sim_route.maybe_select_experts(make_hidden(tokens), TOP_K, NUM_EXPERTS, False)
        out = capsys.readouterr().out
        assert f"T={tokens} K={TOP_K} slots={tokens * TOP_K}" in out
        assert f"SUM_act={tokens * TOP_K}" in out
        # One row per expert, plus the header and summary lines.
        assert sum(1 for line in out.splitlines() if line.startswith("[sim-route r")) >= NUM_EXPERTS

    def test_returning_to_a_seen_token_count_is_reported(self, monkeypatch, capsys):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        moe_sim_route.maybe_select_experts(make_hidden(32), TOP_K, NUM_EXPERTS, False)
        capsys.readouterr()
        # Back to a T seen before: cached, and the line says so rather than staying mute.
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        out = capsys.readouterr().out
        assert "reuse alloc#" in out and "same as previous" in out
        assert "T=64" in out

    def test_reuse_line_names_the_table_being_reused(self, monkeypatch, capsys):
        # The module-global counter names the newest allocation, which after a T switch
        # belongs to a different token count; the line's whole point is "you got the same
        # table as before", so it has to read the id off the cache entry.
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)  # alloc#1
        moe_sim_route.maybe_select_experts(make_hidden(32), TOP_K, NUM_EXPERTS, False)  # alloc#2
        capsys.readouterr()
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        out = capsys.readouterr().out
        assert "T=64" in out and "reuse alloc#1" in out
        assert "alloc#2" not in out

    def test_heartbeat_names_the_table_being_reused(self, monkeypatch, capsys):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "1")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)  # alloc#1
        moe_sim_route.maybe_select_experts(make_hidden(32), TOP_K, NUM_EXPERTS, False)  # alloc#2
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)  # back, alloc#1
        capsys.readouterr()
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        assert "reuse alloc#1 (heartbeat" in capsys.readouterr().out

    def test_consecutive_reuse_stays_quiet_until_the_heartbeat(self, monkeypatch, capsys):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "1")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        capsys.readouterr()
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        assert "heartbeat" in capsys.readouterr().out

    def test_consecutive_reuse_is_silent_within_the_heartbeat(self, monkeypatch, capsys):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "10")
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        capsys.readouterr()
        for _ in range(9):
            moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        assert capsys.readouterr().out == ""
        # The 10th repeat crosses the threshold and confirms the allocation is unchanged.
        moe_sim_route.maybe_select_experts(make_hidden(64), TOP_K, NUM_EXPERTS, False)
        assert "heartbeat, every 10 calls" in capsys.readouterr().out


class TestOtherExpertCounts(PytestBase):
    """Expert count is read from the model, never assumed.

    E=256 / K=6 is just what this checkpoint ships; the module has to hold for any E,
    and the per-expert table and the home-rank split both scale with it.
    """

    @pytest.mark.parametrize("num_experts", [256, 384, 512])
    @pytest.mark.parametrize("tokens", [7, 512, 4096])
    def test_allocation_is_exact_at_other_expert_counts(self, num_experts, tokens):
        profile = moe_sim_route._materialize({0: 16.0, 7: 10.0}, num_experts)
        moe_sim_route._validate(profile, num_experts, TOP_K)
        assert len(profile) == num_experts
        counts = moe_sim_route._apportion(profile, tokens * TOP_K, tokens)
        table = moe_sim_route._assign(counts, tokens, TOP_K, seed=0)
        assert sum(counts) == tokens * TOP_K
        assert table.reshape(-1).bincount(minlength=num_experts).tolist() == counts
        assert int(table.max()) < num_experts
        assert not bool((table.sort(dim=1).values.diff(dim=1) == 0).any())

    def test_the_cap_follows_top_k_not_the_expert_count(self):
        # 100/K is a property of the router, so it must not move when E does.
        for num_experts in (256, 384):
            profile = [CAP + 0.5] + [(100.0 - CAP - 0.5) / (num_experts - 1)] * (num_experts - 1)
            with pytest.raises(ValueError, match="caps any expert at"):
                moe_sim_route._validate(profile, num_experts, TOP_K)

    def test_length_mismatch_names_the_models_count(self):
        with pytest.raises(ValueError, match="384 experts"):
            moe_sim_route._validate([1.0] * 256, 384, TOP_K)

    @pytest.mark.parametrize("world", [8, 6])
    def test_home_rank_split_at_384(self, world):
        moe_sim_route._EP_WORLD = world
        counts = moe_sim_route._apportion(moe_sim_route._materialize({0: 16.0, 7: 10.0}, 384), 4096 * TOP_K, 4096)
        line = moe_sim_route._rank_line(counts)
        assert len(line) == 1
        assert f"epn={384 // world}" in line[0]
        assert sum(ast.literal_eval(line[0].split("home-rank rows = ")[1].split("  max/mean")[0])) == 4096 * TOP_K

    def test_indivisible_world_size_says_so_instead_of_going_mute(self):
        # Silence here would read as "the injection did not run" -- the wrong conclusion.
        moe_sim_route._EP_WORLD = 7
        line = moe_sim_route._rank_line([1] * 384)
        assert len(line) == 1
        assert "不可预测" in line[0] and "R=7" in line[0]


class TestSelectExpertsPassThrough(PytestBase):
    """The shipped path: unset means select_experts behaves exactly as before.

    The two seams below are the non-Ascend ones -- the fused-gating capability probe and
    the weight prefetch -- so this exercises the real function body down to the native
    router without an NPU.
    """

    def _call(self, monkeypatch, hidden):
        monkeypatch.setattr(experts_selector, "check_npu_moe_gating_top_k", lambda **kwargs: False)
        monkeypatch.setattr(experts_selector, "get_weight_prefetch_method", lambda: None)
        return experts_selector.select_experts(
            hidden_states=hidden,
            router_logits=torch.randn(hidden.shape[0], NUM_EXPERTS),
            top_k=TOP_K,
            use_grouped_topk=False,
            renormalize=True,
            # Every real call site passes this; the injection needs it to expand a
            # sparse "e:pct" spec against the model's expert count.
            num_logical_experts=NUM_EXPERTS,
        )

    def test_unset_reaches_the_native_router(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_MOE_SIM_ROUTE", raising=False)
        weights, ids = self._call(monkeypatch, make_hidden(8, dtype=torch.float32))
        assert weights.shape == (8, TOP_K) and ids.shape == (8, TOP_K)
        # No 1/K flattening and no wholesale replacement: this is the real topk.
        assert not bool((weights == weights[0, 0]).all())

    def test_set_replaces_the_router_output(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_MOE_SIM_ROUTE", "0:16,7:10")
        weights, ids = self._call(monkeypatch, make_hidden(8, dtype=torch.float32))
        assert bool((weights == 1.0 / TOP_K).all())
        counts = ids.reshape(-1).bincount(minlength=NUM_EXPERTS)
        assert int(counts.max()) > TOP_K  # expert 0 concentrated well past one row
