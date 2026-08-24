"""Two-phase bet size selection.

Guards the main safety property of the design: the coarse phase may only choose
BETWEEN SIZES of the same action family, and must never discard an entire family
(check/call/fold/all-in). An undertrained coarse phase can at worst pick a suboptimal
size, but never an invalid action.
"""
from __future__ import annotations

import time

import pytest

from pokersolver.ranges.range import Range
from pokersolver.multiway.solver import MultiwayCfrConfig

pytest.importorskip("numpy", reason="vectorized solver requires [solver] extra")

from pokersolver.multiway import policy  # noqa: E402


def _wr(spec: str):
    return tuple((combo, 1.0) for combo in Range.parse(spec).combos())


def _config(players: int = 2, *, stacks: float = 40.0,
            pot: float = 6.0) -> MultiwayCfrConfig:
    ranges = (_wr("AA-TT,AKs"), _wr("99-22,KQs,JTs"),
              _wr("A5s-A2s,T9s"), _wr("QJo,98s"))[:players]
    return MultiwayCfrConfig(
        board=("Js", "Jh", "7d"), ranges=ranges,
        names=tuple(f"p{i}" for i in range(players)),
        pot_bb=pot, bet_bb=4.0, stacks_bb=(stacks,) * players,
        iterations=1, seed=7,
    )


def test_candidate_menu_shrinks_with_player_count() -> None:
    """The limit is tree cost: 5 ms/pass with 2 players vs 72 ms with 4."""
    two = policy.candidate_config(_config(2))
    three = policy.candidate_config(_config(3))
    four = policy.candidate_config(_config(4))

    assert len(two.bet_sizes_pct) > len(three.bet_sizes_pct) > 0
    assert len(four.bet_sizes_pct) == 1
    assert len(two.raise_sizes_pct) == 2
    assert len(four.raise_sizes_pct) == 1
    # Opponents must always have the ability to re-raise — opponent passivity made solver overly aggressive.
    for cfg in (two, three, four):
        assert cfg.opponent_raise_sizes_pct
        assert cfg.opponent_allin
        assert cfg.rake_percent > 0


def test_hero_allin_is_offered_only_below_spr_three() -> None:
    deep = policy.candidate_config(_config(2, stacks=40.0, pot=6.0))    # SPR ~6.7
    shallow = policy.candidate_config(_config(2, stacks=10.0, pot=6.0))  # SPR ~1.7

    assert not deep.hero_allin
    assert shallow.hero_allin


def test_narrowing_keeps_one_size_per_family_and_never_drops_a_family() -> None:
    candidates = policy.candidate_config(_config(2, stacks=10.0))
    evs = {"check": 1.0, "bet@33": 0.2, "bet@66": 0.9, "bet@125": 0.5,
           "raise@200": 0.1, "raise@300": 0.4, "allin": 0.0}

    narrowed = policy.narrow_config(candidates, evs)

    assert narrowed.bet_sizes_pct == (66.0,)
    assert narrowed.raise_sizes_pct == (300.0,)
    # Families remain: neither all-in nor opponent actions change during pruning.
    assert narrowed.hero_allin == candidates.hero_allin
    assert narrowed.opponent_allin
    assert narrowed.opponent_raise_sizes_pct == candidates.opponent_raise_sizes_pct


def test_narrowing_without_ev_data_falls_back_to_the_first_candidate() -> None:
    candidates = policy.candidate_config(_config(2))
    narrowed = policy.narrow_config(candidates, {})

    assert narrowed.bet_sizes_pct == candidates.bet_sizes_pct[:1]
    assert narrowed.raise_sizes_pct == candidates.raise_sizes_pct[:1]


def test_two_phase_returns_decision_within_budget() -> None:
    solver, evs, decision, meta = policy.solve_two_phase(
        _config(2), ("As", "Ah"), budget_s=2.0, workers=1, deals=4_000,
    )

    assert not meta.coarse_skipped
    assert meta.coarse_iterations > 0
    assert len(meta.chosen_bets) == 1
    assert set(evs) >= {"check"}
    assert decision["best_action"] in evs
    # Chosen size must come from candidates.
    assert meta.chosen_bets[0] in policy._CANDIDATE_BETS[2]


def test_two_phase_deadline_is_respected() -> None:
    start = time.monotonic()
    policy.solve_two_phase(_config(2), ("As", "Ah"), budget_s=1.5,
                           workers=1, deals=4_000)
    assert time.monotonic() - start < 3.5


def test_four_players_skip_the_coarse_phase() -> None:
    """With a single candidate there is nothing to select — entire budget goes to solution."""
    solver, _evs, _decision, meta = policy.solve_two_phase(
        _config(4), ("As", "Ah"), budget_s=1.5, workers=1, deals=2_000,
    )

    assert meta.coarse_skipped
    assert meta.coarse_iterations == 0
    assert solver.iterations > 0
