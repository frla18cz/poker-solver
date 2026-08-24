"""Vectorized solver and batch evaluator.

Tests the key empirical findings: that the batch evaluator preserves score_7 ordering,
that the vectorized solver converges to the same results as the reference solver,
and that small fixed deal pools overfit (the reason `batch` exists).
"""
from __future__ import annotations

import random
import time

import pytest

from pokersolver.cards import FULL_DECK, score_7
from pokersolver.ranges.range import Range
from pokersolver.multiway.solver import MultiwayCfrConfig, MultiwayCfrSolver

np = pytest.importorskip("numpy", reason="vectorized solver requires [solver] extra")

from pokersolver.multiway.evaluator import CARD_INDEX, score_batch  # noqa: E402
from pokersolver.multiway.fast import VectorSolver  # noqa: E402


def _index(cards) -> "np.ndarray":
    return np.asarray([[CARD_INDEX[c] for c in cards]], dtype=np.int64)


def _wr(spec: str):
    return tuple((combo, 1.0) for combo in Range.parse(spec).combos())


BASE = dict(
    board=("Js", "Jh", "7d"), names=("Hero", "BB"),
    ranges=(_wr("AA-TT,AKs,AQs,AKo"), _wr("99-22,KQs,JTs,T9s,QJo")),
    pot_bb=6.0, bet_bb=4.0, bet_sizes_pct=(33.0, 75.0), hero_allin=True,
    stacks_bb=(40.0, 40.0), opponent_raise_sizes_pct=(250.0,),
    opponent_allin=True, max_raise_rounds=1,
)
HERO = ("As", "Ah")


# --- evaluator ------------------------------------------------------------
def test_batch_evaluator_keeps_the_order_of_score_7() -> None:
    """Absolute scores may differ, but ordering must match — otherwise showdown changes."""
    rng = random.Random(20260804)
    hands = [rng.sample(FULL_DECK, 7) for _ in range(3_000)]
    packed = np.asarray([[CARD_INDEX[c] for c in hand] for hand in hands],
                        dtype=np.int64)
    fast = score_batch(packed)
    slow = [score_7(hand) for hand in hands]

    for _ in range(20_000):
        a, b = rng.randrange(len(hands)), rng.randrange(len(hands))
        expected = (slow[a] > slow[b]) - (slow[a] < slow[b])
        actual = int(fast[a] > fast[b]) - int(fast[a] < fast[b])
        assert actual == expected, (hands[a], hands[b], slow[a], slow[b])


@pytest.mark.parametrize("cards", [
    ["As", "2d", "3c", "4h", "5s", "Kd", "Qc"],   # wheel — not in _STRAIGHT_MASKS
    ["7s", "7h", "7d", "9c", "9h", "2s", "3d"],   # full house
    ["7s", "7h", "7d", "9c", "9h", "9d", "3d"],   # two trips: lower forms a pair
    ["2s", "3s", "4s", "5s", "6s", "Kd", "Qc"],   # straight flush
    ["As", "Ks", "Qs", "Js", "9s", "2d", "3c"],   # flush, not straight
])
def test_tricky_hands_match_score_7(cards) -> None:
    """Hands where category ranking is most prone to edge-case divergence."""
    others = [c for c in FULL_DECK if c not in cards]
    rng = random.Random(7)
    fast = int(score_batch(_index(cards))[0])
    for _ in range(200):
        rival = rng.sample(others, 7)
        expected = (score_7(cards) > score_7(rival)) - (score_7(cards) < score_7(rival))
        actual = ((fast > int(score_batch(_index(rival))[0]))
                  - (fast < int(score_batch(_index(rival))[0])))
        assert actual == expected, (cards, rival)


# --- solver ---------------------------------------------------------------
def test_vector_solver_lands_where_the_reference_is_heading() -> None:
    """Both versions must converge towards the same point.

    The reference solver is set to 300,000 iterations: at 20,000 it is still far
    from convergence and would show spurious divergence.
    """
    reference = MultiwayCfrSolver(
        MultiwayCfrConfig(iterations=300_000, seed=7, **BASE),
    ).solve_and_result(HERO)
    slow = dict(zip(reference["hero_actions"], reference["hero"]["frequencies"]))

    fast = VectorSolver(MultiwayCfrConfig(iterations=1, seed=7, **BASE),
                        deals=20_000)
    fast.solve(iterations=600)
    quick = fast.hero_strategy(HERO)

    assert set(quick) == set(slow)
    assert quick["check"] == pytest.approx(slow["check"], abs=0.10)


def test_small_deal_pool_makes_the_answer_depend_on_the_seed() -> None:
    """Deal pool size cannot be chosen arbitrarily.

    With a small pool, the solver overfits to the sample rather than solving the game:
    across 8 seeds, 1,000 deals yielded check frequencies from 0.1% to 99.9% (std dev 31%),
    while 20,000 deals remained stable around 5%.
    """
    def spread(pool: int) -> float:
        values = []
        for seed in (1, 3, 11):
            solver = VectorSolver(
                MultiwayCfrConfig(iterations=1, seed=seed, **BASE), deals=pool)
            # Sufficient iterations are needed, otherwise optimization noise obscures sampling error.
            solver.solve(iterations=800)
            values.append(solver.hero_strategy(HERO)["check"])
        return max(values) - min(values)

    assert spread(1_000) > spread(10_000) + 0.10


def test_ev_is_stable_with_more_passes_even_when_frequency_is_not() -> None:
    """Interpretation: EV and action choice are robust, individual combo frequencies may shift.

    Combo frequencies can swing from 99% check to 60% between 1,000 and 8,000 passes
    while EV(check) stays at 4.591 within 3 decimal places. When two actions are near-indifferent,
    equilibrium does not constrain the mix ratio (same unidentifiability as purification).
    Average regret still decays as 1/T, verifying CFR correctness.

    This test therefore checks EV, not raw frequency.
    """
    solver = VectorSolver(MultiwayCfrConfig(iterations=1, seed=7, **BASE),
                          deals=20_000)
    solver.solve(iterations=800)
    early = solver.hero_evs(HERO, deals=40_000)
    solver.solve(iterations=1_700)
    late = solver.hero_evs(HERO, deals=40_000)

    best_early = max(early, key=early.__getitem__)
    best_late = max(late, key=late.__getitem__)
    assert late[best_late] == pytest.approx(early[best_early], abs=0.5)
    if best_early != best_late:
        # Action flips are allowed only when EVs are near identical.
        ranked = sorted(late.values(), reverse=True)
        assert ranked[0] - ranked[1] < 0.3, late


def test_deadline_stops_the_solve_and_keeps_what_it_has() -> None:
    solver = VectorSolver(MultiwayCfrConfig(iterations=1, seed=7, **BASE),
                          deals=20_000)
    start = time.monotonic()
    solver.solve(iterations=1_000_000, deadline=start + 0.5)
    elapsed = time.monotonic() - start

    assert elapsed < 3.0
    assert 0 < solver.iterations < 1_000_000
    assert sum(solver.hero_strategy(HERO).values()) == pytest.approx(1.0)


def test_combo_outside_the_hero_range_is_reported() -> None:
    solver = VectorSolver(MultiwayCfrConfig(iterations=1, seed=7, **BASE),
                          deals=2_000)
    solver.solve(iterations=5)
    with pytest.raises(ValueError, match="hero range"):
        solver.hero_strategy(("2c", "7d"))


def test_batch_restores_the_full_pool_when_it_finishes() -> None:
    """Full deal pool must be restored upon completion, not just the last batch."""
    solver = VectorSolver(MultiwayCfrConfig(iterations=1, seed=7, **BASE),
                          deals=8_000)
    solver.solve(iterations=20, batch=1_000)
    assert solver.deals == 8_000


def test_parallel_solve_matches_single_process_and_does_more_passes() -> None:
    """Multiple workers must not alter results, only accelerate computation.

    Regret is additive across deals, so partitioning deals among processes
    and summing partial regrets is mathematically identical to single-process.
    Benchmarked on 10 cores: 2 cores ~1.8x throughput, 4 cores ~3.0x, 6 cores ~3.9x.
    """
    from pokersolver.multiway.parallel import solve_parallel

    config = MultiwayCfrConfig(iterations=1, seed=7, **BASE)
    single = solve_parallel(config, deals=8_000, workers=1, budget_s=2.0)
    shared = solve_parallel(config, deals=8_000, workers=3, budget_s=2.0)

    lonely = single.hero_evs(HERO, deals=20_000)
    together = shared.hero_evs(HERO, deals=20_000)
    assert max(lonely, key=lonely.__getitem__) == max(together, key=together.__getitem__)
    assert together[max(together, key=together.__getitem__)] == pytest.approx(
        lonely[max(lonely, key=lonely.__getitem__)], abs=0.5)
    assert shared.iterations > single.iterations
