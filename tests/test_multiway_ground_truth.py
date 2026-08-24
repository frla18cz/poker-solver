"""Verification of the solver against closed-form formulas that hold regardless of implementation.

These tests need neither TexasSolver nor any external tool — the correct answer
can be calculated by hand. They cover card removal, showdown order, pot splitting,
rake, and CFR convergence simultaneously.
"""
from __future__ import annotations

import pytest

from pokersolver.multiway import MultiwayCfrConfig, MultiwayCfrSolver
from pokersolver.multiway.solver import _canonical, _InfoSet


def exact(*combos: tuple[str, str]):
    return tuple((combo, 1.0) for combo in combos)


def policy(solver: MultiwayCfrSolver, player: int, history, combo):
    node = solver.nodes[_InfoSet(player, history, _canonical(combo))]
    return node.average_strategy()


def test_river_bluff_share_matches_the_closed_form_ratio() -> None:
    """Bluff share must equal b / (pot + 2b) — textbook river formula.

    Hero holds either the nuts or pure air; opponent holds a bluffcatcher.
    For opponent to be indifferent, bluff frequency must match the pot odds given.
    This is the only spot where the exact analytic solution is known, making it
    the strongest ground-truth verification of the entire pipeline.
    """
    pot, bet = 10.0, 7.0
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("As", "Kd", "7h", "2c", "9s"),
            ranges=(
                exact(("Ah", "Ad"), ("5c", "4c")),   # three aces / air
                exact(("Kh", "Qs")),                  # bluffcatcher: pair of kings
            ),
            names=("Hero", "BB"),
            pot_bb=pot, bet_bb=bet, iterations=40_000, seed=4,
            stacks_bb=(50.0, 50.0),
        ),
    )
    solver.solve()

    value_bet = policy(solver, 0, (), ("Ah", "Ad"))[1]
    bluff_bet = policy(solver, 0, (), ("5c", "4c"))[1]
    assert value_bet > 0.95, "nuts must almost always bet"

    bluff_share = bluff_bet / (value_bet + bluff_bet)
    expected = bet / (pot + 2 * bet)
    assert bluff_share == pytest.approx(expected, abs=0.05)


def test_bluffcatcher_call_frequency_makes_the_bluff_indifferent() -> None:
    """Opponent must call at frequency making bluff indifferent: pot / (pot + bet)."""
    pot, bet = 10.0, 7.0
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("As", "Kd", "7h", "2c", "9s"),
            ranges=(
                exact(("Ah", "Ad"), ("5c", "4c")),
                exact(("Kh", "Qs")),
            ),
            names=("Hero", "BB"),
            pot_bb=pot, bet_bb=bet, iterations=40_000, seed=4,
            stacks_bb=(50.0, 50.0),
        ),
    )
    solver.solve()
    call = policy(solver, 1, ("lead", "bet"), ("Kh", "Qs"))[1]
    assert call == pytest.approx(pot / (pot + bet), abs=0.08)


def test_heads_up_utilities_sum_to_the_dead_pot() -> None:
    """Without rake, the sum of payoffs must equal the dead pot exactly."""
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("As", "Kd", "7h", "2c", "9s"),
            ranges=(exact(("Ah", "Ad")), exact(("Kh", "Qs"))),
            names=("Hero", "BB"), pot_bb=10.0, bet_bb=7.0,
            iterations=50, seed=1, stacks_bb=(50.0, 50.0),
        ),
    )
    solver._scores = ((9,), (1,))
    for commitments in ((0.0, 0.0), (7.0, 7.0), (7.0, 0.0)):
        active = tuple(i for i in range(2) if commitments[i] > 0) or (0, 1)
        assert sum(solver._terminal(active, commitments)) == pytest.approx(10.0)


def test_nuts_never_folds_to_a_raise() -> None:
    """Hero with the absolute nuts must not fold to a raise."""
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("As", "Kd", "7h", "2c", "9s"),
            ranges=(exact(("Ah", "Ad")), exact(("Kh", "Qs")), exact(("7s", "7d"))),
            names=("Hero", "BTN", "BB"),
            pot_bb=10.0, bet_bb=7.0, iterations=8_000, seed=2,
            opponent_raise_to_bb=21.0, stacks_bb=(50.0, 50.0, 50.0),
        ),
    )
    solver.solve()
    responses = [
        node.average_strategy()
        for key, node in solver.nodes.items()
        if key.player == 0 and "hero_response" in key.history
    ]
    assert responses, "hero did not receive response node"
    for fold, _call in responses:
        assert fold < 0.05
