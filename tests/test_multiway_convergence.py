"""Measuring multiway CFR convergence (best-response gain and average regret)."""
from __future__ import annotations

import pytest

from pokersolver.multiway import MultiwayCfrConfig, MultiwayCfrSolver
from pokersolver.multiway import diagnostics as diag


def exact(*combos: tuple[str, str]):
    return tuple((combo, 1.0) for combo in combos)


def solved(iterations: int, *, seed: int = 3) -> MultiwayCfrSolver:
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(
                exact(("As", "Ah"), ("7s", "6s"), ("Kc", "Jc")),
                exact(("Kd", "Kh"), ("Ad", "Qd"), ("9d", "9h")),
                exact(("Qh", "Qs"), ("5h", "5s"), ("Tc", "Td")),
            ),
            names=("Hero", "BTN", "BB"),
            pot_bb=12.0, bet_bb=6.0, iterations=iterations, seed=seed,
        ),
    )
    solver.solve()
    return solver


def test_walker_reproduces_the_solvers_own_evaluation() -> None:
    """Diagnostics traverses the tree again — must match the solver's own evaluation.

    If the tree structure diverged between the two paths, convergence of a different
    game would be measured without warning.
    """
    solver = solved(600)
    deals = diag.sample_deals(solver, 120, seed=99)
    mine = diag._expected_values(solver, deals, {})

    totals = [0.0, 0.0, 0.0]
    config = solver.config
    for holes, scores in deals:
        solver._scores = scores
        commitments = [0.0] * 3
        commitments[0] = min(config.bet_bb, config.player_stacks[0])
        check = solver._terminal((0, 1, 2), (0.0,) * 3)
        bet = solver._evaluate_responses(
            (1, 2), 0, ("lead", "bet"), (0,), tuple(commitments), commitments[0], holes,
            allow_raise=config.opponent_raise_to_bb > commitments[0],
            allow_allin=config.opponent_allin,
        )
        node = solver.nodes.get(diag._InfoSet(0, (), diag._canonical(holes[0])))
        strategy = node.average_strategy() if node else (0.5, 0.5)
        for seat in range(3):
            totals[seat] += strategy[0] * check[seat] + strategy[1] * bet[seat]

    for seat in range(3):
        assert mine[seat] == pytest.approx(totals[seat] / len(deals), abs=1e-9)


def test_average_regret_falls_towards_zero() -> None:
    """External regret MUST decrease towards zero — that is the CFR guarantee."""
    deals_source = solved(200)
    few = max(diag.average_regret(deals_source))
    many = max(diag.average_regret(solved(20_000)))
    assert many < few
    assert many < 1.0


def test_best_response_gain_is_non_negative_and_improves() -> None:
    """BR gain may plateau above zero (n>=3 does not guarantee Nash), but must decrease."""
    reference = solved(200)
    deals = diag.sample_deals(reference, 200, seed=99)

    coarse = diag.measure(reference, deals)
    fine = diag.measure(solved(20_000), deals)

    assert all(gain >= 0.0 for gain in coarse.gain_bb)
    assert all(gain >= 0.0 for gain in fine.gain_bb)
    assert fine.max_gain_bb < coarse.max_gain_bb


def test_uniform_policy_is_visibly_exploitable() -> None:
    """Uniform profile must register as substantially exploitable.

    Tree structure is preserved, only average strategy is overridden — if nodes
    were deleted, the measurement traversal would find nothing and return zero.
    """
    solver = solved(400)
    for node in solver.nodes.values():
        node.strategy_sum = [1.0] * len(node.strategy_sum)
    deals = diag.sample_deals(solver, 200, seed=7)
    report = diag.measure(solver, deals)
    assert report.max_gain_bb > 0.2


def test_measure_reports_percent_of_pot_for_comparability() -> None:
    """Threshold must be comparable with TexasSolver, which measures in % of pot."""
    solver = solved(400)
    deals = diag.sample_deals(solver, 150, seed=11)
    payload = diag.measure(solver, deals).as_dict(solver.config.pot_bb)
    assert payload["max_gain_pct_pot"] == pytest.approx(
        100.0 * payload["max_gain_bb"] / solver.config.pot_bb,
    )
    assert payload["label"] == "train"
    assert len(payload["gain_bb"]) == 3


def test_solve_and_result_can_attach_convergence() -> None:
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(
                exact(("As", "Ah"), ("7s", "6s")),
                exact(("Kd", "Kh"), ("Ad", "Qd")),
                exact(("Qh", "Qs"), ("5h", "5s")),
            ),
            names=("Hero", "BTN", "BB"),
            pot_bb=12.0, bet_bb=6.0, iterations=300, seed=5,
        ),
    )
    payload = solver.solve_and_result(("As", "Ah"), convergence_deals=100)
    convergence = payload["convergence"]
    assert convergence["label"] == "holdout"
    assert convergence["deals"] == 100
    assert convergence["max_gain_bb"] >= 0.0


def test_gain_estimate_does_not_explode_on_a_large_tree() -> None:
    """Split-sample estimation must eliminate optimistic bias.

    Maximum over noisy counterfactual values is always biased high, and the error
    scales with the number of infosets. Naive estimation on a tree with 1,840 infosets
    gave 3.94 bb at 400 deals and 0.28 bb at 20,000 — almost the entire difference was bias,
    not true distance to equilibrium.
    """
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(
                exact(("As", "Ah"), ("7s", "6s"), ("Kc", "Jc")),
                exact(("Kd", "Kh"), ("Ad", "Qd"), ("9d", "9h")),
                exact(("Qh", "Qs"), ("5h", "5s"), ("Tc", "Td")),
            ),
            names=("Hero", "BTN", "BB"),
            pot_bb=12.0, bet_bb=6.0, iterations=20_000, seed=3,
            bet_sizes_pct=(33.0, 75.0), hero_allin=True,
        ),
    )
    solver.solve()
    small = diag.measure(solver, diag.sample_deals(solver, 300, seed=1))
    large = diag.measure(solver, diag.sample_deals(solver, 3000, seed=1))
    # Without correction, the estimate grew as sample size shrank; now the opposite holds.
    assert small.max_gain_bb <= large.max_gain_bb + 0.15


def test_report_says_when_the_estimate_is_below_the_noise_floor() -> None:
    """Truncated zero must not be reported as converged."""
    solver = solved(20_000)
    scarce = diag.measure(solver, diag.sample_deals(solver, 40, seed=2))
    payload = scarce.as_dict(solver.config.pot_bb)
    assert payload["deals_per_infoset"] is not None
    if scarce.max_gain_bb == 0.0:
        assert scarce.resolved is False
        assert payload["resolved"] is False
