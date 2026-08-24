"""Regression tests for the restricted postflop multiway CFR prototype."""
from __future__ import annotations

import pytest

from pokersolver.multiway import MultiwayCfrConfig, MultiwayCfrSolver


def exact(*combos: tuple[str, str]):
    return tuple((combo, 1.0) for combo in combos)


def config(*, seed: int = 17, iterations: int = 400) -> MultiwayCfrConfig:
    return MultiwayCfrConfig(
        board=("2c", "3d", "4h", "8s", "9c"),
        ranges=(
            exact(("As", "Ah"), ("7s", "6s")),
            exact(("Kd", "Kh"), ("Ad", "Qd")),
            exact(("Qh", "Qs"), ("5h", "5s")),
        ),
        names=("Hero", "BTN", "BB"),
        pot_bb=12.0,
        bet_bb=6.0,
        iterations=iterations,
        seed=seed,
    )


def test_solver_is_deterministic_and_returns_normalized_policies() -> None:
    first = MultiwayCfrSolver(config())
    second = MultiwayCfrSolver(config())
    one = first.solve_and_result(("As", "Ah"))
    two = second.solve_and_result(("As", "Ah"))

    assert one["certified_gto"] is False
    assert one["players"] == 3
    assert one["iterations"] == one["valid_deals"] == 400
    assert one["hero"]["check"] == pytest.approx(two["hero"]["check"])
    assert one["hero"]["bet"] == pytest.approx(two["hero"]["bet"])
    assert one["hero"]["check"] + one["hero"]["bet"] == pytest.approx(1.0)
    assert one["hero_equity"] == pytest.approx(1.0)
    assert one["hero_range_equity"] < one["hero_equity"]
    assert one["evaluation_samples"] == 1_000
    for opponent in one["opponents"]:
        assert opponent["fold"] + opponent["call"] == pytest.approx(1.0)


def test_solver_supports_four_players_and_rake() -> None:
    base = config(iterations=240)
    four_way = MultiwayCfrConfig(
        board=base.board,
        ranges=base.ranges + (exact(("Jd", "Jh"), ("Td", "Th")),),
        names=("Hero", "BTN", "SB", "BB"),
        pot_bb=16.0,
        bet_bb=8.0,
        iterations=240,
        seed=31,
        rake_percent=5.0,
        rake_cap_bb=4.0,
    )

    result = MultiwayCfrSolver(four_way).solve_and_result(("As", "Ah"))

    assert result["players"] == 4
    assert len(result["opponents"]) == 3
    assert result["information_sets"] > 4
    assert result["limitations"] == [
        "one fixed bet sizing",
        "no further re-raises",
        "once the action closes, the board runs out to showdown",
        "sampled runouts and self-play convergence",
    ]


def test_facing_wager_solver_returns_fold_call_raise_policy() -> None:
    base = config(iterations=360)
    facing = MultiwayCfrConfig(
        board=base.board,
        ranges=base.ranges,
        names=base.names,
        pot_bb=12.0,
        bet_bb=6.0,
        scenario="facing_wager",
        facing_bet_bb=4.0,
        raise_to_bb=12.0,
        iterations=360,
        seed=23,
    )

    result = MultiwayCfrSolver(facing).solve_and_result(("As", "Ah"))

    assert result["scenario"] == "facing_wager"
    assert result["hero"]["actions"] == ("fold", "call", "raise")
    assert sum(result["hero"]["frequencies"]) == pytest.approx(1.0)
    assert result["hero"]["ev_fold_bb"] == pytest.approx(0.0)
    assert result["hero"]["ev_raise_bb"] > result["hero"]["ev_fold_bb"]


def test_realtime_profile_allows_one_opponent_raise_round() -> None:
    base = config(iterations=120)
    realtime = MultiwayCfrConfig(
        board=base.board, ranges=base.ranges, names=base.names,
        pot_bb=12.0, bet_bb=6.0, iterations=120, seed=29,
        opponent_raise_to_bb=12.0,
    )
    result = MultiwayCfrSolver(realtime).solve_and_result(("As", "Ah"))
    assert result["opponents"][0]["actions"] == ("fold", "call", "raise")
    assert sum(result["opponents"][0]["frequencies"]) == pytest.approx(1.0)
    assert "at most one re-raise per betting round" in result["limitations"]


def test_side_pot_awards_short_stack_only_the_main_pot() -> None:
    study = MultiwayCfrSolver(config(iterations=1))
    study._scores = ((2,), (3,), (1,))

    utilities = study._terminal((0, 1, 2), (10.0, 5.0, 10.0))

    assert utilities == pytest.approx((0.0, 22.0, -10.0))
    assert sum(utilities) == pytest.approx(12.0)


def test_facing_raise_cannot_exceed_hero_stack() -> None:
    base = config(iterations=1)
    with pytest.raises(ValueError, match="Hero stack"):
        MultiwayCfrConfig(
            board=base.board,
            ranges=base.ranges,
            names=base.names,
            pot_bb=12,
            bet_bb=4,
            scenario="facing_wager",
            facing_bet_bb=4,
            raise_to_bb=12,
            stacks_bb=(10, 30, 30),
        )


def test_short_hero_facing_wager_has_only_fold_and_call() -> None:
    base = config(iterations=40)
    short = MultiwayCfrConfig(
        board=base.board,
        ranges=base.ranges,
        names=base.names,
        pot_bb=12,
        bet_bb=4,
        scenario="facing_wager",
        facing_bet_bb=4,
        raise_to_bb=12,
        stacks_bb=(3, 30, 30),
        iterations=40,
    )

    result = MultiwayCfrSolver(short).solve_and_result(("As", "Ah"))

    assert result["hero"]["actions"] == ("fold", "call")
    assert sum(result["hero"]["frequencies"]) == pytest.approx(1.0)


@pytest.mark.parametrize("board", [
    ("2c", "3d", "4h"),
    ("2c", "3d", "4h", "8s"),
    ("2c", "3d", "4h", "8s", "9c"),
])
@pytest.mark.parametrize("scenario", ["checked_to", "facing_wager"])
def test_solver_covers_every_postflop_street_in_both_decision_modes(
    board: tuple[str, ...], scenario: str,
) -> None:
    base = config(iterations=30)
    study = MultiwayCfrConfig(
        board=board,
        ranges=base.ranges + (exact(("Jd", "Jh"), ("Td", "Th")),),
        names=("Hero", "BTN", "SB", "BB"),
        pot_bb=12.0,
        bet_bb=4.0,
        scenario=scenario,
        facing_bet_bb=4.0 if scenario == "facing_wager" else 0.0,
        raise_to_bb=12.0 if scenario == "facing_wager" else 0.0,
        iterations=30,
        seed=41,
    )

    result = MultiwayCfrSolver(study).solve_and_result(("As", "Ah"))

    expected = ("fold", "call", "raise") if scenario == "facing_wager" else ("check", "bet")
    assert result["hero"]["actions"] == expected
    assert sum(result["hero"]["frequencies"]) == pytest.approx(1.0)


def test_config_accepts_heads_up_and_rejects_solo_or_duplicate_board() -> None:
    """Heads-up is 77% of spots — engine must support it, single player is invalid."""
    MultiwayCfrConfig(
        board=("2c", "3d", "4h"), ranges=(exact(("As", "Ah")), exact(("Kd", "Kh"))),
        names=("Hero", "BB"), pot_bb=8, bet_bb=4,
    )
    with pytest.raises(ValueError, match="two, three or four"):
        MultiwayCfrConfig(
            board=("2c", "3d", "4h"), ranges=(exact(("As", "Ah")),),
            names=("Hero",), pot_bb=8, bet_bb=4,
        )
    with pytest.raises(ValueError, match="duplicate"):
        MultiwayCfrConfig(
            board=("2c", "2c", "4h"), ranges=(exact(("As", "Ah")),) * 3,
            names=("Hero", "SB", "BB"), pot_bb=8, bet_bb=4,
        )


def test_hero_must_answer_an_opponent_raise_instead_of_a_free_showdown() -> None:
    """Hero after betting cannot stay committed only to their bet.

    Without a response node, ``_terminal`` would evaluate hero as all-in for a smaller
    amount: hero would win the full main pot while risking only their bet, so a raise
    could never push them off the hand.
    """
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(
                exact(("As", "Ah")), exact(("Kd", "Kh")), exact(("Qh", "Qs")),
            ),
            names=("Hero", "BTN", "BB"),
            pot_bb=24.0, bet_bb=12.0, iterations=200, seed=5,
            opponent_raise_to_bb=36.0, stacks_bb=(100.0, 100.0, 100.0),
        ),
    )
    assert solver._hero_must_respond((0, 1), (12.0, 36.0, 0.0), 36.0)

    solver.solve()
    responses = [key for key in solver.nodes if key.player == 0 and "hero_response" in key.history]
    assert responses, "hero did not receive response node against raise"

    folded = solver._terminal((1,), (12.0, 36.0, 0.0))
    assert folded[0] == pytest.approx(-12.0)
    called = solver._terminal((0, 1), (36.0, 36.0, 0.0))
    assert called[0] == pytest.approx(-36.0) or called[0] > 0


def test_all_in_hero_keeps_side_pot_protection_without_a_response_node() -> None:
    """Hero who is already all-in cannot respond — side pot protects them."""
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(
                exact(("As", "Ah")), exact(("Kd", "Kh")), exact(("Qh", "Qs")),
            ),
            names=("Hero", "BTN", "BB"),
            pot_bb=24.0, bet_bb=12.0, iterations=50, seed=5,
            opponent_raise_to_bb=36.0, stacks_bb=(12.0, 100.0, 100.0),
        ),
    )
    assert not solver._hero_must_respond((0, 1), (12.0, 36.0, 0.0), 36.0)


def test_evaluation_counts_unvisited_information_sets() -> None:
    """Uniform fallback for unvisited infosets must be reported, not hidden."""
    solver = MultiwayCfrSolver(config(iterations=200))
    result = solver.solve_and_result(("As", "Ah"))
    assert "evaluation_unvisited" in result
    assert result["evaluation_unvisited"] >= 0


def test_hero_combo_outside_the_range_fails_loudly() -> None:
    """Combos outside hero range must fail loudly rather than returning None frequencies."""
    solver = MultiwayCfrSolver(config(iterations=100))
    with pytest.raises(ValueError, match="hero range"):
        solver.solve_and_result(("2h", "7c"))


def test_opponents_can_have_several_raise_sizes() -> None:
    """Opponents can have multiple raise sizes."""
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(exact(("As", "Ah")), exact(("Kd", "Kh")), exact(("Qh", "Qs"))),
            names=("Hero", "BTN", "BB"),
            pot_bb=12.0, bet_bb=6.0, iterations=300, seed=4,
            opponent_raise_sizes_pct=(250.0, 400.0), stacks_bb=(100.0, 100.0, 100.0),
        ),
    )
    result = solver.solve_and_result(("As", "Ah"))
    for opponent in result["opponents"]:
        if len(opponent["actions"]) > 2:
            assert "raise@250" in opponent["actions"]
            assert "raise@400" in opponent["actions"]
            break
    else:
        pytest.fail("no opponent received multiple raise sizes")


def test_opponent_raise_gate_reads_both_sources() -> None:
    """Raise size menu must allow raise even without scalar raise-to.

    The gate previously only checked the scalar value which stayed at 0,
    so configured size menus were never applied.
    """
    config = MultiwayCfrConfig(
        board=("2c", "3d", "4h"), ranges=(exact(("As", "Ah")), exact(("Kd", "Kh"))),
        names=("Hero", "BB"), pot_bb=10.0, bet_bb=5.0,
        opponent_raise_sizes_pct=(250.0,),
    )
    solver = MultiwayCfrSolver(config)
    assert solver.opponents_may_raise(5.0) is True
    assert solver.opponent_raise_menu(5.0, 100.0)[0][0] == "raise@250"


def test_aggregated_actions_keep_their_real_names() -> None:
    """Summary must not guess action names from action counts."""
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h", "8s", "9c"),
            ranges=(exact(("As", "Ah")), exact(("Kd", "Kh"))),
            names=("Hero", "BB"), pot_bb=12.0, bet_bb=6.0, iterations=200, seed=4,
            opponent_raise_sizes_pct=(300.0,), stacks_bb=(100.0, 100.0),
        ),
    )
    result = solver.solve_and_result(("As", "Ah"))
    actions = result["opponents"][0]["actions"]
    assert "raise@300" in actions or actions == ("fold", "call")


def test_mix_ev_loss_goes_to_zero_as_the_strategy_converges() -> None:
    """In equilibrium, mixed actions have equal EV, so mixing loses nothing.

    This answers why one would play a mixed action: when converged, it does not lose EV.
    Large loss indicates unconverged frequencies.
    """
    def solve(iterations: int) -> float:
        solver = MultiwayCfrSolver(
            MultiwayCfrConfig(
                board=("Js", "Jh", "7d"),
                ranges=(
                    exact(("As", "Ks"), ("Ac", "Kc"), ("Qd", "Qh")),
                    exact(("Jd", "Jc"), ("7s", "7h"), ("Ad", "Qs")),
                    exact(("Kd", "Kh"), ("Td", "Ts"), ("Ah", "Jd")),
                ),
                names=("Hero", "SB", "BB"),
                pot_bb=6.0, bet_bb=2.0, iterations=iterations, seed=11,
                stacks_bb=(100.0, 100.0, 100.0),
            ),
        )
        return solver.solve_and_result(("As", "Ks"))["hero"]["mix_ev_loss_bb"]

    assert solve(20_000) <= solve(400) + 1e-9
    assert solve(20_000) >= 0.0


def test_deadline_stops_softly_and_keeps_the_strategy() -> None:
    """Budget timeout must not discard progress made so far."""
    import time
    solver = MultiwayCfrSolver(
        MultiwayCfrConfig(
            board=("2c", "3d", "4h"),
            ranges=(exact(("As", "Ah"), ("7s", "6s")), exact(("Kd", "Kh"), ("Ad", "Qd"))),
            names=("Hero", "BB"), pot_bb=10.0, bet_bb=4.0,
            iterations=10_000_000, seed=5, stacks_bb=(50.0, 50.0),
        ),
    )
    started = time.monotonic()
    solver.solve(deadline=started + 0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, "deadline must halt loop"
    assert 0 < solver.iterations < 10_000_000
    result = solver.solve_and_result.__self__ and solver.result(
        ("As", "Ah"), solver.evaluate_hero_combo(("As", "Ah"), 200),
    )
    assert sum(result["hero"]["frequencies"]) == pytest.approx(1.0)


def test_evaluate_respects_the_deadline_but_keeps_samples() -> None:
    """Even on timeout, EV must have at least some samples collected."""
    import time
    solver = MultiwayCfrSolver(config(iterations=200))
    solver.solve()
    evaluation = solver.evaluate_hero_combo(
        ("As", "Ah"), 1_000_000, deadline=time.monotonic() + 0.3,
    )
    assert evaluation["samples"] > 0


def test_evaluate_deadline_holds_even_when_no_deal_succeeds() -> None:
    """Deadline must apply even when hero combination cannot be dealt.

    Previously 'at least one sample' condition kept done at 0 on deal failures,
    bypassing deadline and hanging until end of loop.
    """
    import time
    solver = MultiwayCfrSolver(config(iterations=50))
    solver.solve()
    started = time.monotonic()
    with pytest.raises(ValueError, match="budget ran out|block it"):
        # Combo blocked by board -> no deal succeeds.
        solver.evaluate_hero_combo(("2c", "3d"), 5_000_000, deadline=started + 0.4)
    assert time.monotonic() - started < 5.0, "deadline was not applied"
