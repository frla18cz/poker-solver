"""Regrese pro offline 4max NLHE All-In or Fold lab."""
from __future__ import annotations

import math
import json

import pytest

from pokersolver.aof.solver import (
    AofCfrSolver,
    AofConfig,
    AofInfoSet,
    FOLD,
    JAM,
    canonical_combo,
    parse_history,
    render_matrix,
    settle_actions,
)


def test_walk_pays_bb_the_small_blind_minus_fixed_fee() -> None:
    config = AofConfig(flat_rake_bb=0.12)
    utilities = settle_actions(config, (FOLD, FOLD, FOLD, FOLD))
    assert utilities == pytest.approx((0.0, 0.0, -0.5, 0.38))
    assert sum(utilities) == pytest.approx(-0.12)


def test_uncontested_jam_collects_blinds_and_fee() -> None:
    config = AofConfig(flat_rake_bb=0.12)
    utilities = settle_actions(config, (JAM, FOLD, FOLD, FOLD))
    # CO invests 8bb and collects 9.5bb pot after flat fee.
    assert utilities == pytest.approx((1.38, 0.0, -0.5, -1.0))
    assert sum(utilities) == pytest.approx(-0.12)


def test_multiway_showdown_splits_and_conserves_rake() -> None:
    config = AofConfig(flat_rake_bb=0.12)
    # CO and BTN have identical best scores, blinds fold.
    scores = ((8, 14), (8, 14), (1, 2), (1, 3))
    utilities = settle_actions(config, (JAM, JAM, FOLD, FOLD), scores)
    # 17.5bb pot after 0.12bb fee, each winner gets 8.69bb for 8bb invested.
    assert utilities[0] == pytest.approx(0.69)
    assert utilities[1] == pytest.approx(0.69)
    assert utilities[2:] == pytest.approx((-0.5, -1.0))
    assert sum(utilities) == pytest.approx(-0.12)


def test_combo_and_history_are_canonical_and_validated() -> None:
    assert canonical_combo(("Kd", "As")) == canonical_combo(("As", "Kd"))
    assert parse_history("f, j") == (FOLD, JAM)
    with pytest.raises(ValueError):
        parse_history("FX")
    with pytest.raises(ValueError):
        parse_history("FFFF")


def test_solver_builds_all_tree_matrices_and_is_deterministic() -> None:
    config = AofConfig(flat_rake_bb=0.12)
    first = AofCfrSolver(config, seed=73)
    second = AofCfrSolver(config, seed=73)
    first.solve(160)
    second.solve(160)

    assert first.iterations == 160
    assert len(first.all_matrices()) == 15
    key = AofInfoSet(0, (), canonical_combo(("As", "Ah")))
    assert first.policy_for(key) == pytest.approx(second.policy_for(key))
    root = first.matrix("CO")
    assert len(root) == 169
    assert root["AA"].jam is not None
    assert root["AA"].fold is not None
    assert math.isclose(root["AA"].jam + root["AA"].fold, 1.0, abs_tol=1e-9)


def test_matrix_render_and_average_policy_ev() -> None:
    solver = AofCfrSolver(seed=4)
    solver.solve(120)
    text = "\n".join(render_matrix(solver.matrix("CO"), metric="jam"))
    assert "A" in text and "K" in text
    ev = solver.evaluate_average_strategy(200, seed=12)
    assert set(ev) == {"CO", "BTN", "SB", "BB"}
    # Each simulated hand distributes exact pot after 0.12bb fixed fee.
    assert sum(ev.values()) == pytest.approx(-0.12, abs=1e-9)


def test_json_export_contains_all_context_matrices(tmp_path) -> None:
    solver = AofCfrSolver(seed=11)
    solver.solve(80)
    output = solver.write_json(tmp_path / "aof.json", evaluation_samples=40)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "pokersolver-aof-v1"
    assert payload["iterations"] == 80
    assert set(payload["matrices"]) == {
        "CO:root", "BTN:F", "BTN:J",
        "SB:FF", "SB:FJ", "SB:JF", "SB:JJ",
        "BB:FFF", "BB:FFJ", "BB:FJF", "BB:FJJ",
        "BB:JFF", "BB:JFJ", "BB:JJF", "BB:JJJ",
    }
    assert set(payload["matrices"]["CO:root"]) == set(solver.matrix("CO"))
