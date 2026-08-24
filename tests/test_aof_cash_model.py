"""Unit tests for 4max AoF Cash Game model, side-pots, and 5% rake calculation."""
import pytest
from pokersolver.aof_lab.model import (
    AofCashConfig,
    settle_cash_actions,
    calculate_rake,
    canonical_combo,
    parse_history,
)


def test_config_validation():
    cfg = AofCashConfig(stacks_bb=(20.0, 50.0, 10.0, 100.0), rake_pct=0.05)
    assert cfg.stacks_bb == (20.0, 50.0, 10.0, 100.0)
    assert cfg.rake_pct == 0.05
    assert cfg.blinds() == (0.0, 0.0, 0.5, 1.0)
    assert cfg.effective_stack_bb(0, 1) == 20.0


def test_walk_no_rake_default():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05, rake_on_uncontested=False)
    actions = ("F", "F", "F", "F")
    payoffs = settle_cash_actions(cfg, actions)
    # CO=0, BTN=0, SB=-0.5, BB=+0.5 (blind pot is 1.5, BB committed 1.0 -> net +0.5)
    assert payoffs == (0.0, 0.0, -0.5, 0.5)


def test_single_jam_uncontested():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05, rake_on_uncontested=False)
    actions = ("J", "F", "F", "F")
    payoffs = settle_cash_actions(cfg, actions)
    # CO committed 20, collected 20 + 0 + 0.5 + 1.0 = 21.5 -> net +1.5 BB
    assert payoffs == (1.5, 0.0, -0.5, -1.0)


def test_contested_showdown_symmetric_5pct_rake():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05)
    actions = ("J", "J", "F", "F")
    # CO (p0) wins with higher score
    scores = (1000, 500, 100, 100)
    payoffs = settle_cash_actions(cfg, actions, scores)
    # Total pot: 20 + 20 + 0.5 + 1.0 = 41.5 BB
    # 5% rake: 41.5 * 0.05 = 2.075 BB -> Net pot: 39.425 BB
    # CO net: 39.425 - 20 = +19.425 BB
    # BTN net: -20 BB, SB: -0.5 BB, BB: -1.0 BB
    assert pytest.approx(payoffs[0], 0.001) == 19.425
    assert payoffs[1] == -20.0
    assert payoffs[2] == -0.5
    assert payoffs[3] == -1.0


def test_asymmetric_side_pots():
    # CO 10bb, BTN 50bb, SB 20bb, BB 20bb
    cfg = AofCashConfig(stacks_bb=(10.0, 50.0, 20.0, 20.0), rake_pct=0.0)
    actions = ("J", "J", "F", "F")
    # BTN has 50bb, CO has 10bb. Effective all-in is 10bb. 40bb is uncalled.
    # Scores: CO wins main pot
    scores = (1000, 500, 100, 100)
    payoffs = settle_cash_actions(cfg, actions, scores)
    # Total committed: CO=10, BTN=50 (10 in pot + 40 uncalled returned), SB=0.5, BB=1.0
    # Main pot: 10 + 10 + 0.5 + 1.0 = 21.5 BB
    # CO receives 21.5 -> net +11.5 BB
    # BTN loses 10 BB -> net -10.0 BB
    assert pytest.approx(payoffs[0], 0.001) == 11.5
    assert pytest.approx(payoffs[1], 0.001) == -10.0
    assert payoffs[2] == -0.5
    assert payoffs[3] == -1.0
