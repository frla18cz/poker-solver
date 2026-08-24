"""Unit tests for AoF Spot & Hand Analyzer."""
from pokersolver.aof_lab.model import AofCashConfig
from pokersolver.aof_lab.spot_analyzer import analyze_hand_in_spot, get_top_percent_range


def test_top_percent_range():
    r10 = get_top_percent_range(10.0)
    assert "AA" in r10
    assert "KK" in r10
    assert "72o" not in r10


def test_spot_analyzer_aa_sb_vs_co_jam():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05)
    # Hero in SB facing CO Jam, BTN Fold (history='JF')
    res = analyze_hand_in_spot("SB", "JF", "AA", cfg, monte_carlo_samples=500)
    assert res["hero_pos"] == "SB"
    assert res["hand"] == "AA"
    assert res["ev_delta"] > 5.0  # Massive +EV
    assert res["recommendation"] == "PURE JAM / CALL"
    assert res["rec_color"] == "green"
    assert res["equity_pct"] > 70.0


def test_spot_analyzer_72o_sb_vs_co_jam():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05)
    res = analyze_hand_in_spot("SB", "JF", "72o", cfg, monte_carlo_samples=500)
    assert res["ev_delta"] < -2.0  # Clear -EV
    assert res["recommendation"] == "PURE FOLD"
    assert res["rec_color"] == "red"
