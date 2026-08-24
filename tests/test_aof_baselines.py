"""Unit tests for AoF Baselines and Archetypes."""
import pytest
from pokersolver.aof_lab.baselines import (
    ARCHETYPES,
    STACK_SCALING,
    get_archetype_range,
    get_range_for_percent,
)
from pokersolver.aof_lab.model import AofCashConfig
from pokersolver.aof_lab.spot_analyzer import analyze_hand_in_spot


def test_archetype_ranges():
    gto_co = get_archetype_range("GTO", "CO", (), effective_stack_bb=8.0)
    assert "AA" in gto_co
    assert "AKs" in gto_co
    assert "72o" not in gto_co

    fish_co = get_archetype_range("FISH", "CO", (), effective_stack_bb=8.0)
    assert len(fish_co) > len(gto_co)

    nit_co = get_archetype_range("NIT", "CO", (), effective_stack_bb=8.0)
    assert len(nit_co) < len(gto_co)


def test_stack_depth_scaling():
    # 50BB open should be significantly tighter than 8BB open
    range_8bb = get_archetype_range("GTO", "CO", (), effective_stack_bb=8.0)
    range_50bb = get_archetype_range("GTO", "CO", (), effective_stack_bb=50.0)

    assert len(range_50bb) < len(range_8bb)
    assert "AA" in range_50bb


def test_spot_analyzer_with_archetypes():
    cfg = AofCashConfig(stacks_bb=(8.0, 8.0, 8.0, 8.0), rake_pct=0.05)
    # TJs in SB vs Fish Jam
    res_fish = analyze_hand_in_spot("SB", "JF", "TJs", cfg, opponent_profile="FISH", monte_carlo_samples=400)
    assert res_fish["recommendation"] == "PURE JAM / CALL"

    # TJs in SB vs Nit Jam (should be much worse / fold)
    res_nit = analyze_hand_in_spot("SB", "JF", "TJs", cfg, opponent_profile="NIT", monte_carlo_samples=400)
    assert res_nit["ev_delta"] < res_fish["ev_delta"]
