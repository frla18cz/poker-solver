"""GTO loader tests: is_hero hero consistency, normalization, labels, fallback."""
from __future__ import annotations

import math

import pytest

from pokersolver.ranges.gto import GtoSolutions, default_solutions
from pokersolver.ranges.gto.labels import (
    action_label, decode_sequence, describe_line,
)


@pytest.fixture(scope="module")
def sol() -> GtoSolutions:
    # Explicitly NL100 artifact (not runtime default). Fresh authoritative scrape
    # Cash6mGeneral_6mNL100R25 2.5x /100bb directly from GTO Wizard (233 spots, 0
    # mislabeled — self-inconsistent deep squeeze nodes removed). See README
    # in data/.
    from pokersolver.ranges.gto.loader import DATA_DIR
    return GtoSolutions.load(DATA_DIR / "nl100_100bb.json")


def test_dataset_loads(sol):
    assert len(sol) == 233
    assert sol.meta["gametype"] == "Cash6mGeneral_6mNL100R25"
    # Fresh pull is clean: 0 mislabeled spots
    assert sol.meta["n_label_mismatch"] == 0


def test_all_spots_hero_consistent(sol):
    """Fresh data: declared hero == position derived from sequence (no
    mislabels, no self-inconsistent nodes)."""
    from pokersolver.ranges.gto.labels import decode_sequence
    for sid in sol.ids():
        spot = sol.get(sid)
        assert not spot.label_mismatch, f"{sid} has label_mismatch"
        _events, seq_hero = decode_sequence(spot.sequence)
        assert spot.hero == seq_hero, f"{sid}: hero {spot.hero} != seq {seq_hero}"


def test_coldcaller_range_has_no_premiums(sol):
    """Cold-caller's range properly does NOT contain AA/KK/AKs (would 3-bet them).
    Node: SB cold-calls CO open, BB squeezes -> SB to act."""
    spot = sol.get("sb_vs_sqz_co_bb")
    for premium in ("AA", "KK", "AKs"):
        assert not spot.in_range(premium)
        assert spot.decision(premium) is None
    # but marginal hands are in range
    assert spot.in_range("AJs")


def test_decision_is_normalized(sol):
    """decision() normalizes range-weighted frequencies to sum to 1.0."""
    spot = sol.get("sb_vs_sqz_co_bb")
    dec = spot.decision("AJs")
    assert dec is not None
    assert math.isclose(sum(dec.values()), 1.0, abs_tol=1e-6)
    # AJs mostly folds here (OOP cold-call vs squeeze)
    label, freq = spot.top_action("AJs")
    assert label == "Fold"
    assert freq > 0.5


def test_rfi_spot_full_range_normalizes(sol):
    """Root RFI spot: premiums 100% raise, strategy sum = 1."""
    spot = sol.get("rfi_btn")
    assert spot.hero == "BTN"
    assert not spot.label_mismatch
    dec = spot.decision("AA")
    assert math.isclose(sum(dec.values()), 1.0, abs_tol=1e-6)
    label, freq = spot.top_action("AA")
    assert "Raise" in label or "All-in" in label
    assert freq > 0.99
    # hand out of range folds / is not raise
    assert spot.top_action("72o")[0] == "Fold" or spot.decision("72o") is None


def test_action_labels_present(sol):
    spot = sol.get("sb_vs_sqz_co_bb")
    labels = {a.label for a in spot.actions}
    assert "Fold" in labels
    assert any(l.startswith("Call") for l in labels)
    assert any("All-in" in l for l in labels)


def test_resolve_fallback(sol):
    assert sol.get_or_none("nonexistent") is None
    got = sol.resolve("nonexistent", fallbacks=["also_nonexistent", "rfi_utg"])
    assert got is not None and got.spot_id == "rfi_utg"
    assert sol.resolve("nonexistent", fallbacks=["also_nonexistent"]) is None


# ── pure helpers ──

def test_action_label_bet_levels():
    assert action_label("fold", None, 0, 100) == "Fold"
    assert action_label("call", 15.0, 2, 100) == "Call 15bb"
    assert action_label("raise", 2.5, 0, 100) == "Raise to 2.5bb"
    assert action_label("raise", 13.0, 1, 100) == "3-bet to 13bb"
    assert action_label("raise", 34.0, 2, 100) == "4-bet to 34bb"
    assert action_label("raise", 100.0, 2, 100) == "All-in (100bb)"


def test_decode_sequence_hero():
    # UTG open, HJ/CO fold, BTN call, SB fold, BB squeeze, UTG fold -> BTN to act
    events, hero = decode_sequence("R2.5-F-F-C-F-R15-F")
    assert hero == "BTN"
    assert events[0] == ("UTG", "R2.5")
    assert ("BTN", "C") in events
    assert ("BB", "R15") in events


def test_describe_line_uses_given_hero():
    line = describe_line("R2.5-F-F-C-F-R15-F", hero="BTN")
    assert line.endswith("-> BTN to act")
    assert "UTG raise 2.5" in line
