"""Two cards to a hand class ("AKs", "77", "T9o"), and the grid of all 169.

Cards use the format from ``cards.py``: a rank from '23456789TJQKA' and a suit
from 'shdc'. A hand class puts the higher rank first and adds 's' for suited or
'o' for offsuit; pairs carry no suffix.
"""
from __future__ import annotations

from itertools import combinations

from ..cards import RANKS, RANK_VALUE

# Ranks from the top down, so the notation is canonical.
_ORDER = "AKQJT98765432"


def hand_class(c1: str, c2: str) -> str:
    """'As','Kd' -> 'AKo'; 'As','Ks' -> 'AKs'; '7h','7d' -> '77'."""
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    # higher rank first
    if RANK_VALUE[r1] < RANK_VALUE[r2]:
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def all_hand_classes() -> list[str]:
    """All 169 hand classes: 13 pairs, 78 suited, 78 offsuit."""
    res = []
    for i, hi in enumerate(_ORDER):
        res.append(hi + hi)                       # pair
        for lo in _ORDER[i + 1:]:                 # lo ranks below hi
            res.append(hi + lo + "s")
            res.append(hi + lo + "o")
    return res


def combos_of_class(hc: str) -> list[tuple[str, str]]:
    """The actual card combos in a class — 'AKs' gives its 4 suited combos."""
    suits = "shdc"
    if len(hc) == 2:  # pair
        r = hc[0]
        return [(r + a, r + b) for a, b in combinations(suits, 2)]
    hi, lo, typ = hc[0], hc[1], hc[2]
    res = []
    if typ == "s":
        res = [(hi + s, lo + s) for s in suits]
    else:  # offsuit
        res = [(hi + a, lo + b) for a in suits for b in suits if a != b]
    return res


def is_pair(hc: str) -> bool:
    return len(hc) == 2
