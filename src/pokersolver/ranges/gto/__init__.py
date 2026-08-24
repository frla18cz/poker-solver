"""Precomputed preflop matrices — a dependable layer over stored solutions.

Two things matter: take the hero from is_hero rather than meta.position, and
normalise frequencies through ``GtoSpot.decision()``. See ``loader.py``.

    from pokersolver.ranges.gto import default_solutions
    sol = default_solutions()
    spot = sol.get_or_none("bb_sqz_vs_utg_fold_btn_4b")
    if spot:
        print(spot.hero, spot.decision("AJs"))   # {'Call 15bb': 0.19, 'Fold': 0.81}

Mapping a game state onto a spot (``spot_id_for_state`` and friends) is
deliberately absent: it needs a state type, and that belongs to the caller.
"""
from __future__ import annotations

from .loader import (
    Action, GtoSolutions, GtoSpot, MultiDepthSolutions, MultiSizeSolutions,
    default_multi_solutions, default_solutions,
)

__all__ = [
    "Action", "GtoSolutions", "GtoSpot", "MultiDepthSolutions",
    "MultiSizeSolutions", "default_solutions", "default_multi_solutions",
]
