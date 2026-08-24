"""pokersolver — deterministic poker maths, with no game state involved.

Cards and hand evaluation, Monte Carlo equity against a weighted range, the
grid of 169 hand classes, precomputed preflop matrices, and CFR solvers for
all-in-or-fold and multiway postflop spots.

The library has no notion of "the player to act" and no game-state type. It
takes primitives — cards, stacks in big blinds, positions as strings — and
returns numbers. Wiring it to an engine is the caller's job.
"""
from __future__ import annotations

__all__ = ["cards", "equity", "ranges"]
