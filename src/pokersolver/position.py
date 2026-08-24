"""Table position (BTN/SB/BB/EP/MP/CO) worked out from primitives.

Nothing here knows about a game-state type: it takes seat indexes, the button
and the blinds, so any engine can use it as-is.
"""
from __future__ import annotations

# Positions in order of how early you act; earlier is worse postflop.
POSITIONS = ["EP", "MP", "CO", "BTN", "SB", "BB"]


def position_label(hero: int, dealer: int | None, sb: int | None, bb: int | None,
                   seats_in_hand: list[int]) -> str:
    """The position of ``hero``. ``seats_in_hand`` are the occupied seat indexes."""
    # Heads-up, the button also posts the small blind — and SB is the label
    # that matters to strategy.
    if len(seats_in_hand) == 2 and hero == dealer and hero == sb:
        return "SB"
    if hero == dealer:
        return "BTN"
    if hero == sb:
        return "SB"
    if hero == bb:
        return "BB"
    if not seats_in_hand or dealer is None or hero not in seats_in_hand:
        return "?"
    order = sorted(seats_in_hand)
    start = order.index(dealer) if dealer in order else 0
    rot = order[start:] + order[:start]  # BTN first, then SB, BB, ...
    try:
        idx = rot.index(hero)
    except ValueError:
        return "?"
    n = len(rot)
    if idx <= 2:
        return ["BTN", "SB", "BB"][idx]
    # Non-blind seats (idx 3..n-1): the last is CO, the one before it is MP at
    # a fullish table, and the earliest is EP (UTG). Higher idx acts earlier.
    if idx == n - 1:
        return "CO"
    if idx == n - 2 and n >= 6:
        return "MP"
    return "EP"


def _next_seat(seat: int | None, seats_in: list[int]) -> int | None:
    if seat is None or seat not in seats_in:
        return None
    i = seats_in.index(seat)
    return seats_in[(i + 1) % len(seats_in)]
