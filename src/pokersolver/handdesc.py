"""Describing how strong a hand is on a board, in words.

Returns a readable summary: the made hand plus any draws.

The wording is the library's output format, not a display string — callers
that need another language translate it themselves.
"""
from __future__ import annotations

from .cards import (
    FLUSH, PAIR, QUADS, RANK_VALUE, STRAIGHT, TRIPS, TWO_PAIR,
    best_hand, parse_card,
)

_CAT = {
    0: "high card", 1: "pair", 2: "two pair", 3: "three of a kind",
    4: "straight", 5: "flush", 6: "full house", 7: "four of a kind",
    8: "straight flush",
}

_RANK_NAME = {value: rank for rank, value in RANK_VALUE.items()}


def board_only_made_hand(hole: list[str], board: list[str]) -> dict | None:
    """Describe a made pair that comes entirely from the board.

    The category of the best five-card hand is not enough on its own: on
    ``Q33`` a player holding ``K2`` technically has a pair, but neither hole
    card paired anything. Anything reading the description can mistake a shared
    pair for second pair, so this reports where the made hand actually came
    from, plus the hero's kicker if there is one.
    """
    if len(hole) != 2 or len(board) < 3:
        return None
    hole_vals = [parse_card(card)[0] for card in hole]
    board_vals = [parse_card(card)[0] for card in board]
    # A pocket pair, or a hole card matching the board, means the hand really
    # contributes to the pairing — not a board-only case.
    if hole_vals[0] == hole_vals[1] or set(hole_vals) & set(board_vals):
        return None
    score = best_hand(hole + board)
    category = score[0]
    if category not in (PAIR, TWO_PAIR, TRIPS, QUADS):
        return None
    counts = {value: board_vals.count(value) for value in set(board_vals)}
    required = {PAIR: 2, TWO_PAIR: 2, TRIPS: 3, QUADS: 4}[category]
    made_ranks = sorted(
        (value for value, count in counts.items() if count >= required),
        reverse=True,
    )
    if category == TWO_PAIR:
        made_ranks = sorted(
            (value for value, count in counts.items() if count >= 2),
            reverse=True,
        )[:2]
    elif made_ranks:
        made_ranks = made_ranks[:1]
    if not made_ranks or (category == TWO_PAIR and len(made_ranks) < 2):
        return None

    # score[1] holds the paired ranks followed by the relevant kickers.
    group_count = 2 if category == TWO_PAIR else 1
    tiebreak_values = score[1] if isinstance(score[1], tuple) else (score[1],)
    kicker_values = set(tiebreak_values[group_count:])
    hero_kickers = sorted(
        (value for value in hole_vals if value in kicker_values), reverse=True,
    )
    return {
        "category": category,
        "category_name": _CAT[category],
        "board_ranks": [_RANK_NAME[value] for value in made_ranks],
        "hero_kicker": _RANK_NAME[hero_kickers[0]] if hero_kickers else None,
    }


def _has_flush_draw(cards: list[str]) -> bool:
    suits = [parse_card(c)[1] for c in cards]
    return any(suits.count(s) == 4 for s in set(suits))


def _has_straight_draw(cards: list[str]) -> bool:
    vals = set(parse_card(c)[0] for c in cards)
    if 14 in vals:
        vals = vals | {1}  # eso i jako low
    ordered = sorted(vals)
    # a window of 5 values holding at least 4 — open-ended or gutshot
    for low in range(1, 11):
        window = [v for v in ordered if low <= v <= low + 4]
        if len(set(window)) >= 4:
            return True
    return False


def _pair_kind(hole: list[str], board: list[str]) -> str:
    """Tell top, second, bottom and pocket pairs apart against the board."""
    board_vals = sorted((parse_card(c)[0] for c in board), reverse=True)
    hole_vals = [parse_card(c)[0] for c in hole]
    if hole_vals[0] == hole_vals[1]:  # pocket pair
        if board_vals and hole_vals[0] > board_vals[0]:
            return "overpair"
        return "pocket pair"
    # paired with the board?
    for hv in sorted(hole_vals, reverse=True):
        if hv in board_vals:
            if board_vals and hv == board_vals[0]:
                return "top pair"
            if len(board_vals) > 1 and hv == board_vals[1]:
                return "second pair"
            return "bottom pair"
    return "pair"


def describe_hand(hole: list[str], board: list[str]) -> str:
    """For example 'top pair', 'flush draw', 'two pair + flush draw'."""
    if len(hole) != 2 or len(board) < 3:
        return "?"
    cat = best_hand(hole + board)[0]
    board_only = board_only_made_hand(hole, board)
    if board_only:
        ranks = " and ".join(board_only["board_ranks"])
        kicker = (f"; hero kicker {board_only['hero_kicker']}"
                  if board_only["hero_kicker"] else "; hero plays the board")
        made = (f"{board_only['category_name']} {ranks} on the board only; "
                f"hero has no private pair{kicker}")
    elif cat == PAIR:
        made = _pair_kind(hole, board)
    else:
        made = _CAT.get(cat, "?")
    draws = []
    if len(board) < 5:  # nothing left to draw to on the river
        if cat < FLUSH and _has_flush_draw(hole + board):
            draws.append("flush draw")
        if cat < STRAIGHT and _has_straight_draw(hole + board):
            draws.append("straight draw")
    if draws and cat == 0:            # high card plus a draw: show just the draw
        return " + ".join(draws)
    return made + (" + " + " + ".join(draws) if draws else "")
