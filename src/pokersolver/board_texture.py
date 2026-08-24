"""Board texture, and the bet sizes that follow from it.

Sizing — and whether the caller may lead out — should not be one constant for
every board. It depends on WHOSE range the board favours and HOW MANY draws it
brings. What follows is the usual consensus from solvers and the literature
(PioSolver / Upswing Lab /
Modern Poker Theory (Acevedo) / Play Optimal Poker (Brokos)):

  * Dry high boards (A/K/Q-high, unconnected) favour the PREFLOP RAISER's
    range: a small range c-bet, about a third of the pot, at high frequency.
    The caller does not lead.
  * Paired boards miss nearly everyone, so small bets at high frequency.
  * Low connected boards (654, 873) hit the CALLER harder — more sets, two
    pairs and straights — so the caller MAY lead and the raiser checks more.
  * Dynamic, wet boards (connected, draw-heavy) want big polarised bets, to
    charge the draws and protect equity.
  * Monotone boards (three of a suit) call for caution and smaller bets;
    blockers decide them.

The balance principle underneath all of it: on a given texture the SAME size is
used for value and for bluffs, or the size itself gives the hand away. That is
why `bet_fraction` takes only the texture and the street, never the intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cards import parse_card


class BoardTexture(Enum):
    """Board archetypes; they drive both sizing and the right to lead."""
    DRY_HIGH = "dry_high"            # A/K/Q-high, unconnected: raiser ahead, small range bet
    DRY_PAIRED = "dry_paired"        # paired but otherwise dry: small bets, high frequency
    LOW_CONNECTED = "low_connected"  # low and connected (<=9): caller ahead, leading allowed
    DYNAMIC = "dynamic"              # connected high or draw-heavy: big polarised bets
    MONOTONE = "monotone"            # three of a suit: cautious, smaller bets
    MIDDLING = "middling"            # everything else: a middling bet


class RangeEdge(Enum):
    """Whose preflop range the board favours."""
    RAISER = "raiser"      # the preflop aggressor
    CALLER = "caller"      # whoever called; a leading spot
    NEUTRAL = "neutral"    # neither, particularly


_EDGE: dict[BoardTexture, RangeEdge] = {
    BoardTexture.DRY_HIGH: RangeEdge.RAISER,
    BoardTexture.DRY_PAIRED: RangeEdge.RAISER,
    BoardTexture.LOW_CONNECTED: RangeEdge.CALLER,
    BoardTexture.DYNAMIC: RangeEdge.NEUTRAL,
    BoardTexture.MONOTONE: RangeEdge.NEUTRAL,
    BoardTexture.MIDDLING: RangeEdge.NEUTRAL,
}

# Flop bet size as a share of the pot, per texture — a range (lo, hi) rather
# than one number. Dry boards get small range bets, low connected boards middling
# ones, dynamic boards big polarised ones. The actual bet is sampled from the
# range, because varying it is harder to read. Dry and wet ranges never overlap,
# so a dry board always bets smaller than a dynamic one.
_FLOP_RANGE: dict[BoardTexture, tuple[float, float]] = {
    BoardTexture.DRY_HIGH: (0.25, 0.40),
    BoardTexture.DRY_PAIRED: (0.25, 0.40),
    BoardTexture.MONOTONE: (0.33, 0.50),
    BoardTexture.MIDDLING: (0.45, 0.60),
    BoardTexture.LOW_CONNECTED: (0.45, 0.66),
    BoardTexture.DYNAMIC: (0.60, 0.85),
}
# Bets grow on later streets: ranges polarise and the pot geometry demands it.
_STREET_MULT: dict[int, float] = {3: 1.0, 4: 1.05, 5: 1.12}

_FRACTION_MIN, _FRACTION_MAX = 0.25, 1.25  # the cap allows a wet-river overbet


def _connectedness(vals: set[int]) -> int:
    """How many distinct ranks fit in the busiest five-value window.

    2 means isolated, 3 means three to a straight, 4-5 means very connected.
    The ace counts as low as well.
    """
    uniq = set(vals)
    if 14 in uniq:
        uniq = uniq | {1}  # the ace also plays low (A-2-3-4-5)
    best = 0
    for low in range(1, 11):
        window = {v for v in uniq if low <= v <= low + 4}
        best = max(best, len(window))
    return best


@dataclass(frozen=True)
class BoardInfo:
    """What reading a board produced."""
    texture: BoardTexture
    edge: RangeEdge
    paired: bool
    suit: str          # "rainbow" | "two_tone" | "monotone"
    connected: int     # the _connectedness score (2..5)
    high: int          # the highest rank on the board (2..14)

    @property
    def favors_caller(self) -> bool:
        """Does the board favour the caller — may they lead out?"""
        return self.edge is RangeEdge.CALLER

    def bet_fraction_range(self, street_cards: int) -> tuple[float, float]:
        """The (lo, hi) share of pot for a street: 3 flop, 4 turn, 5 river."""
        lo, hi = _FLOP_RANGE[self.texture]
        mult = _STREET_MULT.get(street_cards, 1.0)
        clamp = lambda x: max(_FRACTION_MIN, min(_FRACTION_MAX, x * mult))
        return clamp(lo), clamp(hi)

    def bet_fraction(self, street_cards: int) -> float:
        """The middle of the range — deterministic, for tests and previews."""
        lo, hi = self.bet_fraction_range(street_cards)
        return (lo + hi) / 2


def read_board(board: list[str]) -> BoardInfo:
    """Read a board of 3-5 cards: texture, whose range it favours, and sizing.

    Fewer than three cards (preflop) reads as a neutral MIDDLING board.
    """
    cards = [c for c in board if isinstance(c, str) and len(c) == 2]
    if len(cards) < 3:
        return BoardInfo(BoardTexture.MIDDLING, RangeEdge.NEUTRAL, False, "rainbow", 2, 2)

    vals = [parse_card(c)[0] for c in cards]
    suits = [parse_card(c)[1] for c in cards]

    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    paired = any(c >= 2 for c in counts.values())

    max_suit = max(suits.count(s) for s in set(suits))
    suit = "monotone" if max_suit >= 3 else "two_tone" if max_suit == 2 else "rainbow"

    high = max(vals)
    connected = _connectedness(set(vals))
    straighty = connected >= 3  # at least three cards to a straight

    if suit == "monotone":
        tex = BoardTexture.MONOTONE
    elif paired:
        tex = BoardTexture.DRY_PAIRED
    elif straighty and high <= 9:
        tex = BoardTexture.LOW_CONNECTED
    elif straighty:                     # connected and high (JT9, QJ8): wet
        tex = BoardTexture.DYNAMIC
    elif high >= 12:                    # A/K/Q-high, unconnected: dry and high
        tex = BoardTexture.DRY_HIGH
    else:                               # middling or plain two-tone, unconnected
        tex = BoardTexture.MIDDLING

    return BoardInfo(tex, _EDGE[tex], paired, suit, connected, high)
