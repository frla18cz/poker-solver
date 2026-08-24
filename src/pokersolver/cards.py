"""Cards and hand evaluation, with no dependencies beyond the standard library.

A card is a two-character string: rank + suit — "As", "Td", "9c", "2h".
  rank: 2 3 4 5 6 7 8 9 T J Q K A
  suit: s(pades) h(earts) d(iamonds) c(lubs)
"""
from __future__ import annotations

from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}  # 2..14
_SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}

FULL_DECK = [r + s for r in RANKS for s in SUITS]

# Hand categories (higher is better)
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8


def parse_card(card: str) -> tuple[int, str]:
    """(rank value, suit) for a card like 'As'."""
    card = card.strip()
    if len(card) != 2 or card[0] not in RANK_VALUE or card[1] not in SUITS:
        raise ValueError(f"not a card: {card!r}")
    return RANK_VALUE[card[0]], card[1]


def _straight_high(values: set[int]) -> int | None:
    """Top card of the straight, or None. Handles the A-2-3-4-5 wheel."""
    if 14 in values:
        values = values | {1}  # the ace also plays low
    ordered = sorted(values, reverse=True)
    run = 1
    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1] - 1:
            run += 1
            if run >= 5:
                return ordered[i] + 4
        elif ordered[i] != ordered[i - 1]:
            run = 1
    return None


def score_5(cards: list[str]) -> tuple:
    """Score exactly 5 cards as a comparable tuple: (category, tiebreakers...).

    A higher tuple is a stronger hand — ordinary tuple comparison just works.
    """
    vals = sorted((parse_card(c)[0] for c in cards), reverse=True)
    suits = [parse_card(c)[1] for c in cards]

    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(set(vals))

    # rank counts, ordered by (count, value)
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # sort ranks by count first, then by value — both descending
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    count_pattern = tuple(c for _, c in by_count)
    ordered_vals = tuple(v for v, _ in by_count)

    if is_flush and straight_high:
        return (STRAIGHT_FLUSH, straight_high)
    if count_pattern == (4, 1):
        return (QUADS, ordered_vals)
    if count_pattern == (3, 2):
        return (FULL_HOUSE, ordered_vals)
    if is_flush:
        return (FLUSH, tuple(vals))
    if straight_high:
        return (STRAIGHT, straight_high)
    if count_pattern == (3, 1, 1):
        return (TRIPS, ordered_vals)
    if count_pattern == (2, 2, 1):
        return (TWO_PAIR, ordered_vals)
    if count_pattern == (2, 1, 1, 1):
        return (PAIR, ordered_vals)
    return (HIGH_CARD, tuple(vals))


def best_hand(cards: list[str]) -> tuple:
    """Best five-card hand out of 5-7 cards, as a comparable score."""
    if len(cards) < 5:
        raise ValueError("need at least 5 cards")
    if len(cards) == 5:
        return score_5(cards)
    return max(score_5(list(combo)) for combo in combinations(cards, 5))


# --- fast path for solvers ----------------------------------------------------
# ``best_hand`` walks all 21 five-card combinations, which is over 80% of the
# time a CFR solver spends. ``score_7`` computes the same answer in one pass
# over a rank histogram and a suit bitmask.

_STRAIGHT_MASKS: tuple[tuple[int, int], ...] = tuple(
    # (mask of five consecutive ranks, top card of that straight)
    (sum(1 << (high - offset) for offset in range(5)), high)
    for high in range(14, 5, -1)
)
_WHEEL_MASK = (1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)


def _straight_from_mask(mask: int) -> int | None:
    """Highest straight in a rank bitmask, or None. Handles the wheel."""
    for pattern, high in _STRAIGHT_MASKS:
        if mask & pattern == pattern:
            return high
    return 5 if mask & _WHEEL_MASK == _WHEEL_MASK else None


def score_7(cards) -> tuple:
    """Score 5-7 cards in a single pass.

    Returns a tuple that orders **identically** to :func:`best_hand`; the
    shapes match, so results from the two can be compared with each other.
    """
    by_rank = [0] * 15          # rank -> count
    by_suit = [0] * 4           # suit -> count
    suit_masks = [0, 0, 0, 0]   # suit -> rank bitmask
    mask = 0
    for card in cards:
        value = RANK_VALUE[card[0]]
        suit = _SUIT_INDEX[card[1]]
        by_rank[value] += 1
        by_suit[suit] += 1
        suit_masks[suit] |= 1 << value
        mask |= 1 << value

    for suit, count in enumerate(by_suit):
        if count >= 5:
            flush_mask = suit_masks[suit]
            straight_high = _straight_from_mask(flush_mask)
            if straight_high:
                return (STRAIGHT_FLUSH, straight_high)
            ranks = [v for v in range(14, 1, -1) if flush_mask >> v & 1]
            return (FLUSH, tuple(ranks[:5]))

    quads = trips = 0
    pairs: list[int] = []
    for value in range(14, 1, -1):
        count = by_rank[value]
        if count == 4 and not quads:
            quads = value
        elif count == 3 and not trips:
            trips = value
        elif count >= 2:
            pairs.append(value)

    if quads:
        kicker = max(v for v in range(14, 1, -1) if by_rank[v] and v != quads)
        return (QUADS, (quads, kicker))
    if trips and pairs:
        return (FULL_HOUSE, (trips, pairs[0]))

    straight_high = _straight_from_mask(mask)
    if straight_high:
        return (STRAIGHT, straight_high)

    if trips:
        kickers = [v for v in range(14, 1, -1) if by_rank[v] and v != trips]
        return (TRIPS, (trips, *kickers[:2]))
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = max(v for v in range(14, 1, -1) if by_rank[v] and v not in (high, low))
        return (TWO_PAIR, (high, low, kicker))
    if pairs:
        pair = pairs[0]
        kickers = [v for v in range(14, 1, -1) if by_rank[v] and v != pair]
        return (PAIR, (pair, *kickers[:3]))

    ranks = [v for v in range(14, 1, -1) if by_rank[v]]
    return (HIGH_CARD, tuple(ranks[:5]))
