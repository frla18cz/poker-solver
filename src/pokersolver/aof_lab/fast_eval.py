"""Fast 7-card poker hand evaluator optimized for multiway showdowns.

Represents cards as 0..51 integers or standard strings ('As', 'Kd', etc.) and
evaluates any 7-card hand into a comparable 32-bit integer score (higher = better).
Provides ~20-40x speedup over combinations(7, 5) + score_5.
"""
from __future__ import annotations

from typing import Final, Iterable
from ..cards import FULL_DECK, RANKS, SUITS, RANK_VALUE

# Card mappings: 0..51 -> rank (0..12 for 2..A), suit (0..3)
_CARD_TO_INT: Final[dict[str, int]] = {
    card: (RANK_VALUE[card[0]] - 2) * 4 + "shdc".index(card[1])
    for card in FULL_DECK
}
_INT_TO_CARD: Final[list[str]] = sorted(FULL_DECK, key=lambda c: _CARD_TO_INT[c])


def card_to_int(card: str) -> int:
    return _CARD_TO_INT[card]


def int_to_card(idx: int) -> str:
    return _INT_TO_CARD[idx]


def _build_straight_lookup() -> dict[int, int]:
    """Rank bitmask (13 bits) -> high card of straight (5..14) or 0."""
    res = {}
    for mask in range(1 << 13):
        straight_high = 0
        for high in range(12, 3, -1):  # 12 (Ace) down to 4 (6)
            pattern = (0x1F << (high - 4))
            if (mask & pattern) == pattern:
                straight_high = high + 2  # rank value 6..14
                break
        if not straight_high:
            # Check wheel (5-4-3-2-A): bits 3, 2, 1, 0 and 12
            wheel = (1 << 12) | 0x0F
            if (mask & wheel) == wheel:
                straight_high = 5
        res[mask] = straight_high
    return res


_STRAIGHT_LOOKUP: Final[dict[int, int]] = _build_straight_lookup()


def evaluate_7cards_int(cards: list[int] | tuple[int, ...]) -> int:
    """Evaluate exactly 7 card integers (0..51) into a single integer score.

    Score encoding (higher is strictly better):
    - [Category 0..8] << 24
    - Tie-breakers encoded in lower 24 bits
    Categories:
      8: Straight Flush
      7: Four of a Kind
      6: Full House
      5: Flush
      4: Straight
      3: Three of a Kind
      2: Two Pair
      1: One Pair
      0: High Card
    """
    suit_counts = [0, 0, 0, 0]
    rank_counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    rank_mask = 0

    for c in cards:
        r = c >> 2  # 0..12
        s = c & 3   # 0..3
        suit_counts[s] += 1
        rank_counts[r] += 1
        suit_masks[s] |= (1 << r)
        rank_mask |= (1 << r)

    # 1. Check Flush & Straight Flush
    flush_suit = -1
    for s in range(4):
        if suit_counts[s] >= 5:
            flush_suit = s
            break

    if flush_suit != -1:
        flush_mask = suit_masks[flush_suit]
        sf_high = _STRAIGHT_LOOKUP[flush_mask]
        if sf_high:
            return (8 << 24) | sf_high
        # Regular flush: take top 5 bits of flush_mask
        bits = []
        for r in range(12, -1, -1):
            if flush_mask & (1 << r):
                bits.append(r + 2)
                if len(bits) == 5:
                    break
        return (5 << 24) | (bits[0] << 16) | (bits[1] << 12) | (bits[2] << 8) | (bits[3] << 4) | bits[4]

    # 2. Check Straight
    straight_high = _STRAIGHT_LOOKUP[rank_mask]

    # 3. Check rank multiplicity
    quad_rank = -1
    trip_ranks = []
    pair_ranks = []
    single_ranks = []

    for r in range(12, -1, -1):
        cnt = rank_counts[r]
        if cnt == 4:
            quad_rank = r + 2
        elif cnt == 3:
            trip_ranks.append(r + 2)
        elif cnt == 2:
            pair_ranks.append(r + 2)
        elif cnt == 1:
            single_ranks.append(r + 2)

    # Quads
    if quad_rank != -1:
        kicker = 0
        for r in range(12, -1, -1):
            if (r + 2) != quad_rank and rank_counts[r] > 0:
                kicker = r + 2
                break
        return (7 << 24) | (quad_rank << 8) | kicker

    # Full House: trips + trips or trips + pair
    if trip_ranks:
        top_trip = trip_ranks[0]
        if len(trip_ranks) > 1:
            return (6 << 24) | (top_trip << 8) | trip_ranks[1]
        if pair_ranks:
            return (6 << 24) | (top_trip << 8) | pair_ranks[0]

    # Straight
    if straight_high:
        return (4 << 24) | straight_high

    # Three of a Kind
    if trip_ranks:
        top_trip = trip_ranks[0]
        kickers = single_ranks + pair_ranks
        kickers.sort(reverse=True)
        return (3 << 24) | (top_trip << 12) | (kickers[0] << 6) | kickers[1]

    # Two Pair
    if len(pair_ranks) >= 2:
        top_pair = pair_ranks[0]
        second_pair = pair_ranks[1]
        kickers = single_ranks + pair_ranks[2:]
        kickers.sort(reverse=True)
        return (2 << 24) | (top_pair << 12) | (second_pair << 6) | kickers[0]

    # One Pair
    if pair_ranks:
        pair = pair_ranks[0]
        return (1 << 24) | (pair << 16) | (single_ranks[0] << 12) | (single_ranks[1] << 8) | (single_ranks[2] << 4)

    # High Card
    return (0 << 24) | (single_ranks[0] << 16) | (single_ranks[1] << 12) | (single_ranks[2] << 8) | (single_ranks[3] << 4) | single_ranks[4]


def fast_best_hand(cards_str: Iterable[str]) -> int:
    """Evaluate 7 card strings ('As', 'Kd', etc.) into fast 32-bit score."""
    ints = [_CARD_TO_INT[c] for c in cards_str]
    return evaluate_7cards_int(ints)
