"""Seven-card evaluation over arrays — thousands of hands in one numpy pass.

It exists because building the deal pool was the most expensive part of a live
budget: 60,000 deals took 0.97s in pure Python, leaving two iterations out of a
one-second budget.

It returns **a single int64 score**, not the tuple :func:`score_7` gives. The
absolute value means nothing; scores are only comparable with each other — and
they order identically to ``score_7``, which a test over random hands checks.
The category and five kickers are packed four bits each; ranks run 2..14, so
each fits a nibble.

Nearly everything goes through tables indexed by a 13-bit rank mask. The first
attempt computed kickers with ``argsort`` over an (N, 13) array and was no
faster than pure Python; precomputed tables (8192 entries, built at import) are
what made the order-of-magnitude difference.

It lives here rather than in ``cards.py`` on purpose: the core stays
stdlib-only, and this needs numpy.
"""
from __future__ import annotations

import numpy as np

from pokersolver.cards import (
    FLUSH,
    FULL_DECK,
    FULL_HOUSE,
    PAIR,
    QUADS,
    RANK_VALUE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    _STRAIGHT_MASKS,
    _SUIT_INDEX,
    _WHEEL_MASK,
)

_RANKS = range(2, 15)
_SIZE = 1 << 15  # the mask uses bits 2..14

CARD_RANK = np.asarray([RANK_VALUE[c[0]] for c in FULL_DECK], dtype=np.int64)
CARD_SUIT = np.asarray([_SUIT_INDEX[c[1]] for c in FULL_DECK], dtype=np.int64)
CARD_INDEX = {card: i for i, card in enumerate(FULL_DECK)}


def _build_tables() -> tuple[np.ndarray, list[np.ndarray]]:
    """Per rank mask: the highest straight and the top-k kickers, packed."""
    straight = np.zeros(_SIZE, dtype=np.int64)
    tops = [np.zeros(_SIZE, dtype=np.int64) for _k in range(6)]
    for mask in range(_SIZE):
        for pattern, high in _STRAIGHT_MASKS:
            if mask & pattern == pattern:
                straight[mask] = high
                break
        else:
            if mask & _WHEEL_MASK == _WHEEL_MASK:
                straight[mask] = 5
        present = [value for value in range(14, 1, -1) if mask >> value & 1]
        for k in range(1, 6):
            packed = 0
            for slot, value in enumerate(present[:k]):
                packed |= value << (16 - 4 * slot)
            tops[k][mask] = packed
    return straight, tops


_STRAIGHT_HIGH, _TOP = _build_tables()


def _cat(category: int) -> int:
    return category << 20


def score_batch(cards: "np.ndarray") -> "np.ndarray":
    """Score ``(N, 7)`` indexes into ``FULL_DECK``, returning ``(N,)`` scores."""
    if cards.ndim != 2 or cards.shape[1] != 7:
        raise ValueError(f"expected shape (N, 7), got {cards.shape}")
    ranks = CARD_RANK[cards]
    suits = CARD_SUIT[cards]
    one = np.int64(1)

    # Rank counts as nibbles in one int64: 13 ranks x 4 bits fits.
    tally = (one << (4 * ranks)).sum(axis=1)
    mask = np.zeros(cards.shape[0], dtype=np.int64)
    mask2 = np.zeros_like(mask)
    mask3 = np.zeros_like(mask)
    mask4 = np.zeros_like(mask)
    for value in _RANKS:
        count = (tally >> (4 * value)) & 0xF
        bit = one << value
        mask |= np.where(count >= 1, bit, 0)
        mask2 |= np.where(count >= 2, bit, 0)
        mask3 |= np.where(count >= 3, bit, 0)
        mask4 |= np.where(count >= 4, bit, 0)

    # --- barva ---
    suit_tally = (one << (4 * suits)).sum(axis=1)
    flush_mask = np.zeros_like(mask)
    has_flush = np.zeros(cards.shape[0], dtype=bool)
    for suit in range(4):
        hit = ((suit_tally >> (4 * suit)) & 0xF) >= 5
        if not hit.any():
            continue
        of_suit = np.bitwise_or.reduce(
            np.where(suits == suit, one << ranks, 0), axis=1)
        flush_mask = np.where(hit, of_suit, flush_mask)
        has_flush |= hit

    # --- categories, weakest first; a stronger one overwrites ---
    score = _cat(0) | _TOP[5][mask]

    top_pair = _TOP[1][mask2] >> 16
    has_pair = mask2 != 0
    if has_pair.any():
        rest = mask & ~(one << top_pair)
        score = np.where(has_pair,
                         _cat(PAIR) | top_pair << 16 | _TOP[3][rest] >> 4, score)

    two = _TOP[2][mask2]
    second_pair = (two >> 12) & 0xF
    has_two = second_pair > 0
    if has_two.any():
        rest = mask & ~(one << top_pair) & ~(one << second_pair)
        score = np.where(has_two,
                         _cat(TWO_PAIR) | two | _TOP[1][rest] >> 8, score)

    trip = _TOP[1][mask3] >> 16
    has_trip = mask3 != 0
    if has_trip.any():
        rest = mask & ~(one << trip)
        score = np.where(has_trip,
                         _cat(TRIPS) | trip << 16 | _TOP[2][rest] >> 4, score)

    straight = _STRAIGHT_HIGH[mask]
    score = np.where(straight > 0, _cat(STRAIGHT) | straight << 16, score)
    score = np.where(has_flush, _cat(FLUSH) | _TOP[5][flush_mask], score)

    # Full house: the trips plus the best pair. A second set of trips counts as
    # that pair, which is why the search uses ``mask2`` — anything at least a
    # pair — with the top trips removed.
    boat_pair = _TOP[1][mask2 & ~(one << trip)] >> 16
    boat = has_trip & (boat_pair > 0)
    score = np.where(boat, _cat(FULL_HOUSE) | trip << 16 | boat_pair << 12, score)

    quad = _TOP[1][mask4] >> 16
    has_quad = mask4 != 0
    if has_quad.any():
        rest = mask & ~(one << quad)
        score = np.where(has_quad,
                         _cat(QUADS) | quad << 16 | _TOP[1][rest] >> 4, score)

    sf = np.where(has_flush, _STRAIGHT_HIGH[flush_mask], 0)
    return np.where(sf > 0, _cat(STRAIGHT_FLUSH) | sf << 16, score)
