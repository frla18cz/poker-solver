"""A range of hands, and a parser for the usual notation.

Notation, separated by commas or spaces:
  pairs:     "AA", "22+", "77-TT"
  suited:    "AKs", "ATs+", "A2s-A5s"
  offsuit:   "AJo+", "KQo"
  both:      "AK", "AJ+", "AJ-A9"  = suited and offsuit, the common shorthand
Kombinace: "22+, ATs+, KQs, AJo+"

Notation without ``s``/``o`` (``QJ``, ``AJ+``) used to be read silently as a
PAIR of the first card — ``QJ`` became QQ, six combos — because
``combos_of_class`` treats any two characters as a pair. People write the
shorthand constantly, so equity against such a range came out tens of percent
wrong and nothing said so. It now means suited + offsuit, the way any player
reads it.
"""
from __future__ import annotations

from .hand_grid import combos_of_class

_ORDER = "AKQJT98765432"          # highest first
_IDX = {r: i for i, r in enumerate(_ORDER)}   # 0=A .. 12=2


def _pairs_from(low: str, high: str = "A") -> list[str]:
    """Pairs from 'low' through 'high', inclusive. Both are ranks."""
    lo, hi = _IDX[low], _IDX[high]
    return [r + r for r in _ORDER if hi <= _IDX[r] <= lo]


def _both_suits(core: str) -> str:
    """``JQ`` becomes ``QJ`` — higher rank first. The caller adds ``s``/``o``."""
    a, b = core[0], core[1]
    return (a + b) if _IDX[a] <= _IDX[b] else (b + a)


def _expand_token(tok: str) -> set[str]:
    tok = tok.strip()
    if not tok:
        return set()
    # hyphenated range: "A2s-A5s", "77-TT"
    if "-" in tok:
        a, b = tok.split("-", 1)
        return _expand_range(a.strip(), b.strip())
    plus = tok.endswith("+")
    core = tok[:-1] if plus else tok

    if len(core) == 2 and core[0] == core[1]:            # pair
        if plus:
            return set(_pairs_from(core[0]))              # "22+" -> 22..AA
        return {core}
    if len(core) == 2 and core[0] in _IDX and core[1] in _IDX:   # "QJ", "AJ+"
        base = _both_suits(core)
        return _expand_token(base + "s" + ("+" if plus else "")) \
            | _expand_token(base + "o" + ("+" if plus else ""))
    if len(core) == 3 and core[2] in "so":               # suited/offsuit
        hi, lo, typ = core[0], core[1], core[2]
        if plus:
            # hi stays fixed; lo runs from here up to hi-1
            out = set()
            for r in _ORDER:
                if _IDX[hi] < _IDX[r] <= _IDX[lo]:
                    out.add(hi + r + typ)
            out.add(core)
            return out
        return {core}
    return {core}  # leave anything unrecognised alone


def _expand_range(a: str, b: str) -> set[str]:
    # pairs, "77-TT"
    if len(a) == 2 and a[0] == a[1]:
        lo_rank = a[0] if _IDX[a[0]] > _IDX[b[0]] else b[0]
        hi_rank = b[0] if _IDX[b[0]] < _IDX[a[0]] else a[0]
        return set(_pairs_from(lo_rank, hi_rank))
    # suited/offsuit "A2s-A5s": same hi and type, lo in between
    if len(a) == 3 and len(b) == 3 and a[0] == b[0] and a[2] == b[2]:
        hi, typ = a[0], a[2]
        l1, l2 = _IDX[a[1]], _IDX[b[1]]
        lo_i, hi_i = min(l1, l2), max(l1, l2)
        return {hi + _ORDER[i] + typ for i in range(lo_i, hi_i + 1)}
    # both, "AJ-A9": the range in suited and offsuit
    if (len(a) == 2 and len(b) == 2 and a[0] != a[1] and b[0] != b[1]
            and all(c in _IDX for c in a + b) and a[0] == b[0]):
        return _expand_range(a + "s", b + "s") | _expand_range(a + "o", b + "o")
    return {a, b}


class Range:
    def __init__(self, classes: set[str] | None = None):
        self.classes: set[str] = classes or set()

    @classmethod
    def parse(cls, text: str) -> "Range":
        out: set[str] = set()
        for tok in text.replace(",", " ").split():
            out |= _expand_token(tok)
        return cls(out)

    def contains(self, hand_class: str) -> bool:
        return hand_class in self.classes

    def combos(self, dead_cards: set[str] | None = None) -> list[tuple[str, str]]:
        """The actual card combos in the range, minus anything dead_cards block."""
        dead = dead_cards or set()
        res = []
        # A set iterates in a different order in each process (hash randomisation),
        # which changed Monte Carlo results even with a fixed RNG seed. Sorting
        # makes simulations and tests genuinely reproducible.
        for hc in sorted(self.classes, reverse=True):
            for a, b in combos_of_class(hc):
                if a not in dead and b not in dead:
                    res.append((a, b))
        return res

    def __len__(self) -> int:
        return len(self.classes)

    def __or__(self, other: "Range") -> "Range":
        return Range(self.classes | other.classes)
