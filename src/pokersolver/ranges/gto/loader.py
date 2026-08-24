"""Reading precomputed preflop matrices (NL100, 100bb, 2.5x open).

Loads the consolidated `data/nl100_100bb.json` artefact built by `build.py` and
puts a dependable query API on top of it. Three things the raw data does not
handle on its own:

1. THE HERO comes from `is_hero`, not from `meta.position`. In 16 of 197 spots
   `meta.position` is wrong — the label belongs to a different player than the
   range does — so the build stores the real hero as `hero`.
2. NORMALISATION per hand. Frequencies in the file are range-weighted: a hand's
   actions sum to the share of combos that reach the node, which is below 1 in
   deeper spots. `decision()` rescales them to the strategy you actually want:
   "what do I do holding this hand here".
3. FALLBACK. A missing spot (the dataset covers 197 of 276) returns None rather
   than raising, so `get_or_none` plus `warnings()` let a caller degrade safely.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Where the matrices live. By default next to the code, so they ship with the
# package; `PF_GTO_DATA_DIR` overrides that, which lets you swap the dataset
# without reinstalling and run with matrices that are not part of any release.
# Individual missing files are skipped quietly, but a directory with no
# matrices at all raises on the first `default_multi_solutions()` — staying
# silent would mean playing without matrices and never finding out.
DATA_DIR = Path(os.environ.get("PF_GTO_DATA_DIR")
                or Path(__file__).resolve().parent / "data")
# The active dataset: 6-max NL100, 2.5x open, 100bb — 243 nodes, written by
# build.py. A rebuild lands straight here, so there is no merge step to forget.
DEFAULT_DATASET = DATA_DIR / "nl100_100bb.json"

# Below this share of combos a spot counts as "narrow" — a deep node. Advisory.
LOW_COVERAGE = 0.25


@dataclass(frozen=True)
class Action:
    """One action available at a node."""
    key: str          # 'fold' | 'call_15bb' | 'raise_34bb'
    type: str         # 'fold' | 'call' | 'raise'
    size: float | None
    label: str        # readable: 'Call 15bb', '4-bet to 34bb', 'All-in (100bb)'


@dataclass(frozen=True)
class GtoSpot:
    """A single preflop spot: hero, available actions, and strategy per hand."""
    spot_id: str
    hero: str                          # correct hero (from is_hero)
    position_meta: str                 # the original meta.position; may differ
    label: str                         # corrected readable description
    line: str                          # decoded action line
    sequence: str
    actions: tuple[Action, ...]        # in the order the data listed them
    _strategy: dict[str, dict[str, float]]   # hand -> {action_key: raw freq}
    _ev: dict[str, float]              # hand -> EV in bb
    range_fraction: float              # share of combos in the range (0..1)
    label_mismatch: bool               # meta.position != hero

    # ── queries ──
    def in_range(self, hand: str) -> bool:
        """Is this hand in the node's range at all — does it arrive with any weight?"""
        freqs = self._strategy.get(hand)
        return bool(freqs) and sum(freqs.values()) > 1e-6

    def decision(self, hand: str) -> dict[str, float] | None:
        """The NORMALISED strategy for a hand: {action label: freq}, summing to 1.

        Returns None when the hand is not in the range and never reaches this
        node — so a caller can tell "fold 100%" apart from "I never hold this
        hand here".
        """
        freqs = self._strategy.get(hand)
        if not freqs:
            return None
        total = sum(freqs.values())
        if total <= 1e-9:
            return None
        label = {a.key: a.label for a in self.actions}
        return {label.get(k, k): v / total for k, v in freqs.items() if v > 1e-9}

    def top_action(self, hand: str) -> tuple[str, float] | None:
        """The most frequent action as (label, normalised freq), or None."""
        dec = self.decision(hand)
        if not dec:
            return None
        label, freq = max(dec.items(), key=lambda kv: kv[1])
        return label, freq

    def ev(self, hand: str) -> float | None:
        return self._ev.get(hand)

    def warnings(self) -> list[str]:
        """Things worth knowing before trusting this spot."""
        w: list[str] = []
        if self.label_mismatch:
            w.append(f"the label belongs to {self.position_meta}, but the hero is "
                     f"{self.hero} (bereme {self.hero})")
        if self.range_fraction < LOW_COVERAGE:
            w.append(f"narrow node — only {self.range_fraction:.0%} of combos reach it")
        return w


class GtoSolutions:
    """A loaded set of spots, with lookup and fallback."""

    def __init__(self, meta: dict[str, Any], spots: dict[str, GtoSpot]):
        self.meta = meta
        self._spots = spots

    # -- loading --
    @classmethod
    def load(cls, path: Path | str = DEFAULT_DATASET) -> "GtoSolutions":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        spots: dict[str, GtoSpot] = {}
        for spot_id, s in raw["spots"].items():
            actions = tuple(
                Action(key=a["key"], type=a["type"], size=a.get("size"),
                       label=a["label"])
                for a in s["actions"]
            )
            spots[spot_id] = GtoSpot(
                spot_id=spot_id,
                hero=s["hero"],
                position_meta=s["position_meta"],
                label=s["label"],
                line=s["line"],
                sequence=s["sequence"],
                actions=actions,
                _strategy=s["strategy"],
                _ev=s.get("ev", {}),
                range_fraction=s["range_fraction"],
                label_mismatch=s["label_mismatch"],
            )
        return cls(raw["meta"], spots)

    # ── lookup ──
    def get(self, spot_id: str) -> GtoSpot:
        """A spot by id; raises KeyError when missing (the dataset covers 197 of 276)."""
        return self._spots[spot_id]

    def get_or_none(self, spot_id: str) -> GtoSpot | None:
        """Lookup that answers None instead of raising, for callers that degrade."""
        return self._spots.get(spot_id)

    def resolve(self, spot_id: str, fallbacks: list[str] | None = None
                ) -> GtoSpot | None:
        """Try spot_id, then each of `fallbacks`; the first that exists, or None."""
        for candidate in (spot_id, *(fallbacks or [])):
            spot = self._spots.get(candidate)
            if spot is not None:
                return spot
        return None

    def ids(self) -> list[str]:
        return sorted(self._spots)

    def by_hero(self, hero: str) -> list[GtoSpot]:
        return [s for s in self._spots.values() if s.hero == hero]

    def __len__(self) -> int:
        return len(self._spots)

    def __contains__(self, spot_id: str) -> bool:
        return spot_id in self._spots


@lru_cache(maxsize=1)
def default_solutions() -> GtoSolutions:
    """The shared default (100bb) dataset, loaded once."""
    return GtoSolutions.load()


# Available depths (effective stack in bb) -> data file. 100bb is the baseline;
# 50bb short and 200bb deep are separate solves with their own sizings.
DEPTH_DATASETS: dict[int, Path] = {
    50: DATA_DIR / "nl100_50bb.json",
    100: DATA_DIR / "nl100_100bb.json",
    200: DATA_DIR / "nl100_200bb.json",
}


class MultiDepthSolutions:
    """Matrices at several stack depths, picked by the effective stack.

    Proxies the :class:`GtoSolutions` interface to the 100bb baseline, so code
    that treats it as a single dataset keeps working. Callers that care call
    :meth:`for_stack` and get the nearest depth.
    """

    def __init__(self, by_depth: dict[int, GtoSolutions], base_depth: int = 100,
                 open_size: float = 2.5):
        self._by_depth = by_depth
        self._base = by_depth[base_depth]
        self._base_depth = base_depth
        # The family's nominal open size in bb — 2.5 for the baseline, 2.0 for
        # the min-raise family. It is how a caller tells whether the open it is
        # facing matches this matrix at all (see facing_open_sizing_mismatch).
        self.open_size = open_size
        self.meta = dict(self._base.meta, depths=sorted(by_depth),
                         open_size=open_size)

    @classmethod
    def load(cls, datasets: dict[int, Path] | None = None,
             open_size: float = 2.5) -> "MultiDepthSolutions":
        datasets = datasets or DEPTH_DATASETS
        return cls({d: GtoSolutions.load(p) for d, p in datasets.items()
                    if Path(p).exists()}, open_size=open_size)

    def depth_for(self, bb: float | None) -> int:
        """The depth key :meth:`for_stack` would choose.

        Separate from `for_stack` so an audit or a dashboard can report WHICH
        matrix a node came from without duplicating the selection rule.
        """
        if bb is None or bb <= 0:
            return self._base_depth
        # nearest depth; on a tie prefer the deeper, more conservative one
        return min(self._by_depth, key=lambda d: (abs(d - bb), -d))

    def for_stack(self, bb: float | None) -> GtoSolutions:
        """The dataset at the nearest depth; unknown or None gives the baseline."""
        return self._by_depth[self.depth_for(bb)]

    # -- proxy to the baseline, so this works anywhere GtoSolutions does --
    def get(self, spot_id: str) -> GtoSpot: return self._base.get(spot_id)
    def get_or_none(self, spot_id: str) -> GtoSpot | None:
        return self._base.get_or_none(spot_id)
    def resolve(self, spot_id: str, fallbacks: list[str] | None = None) -> GtoSpot | None:
        return self._base.resolve(spot_id, fallbacks)
    def ids(self) -> list[str]: return self._base.ids()
    def __len__(self) -> int: return len(self._base)
    def __contains__(self, spot_id: str) -> bool: return spot_id in self._base


# Matrix families by open size. The key is the nominal open in bb, and each
# family carries its own 50/100/200bb set. The 2.5 baseline is complete; the
# 2.0 (min-raise) family only covers blind defence spots — see data/README.md.
# Missing files are skipped, so an incomplete family behaves exactly as if only
# the baseline existed.
BASE_OPEN_SIZE = 2.5
OPEN_SIZE_FAMILIES: dict[float, dict[int, Path]] = {
    2.5: DEPTH_DATASETS,
    2.0: {
        50: DATA_DIR / "nl100_2bbopen_50bb.json",
        100: DATA_DIR / "nl100_2bbopen_100bb.json",
        200: DATA_DIR / "nl100_2bbopen_200bb.json",
    },
}
# How far an actual open may sit from the family's nominal size and still count
# as the same family. Half a blind, matching the reraise guard.
OPEN_SIZE_TOLERANCE_BB = 0.25


class MultiSizeSolutions:
    """Matrices across several open-size families (2.0 min-raise, 2.5 baseline).

    Selects the family based on the actual size of the live open (:meth:`for_open`),
    then the depth within it. Proxies the baseline family, so code that treats
    it as a single dataset keeps working.
    """

    def __init__(self, by_open: dict[float, MultiDepthSolutions],
                 base_open: float = BASE_OPEN_SIZE):
        if base_open not in by_open:          # the baseline must always exist
            base_open = min(by_open, key=lambda o: abs(o - BASE_OPEN_SIZE))
        self._by_open = by_open
        self._base = by_open[base_open]
        self.meta = dict(self._base.meta, open_sizes=sorted(by_open))

    @classmethod
    def load(cls, families: dict[float, dict[int, Path]] | None = None
             ) -> "MultiSizeSolutions":
        families = families or OPEN_SIZE_FAMILIES
        by_open: dict[float, MultiDepthSolutions] = {}
        for open_size, datasets in families.items():
            if any(Path(p).exists() for p in datasets.values()):
                by_open[open_size] = MultiDepthSolutions.load(
                    datasets, open_size=open_size)
        return cls(by_open)

    def for_open(self, open_bb: float | None) -> MultiDepthSolutions:
        """The family nearest the open being faced; unknown or None gives baseline.

        It answers with the baseline even when nothing is within tolerance, so
        the spot is still found — the caller then uses
        :func:`facing_open_sizing_mismatch` to see the price does not match and
        can decide for itself.
        """
        if open_bb is None or open_bb <= 0 or len(self._by_open) == 1:
            return self._base
        return self._by_open[min(self._by_open, key=lambda o: abs(o - open_bb))]

    def open_sizes(self) -> list[float]:
        return sorted(self._by_open)

    # -- proxy to the baseline family, so this works anywhere GtoSolutions does --
    def get(self, spot_id: str) -> GtoSpot: return self._base.get(spot_id)
    def get_or_none(self, spot_id: str) -> GtoSpot | None:
        return self._base.get_or_none(spot_id)
    def resolve(self, spot_id: str, fallbacks: list[str] | None = None) -> GtoSpot | None:
        return self._base.resolve(spot_id, fallbacks)
    def ids(self) -> list[str]: return self._base.ids()
    def __len__(self) -> int: return len(self._base)
    def __contains__(self, spot_id: str) -> bool: return spot_id in self._base


@lru_cache(maxsize=1)
def default_multi_solutions() -> MultiSizeSolutions:
    """The shared instance: open-size families by stack depth.

    With only the 2.5 family present it behaves as a plain multi-depth set."""
    return MultiSizeSolutions.load()
