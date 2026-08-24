"""4max NLHE AoF Cash Game model.

Supports arbitrary / asymmetric player stacks, customizable 5% rake with cap,
side-pot settlements, and tree action definitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Iterable

from ..cards import RANK_VALUE, SUITS
from ..ranges.hand_grid import all_hand_classes, combos_of_class

FOLD: Final = "F"
JAM: Final = "J"
ACTIONS: Final = (FOLD, JAM)
POSITIONS: Final = ("CO", "BTN", "SB", "BB")
_ORDER: Final = "AKQJT98765432"
_SUIT_ORDER: Final = {suit: i for i, suit in enumerate(SUITS)}


@dataclass(frozen=True)
class AofCashConfig:
    """Konfigurace 4max AoF cash game stolu.

    Everything is in big blinds.
    - `stacks_bb`: four stacks, for (CO, BTN, SB, BB).
    - `rake_pct`: rake as a fraction, e.g. 0.05 for 5%.
    - `rake_cap_bb`: the cap in bb, or None for no cap.
    - `rake_on_uncontested`: whether a walk or uncontested pot is raked too.
    """

    positions: tuple[str, str, str, str] = POSITIONS
    stacks_bb: tuple[float, float, float, float] = (8.0, 8.0, 8.0, 8.0)
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    ante_bb: float = 0.0
    rake_pct: float = 0.05
    rake_cap_bb: float | None = None
    rake_on_uncontested: bool = False

    def __post_init__(self) -> None:
        if len(self.positions) != 4 or len(set(self.positions)) != 4:
            raise ValueError("the model needs exactly four distinct positions")
        if len(self.stacks_bb) != 4 or any(s <= 0 for s in self.stacks_bb):
            raise ValueError("stacks_bb needs four positive numbers")
        if not (0 <= self.small_blind_bb <= self.big_blind_bb):
            raise ValueError("the blinds must satisfy 0 <= SB <= BB")
        if self.rake_pct < 0.0 or self.rake_pct > 1.0:
            raise ValueError("rake_pct is a fraction between 0.0 and 1.0, e.g. 0.05")

    def blinds(self) -> tuple[float, float, float, float]:
        """Forced contribution (ante + blind) pro CO, BTN, SB, BB."""
        return (
            self.ante_bb,
            self.ante_bb,
            self.small_blind_bb + self.ante_bb,
            self.big_blind_bb + self.ante_bb,
        )

    def effective_stack_bb(self, pos1_idx: int, pos2_idx: int) -> float:
        """The effective stack between two players."""
        return min(self.stacks_bb[pos1_idx], self.stacks_bb[pos2_idx])


@dataclass(frozen=True)
class AofCashInfoSet:
    """What a player knows at the moment they decide."""

    player: int
    history: tuple[str, ...]
    combo: tuple[str, str]


@dataclass(frozen=True)
class MatrixCell:
    """Aggregated strategy and EV for one of the 169 hand classes."""

    fold: float | None
    jam: float | None
    ev_fold: float | None
    ev_jam: float | None
    samples: float

    @property
    def ev_delta(self) -> float | None:
        if self.ev_fold is None or self.ev_jam is None:
            return None
        return self.ev_jam - self.ev_fold

    def as_dict(self) -> dict[str, float | None]:
        return {
            "fold": self.fold,
            "jam": self.jam,
            "ev_fold": self.ev_fold,
            "ev_jam": self.ev_jam,
            "ev_delta": self.ev_delta,
            "samples": self.samples,
        }


def canonical_combo(cards: Iterable[str]) -> tuple[str, str]:
    """Cards in canonical order, so the info-set key is stable."""
    combo = tuple(cards)
    if len(combo) != 2 or combo[0] == combo[1]:
        raise ValueError("a combo needs two different cards")
    ordered = sorted(
        combo,
        key=lambda card: (RANK_VALUE[card[0]], -_SUIT_ORDER[card[1]]),
        reverse=True,
    )
    return ordered[0], ordered[1]


def parse_history(raw: str | Iterable[str]) -> tuple[str, ...]:
    """Parse 'FJ' / 'F,J' na tuple ('F', 'J')."""
    if isinstance(raw, str):
        tokens = tuple(char for char in raw.upper() if char in ACTIONS)
        invalid = [char for char in raw.upper() if char not in {*ACTIONS, ",", " ", "-", "/"}]
        if invalid:
            raise ValueError(f"not a valid history: {raw!r}")
    else:
        tokens = tuple(str(token).upper() for token in raw)
    if len(tokens) > 3 or any(token not in ACTIONS for token in tokens):
        raise ValueError("a history holds at most three F/J actions")
    return tokens


def calculate_rake(config: AofCashConfig, actions: tuple[str, ...], pot: float) -> float:
    """Rake from the pot, honouring the uncontested-pot rule."""
    contested = sum(action == JAM for action in actions) >= 2
    if not contested and not config.rake_on_uncontested:
        return 0.0
    rake = pot * config.rake_pct
    if config.rake_cap_bb is not None:
        rake = min(rake, config.rake_cap_bb)
    return min(rake, pot)


def settle_cash_actions(
    config: AofCashConfig,
    actions: tuple[str, str, str, str],
    scores: tuple[int, int, int, int] | tuple[tuple, tuple, tuple, tuple] | None = None,
) -> tuple[float, float, float, float]:
    """Payoffs for all four players, with uneven stacks and side pots.

    Folding risks only the forced blind; jamming risks all of
    `stacks_bb[player]`. Uneven stacks produce a main pot and side pots split
    by what each player put in.
    """
    if len(actions) != 4 or any(action not in ACTIONS for action in actions):
        raise ValueError("settle_cash_actions needs four F/J actions")

    blinds = config.blinds()
    committed = [
        config.stacks_bb[i] if action == JAM else blinds[i]
        for i, action in enumerate(actions)
    ]
    jammed = [i for i, action in enumerate(actions) if action == JAM]

    # Case 1: everyone folds — the BB takes the blinds, a walk
    if not jammed:
        pot = sum(committed)
        rake = calculate_rake(config, actions, pot)
        utilities = [-amount for amount in committed]
        utilities[3] += (pot - rake)
        return (utilities[0], utilities[1], utilities[2], utilities[3])

    # Case 2: exactly one jam — the jammer collects the blinds, no showdown
    if len(jammed) == 1:
        winner = jammed[0]
        pot = sum(committed)
        rake = calculate_rake(config, actions, pot)
        utilities = [-amount for amount in committed]
        utilities[winner] += (pot - rake)
        return (utilities[0], utilities[1], utilities[2], utilities[3])

    # Case 3: two or more jams — a showdown, possibly with side pots
    if scores is None:
        raise ValueError("a multiway showdown needs hand scores")

    # Build the side pots. Each player contributed `committed[i]`, and only the
    # players in `jammed` can win any of it.
    utilities = [-c for c in committed]

    # The distinct contribution levels among the jammers
    # a rozpadneme pot na vrstvy (pot slices)
    active_contributors = list(range(4))
    remaining_committed = list(committed)
    total_net_pot = 0.0

    # The gross pot and the total rake
    total_gross_pot = sum(committed)
    total_rake = calculate_rake(config, actions, total_gross_pot)
    rake_ratio = (total_gross_pot - total_rake) / total_gross_pot if total_gross_pot > 0 else 1.0

    # Split the pot into layers by contribution
    sorted_jam_levels = sorted({remaining_committed[j] for j in jammed if remaining_committed[j] > 0})

    prev_level = 0.0
    for level in sorted_jam_levels:
        slice_amount = level - prev_level
        if slice_amount <= 1e-9:
            continue

        # Who paid into this layer?
        slice_pot = 0.0
        for p in range(4):
            contrib = min(max(0.0, remaining_committed[p]), slice_amount)
            slice_pot += contrib
            remaining_committed[p] -= contrib

        # Which jammers are contesting it?
        contenders = [j for j in jammed if committed[j] >= level]
        if not contenders:
            # Nobody contests it — leftover money from folded players — so it goes back
            continue

        best_score = max(scores[c] for c in contenders)
        winners = [c for c in contenders if scores[c] == best_score]

        net_slice_pot = slice_pot * rake_ratio
        share = net_slice_pot / len(winners)
        for w in winners:
            utilities[w] += share

        prev_level = level

    # Return any uncalled portion
    for p in range(4):
        if remaining_committed[p] > 0:
            utilities[p] += remaining_committed[p]

    return (utilities[0], utilities[1], utilities[2], utilities[3])


def render_matrix_lines(matrix: dict[str, MatrixCell], *, metric: str = "jam") -> list[str]:
    """The 13x13 matrix as text: frequencies or EV deltas."""
    if metric not in {"jam", "delta"}:
        raise ValueError("metric must be 'jam' or 'delta'")
    header = "       " + " ".join(f"{rank:>6}" for rank in _ORDER)
    lines = [header]
    for row, rank in enumerate(_ORDER):
        cells = []
        for column in range(13):
            high, low = _ORDER[row], _ORDER[column]
            if row == column:
                hand = high + high
            elif row < column:
                hand = high + low + "s"
            else:
                hand = low + high + "o"
            cell = matrix[hand]
            value = cell.jam if metric == "jam" else cell.ev_delta
            if value is None:
                text = "   .  "
            elif metric == "jam":
                text = f"{value:5.0%} "
            else:
                text = f"{value:+5.2f} "
            cells.append(text)
        lines.append(f"{rank:>3}  " + "".join(cells))
    return lines
