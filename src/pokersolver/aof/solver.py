"""4max NLHE All-In or Fold: payoff model, CFR self-play a export matic.

All-in-or-fold NLHE gives each player a single preflop choice. This models the
usual order CO -> BTN -> SB -> BB, where everyone picks ``fold`` or ``jam``;
after the jams a board runs out and the showdown is settled multiway.

It is an *offline study tool*. Multiplayer poker with rake is not zero-sum, so
what comes out is a strategy from regret-minimising self-play — not a certified
equilibrium and not an exploitability figure. A room's exact rules, especially
whether a walk is charged a fee, are parameters of :class:`AofConfig`.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ..cards import FULL_DECK, RANK_VALUE, SUITS, best_hand
from ..ranges.hand_grid import all_hand_classes, combos_of_class


FOLD: Final = "F"
JAM: Final = "J"
ACTIONS: Final = (FOLD, JAM)
POSITIONS: Final = ("CO", "BTN", "SB", "BB")
_ORDER: Final = "AKQJT98765432"
_SUIT_ORDER: Final = {suit: i for i, suit in enumerate(SUITS)}


@dataclass(frozen=True)
class AofConfig:
    """The rules of one 4-max all-in-or-fold table, all in big blinds.

    ``stack_bb`` is the full stack including any blind already posted. The flat
    rake comes out of the pot exactly once per hand. The defaults match a
    common 8bb game; check any specific stake against its hand history.
    """

    positions: tuple[str, str, str, str] = POSITIONS
    stack_bb: float = 8.0
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    flat_rake_bb: float = 0.12
    rake_on_uncontested: bool = True

    def __post_init__(self) -> None:
        if len(self.positions) != 4 or len(set(self.positions)) != 4:
            raise ValueError("the model needs exactly four distinct positions")
        if self.stack_bb <= 0:
            raise ValueError("stack_bb must be positive")
        if not 0 <= self.small_blind_bb <= self.big_blind_bb:
            raise ValueError("the blinds must satisfy 0 <= SB <= BB")
        if self.stack_bb < self.big_blind_bb:
            raise ValueError("the stack must cover the big blind")
        if self.flat_rake_bb < 0:
            raise ValueError("flat_rake_bb cannot be negative")

    def blinds(self) -> tuple[float, float, float, float]:
        """Forced contribution pro CO, BTN, SB, BB."""
        return (0.0, 0.0, self.small_blind_bb, self.big_blind_bb)


@dataclass(frozen=True)
class AofInfoSet:
    """What a player knows when making their single decision."""

    player: int
    history: tuple[str, ...]
    combo: tuple[str, str]


@dataclass
class _RegretNode:
    """Regret-matching stav jednoho private-card information setu."""

    regrets: list[float]
    strategy_sum: list[float]
    action_value_sum: list[float]
    action_value_weight: float = 0.0

    @classmethod
    def new(cls) -> "_RegretNode":
        return cls([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    def strategy(self) -> tuple[float, float]:
        """The regret-matching policy, in the order fold, jam."""
        positive = [max(0.0, value) for value in self.regrets]
        total = sum(positive)
        if total <= 1e-15:
            return (0.5, 0.5)
        return (positive[0] / total, positive[1] / total)

    def average_strategy(self) -> tuple[float, float]:
        total = sum(self.strategy_sum)
        if total <= 1e-15:
            return self.strategy()
        return (self.strategy_sum[0] / total, self.strategy_sum[1] / total)

    def action_values(self) -> tuple[float | None, float | None]:
        """A counterfactual estimate of fold/jam EV during training.

        Earlier opponent actions are weighted by their reach probability. Early
        in a solve this is rough; once converged it is a useful diagnostic to
        export alongside the matrix.
        """
        if self.action_value_weight <= 1e-15:
            return (None, None)
        return (
            self.action_value_sum[0] / self.action_value_weight,
            self.action_value_sum[1] / self.action_value_weight,
        )


@dataclass(frozen=True)
class MatrixCell:
    """The aggregate for one cell of the 13x13 grid."""

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
    """Two cards in canonical order, for a stable information-set key."""
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
    """Parse ``FJ`` or ``F,J`` into a history of 0 to 3 actions."""
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


def _rake(config: AofConfig, actions: tuple[str, ...], pot: float) -> float:
    contested = sum(action == JAM for action in actions) >= 2
    if contested or config.rake_on_uncontested:
        return min(config.flat_rake_bb, pot)
    return 0.0


def settle_actions(
    config: AofConfig,
    actions: tuple[str, str, str, str],
    scores: tuple[tuple, tuple, tuple, tuple] | None = None,
) -> tuple[float, float, float, float]:
    """Settle one finished hand from its actions and, if needed, hand scores.

    Folding leaves only the forced blind; jamming commits the whole
    ``stack_bb``. If everyone folds, the BB takes the blinds — a walk. With a
    single jam the jammer wins without a showdown. With two or more, scores are
    required and the best hand takes the pot, split as many ways as it ties.
    """
    if len(actions) != 4 or any(action not in ACTIONS for action in actions):
        raise ValueError("settle_actions needs four F/J actions")
    blinds = config.blinds()
    committed = [config.stack_bb if action == JAM else blind
                 for action, blind in zip(actions, blinds, strict=True)]
    pot = sum(committed)
    jammed = [i for i, action in enumerate(actions) if action == JAM]
    if not jammed:
        winners = [3]  # the BB takes the blinds when nobody jams
    elif len(jammed) == 1:
        winners = jammed
    else:
        if scores is None:
            raise ValueError("a multiway showdown needs hand scores")
        best = max(scores[i] for i in jammed)
        winners = [i for i in jammed if scores[i] == best]
    net_pot = pot - _rake(config, actions, pot)
    utilities = [-amount for amount in committed]
    share = net_pot / len(winners)
    for winner in winners:
        utilities[winner] += share
    return tuple(utilities)  # type: ignore[return-value]


class AofCfrSolver:
    """Regret-matching self-play over random deals and runouts.

    One step samples four private hands and a board, then walks the whole
    binary tree of 16 terminal nodes. Because player actions are not sampled,
    the updates are markedly less noisy than outcome sampling. Chance — the
    deal and the runout — is still sampled, so use tens to hundreds of
    thousands of iterations for anything you rely on.
    """

    def __init__(self, config: AofConfig | None = None, *, seed: int = 1):
        self.config = config or AofConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes: dict[AofInfoSet, _RegretNode] = {}
        self.iterations = 0
        self._scores: tuple[tuple, tuple, tuple, tuple] | None = None
        self._payoffs: dict[tuple[str, str, str, str], tuple[float, float, float, float]] = {}

    # ── solve ──────────────────────────────────────────────────────────

    def solve(
        self,
        iterations: int,
        *,
        progress: Callable[[int, int], None] | None = None,
        progress_every: int = 10_000,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if progress_every <= 0:
            raise ValueError("progress_every must be positive")
        for index in range(1, iterations + 1):
            holes, board = self._deal()
            scores = tuple(best_hand(list(hole) + board) for hole in holes)
            self._scores = scores[0], scores[1], scores[2], scores[3]
            self._payoffs = {}
            self._cfr((), holes, (1.0, 1.0, 1.0, 1.0))
            self.iterations += 1
            if progress and (index == iterations or index % progress_every == 0):
                progress(index, iterations)

    def _deal(self) -> tuple[tuple[tuple[str, str], ...], list[str]]:
        drawn = self.rng.sample(FULL_DECK, 13)
        holes = tuple(canonical_combo(drawn[i:i + 2]) for i in range(0, 8, 2))
        return holes, drawn[8:]

    def _node(self, player: int, history: tuple[str, ...], combo: tuple[str, str]) -> _RegretNode:
        key = AofInfoSet(player, history, combo)
        node = self.nodes.get(key)
        if node is None:
            node = _RegretNode.new()
            self.nodes[key] = node
        return node

    def _terminal_utility(self, history: tuple[str, ...]) -> tuple[float, float, float, float]:
        actions = history[0], history[1], history[2], history[3]
        cached = self._payoffs.get(actions)
        if cached is None:
            if self._scores is None:
                raise RuntimeError("the current deal has no scores")
            cached = settle_actions(self.config, actions, self._scores)
            self._payoffs[actions] = cached
        return cached

    def _cfr(
        self,
        history: tuple[str, ...],
        holes: tuple[tuple[str, str], ...],
        reach: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        player = len(history)
        if player == 4:
            return self._terminal_utility(history)

        node = self._node(player, history, holes[player])
        strategy = node.strategy()
        action_utilities: list[tuple[float, float, float, float]] = []
        for action_index, action in enumerate(ACTIONS):
            next_reach = list(reach)
            next_reach[player] *= strategy[action_index]
            action_utilities.append(self._cfr(
                history + (action,), holes, tuple(next_reach),
            ))

        expected = tuple(
            sum(strategy[action_index] * action_utilities[action_index][p]
                for action_index in range(2))
            for p in range(4)
        )
        counterfactual_reach = 1.0
        for opponent, probability in enumerate(reach):
            if opponent != player:
                counterfactual_reach *= probability
        own_reach = reach[player]
        for action_index in range(2):
            node.regrets[action_index] += counterfactual_reach * (
                action_utilities[action_index][player] - expected[player]
            )
            node.strategy_sum[action_index] += own_reach * strategy[action_index]
            node.action_value_sum[action_index] += counterfactual_reach * action_utilities[action_index][player]
        node.action_value_weight += counterfactual_reach
        return expected[0], expected[1], expected[2], expected[3]

    # ── policies, evaluation a charty ──────────────────────────────────

    def info_set(
        self, position: str, history: str | Iterable[str], combo: Iterable[str],
    ) -> AofInfoSet:
        try:
            player = self.config.positions.index(position.upper())
        except ValueError as exc:
            raise ValueError(f"unknown position: {position!r}") from exc
        parsed = parse_history(history)
        if len(parsed) != player:
            raise ValueError(
                f"{position.upper()} acts after {player} actions, not {len(parsed)}",
            )
        return AofInfoSet(player, parsed, canonical_combo(combo))

    def policy_for(self, info_set: AofInfoSet) -> tuple[float, float]:
        node = self.nodes.get(info_set)
        return node.average_strategy() if node else (0.5, 0.5)

    def matrix(self, position: str, history: str | Iterable[str] = "") -> dict[str, MatrixCell]:
        """Aggregate per-combo strategies into the 169 hand classes."""
        try:
            player = self.config.positions.index(position.upper())
        except ValueError as exc:
            raise ValueError(f"unknown position: {position!r}") from exc
        parsed = parse_history(history)
        if len(parsed) != player:
            raise ValueError(
                f"{position.upper()} needs a history of length {player}",
            )
        output: dict[str, MatrixCell] = {}
        for cls in all_hand_classes():
            nodes = [self.nodes.get(AofInfoSet(player, parsed, canonical_combo(combo)))
                     for combo in combos_of_class(cls)]
            usable = [node for node in nodes if node is not None]
            if not usable:
                output[cls] = MatrixCell(None, None, None, None, 0.0)
                continue
            policies = [node.average_strategy() for node in usable]
            values = [node.action_values() for node in usable]
            samples = sum(node.action_value_weight for node in usable)
            fold = sum(policy[0] for policy in policies) / len(policies)
            jam = sum(policy[1] for policy in policies) / len(policies)
            ev_fold = _mean_optional(value[0] for value in values)
            ev_jam = _mean_optional(value[1] for value in values)
            output[cls] = MatrixCell(fold, jam, ev_fold, ev_jam, samples)
        return output

    def all_matrices(self) -> dict[str, dict[str, MatrixCell]]:
        """Every node of the tree: 1 + 2 + 4 + 8 = 15 matrices."""
        result: dict[str, dict[str, MatrixCell]] = {}
        for player, position in enumerate(self.config.positions):
            for history_bits in range(1 << player):
                history = tuple(
                    JAM if history_bits & (1 << (player - 1 - index)) else FOLD
                    for index in range(player)
                )
                key = f"{position}:{''.join(history) or 'root'}"
                result[key] = self.matrix(position, history)
        return result

    def evaluate_average_strategy(self, samples: int = 10_000, *, seed: int | None = None) -> dict[str, float]:
        """An independent Monte Carlo check of the average policy's EV per position."""
        if samples <= 0:
            raise ValueError("samples must be positive")
        rng = random.Random(self.seed + 1 if seed is None else seed)
        totals = [0.0] * 4
        for _ in range(samples):
            drawn = rng.sample(FULL_DECK, 13)
            holes = tuple(canonical_combo(drawn[i:i + 2]) for i in range(0, 8, 2))
            scores = tuple(best_hand(list(hole) + drawn[8:]) for hole in holes)
            history: tuple[str, ...] = ()
            for player in range(4):
                probabilities = self.policy_for(AofInfoSet(player, history, holes[player]))
                action = FOLD if rng.random() < probabilities[0] else JAM
                history += (action,)
            payoff = settle_actions(self.config, history, scores)
            for player, utility in enumerate(payoff):
                totals[player] += utility
        return {position: total / samples
                for position, total in zip(self.config.positions, totals, strict=True)}

    def export_payload(self, *, evaluation_samples: int = 0) -> dict:
        matrices = {
            name: {cls: cell.as_dict() for cls, cell in matrix.items()}
            for name, matrix in self.all_matrices().items()
        }
        payload = {
            "format": "pokersolver-aof-v1",
            "game": "4max NLHE All-In or Fold",
            "method": "sampled-chance regret-minimisation self-play",
            "warning": (
                "4-player poker with rake is not zero-sum; this is a self-play "
                "study policy, not a certified GTO equilibrium."
            ),
            "config_bb": asdict(self.config),
            "seed": self.seed,
            "iterations": self.iterations,
            "information_sets": len(self.nodes),
            "matrices": matrices,
        }
        if evaluation_samples:
            payload["average_strategy_ev_bb"] = self.evaluate_average_strategy(evaluation_samples)
            payload["evaluation_samples"] = evaluation_samples
        return payload

    def write_json(self, path: str | Path, *, evaluation_samples: int = 0) -> Path:
        """Write a readable export of the finished charts."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.export_payload(evaluation_samples=evaluation_samples), handle,
                      ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return output


def _mean_optional(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _cell_hand(row: int, column: int) -> str:
    high, low = _ORDER[row], _ORDER[column]
    if row == column:
        return high + high
    if row < column:
        return high + low + "s"
    return low + high + "o"


def render_matrix(
    matrix: dict[str, MatrixCell], *, metric: str = "jam",
) -> list[str]:
    """The 13x13 matrix as text: jam frequency, or ``EV(jam)-EV(fold)``.

    ``metric='jam'`` prints percentages, ``metric='delta'`` a difference in bb.
    A dot means the combo has not been seen yet — run more iterations.
    """
    if metric not in {"jam", "delta"}:
        raise ValueError("metric must be 'jam' or 'delta'")
    header = "       " + " ".join(f"{rank:>6}" for rank in _ORDER)
    lines = [header]
    for row, rank in enumerate(_ORDER):
        cells = []
        for column in range(13):
            cell = matrix[_cell_hand(row, column)]
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
