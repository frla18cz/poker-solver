"""Restricted sampled-chance CFR prototype for multiway postflop study.

The model resolves either a checked-to Hero decision (check/bet) or a Hero
decision facing a wager (fold/call/raise). Opponents may fold/call and, when a
raise-to sizing is configured, take one additional raise branch before the
remaining board is dealt without further betting.
It is useful as a transparent continual-resolve prototype, but is deliberately
not presented as a complete multiway NLHE equilibrium solver.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import TypeAlias

from pokersolver.cards import FULL_DECK, score_7


WeightedRange: TypeAlias = tuple[tuple[tuple[str, str], float], ...]
CHECK = "check"
BET = "bet"
FOLD = "fold"
CALL = "call"
RAISE = "raise"
ALLIN = "allin"
HERO = 0
HERO_RESPONSE = "hero_response"


@dataclass(frozen=True)
class MultiwayCfrConfig:
    """One restricted multiway c-bet subgame, expressed in big blinds."""

    board: tuple[str, ...]
    ranges: tuple[WeightedRange, ...]
    names: tuple[str, ...]
    pot_bb: float
    bet_bb: float
    iterations: int = 10_000
    seed: int = 1
    rake_percent: float = 0.0
    rake_cap_bb: float = 0.0
    scenario: str = "checked_to"
    facing_bet_bb: float = 0.0
    raise_to_bb: float = 0.0
    stacks_bb: tuple[float, ...] = ()
    opponent_raise_to_bb: float = 0.0
    opponent_allin: bool = False
    # A richer tree. Empty tuples mean the old behaviour: a single
    # velikost z bet_bb / raise_to_bb / opponent_raise_to_bb.
    bet_sizes_pct: tuple[float, ...] = ()
    raise_sizes_pct: tuple[float, ...] = ()
    opponent_raise_sizes_pct: tuple[float, ...] = ()
    hero_allin: bool = False
    max_raise_rounds: int = 1

    def __post_init__(self) -> None:
        if len(self.board) not in (3, 4, 5):
            raise ValueError("multiway CFR requires a flop, turn, or river board")
        if len(set(self.board)) != len(self.board):
            raise ValueError("board contains duplicate cards")
        if not 2 <= len(self.ranges) <= 4:
            # Heads-up is 77% of all postflop spots. Side pots, rake and
            # showdown are all written generically over active players and
            # commitments, so two players are a degenerate case of the same
            # tree rather than a branch of their own.
            raise ValueError("multiway CFR supports two, three or four players")
        if len(self.names) != len(self.ranges) or len(set(self.names)) != len(self.names):
            raise ValueError("each player needs a unique name")
        if any(not entries for entries in self.ranges):
            raise ValueError("each player range must contain at least one combo")
        deck = set(FULL_DECK)
        if any(
            len(combo) != 2 or combo[0] == combo[1]
            or combo[0] not in deck or combo[1] not in deck or weight <= 0
            for entries in self.ranges for combo, weight in entries
        ):
            raise ValueError("ranges contain an invalid combo or weight")
        if self.pot_bb <= 0 or self.bet_bb <= 0:
            raise ValueError("pot and bet must be positive")
        if self.scenario not in ("checked_to", "facing_wager"):
            raise ValueError("scenario must be checked_to or facing_wager")
        if self.scenario == "facing_wager":
            if self.facing_bet_bb <= 0:
                raise ValueError("facing wager must be positive")
            # The absolute raise-to only matters when no percentage sizes are
            # given. With percentages present the menu is built from those and
            # this field goes unused, so demanding it would be a nuisance.
            if (not self.raise_sizes_pct
                    and self.player_stacks[0] > self.facing_bet_bb
                    and self.raise_to_bb <= self.facing_bet_bb):
                raise ValueError("raise-to amount must exceed the facing wager")
        if self.stacks_bb and len(self.stacks_bb) != len(self.ranges):
            raise ValueError("supply one stack for every player")
        if any(stack < 0 for stack in self.stacks_bb):
            raise ValueError("player stacks cannot be negative")
        if self.opponent_raise_to_bb < 0:
            raise ValueError("opponent raise-to amount cannot be negative")
        if (
            self.scenario == "facing_wager"
            and not self.raise_sizes_pct
            and self.player_stacks[0] > self.facing_bet_bb
            and self.raise_to_bb > self.player_stacks[0]
        ):
            raise ValueError(
                f"raise-to amount exceeds Hero stack ({self.player_stacks[0]:g} BB)"
            )
        if any(pct <= 0 for pct in (*self.bet_sizes_pct, *self.raise_sizes_pct,
                                    *self.opponent_raise_sizes_pct)):
            raise ValueError("bet and raise sizes must be positive percentages")
        if self.max_raise_rounds < 0:
            raise ValueError("max_raise_rounds cannot be negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0 <= self.rake_percent <= 20 or self.rake_cap_bb < 0:
            raise ValueError("invalid rake configuration")

    @property
    def player_stacks(self) -> tuple[float, ...]:
        return self.stacks_bb or (100.0,) * len(self.ranges)


@dataclass(frozen=True)
class _InfoSet:
    player: int
    history: tuple[str, ...]
    combo: tuple[str, str]


@dataclass
class _Node:
    regrets: list[float]
    strategy_sum: list[float]
    action_value_sum: list[float]
    action_value_weight: float = 0.0
    visits: int = 0
    names: tuple[str, ...] = ()

    @classmethod
    def new(cls, action_count: int, names: tuple[str, ...] = ()) -> "_Node":
        return cls([0.0] * action_count, [0.0] * action_count, [0.0] * action_count,
                   names=names)

    def current_strategy(self) -> tuple[float, ...]:
        positive = [max(0.0, value) for value in self.regrets]
        total = sum(positive)
        if total <= 1e-15:
            return tuple(1.0 / len(positive) for _value in positive)
        return tuple(value / total for value in positive)

    def average_strategy(self) -> tuple[float, ...]:
        total = sum(self.strategy_sum)
        if total <= 1e-15:
            return self.current_strategy()
        return tuple(value / total for value in self.strategy_sum)

    def action_values(self) -> tuple[float | None, ...]:
        if self.action_value_weight <= 1e-15:
            return tuple(None for _value in self.action_value_sum)
        return tuple(
            value / self.action_value_weight for value in self.action_value_sum
        )


def _distinct_options(
    options: list[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    """Drop sizes that collapse to the same amount once capped by the stack.

    Without this there would be two differently named actions with identical
    effect; CFR would split frequency between them and the result would look
    like a mix where there is no actual choice.
    """
    kept: list[tuple[str, float]] = []
    for name, amount in options:
        if amount <= 1e-9:
            continue
        if any(abs(amount - seen) <= 1e-9 for _name, seen in kept):
            continue
        kept.append((name, amount))
    return tuple(kept)


def _canonical(combo: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(combo, reverse=True))  # type: ignore[return-value]


class MultiwayCfrSolver:
    """Sampled-chance regret minimisation for one restricted decision node."""

    def __init__(self, config: MultiwayCfrConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.nodes: dict[_InfoSet, _Node] = {}
        self.iterations = 0
        self.valid_deals = 0
        self.evaluation_unvisited = 0
        self._scores: tuple[tuple, ...] = ()
        self._hero_equity_sum = 0.0

    def solve(
        self,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        deadline: float | None = None,
    ) -> None:
        """Train the strategy. ``deadline`` is a ``time.monotonic()`` timestamp.

        The deadline is **soft**: the loop stops and the average strategy built
        so far stays usable. Cancelling through ``cancelled`` is hard by
        contrast, because there nobody wants the result.
        """
        for _ in range(self.config.iterations):
            if cancelled and cancelled():
                raise InterruptedError("multiway solve cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                break
            deal = self._deal()
            if deal is None:
                continue
            holes, full_board = deal
            self._scores = tuple(score_7(list(hole) + list(full_board)) for hole in holes)
            best = max(self._scores)
            winners = sum(score == best for score in self._scores)
            if self._scores[0] == best:
                self._hero_equity_sum += 1.0 / winners
            self.valid_deals += 1
            self._cfr_root(holes, (1.0,) * len(holes))
            self.iterations += 1
            if progress and (self.iterations == 1 or self.iterations % 100 == 0):
                progress(self.iterations, self.config.iterations, "solving")
        if not self.valid_deals:
            raise ValueError("could not sample collision-free hands from the supplied ranges")

    def _weighted_pick(
        self, entries: WeightedRange, rng: random.Random | None = None,
    ) -> tuple[str, str]:
        rng = rng or self.rng
        total = sum(weight for _combo, weight in entries)
        needle = rng.random() * total
        for combo, weight in entries:
            needle -= weight
            if needle <= 0:
                return combo
        return entries[-1][0]

    def _deal(self) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
        board_cards = set(self.config.board)
        for _attempt in range(200):
            holes = tuple(self._weighted_pick(entries) for entries in self.config.ranges)
            private = [card for combo in holes for card in combo]
            if len(set(private)) != len(private) or board_cards.intersection(private):
                continue
            remaining = [card for card in FULL_DECK if card not in board_cards and card not in private]
            runout = self.rng.sample(remaining, 5 - len(self.config.board))
            return holes, self.config.board + tuple(runout)
        return None

    def _deal_with_hero(
        self, hero_combo: tuple[str, str], rng: random.Random,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
        board_cards = set(self.config.board)
        if board_cards.intersection(hero_combo) or hero_combo[0] == hero_combo[1]:
            return None
        for _attempt in range(200):
            opponents = tuple(
                self._weighted_pick(entries, rng) for entries in self.config.ranges[1:]
            )
            holes = (hero_combo,) + opponents
            private = [card for combo in holes for card in combo]
            if len(set(private)) != len(private) or board_cards.intersection(private):
                continue
            remaining = [card for card in FULL_DECK if card not in board_cards and card not in private]
            runout = rng.sample(remaining, 5 - len(self.config.board))
            return holes, self.config.board + tuple(runout)
        return None

    def _node(
        self,
        player: int,
        history: tuple[str, ...],
        combo: tuple[str, str],
        action_count: int = 2,
        names: tuple[str, ...] = (),
    ) -> _Node:
        key = _InfoSet(player, history, _canonical(combo))
        node = self.nodes.get(key)
        if node is None:
            node = _Node.new(action_count, names)
            self.nodes[key] = node
        elif len(node.regrets) != action_count:
            raise RuntimeError("information set has inconsistent legal actions")
        return node

    def _update_node(
        self,
        node: _Node,
        player: int,
        strategy: tuple[float, ...],
        action_utilities: tuple[tuple[float, ...], ...],
        expected: tuple[float, ...],
        reach: tuple[float, ...],
    ) -> None:
        counterfactual_reach = 1.0
        for other, probability in enumerate(reach):
            if other != player:
                counterfactual_reach *= probability
        for action_index in range(len(strategy)):
            node.regrets[action_index] += counterfactual_reach * (
                action_utilities[action_index][player] - expected[player]
            )
            node.strategy_sum[action_index] += reach[player] * strategy[action_index]
            node.action_value_sum[action_index] += (
                counterfactual_reach * action_utilities[action_index][player]
            )
        node.action_value_weight += counterfactual_reach
        node.visits += 1

    def bet_menu(self) -> tuple[tuple[str, float], ...]:
        """The hero's bets as ``(action name, total commitment in bb)``.

        Without ``bet_sizes_pct`` there is a single action called ``bet``, so
        older callers and stored results keep working. With several sizes the
        names become ``bet@33``, ``bet@75`` and so on.
        """
        config = self.config
        cap = config.player_stacks[HERO]
        options: list[tuple[str, float]] = []
        if config.bet_sizes_pct:
            for pct in config.bet_sizes_pct:
                amount = min(round(config.pot_bb * pct / 100.0, 4), cap)
                options.append((f"{BET}@{pct:g}", amount))
        else:
            options.append((BET, min(config.bet_bb, cap)))
        if config.hero_allin:
            options.append((ALLIN, cap))
        return _distinct_options(options)

    def raise_menu(self) -> tuple[tuple[str, float], ...]:
        """The hero's raises as ``(action name, total commitment in bb)``.

        Percentages are of the opponent's bet, which is how raise sizing is
        normally expressed: raising to 250% means 2.5x their bet.
        """
        config = self.config
        cap = config.player_stacks[HERO]
        options: list[tuple[str, float]] = []
        if config.raise_sizes_pct:
            for pct in config.raise_sizes_pct:
                amount = min(round(config.facing_bet_bb * pct / 100.0, 4), cap)
                if amount > config.facing_bet_bb + 1e-12:
                    options.append((f"{RAISE}@{pct:g}", amount))
        elif config.raise_to_bb > config.facing_bet_bb:
            # Once capped by the stack there may be nothing left to raise with,
            # which makes it a call rather than a raise.
            amount = min(config.raise_to_bb, cap)
            if amount > config.facing_bet_bb + 1e-12:
                options.append((RAISE, amount))
        if config.hero_allin and cap > config.facing_bet_bb + 1e-12:
            options.append((ALLIN, cap))
        return _distinct_options(options)

    def opponents_may_raise(self, target: float) -> bool:
        """May the opponents re-raise at all?

        Both sources have to be consulted: the old scalar size and the newer
        list. Checking only the scalar meant the list never took effect, because
        the scalar stayed at zero.
        """
        return bool(self.config.opponent_raise_sizes_pct) or (
            self.config.opponent_raise_to_bb > target
        )

    def opponent_raise_menu(
        self, target: float, maximum: float,
    ) -> tuple[tuple[str, float], ...]:
        """Opponent raises against commitment ``target``, capped by their stack.

        Without ``opponent_raise_sizes_pct`` there is a single size named
        ``raise``, as before.
        """
        config = self.config
        options: list[tuple[str, float]] = []
        if config.opponent_raise_sizes_pct:
            for pct in config.opponent_raise_sizes_pct:
                amount = min(round(target * pct / 100.0, 4), maximum)
                if amount > target + 1e-12:
                    options.append((f"{RAISE}@{pct:g}", amount))
        else:
            amount = min(config.opponent_raise_to_bb, maximum)
            if amount > target + 1e-12:
                options.append((RAISE, amount))
        return _distinct_options(options)

    def _cfr_root(
        self, holes: tuple[tuple[str, str], ...], reach: tuple[float, ...],
    ) -> tuple[float, ...]:
        if self.config.scenario == "facing_wager":
            return self._cfr_facing_root(holes, reach)
        seats = len(holes)
        menu = self.bet_menu()
        node = self._node(0, (), holes[0], 1 + len(menu))
        strategy = node.current_strategy()
        utilities = [self._terminal(tuple(range(seats)), (0.0,) * seats)]
        for index, (_name, amount) in enumerate(menu, start=1):
            next_reach = list(reach)
            next_reach[0] *= strategy[index]
            commitments = [0.0] * seats
            commitments[0] = amount
            utilities.append(self._cfr_responses(
                tuple(range(1, seats)), 0, ("lead", _name), (0,), tuple(commitments),
                amount, holes, tuple(next_reach),
                allow_raise=self.opponents_may_raise(amount),
                allow_allin=self.config.opponent_allin,
                raise_rounds=self.config.max_raise_rounds,
            ))
        actions = tuple(utilities)
        expected = tuple(
            sum(strategy[index] * actions[index][seat] for index in range(len(strategy)))
            for seat in range(seats)
        )
        self._update_node(node, 0, strategy, actions, expected, reach)
        return expected

    def _cfr_facing_root(
        self, holes: tuple[tuple[str, str], ...], reach: tuple[float, ...],
    ) -> tuple[float, ...]:
        node = self._node(0, ("facing",), holes[0], len(self.hero_actions))
        strategy = node.current_strategy()
        aggressor = len(holes) - 1

        fold_commitments = [0.0] * len(holes)
        fold_commitments[aggressor] = self.config.facing_bet_bb
        fold_utility = self._terminal((aggressor,), tuple(fold_commitments))

        call_reach = list(reach)
        call_reach[0] *= strategy[1]
        call_commitments = list(fold_commitments)
        call_commitments[0] = min(
            self.config.facing_bet_bb, self.config.player_stacks[0],
        )
        call_responders = tuple(range(1, aggressor))
        call_utility = self._cfr_responses(
            call_responders, 0, ("facing", CALL), (0, aggressor),
            tuple(call_commitments), self.config.facing_bet_bb, holes, tuple(call_reach),
            allow_raise=self.opponents_may_raise(self.config.facing_bet_bb),
            allow_allin=self.config.opponent_allin,
            raise_rounds=self.config.max_raise_rounds,
        )

        actions = [fold_utility, call_utility]
        raise_responders = tuple(range(1, aggressor)) + (aggressor,)
        for offset, (name, amount) in enumerate(self.raise_menu(), start=2):
            raise_reach = list(reach)
            raise_reach[0] *= strategy[offset]
            raise_commitments = list(fold_commitments)
            raise_commitments[0] = amount
            actions.append(self._cfr_responses(
                raise_responders, 0, ("facing", name), (0,), tuple(raise_commitments),
                amount, holes, tuple(raise_reach),
                allow_raise=self.opponents_may_raise(amount),
                allow_allin=self.config.opponent_allin,
                raise_rounds=self.config.max_raise_rounds - 1,
            ))
        action_tuple = tuple(actions)
        expected = tuple(
            sum(strategy[index] * action_tuple[index][seat] for index in range(len(strategy)))
            for seat in range(len(holes))
        )
        self._update_node(node, 0, strategy, action_tuple, expected, reach)
        return expected

    def _cfr_responses(
        self,
        responders: tuple[int, ...],
        responder_index: int,
        history: tuple[str, ...],
        active: tuple[int, ...],
        commitments: tuple[float, ...],
        target: float,
        holes: tuple[tuple[str, str], ...],
        reach: tuple[float, ...],
        allow_raise: bool = False,
        allow_allin: bool = False,
        raise_rounds: int = 1,
    ) -> tuple[float, ...]:
        if responder_index == len(responders):
            return self._close_round(
                history, active, commitments, target, holes, reach, raise_rounds,
            )
        player = responders[responder_index]
        maximum = commitments[player] + self.config.player_stacks[player]
        if maximum <= commitments[player] + 1e-12:
            next_active = active + (player,) if commitments[player] > 0 else active
            return self._cfr_responses(
                responders, responder_index + 1, history + ((CALL if commitments[player] > 0 else FOLD),),
                next_active, commitments, target, holes, reach, allow_raise=allow_raise,
                allow_allin=allow_allin, raise_rounds=raise_rounds,
            )
        raises = (self.opponent_raise_menu(target, maximum)
                  if allow_raise and raise_rounds > 0 else ())
        amounts = {CALL: min(target, maximum), **{name: amount for name, amount in raises}}
        actions_available = [FOLD, CALL, *(name for name, _amount in raises)]
        if allow_allin and maximum > target + 1e-12:
            actions_available.append(ALLIN)
            amounts[ALLIN] = maximum
        node = self._node(player, history, holes[player], len(actions_available),
                          tuple(actions_available))
        strategy = node.current_strategy()
        action_utilities: list[tuple[float, ...]] = []
        for action_index, action in enumerate(actions_available):
            next_reach = list(reach)
            next_reach[player] *= strategy[action_index]
            next_active = active if action == FOLD else active + (player,)
            next_commitments = list(commitments)
            if action != FOLD:
                next_commitments[player] = amounts[action]
            is_raise = action.startswith(RAISE)
            action_utilities.append(self._cfr_responses(
                responders, responder_index + 1, history + (action,), next_active,
                tuple(next_commitments),
                amounts[action] if action != FOLD and (is_raise or action == ALLIN) else target,
                holes, tuple(next_reach),
                allow_raise=allow_raise and not is_raise and action != ALLIN,
                allow_allin=allow_allin and action != ALLIN,
                raise_rounds=raise_rounds - 1 if is_raise else raise_rounds,
            ))
        actions = tuple(action_utilities)
        expected = tuple(sum(strategy[index] * action[seat] for index, action in enumerate(actions)) for seat in range(len(holes)))
        self._update_node(node, player, strategy, actions, expected, reach)
        return expected

    def _evaluation_strategy(
        self, node: _Node | None, action_count: int,
    ) -> tuple[float, ...]:
        """A node's average strategy; unvisited infosets fall back to uniform.

        That fallback is a fiction training never saw, so it is counted and
        reported in ``result()``. A high count means the pool of deals left part
        of the tree uncovered, and EVs from that branch are not trustworthy.
        """
        if node is not None and len(node.regrets) == action_count:
            return node.average_strategy()
        self.evaluation_unvisited += 1
        return tuple(1.0 / action_count for _index in range(action_count))

    def _evaluate_close_round(
        self,
        history: tuple[str, ...],
        active: tuple[int, ...],
        commitments: tuple[float, ...],
        target: float,
        holes: tuple[tuple[str, str], ...],
        raise_rounds: int = 0,
    ) -> tuple[float, ...]:
        """The evaluation counterpart to :meth:`_close_round`; same tree shape."""
        if not self._hero_must_respond(active, commitments, target):
            return self._terminal(active, commitments)
        maximum = self.config.player_stacks[HERO]
        menu = self.reraise_menu(target) if raise_rounds > 0 else ()
        node = self.nodes.get(
            _InfoSet(HERO, history + (HERO_RESPONSE,), _canonical(holes[HERO])),
        )
        strategy = self._evaluation_strategy(node, 2 + len(menu))
        utilities = [
            self._terminal(tuple(seat for seat in active if seat != HERO), commitments),
        ]
        call_commitments = list(commitments)
        call_commitments[HERO] = min(target, maximum)
        utilities.append(self._terminal(active, tuple(call_commitments)))
        responders = tuple(seat for seat in active if seat != HERO)
        for name, amount in menu:
            next_commitments = list(commitments)
            next_commitments[HERO] = amount
            utilities.append(self._evaluate_responses(
                responders, 0, history + (HERO_RESPONSE, name), (HERO,),
                tuple(next_commitments), amount, holes,
                allow_raise=self.opponents_may_raise(amount),
                allow_allin=self.config.opponent_allin,
                raise_rounds=raise_rounds - 1,
            ))
        return tuple(
            sum(strategy[index] * utilities[index][seat] for index in range(len(strategy)))
            for seat in range(len(holes))
        )

    def _hero_must_respond(
        self, active: tuple[int, ...], commitments: tuple[float, ...], target: float,
    ) -> bool:
        """The hero faces a raise after acting, and has something left to call with.

        ``player_stacks`` is the TOTAL commitment a player can carry this round —
        the same way ``_cfr_root`` caps it — not what remains behind a bet. For
        opponents the commitment is zero when they decide, so the difference
        does not show; the hero has already bet, and adding the stack to their
        commitment would apply the cap twice.
        """
        return (
            HERO in active
            and commitments[HERO] < target - 1e-12
            and self.config.player_stacks[HERO] > commitments[HERO] + 1e-12
        )

    def reraise_menu(self, target: float) -> tuple[tuple[str, float], ...]:
        """The hero's re-raises against commitment ``target``."""
        config = self.config
        cap = config.player_stacks[HERO]
        options: list[tuple[str, float]] = []
        for pct in config.raise_sizes_pct or ():
            amount = min(round(target * pct / 100.0, 4), cap)
            if amount > target + 1e-12:
                options.append((f"{RAISE}@{pct:g}", amount))
        if config.hero_allin and cap > target + 1e-12:
            options.append((ALLIN, cap))
        return _distinct_options(options)

    def _close_round(
        self,
        history: tuple[str, ...],
        active: tuple[int, ...],
        commitments: tuple[float, ...],
        target: float,
        holes: tuple[tuple[str, str], ...],
        reach: tuple[float, ...],
        raise_rounds: int = 0,
    ) -> tuple[float, ...]:
        """Close the round. If someone out-bet the hero, they get to fold or call.

        Without this node the hero would stay committed only to their original
        bet, and ``_terminal`` would settle them as all-in for the smaller
        amount — a free showdown, where no raise could ever push them off a hand.
        """
        if not self._hero_must_respond(active, commitments, target):
            return self._terminal(active, commitments)
        maximum = self.config.player_stacks[HERO]
        menu = self.reraise_menu(target) if raise_rounds > 0 else ()
        node = self._node(HERO, history + (HERO_RESPONSE,), holes[HERO], 2 + len(menu))
        strategy = node.current_strategy()

        utilities = [
            self._terminal(tuple(seat for seat in active if seat != HERO), commitments),
        ]
        call_commitments = list(commitments)
        call_commitments[HERO] = min(target, maximum)
        utilities.append(self._terminal(active, tuple(call_commitments)))

        # The hero re-raises: everyone still in has to answer the new amount.
        responders = tuple(seat for seat in active if seat != HERO)
        for offset, (name, amount) in enumerate(menu, start=2):
            next_reach = list(reach)
            next_reach[HERO] *= strategy[offset]
            next_commitments = list(commitments)
            next_commitments[HERO] = amount
            utilities.append(self._cfr_responses(
                responders, 0, history + (HERO_RESPONSE, name), (HERO,),
                tuple(next_commitments), amount, holes, tuple(next_reach),
                allow_raise=self.opponents_may_raise(amount),
                allow_allin=self.config.opponent_allin,
                raise_rounds=raise_rounds - 1,
            ))

        actions = tuple(utilities)
        expected = tuple(
            sum(strategy[index] * actions[index][seat] for index in range(len(strategy)))
            for seat in range(len(holes))
        )
        self._update_node(node, HERO, strategy, actions, expected, reach)
        return expected

    def _terminal(
        self, active: tuple[int, ...], commitments: tuple[float, ...],
    ) -> tuple[float, ...]:
        total_pot = self.config.pot_bb + sum(commitments)
        raw_rake = total_pot * self.config.rake_percent / 100.0
        rake = min(raw_rake, self.config.rake_cap_bb) if self.config.rake_cap_bb > 0 else raw_rake
        utilities = [-amount for amount in commitments]
        pots: list[tuple[float, tuple[int, ...]]] = [(self.config.pot_bb, active)]
        levels = sorted({amount for amount in commitments if amount > 0})
        previous = 0.0
        for level in levels:
            contributors = tuple(
                player for player, amount in enumerate(commitments) if amount + 1e-12 >= level
            )
            eligible = tuple(player for player in active if player in contributors)
            pots.append(((level - previous) * len(contributors), eligible))
            previous = level
        remaining_rake = rake
        for amount, eligible in pots:
            award = max(0.0, amount - min(amount, remaining_rake))
            remaining_rake = max(0.0, remaining_rake - amount)
            if award <= 0 or not eligible:
                continue
            best = max(self._scores[player] for player in eligible)
            winners = [player for player in eligible if self._scores[player] == best]
            share = award / len(winners)
            for winner in winners:
                utilities[winner] += share
        return tuple(utilities)

    def _evaluate_responses(
        self,
        responders: tuple[int, ...],
        responder_index: int,
        history: tuple[str, ...],
        active: tuple[int, ...],
        commitments: tuple[float, ...],
        target: float,
        holes: tuple[tuple[str, str], ...],
        allow_raise: bool = False,
        allow_allin: bool = False,
        raise_rounds: int = 1,
    ) -> tuple[float, ...]:
        if responder_index == len(responders):
            return self._evaluate_close_round(
                history, active, commitments, target, holes, raise_rounds,
            )
        player = responders[responder_index]
        maximum = commitments[player] + self.config.player_stacks[player]
        if maximum <= commitments[player] + 1e-12:
            next_active = active + (player,) if commitments[player] > 0 else active
            return self._evaluate_responses(
                responders, responder_index + 1,
                history + ((CALL if commitments[player] > 0 else FOLD),),
                next_active, commitments, target, holes, allow_raise=allow_raise,
                allow_allin=allow_allin, raise_rounds=raise_rounds,
            )
        node = self.nodes.get(_InfoSet(player, history, _canonical(holes[player])))
        raises = (self.opponent_raise_menu(target, maximum)
                  if allow_raise and raise_rounds > 0 else ())
        amounts = {CALL: min(target, maximum), **{name: amount for name, amount in raises}}
        actions_available = [FOLD, CALL, *(name for name, _amount in raises)]
        if allow_allin and maximum > target + 1e-12:
            actions_available.append(ALLIN)
            amounts[ALLIN] = maximum
        strategy = self._evaluation_strategy(node, len(actions_available))
        utilities=[]
        for action in actions_available:
            next_commitments=list(commitments)
            if action != FOLD:
                next_commitments[player]=amounts[action]
            is_raise = action.startswith(RAISE)
            utilities.append(self._evaluate_responses(
                responders,responder_index+1,history+(action,),
                active if action==FOLD else active+(player,),tuple(next_commitments),
                amounts[action] if action != FOLD and (is_raise or action == ALLIN) else target,
                holes,
                allow_raise=allow_raise and not is_raise and action != ALLIN,
                allow_allin=allow_allin and action != ALLIN,
                raise_rounds=raise_rounds - 1 if is_raise else raise_rounds,
            ))
        return tuple(sum(strategy[index]*utility[seat] for index,utility in enumerate(utilities)) for seat in range(len(holes)))

    def evaluate_hero_combo(
        self,
        hero_combo: tuple[str, str],
        samples: int,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        deadline: float | None = None,
    ) -> dict[str, float | int]:
        """Independent EV/equity check against the final average policies."""
        rng = random.Random(self.config.seed ^ 0x4D554C54)
        equity = 0.0
        action_ev = [0.0] * len(self.hero_actions)
        done = 0
        for _ in range(samples):
            if cancelled and cancelled():
                raise InterruptedError("multiway evaluation cancelled")
            # The deadline applies UNCONDITIONALLY. This used to read "done > 0"
            # so the EV would have at least some samples — but when the hero's
            # combo is hard to deal (heavy blockers), done stays zero, the limit
            # never applies and the loop runs every sample. In a live hand that
            # would blow the time to act. If not one sample lands, the caller
            # sees samples == 0 and handles it.
            if deadline is not None and time.monotonic() >= deadline:
                break
            deal = self._deal_with_hero(hero_combo, rng)
            if deal is None:
                continue
            holes, full_board = deal
            self._scores = tuple(score_7(list(hole) + list(full_board)) for hole in holes)
            best = max(self._scores)
            winners = sum(score == best for score in self._scores)
            if self._scores[0] == best:
                equity += 1.0 / winners
            if self.config.scenario == "checked_to":
                action_ev[0] += self._terminal(
                    tuple(range(len(holes))), (0.0,) * len(holes),
                )[0]
                for index, (name, amount) in enumerate(self.bet_menu(), start=1):
                    commitments = [0.0] * len(holes)
                    commitments[0] = amount
                    action_ev[index] += self._evaluate_responses(
                        tuple(range(1, len(holes))), 0, ("lead", name), (0,),
                        tuple(commitments), amount, holes,
                        allow_raise=self.opponents_may_raise(amount),
                        allow_allin=self.config.opponent_allin,
                        raise_rounds=self.config.max_raise_rounds,
                    )[0]
            else:
                aggressor = len(holes) - 1
                fold_commitments = [0.0] * len(holes)
                fold_commitments[aggressor] = self.config.facing_bet_bb
                action_ev[0] += self._terminal((aggressor,), tuple(fold_commitments))[0]
                call_commitments = list(fold_commitments)
                call_commitments[0] = min(
                    self.config.facing_bet_bb, self.config.player_stacks[0],
                )
                action_ev[1] += self._evaluate_responses(
                    tuple(range(1, aggressor)), 0, ("facing", CALL), (0, aggressor),
                    tuple(call_commitments), self.config.facing_bet_bb, holes,
                    allow_raise=self.opponents_may_raise(self.config.facing_bet_bb),
                    allow_allin=self.config.opponent_allin,
                    raise_rounds=self.config.max_raise_rounds,
                )[0]
                for offset, (name, amount) in enumerate(self.raise_menu(), start=2):
                    raise_commitments = list(fold_commitments)
                    raise_commitments[0] = amount
                    action_ev[offset] += self._evaluate_responses(
                        tuple(range(1, aggressor)) + (aggressor,), 0,
                        ("facing", name), (0,), tuple(raise_commitments),
                        amount, holes,
                        allow_raise=self.opponents_may_raise(amount),
                        allow_allin=self.config.opponent_allin,
                        raise_rounds=self.config.max_raise_rounds - 1,
                    )[0]
            done += 1
            if progress and (done == 1 or done % 100 == 0):
                progress(done, samples, "evaluating")
        if not done:
            raise ValueError(
                "could not evaluate the chosen hero combo — the budget ran out, "
                "or cards in play block it",
            )
        result: dict[str, float | int] = {
            "samples": done,
            "equity": equity / done,
        }
        for action, value in zip(self.hero_actions, action_ev, strict=True):
            result[f"ev_{action}_bb"] = value / done
        return result

    @property
    def hero_actions(self) -> tuple[str, ...]:
        # The menu is the single source of truth: if hero_actions and the tree
        # disagreed on how many actions there are, _node would only hit it at
        # runtime.
        if self.config.scenario != "facing_wager":
            return (CHECK, *(name for name, _amount in self.bet_menu()))
        return (FOLD, CALL, *(name for name, _amount in self.raise_menu()))

    def hero_class_matrix(self) -> dict[str, dict]:
        """The whole hero range's strategy, by the 169 hand classes.

        Per-combo frequencies are weighted by visit count, because that is how
        often the combo comes up in the range. A plain average would give a
        combo seen five times the same weight as one seen a hundred.
        """
        from pokersolver.ranges.hand_grid import hand_class

        root = ("facing",) if self.config.scenario == "facing_wager" else ()
        actions = self.hero_actions
        buckets: dict[str, dict] = {}
        for key, node in self.nodes.items():
            if key.player != HERO or key.history != root:
                continue
            if len(node.regrets) != len(actions):
                continue
            name = hand_class(*key.combo)
            if name is None:
                continue
            slot = buckets.setdefault(
                name, {"weight": 0.0, "sums": [0.0] * len(actions), "combos": 0},
            )
            weight = float(node.visits) or 1.0
            average = node.average_strategy()
            slot["weight"] += weight
            slot["combos"] += 1
            for index in range(len(actions)):
                slot["sums"][index] += weight * average[index]
        return {
            name: {
                "frequencies": [value / slot["weight"] for value in slot["sums"]],
                "combos": slot["combos"],
                "visits": int(slot["weight"]),
            }
            for name, slot in buckets.items() if slot["weight"] > 0
        }

    def _aggregate_player(self, player: int) -> dict[str, float | int | None]:
        entries = [(node, node.action_value_weight) for key, node in self.nodes.items()
                   if key.player == player]
        total_weight = sum(weight for _node, weight in entries)
        if total_weight <= 0:
            return {"actions": (FOLD, CALL), "frequencies": (None, None), "ev_bb": (None, None),
                    "first": None, "second": None, "ev_first_bb": None, "ev_second_bb": None,
                    "nodes": 0, "samples": 0.0}
        action_count = max(len(node.regrets) for node, _weight in entries)
        frequencies = tuple(sum((node.average_strategy()[index] if index < len(node.regrets) else 0.0) * weight for node, weight in entries) / total_weight for index in range(action_count))
        ev_values = []
        for index in range(action_count):
            usable = [(values[index], weight) for values, weight in ((node.action_values(), weight) for node, weight in entries) if index < len(values) and values[index] is not None]
            ev_values.append(sum(value * weight for value, weight in usable) / sum(weight for _value, weight in usable) if usable else None)
        # The node remembers the action names. They used to be derived from
        # their COUNT via a hardcoded table that could not express raise@250 —
        # with several sizes, an opponent's raise simply vanished.
        named = next((node.names for node, _weight in entries
                      if len(node.names) == action_count), ())
        actions = named or ((FOLD, CALL, RAISE, ALLIN)[:action_count]
                            if action_count <= 4 else tuple(
                                f"action{index}" for index in range(action_count)))
        return {
            "actions": actions, "frequencies": frequencies, "ev_bb": tuple(ev_values),
            "first": frequencies[0], "second": frequencies[1] if len(frequencies) > 1 else None,
            "ev_first_bb": ev_values[0], "ev_second_bb": ev_values[1] if len(ev_values) > 1 else None,
            "nodes": len(entries),
            "samples": total_weight,
        }

    def result(
        self,
        hero_combo: tuple[str, str],
        evaluation: dict[str, float | int],
        *,
        elapsed_s: float | None = None,
    ) -> dict:
        root_history = ("facing",) if self.config.scenario == "facing_wager" else ()
        key = _InfoSet(0, root_history, _canonical(hero_combo))
        hero_node = self.nodes.get(key)
        if hero_node is None:
            # With no node there is no strategy to return. This used to hand
            # back None frequencies, which rendered as blanks in a UI and only
            # failed much later in the maths, far from the actual cause.
            raise ValueError(
                f"combo {''.join(hero_combo)} carries no weight in the hero range "
                f"(nebo ji blokuje board {' '.join(self.config.board)}), "
                "so the solver has no strategy for it",
            )
        hero_policy = hero_node.average_strategy()
        hero_result: dict[str, object] = {
            "actions": self.hero_actions,
            "frequencies": hero_policy,
            "ev_bb": tuple(evaluation[f"ev_{action}_bb"] for action in self.hero_actions),
            "information_sets": sum(key.player == 0 for key in self.nodes),
            # How often THIS combo's infoset came up during training. The
            # frequencies are learned only from these visits, while EV is
            # sampled separately with the hero's hand fixed — which is why the
            # two can disagree.
            "visits": hero_node.visits,
            "samples_per_action": round(hero_node.visits / max(1, len(hero_policy)), 1),
        }
        # How many bb are given up by playing the listed mix instead of the best
        # action. At equilibrium every mixed action has the same EV, so this goes
        # to zero. A large value means the mix is under-solved — which is exactly
        # what "the solver recommends an action its own EV says loses" looks like.
        known = [(freq, ev) for freq, ev in zip(hero_policy, hero_result["ev_bb"])
                 if ev is not None]
        if known:
            mix_ev = sum(freq * ev for freq, ev in known)
            best_ev = max(ev for _freq, ev in known)
            hero_result["mix_ev_bb"] = round(mix_ev, 4)
            hero_result["best_ev_bb"] = round(best_ev, 4)
            hero_result["mix_ev_loss_bb"] = round(max(0.0, best_ev - mix_ev), 4)
        for index, action in enumerate(self.hero_actions):
            hero_result[action] = hero_policy[index]
            hero_result[f"ev_{action}_bb"] = evaluation[f"ev_{action}_bb"]
        opponents = []
        for player in range(1, len(self.config.ranges)):
            aggregate = self._aggregate_player(player)
            opponent = {
                "name": self.config.names[player],
                "actions": aggregate["actions"],
                "frequencies": aggregate["frequencies"],
                "ev_bb": aggregate["ev_bb"],
                "fold": aggregate["first"],
                "call": aggregate["second"],
                "ev_fold_bb": aggregate["ev_first_bb"],
                "ev_call_bb": aggregate["ev_second_bb"],
                "information_sets": aggregate["nodes"],
                "samples": aggregate["samples"],
            }
            if len(aggregate["frequencies"]) > 2:
                if RAISE in aggregate["actions"]:
                    index = aggregate["actions"].index(RAISE)
                    opponent["raise"] = aggregate["frequencies"][index]
                    opponent["ev_raise_bb"] = aggregate["ev_bb"][index]
                if ALLIN in aggregate["actions"]:
                    index = aggregate["actions"].index(ALLIN)
                    opponent["allin"] = aggregate["frequencies"][index]
                    opponent["ev_allin_bb"] = aggregate["ev_bb"][index]
            opponents.append(opponent)
        return {
            "format": "pokersolver-multiway-cfr-v1",
            "method": "sampled-chance regret-minimisation self-play",
            "certified_gto": False,
            "warning": (
                "A limited multiplayer self-play policy, not certified GTO. "
                + (
                    "The hero faces a bet with fold/call/raise; opponents have "
                    "fold/call and a limited raise."
                    if self.config.scenario == "facing_wager"
                    else "The hero picks check/bet; opponents answer fold/call."
                )
            ),
            "players": len(self.config.ranges),
            "names": self.config.names,
            "board": self.config.board,
            "pot_bb": self.config.pot_bb,
            "bet_bb": min(self.config.bet_bb, self.config.player_stacks[0]),
            "requested_bet_bb": self.config.bet_bb,
            "scenario": self.config.scenario,
            "facing_bet_bb": self.config.facing_bet_bb,
            "raise_to_bb": self.config.raise_to_bb,
            "stacks_bb": self.config.player_stacks,
            "iterations": self.iterations,
            "valid_deals": self.valid_deals,
            "seed": self.config.seed,
            "elapsed_s": elapsed_s,
            "hero_combo": hero_combo,
            "hero_equity": evaluation["equity"],
            "hero_range_equity": self._hero_equity_sum / self.valid_deals,
            "evaluation_samples": evaluation["samples"],
            "evaluation_unvisited": self.evaluation_unvisited,
            "hero_matrix": self.hero_class_matrix(),
            "hero_actions": list(self.hero_actions),
            "hero": hero_result,
            "opponents": opponents,
            "information_sets": len(self.nodes),
            "limitations": [
                (
                    "one fixed facing-bet and raise-to sizing"
                    if self.config.scenario == "facing_wager"
                    else "one fixed bet sizing"
                ),
                ("at most one re-raise per betting round"
                 if self.config.opponent_raise_to_bb > 0 else "no further re-raises"),
                "once the action closes, the board runs out to showdown",
                "sampled runouts and self-play convergence",
            ],
        }

    def solve_and_result(
        self,
        hero_combo: tuple[str, str],
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        convergence_deals: int = 0,
    ) -> dict:
        """Solve the spot and return the result.

        ``convergence_deals > 0`` adds a ``convergence`` block with per-player
        best-response gain. It is off by default, because measuring costs extra
        time and the caller should choose to spend it.
        """
        started = time.perf_counter()
        self.solve(cancelled=cancelled, progress=progress)
        evaluation = self.evaluate_hero_combo(
            hero_combo,
            min(10_000, max(1_000, self.config.iterations // 5)),
            cancelled=cancelled,
            progress=progress,
        )
        solve_s = round(time.perf_counter() - started, 3)
        payload = self.result(hero_combo, evaluation, elapsed_s=solve_s)
        payload["solve_s"] = solve_s
        payload["convergence_s"] = 0.0
        if convergence_deals > 0:
            from .diagnostics import measure, sample_deals

            if progress:
                progress(0, convergence_deals, "measuring")
            measure_started = time.perf_counter()
            deals = sample_deals(self, convergence_deals, seed=self.config.seed ^ 0xE7A1)
            payload["convergence"] = measure(self, deals, label="holdout").as_dict(
                self.config.pot_bb,
            )
            payload["convergence_s"] = round(time.perf_counter() - measure_started, 3)
        # elapsed_s has to be ALL the time the caller waited. Convergence is
        # measured after the solve and can take longer than the solve itself —
        # reporting only the solve meant waiting 32s on 20k deals and seeing 1.1s.
        payload["elapsed_s"] = round(time.perf_counter() - started, 3)
        return payload
