"""Measuring convergence of multiway CFR.

Not "exploitability": with three or more players there is no minimax value, so
exploitability in the two-player sense is undefined. (Rake is beside the point
here — even a raked heads-up game is non-zero-sum.) What is measured instead is
**best-response gain per player**: how many big blinds a player would gain by
best-responding while everyone else holds their average strategy. That is the
epsilon of an epsilon-Nash profile, and it is defined for any ``n``.

The honest caveat, which belongs next to any number this produces: CFR does not
guarantee convergence to Nash for ``n >= 3``. Minimising external regret yields
a coarse correlated equilibrium, and the product of the average marginals need
not be a Nash profile. **BR gain may therefore stall at a positive value, and
that is theoretically expected rather than a bug.** ``avg_regret_bb`` is
reported alongside it: that one must go to zero, and it is free from the
regrets already stored.

The tree is walked again here, but the payoffs come from
``MultiwayCfrSolver._terminal`` — side pots and rake are the risky part and
there is no sense in having two copies of them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from pokersolver.cards import score_7

from .solver import (
    ALLIN,
    CALL,
    FOLD,
    HERO,
    HERO_RESPONSE,
    RAISE,
    MultiwayCfrSolver,
    _canonical,
    _InfoSet,
)

Deal = tuple[tuple[tuple[str, str], ...], tuple[tuple, ...]]


@dataclass(frozen=True)
class BestResponseReport:
    """One measurement, over one particular pool of deals."""

    label: str
    deals: int
    ev_bb: tuple[float, ...]
    gain_bb: tuple[float, ...]
    avg_regret_bb: tuple[float, ...]
    gain_raw_bb: tuple[float, ...] = ()
    infosets: int = 0

    @property
    def max_gain_bb(self) -> float:
        return max(self.gain_bb) if self.gain_bb else 0.0

    @property
    def reliability(self) -> str:
        """How much this number can be trusted.

        What decides it is deals per infoset. Splitting the sample removes the
        bias, but on thin data the variance remains large — and a number that
        cannot be trusted must not present itself as a result.
        """
        if not self.infosets:
            return "unknown"
        per_infoset = self.deals / self.infosets
        if per_infoset >= 10:
            return "ok"
        return "low" if per_infoset >= 3 else "insufficient"

    @property
    def resolved(self) -> bool:
        """Can this estimate be told apart from noise at all?

        A split-sample estimate can come out negative when the true gain is near
        zero and there are few deals. A clipped ``0.000`` then means
        "indistinguishable from noise", not "converged" — two very different
        messages.
        """
        return bool(self.gain_raw_bb) and max(self.gain_raw_bb) > 0.0

    def max_gain_pct_pot(self, pot_bb: float) -> float:
        return 100.0 * self.max_gain_bb / pot_bb if pot_bb > 0 else 0.0

    def as_dict(self, pot_bb: float) -> dict:
        return {
            "label": self.label,
            "deals": self.deals,
            "infosets": self.infosets,
            "deals_per_infoset": round(self.deals / self.infosets, 2) if self.infosets else None,
            "ev_bb": list(self.ev_bb),
            "gain_bb": list(self.gain_bb),
            "gain_raw_bb": list(self.gain_raw_bb),
            "avg_regret_bb": list(self.avg_regret_bb),
            "max_gain_bb": self.max_gain_bb,
            "max_gain_pct_pot": self.max_gain_pct_pot(pot_bb),
            "resolved": self.resolved,
            "reliability": self.reliability,
        }


def sample_deals(solver: MultiwayCfrSolver, count: int, seed: int) -> list[Deal]:
    """A fixed pool of deals, shared between measurements.

    Sharing it matters: without that, the difference between two configurations
    would be mostly sampling noise rather than signal.
    """
    if count <= 0:
        raise ValueError("the pool of deals cannot be empty")
    saved = solver.rng
    solver.rng = random.Random(seed)
    try:
        deals: list[Deal] = []
        for _attempt in range(count * 4):
            if len(deals) >= count:
                break
            dealt = solver._deal()
            if dealt is None:
                continue
            holes, board = dealt
            scores = tuple(score_7(list(hole) + list(board)) for hole in holes)
            deals.append((holes, scores))
    finally:
        solver.rng = saved
    if not deals:
        raise ValueError("cannot draw a conflict-free deal from these ranges")
    return deals


class _Walk:
    """Walk the tree under a policy, optionally collecting counterfactual values.

    ``policy(key, action_count)`` returns the distribution at a node. When
    ``target`` is given, ``cfv`` accumulates, for each of that player's
    infosets,
    ``w_d * pi_ostatnich * hodnota_akce`` — tedy counterfactual hodnota, ne
    the value conditioned on reaching the node.
    """

    def __init__(self, solver: MultiwayCfrSolver, policy, target: int | None) -> None:
        self.solver = solver
        self.policy = policy
        self.target = target
        self.cfv: dict[_InfoSet, list[float]] = {}
        self.reached: dict[_InfoSet, float] = {}

    # -- helpers -----------------------------------------------------------
    def _strategy(self, key: _InfoSet, action_count: int) -> tuple[float, ...]:
        return self.policy(key, action_count)

    def _record(
        self, key: _InfoSet, values: list[tuple[float, ...]], weight: float,
    ) -> None:
        if self.target is None or key.player != self.target:
            return
        slot = self.cfv.setdefault(key, [0.0] * len(values))
        for index, utility in enumerate(values):
            slot[index] += weight * utility[self.target]
        self.reached[key] = self.reached.get(key, 0.0) + weight

    def _mix(self, strategy, values, seats: int) -> tuple[float, ...]:
        return tuple(
            sum(strategy[index] * values[index][seat] for index in range(len(strategy)))
            for seat in range(seats)
        )

    # -- root --------------------------------------------------------------
    def root(self, holes, weight: float) -> tuple[float, ...]:
        config = self.solver.config
        seats = len(holes)
        if config.scenario == "facing_wager":
            return self._facing_root(holes, weight, seats)

        menu = self.solver.bet_menu()
        key = _InfoSet(HERO, (), _canonical(holes[HERO]))
        strategy = self._strategy(key, 1 + len(menu))
        values = [self.solver._terminal(tuple(range(seats)), (0.0,) * seats)]
        for index, (name, amount) in enumerate(menu, start=1):
            commitments = [0.0] * seats
            commitments[HERO] = amount
            child = weight if self.target == HERO else weight * strategy[index]
            values.append(self._responses(
                tuple(range(1, seats)), 0, ("lead", name), (HERO,), tuple(commitments),
                amount, holes, child,
                allow_raise=self.solver.opponents_may_raise(amount),
                allow_allin=config.opponent_allin,
                raise_rounds=config.max_raise_rounds,
            ))
        self._record(key, values, weight)
        return self._mix(strategy, values, seats)

    def _facing_root(self, holes, weight: float, seats: int) -> tuple[float, ...]:
        config = self.solver.config
        solver = self.solver
        aggressor = seats - 1
        key = _InfoSet(HERO, ("facing",), _canonical(holes[HERO]))
        strategy = self._strategy(key, len(solver.hero_actions))

        fold_commitments = [0.0] * seats
        fold_commitments[aggressor] = config.facing_bet_bb
        fold = solver._terminal((aggressor,), tuple(fold_commitments))

        call_commitments = list(fold_commitments)
        call_commitments[HERO] = min(config.facing_bet_bb, config.player_stacks[HERO])
        call_weight = weight if self.target == HERO else weight * strategy[1]
        call = self._responses(
            tuple(range(1, aggressor)), 0, ("facing", CALL), (HERO, aggressor),
            tuple(call_commitments), config.facing_bet_bb, holes, call_weight,
            allow_raise=solver.opponents_may_raise(config.facing_bet_bb),
            allow_allin=config.opponent_allin,
            raise_rounds=config.max_raise_rounds,
        )

        values = [fold, call]
        for offset, (name, amount) in enumerate(solver.raise_menu(), start=2):
            raise_commitments = list(fold_commitments)
            raise_commitments[HERO] = amount
            raise_weight = weight if self.target == HERO else weight * strategy[offset]
            values.append(self._responses(
                tuple(range(1, aggressor)) + (aggressor,), 0, ("facing", name),
                (HERO,), tuple(raise_commitments), amount, holes,
                raise_weight,
                allow_raise=solver.opponents_may_raise(amount),
                allow_allin=config.opponent_allin,
                raise_rounds=config.max_raise_rounds - 1,
            ))
        self._record(key, values, weight)
        return self._mix(strategy, values, seats)

    # -- opponent responses -------------------------------------------------
    def _responses(
        self, responders, responder_index, history, active, commitments, target,
        holes, weight, *, allow_raise: bool, allow_allin: bool, raise_rounds: int = 1,
    ) -> tuple[float, ...]:
        solver = self.solver
        config = solver.config
        seats = len(holes)
        if responder_index == len(responders):
            return self._close_round(
                history, active, commitments, target, holes, weight, raise_rounds,
            )

        player = responders[responder_index]
        maximum = commitments[player] + config.player_stacks[player]
        if maximum <= commitments[player] + 1e-12:
            token = CALL if commitments[player] > 0 else FOLD
            next_active = active + (player,) if commitments[player] > 0 else active
            return self._responses(
                responders, responder_index + 1, history + (token,), next_active,
                commitments, target, holes, weight,
                allow_raise=allow_raise, allow_allin=allow_allin,
                raise_rounds=raise_rounds,
            )

        raises = (solver.opponent_raise_menu(target, maximum)
                  if allow_raise and raise_rounds > 0 else ())
        amounts = {CALL: min(target, maximum), **{name: amount for name, amount in raises}}
        available = [FOLD, CALL, *(name for name, _amount in raises)]
        if allow_allin and maximum > target + 1e-12:
            available.append(ALLIN)
            amounts[ALLIN] = maximum

        key = _InfoSet(player, history, _canonical(holes[player]))
        strategy = self._strategy(key, len(available))
        values: list[tuple[float, ...]] = []
        for index, action in enumerate(available):
            next_commitments = list(commitments)
            if action != FOLD:
                next_commitments[player] = amounts[action]
            is_raise = action.startswith(RAISE)
            next_target = (
                amounts[action] if action != FOLD and (is_raise or action == ALLIN)
                else target
            )
            # Only players OTHER than the target scale the weight; including the
            # target would turn a counterfactual value into one conditioned on
            # their own strategy.
            child_weight = weight if self.target == player else weight * strategy[index]
            values.append(self._responses(
                responders, responder_index + 1, history + (action,),
                active if action == FOLD else active + (player,),
                tuple(next_commitments), next_target, holes, child_weight,
                allow_raise=allow_raise and not is_raise and action != ALLIN,
                allow_allin=allow_allin and action != ALLIN,
                raise_rounds=raise_rounds - 1 if is_raise else raise_rounds,
            ))
        self._record(key, values, weight)
        return self._mix(strategy, values, seats)

    def _close_round(
        self, history, active, commitments, target, holes, weight: float,
        raise_rounds: int = 0,
    ) -> tuple[float, ...]:
        solver = self.solver
        config = solver.config
        seats = len(holes)
        if not solver._hero_must_respond(active, commitments, target):
            return solver._terminal(active, commitments)

        maximum = config.player_stacks[HERO]
        menu = solver.reraise_menu(target) if raise_rounds > 0 else ()
        key = _InfoSet(HERO, history + (HERO_RESPONSE,), _canonical(holes[HERO]))
        strategy = self._strategy(key, 2 + len(menu))
        values = [
            solver._terminal(tuple(seat for seat in active if seat != HERO), commitments),
        ]
        call_commitments = list(commitments)
        call_commitments[HERO] = min(target, maximum)
        values.append(solver._terminal(active, tuple(call_commitments)))
        responders = tuple(seat for seat in active if seat != HERO)
        for offset, (name, amount) in enumerate(menu, start=2):
            next_commitments = list(commitments)
            next_commitments[HERO] = amount
            child = weight if self.target == HERO else weight * strategy[offset]
            values.append(self._responses(
                responders, 0, history + (HERO_RESPONSE, name), (HERO,),
                tuple(next_commitments), amount, holes, child,
                allow_raise=self.solver.opponents_may_raise(amount),
                allow_allin=config.opponent_allin,
                raise_rounds=raise_rounds - 1,
            ))
        self._record(key, values, weight)
        return self._mix(strategy, values, seats)


def _average_policy(solver: MultiwayCfrSolver, overrides: dict[_InfoSet, int]):
    """The average strategy; ``overrides`` forces a pure action for best response."""

    def policy(key: _InfoSet, action_count: int) -> tuple[float, ...]:
        forced = overrides.get(key)
        if forced is not None:
            return tuple(1.0 if i == forced else 0.0 for i in range(action_count))
        node = solver.nodes.get(key)
        if node is not None and len(node.regrets) == action_count:
            return node.average_strategy()
        return tuple(1.0 / action_count for _index in range(action_count))

    return policy


def _expected_values(
    solver: MultiwayCfrSolver, deals: list[Deal], overrides: dict[_InfoSet, int],
) -> tuple[float, ...]:
    seats = len(solver.config.ranges)
    totals = [0.0] * seats
    walk = _Walk(solver, _average_policy(solver, overrides), target=None)
    for holes, scores in deals:
        solver._scores = scores
        values = walk.root(holes, 1.0)
        for seat in range(seats):
            totals[seat] += values[seat]
    return tuple(total / len(deals) for total in totals)


def best_response_gain(
    solver: MultiwayCfrSolver, player: int, deals: list[Deal],
) -> float:
    """How many bb a player gains by best-responding to the average profile.

    The maximum is taken **at infoset level**, over counterfactual values summed
    across every deal that reaches the infoset. Taking it per deal would model a
    clairvoyant player who sees the opponents' cards, and overstate the number.

    Infosets are processed deepest-first: a player can have two decision nodes
    on one path, so the shallower one has to account for already
    best-responding at the deeper one.

    A limitation worth stating: only infosets that training actually visited and
    stored in ``solver.nodes`` are searched. A node sampling never reached is
    invisible here and contributes no gain, so the number is a lower bound.

    **The sample is split in half.** The best response is chosen on one half and
    scored on the other. Without that the estimate is systematically optimistic:
    a maximum over noisy values always lands too high, and the error grows with
    the number of infosets. On a tree with 1840 infosets the same strategy
    measured 3.94bb over 400 deals and 0.28bb over 20,000 — nearly all of that
    difference was bias.
    """
    if len(deals) < 2:
        raise ValueError("measuring best response needs at least two deals")
    half = len(deals) // 2
    folds = ((deals[:half], deals[half:]), (deals[half:], deals[:half]))
    gains = [
        _one_sided_gain(solver, player, pick_on, score_on)
        for pick_on, score_on in folds
    ]
    return sum(gains) / len(gains)


def _one_sided_gain(
    solver: MultiwayCfrSolver, player: int, pick_on: list[Deal], score_on: list[Deal],
) -> float:
    """Choose the best response on ``pick_on`` and score it on ``score_on``."""
    depths = sorted(
        {len(key.history) for key in solver.nodes if key.player == player},
        reverse=True,
    )
    overrides: dict[_InfoSet, int] = {}
    for depth in depths:
        walk = _Walk(solver, _average_policy(solver, overrides), target=player)
        for holes, scores in pick_on:
            solver._scores = scores
            walk.root(holes, 1.0)
        for key, values in walk.cfv.items():
            if len(key.history) == depth:
                overrides[key] = max(range(len(values)), key=values.__getitem__)

    best = _expected_values(solver, score_on, overrides)[player]
    average = _expected_values(solver, score_on, {})[player]
    return best - average


def average_regret(solver: MultiwayCfrSolver) -> tuple[float, ...]:
    """Average external regret per player — this must go to zero, unlike BR gain."""
    seats = len(solver.config.ranges)
    totals = [0.0] * seats
    for key, node in solver.nodes.items():
        totals[key.player] += max([0.0, *(max(0.0, r) for r in node.regrets)])
    iterations = max(1, solver.iterations)
    return tuple(total / iterations for total in totals)


def measure(
    solver: MultiwayCfrSolver, deals: list[Deal], *, label: str = "train",
) -> BestResponseReport:
    """Measure a profile over a pool of deals."""
    seats = len(solver.config.ranges)
    raw = tuple(best_response_gain(solver, seat, deals) for seat in range(seats))
    return BestResponseReport(
        label=label,
        deals=len(deals),
        ev_bb=_expected_values(solver, deals, {}),
        gain_bb=tuple(max(0.0, value) for value in raw),
        gain_raw_bb=raw,
        avg_regret_bb=average_regret(solver),
        infosets=len(solver.nodes),
    )
