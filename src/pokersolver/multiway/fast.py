"""Vectorised multiway CFR — the same tree, but every deal at once.

The reference solver handles one deal per iteration and walks the whole tree in
Python. Here the pool of deals is sampled once up front and each node computes
all of them as a single array, which drops the inner loop into numpy.

The speedup comes less from numpy than from computing **the showdown once**.
The reference solver re-evaluates the cards every iteration even though the
deals never change; here the payoffs at every terminal node are computed while
building the pool. Measured against the reference on identical answers: 15x for
two players, 9.5x for three, 9x for four.

A fixed pool has a cost worth knowing: **it splits the error into optimisation
error, which falls with iterations, and sampling error, which falls with pool
size and no amount of iterating removes**. With 1000 deals, eight seeds gave
answers from 0.1% to 99.9% check — the solver solved its sample, not the game.
From 20,000 deals it settles under 5%. If you need a large pool and cheap
passes, ``solve(batch=...)`` takes a different subset each pass.

The measured trade-off at live budgets (two players, deviation from a 1M-iteration
reference): a pool of 20,000 without batching is off by 0.6-4.6% at 2s and
0.5-1.2% at 5s. Larger pools do worse in the same time, because they cost
iterations.

Needs numpy: ``pip install -e '.[cfr]'``.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from .evaluator import CARD_INDEX, score_batch

from .solver import (
    ALLIN,
    CALL,
    CHECK,
    FOLD,
    HERO,
    HERO_RESPONSE,
    RAISE,
    MultiwayCfrConfig,
    _canonical,
)

try:  # numpy is an optional extra
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
    raise ModuleNotFoundError(
        "the vectorised solver needs numpy: pip install -e '.[cfr]'",
    ) from exc


def _popcount(values: "np.ndarray") -> "np.ndarray":
    """Population count; ``np.bitwise_count`` only exists from numpy 2.0."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    total = np.zeros_like(values)
    for bit in range(52):
        total += (values >> bit) & 1
    return total


@dataclass
class _Table:
    """Regrets and average strategy for one node, per combo of the player."""

    regrets: "np.ndarray"        # (combos, actions)
    strategy_sum: "np.ndarray"   # (combos, actions)
    actions: tuple[str, ...]

    def current(self) -> "np.ndarray":
        positive = np.maximum(self.regrets, 0.0)
        total = positive.sum(axis=1, keepdims=True)
        uniform = 1.0 / positive.shape[1]
        return np.where(total > 1e-15, positive / np.maximum(total, 1e-15), uniform)

    def average(self) -> "np.ndarray":
        total = self.strategy_sum.sum(axis=1, keepdims=True)
        return np.where(total > 1e-15, self.strategy_sum / np.maximum(total, 1e-15),
                        self.current())


class VectorSolver:
    """The same model as :class:`MultiwayCfrSolver`, over arrays."""

    def __init__(self, config: MultiwayCfrConfig, deals: int = 20_000):
        self.config = config
        self.deals = deals
        self.players = len(config.ranges)
        self.iterations = 0
        # Per player: the list of their combos, and an index into it per deal.
        self._combos: list[list[tuple[str, str]]] = []
        self._combo_index: list[dict[tuple[str, str], int]] = []
        self._deal_combo: list["np.ndarray"] = []
        self._tables: dict[tuple[int, tuple[str, ...]], _Table] = {}
        self._terminals: dict[tuple, "np.ndarray"] = {}
        self._scores: "np.ndarray" | None = None
        # The pool is sampled once; ``_batch`` selects the active slice of it.
        self._pool_combo: list["np.ndarray"] = []
        self._pool_scores: "np.ndarray" | None = None
        self._pool_size = 0
        self._batch: "np.ndarray | None" = None
        # Evaluation mode: the average strategy, no learning.
        self._eval = False
        self._root_values: list["np.ndarray"] = []
        self.eval_unvisited = 0
        self.eval_equity = 0.0
        self.eval_samples = 0
        self._build_deals()

    # -- setup -----------------------------------------------------------
    def _build_deals(self) -> None:
        config = self.config
        rng = random.Random(config.seed)
        board = set(config.board)
        for entries in config.ranges:
            combos = [combo for combo, _w in entries]
            self._combos.append(combos)
            self._combo_index.append({_canonical(c): i for i, c in enumerate(combos)})

        generator = np.random.default_rng(config.seed)
        board_index = np.asarray([CARD_INDEX[c] for c in config.board],
                                 dtype=np.int64)
        board_mask = int(np.bitwise_or.reduce(np.int64(1) << board_index)) \
            if len(board_index) else 0
        missing = 5 - len(config.board)

        # Each combo's cards as deck indexes — two per combo.
        combo_cards = [
            np.asarray([[CARD_INDEX[a], CARD_INDEX[b]] for a, b in combos],
                       dtype=np.int64)
            for combos in self._combos
        ]
        probabilities = []
        for entries in config.ranges:
            weight = np.asarray([w for _c, w in entries], dtype=np.float64)
            probabilities.append(weight / weight.sum())

        kept_combo: list[list["np.ndarray"]] = [[] for _p in range(self.players)]
        kept_scores: list["np.ndarray"] = []
        have = 0
        rounds = 0
        while have < self.deals and rounds < 40:
            rounds += 1
            draw = max(1024, int((self.deals - have) * 1.6))
            choice = [generator.choice(len(self._combos[p]), size=draw,
                                       p=probabilities[p])
                      for p in range(self.players)]
            holes = [combo_cards[p][choice[p]] for p in range(self.players)]
            # Collisions show up in a deck bitmask: when a card repeats, the
            # mask has fewer bits set than the number of cards dealt.
            used = np.full(draw, board_mask, dtype=np.int64)
            for hole in holes:
                used |= (np.int64(1) << hole[:, 0]) | (np.int64(1) << hole[:, 1])
            expected = len(config.board) + 2 * self.players
            ok = _popcount(used) == expected
            if not ok.any():
                continue
            used, choice = used[ok], [c[ok] for c in choice]
            holes = [h[ok] for h in holes]
            live = int(ok.sum())

            # Completing the board: give every card a random key, push the used
            # ones to the end and take the ``missing`` smallest. This also works
            # on turn and river, where nothing is drawn.
            keys = generator.random((live, 52))
            blocked = ((used[:, None] >> np.arange(52)[None, :]) & 1).astype(bool)
            keys[blocked] = np.inf
            runout = np.argsort(keys, axis=1)[:, :missing] if missing else \
                np.zeros((live, 0), dtype=np.int64)
            full = np.concatenate(
                [np.tile(board_index, (live, 1)), runout], axis=1)

            scores = np.stack([
                score_batch(np.concatenate([holes[p], full], axis=1))
                for p in range(self.players)
            ], axis=1)
            for p in range(self.players):
                kept_combo[p].append(choice[p])
            kept_scores.append(scores)
            have += live

        if not kept_scores:
            raise ValueError("cannot draw a conflict-free deal from these ranges")
        self._pool_combo = [np.concatenate([c.astype(np.int64) for c in kept_combo[p]])[:self.deals]
                            for p in range(self.players)]
        # Scores are only ever compared within a deal, so they are stored raw;
        # converting them to ranks would buy nothing.
        self._pool_scores = np.concatenate(kept_scores)[:self.deals]
        self.deals = self._pool_size = len(self._pool_scores)
        self._deal_combo = list(self._pool_combo)
        self._scores = self._pool_scores

    def _table(self, player: int, history: tuple[str, ...],
               actions: tuple[str, ...]) -> _Table:
        key = (player, history)
        table = self._tables.get(key)
        if table is None:
            shape = (len(self._combos[player]), len(actions))
            table = _Table(np.zeros(shape), np.zeros(shape), actions)
            self._tables[key] = table
        return table

    def _terminal(self, active: tuple[int, ...],
                  commitments: tuple[float, ...]) -> "np.ndarray":
        """Payoffs for every deal at one terminal node, computed once."""
        key = (active, commitments)
        cached = self._terminals.get(key)
        if cached is not None:
            return cached
        config = self.config
        total = config.pot_bb + sum(commitments)
        raw = total * config.rake_percent / 100.0
        rake = min(raw, config.rake_cap_bb) if config.rake_cap_bb > 0 else raw

        payoff = np.tile(np.asarray([-a for a in commitments], dtype=np.float64),
                         (self.deals, 1))
        pots: list[tuple[float, tuple[int, ...]]] = [(config.pot_bb, active)]
        levels = sorted({a for a in commitments if a > 0})
        previous = 0.0
        for level in levels:
            contributors = tuple(p for p, a in enumerate(commitments) if a + 1e-12 >= level)
            eligible = tuple(p for p in active if p in contributors)
            pots.append(((level - previous) * len(contributors), eligible))
            previous = level
        remaining_rake = rake
        for amount, eligible in pots:
            award = max(0.0, amount - min(amount, remaining_rake))
            remaining_rake = max(0.0, remaining_rake - amount)
            if award <= 0 or not eligible:
                continue
            sub = self._scores[:, list(eligible)]
            mask = sub == sub.max(axis=1, keepdims=True)
            share = award / mask.sum(axis=1)
            for slot, player in enumerate(eligible):
                payoff[:, player] += share * mask[:, slot]
        self._terminals[key] = payoff
        return payoff

    # -- walking the tree --------------------------------------------------
    def _strategy(self, table: _Table, player: int) -> "np.ndarray":
        """The node's strategy for every deal.

        During training the current one from regret matching; during evaluation
        the average, which is what converges to equilibrium. An unvisited node
        falls back to uniform — a fiction training never saw, so it is counted.
        """
        if self._eval:
            if not table.strategy_sum.any():
                self.eval_unvisited += 1
            return table.average()[self._deal_combo[player]]
        return table.current()[self._deal_combo[player]]

    def _mix(self, strategy: "np.ndarray", values: list["np.ndarray"]) -> "np.ndarray":
        out = np.zeros_like(values[0])
        for index, value in enumerate(values):
            out += strategy[:, index, None] * value
        return out

    def _update(self, table: _Table, player: int, strategy: "np.ndarray",
                values: list["np.ndarray"], expected: "np.ndarray",
                reach: "np.ndarray", weight: float) -> None:
        if self._eval:
            return
        counter = np.ones(self.deals)
        for other in range(self.players):
            if other != player:
                counter *= reach[:, other]
        rows = self._deal_combo[player]
        size = table.regrets.shape[0]
        own = reach[:, player]
        # bincount rather than np.add.at: the same operation (sum by combo
        # index) but measured 5x faster — add.at handles repeated indexes one at
        # a time.
        for index, value in enumerate(values):
            table.regrets[:, index] += np.bincount(
                rows, counter * (value[:, player] - expected[:, player]), minlength=size)
            table.strategy_sum[:, index] += np.bincount(
                rows, weight * own * strategy[:, index], minlength=size)

    def _responses(self, responders, index, history, active, commitments, target,
                   reach, weight, *, allow_raise, allow_allin, rounds):
        if index == len(responders):
            return self._close(history, active, commitments, target, reach, weight,
                               rounds)
        player = responders[index]
        config = self.config
        maximum = commitments[player] + config.player_stacks[player]
        if maximum <= commitments[player] + 1e-12:
            token = CALL if commitments[player] > 0 else FOLD
            nxt = active + (player,) if commitments[player] > 0 else active
            return self._responses(responders, index + 1, history + (token,), nxt,
                                   commitments, target, reach, weight,
                                   allow_raise=allow_raise, allow_allin=allow_allin,
                                   rounds=rounds)
        raises = (self._opponent_raises(target, maximum)
                  if allow_raise and rounds > 0 else ())
        amounts = {CALL: min(target, maximum), **dict(raises)}
        available = [FOLD, CALL, *(name for name, _a in raises)]
        if allow_allin and maximum > target + 1e-12:
            available.append(ALLIN)
            amounts[ALLIN] = maximum

        table = self._table(player, history, tuple(available))
        strategy = self._strategy(table, player)
        values = []
        for slot, action in enumerate(available):
            nxt_commit = list(commitments)
            if action != FOLD:
                nxt_commit[player] = amounts[action]
            is_raise = action.startswith(RAISE)
            nxt_reach = reach.copy()
            nxt_reach[:, player] *= strategy[:, slot]
            values.append(self._responses(
                responders, index + 1, history + (action,),
                active if action == FOLD else active + (player,),
                tuple(nxt_commit),
                amounts[action] if action != FOLD and (is_raise or action == ALLIN) else target,
                nxt_reach, weight,
                allow_raise=allow_raise and not is_raise and action != ALLIN,
                allow_allin=allow_allin and action != ALLIN,
                rounds=rounds - 1 if is_raise else rounds,
            ))
        expected = self._mix(strategy, values)
        self._update(table, player, strategy, values, expected, reach, weight)
        return expected

    def _opponent_raises(self, target: float, maximum: float):
        config = self.config
        options = []
        if config.opponent_raise_sizes_pct:
            for pct in config.opponent_raise_sizes_pct:
                amount = min(round(target * pct / 100.0, 4), maximum)
                if amount > target + 1e-12:
                    options.append((f"{RAISE}@{pct:g}", amount))
        else:
            amount = min(config.opponent_raise_to_bb, maximum)
            if amount > target + 1e-12:
                options.append((RAISE, amount))
        seen, kept = set(), []
        for name, amount in options:
            if amount in seen:
                continue
            seen.add(amount)
            kept.append((name, amount))
        return tuple(kept)

    def _close(self, history, active, commitments, target, reach, weight, rounds=0):
        config = self.config
        if not (HERO in active and commitments[HERO] < target - 1e-12
                and config.player_stacks[HERO] > commitments[HERO] + 1e-12):
            return self._terminal(active, commitments)
        maximum = config.player_stacks[HERO]
        menu = self._reraise_menu(target) if rounds > 0 else ()
        table = self._table(HERO, history + (HERO_RESPONSE,),
                            (FOLD, CALL, *(n for n, _a in menu)))
        strategy = self._strategy(table, HERO)
        call_commit = list(commitments)
        call_commit[HERO] = min(target, maximum)
        values = [
            self._terminal(tuple(s for s in active if s != HERO), commitments),
            self._terminal(active, tuple(call_commit)),
        ]
        responders = tuple(s for s in active if s != HERO)
        for slot, (name, amount) in enumerate(menu, start=2):
            nxt_commit = list(commitments)
            nxt_commit[HERO] = amount
            nxt_reach = reach.copy()
            nxt_reach[:, HERO] *= strategy[:, slot]
            values.append(self._responses(
                responders, 0, history + (HERO_RESPONSE, name), (HERO,),
                tuple(nxt_commit), amount, nxt_reach, weight,
                allow_raise=(bool(config.opponent_raise_sizes_pct)
                             or config.opponent_raise_to_bb > amount),
                allow_allin=config.opponent_allin, rounds=rounds - 1,
            ))
        expected = self._mix(strategy, values)
        self._update(table, HERO, strategy, values, expected, reach, weight)
        return expected

    def _reraise_menu(self, target: float) -> tuple[tuple[str, float], ...]:
        config = self.config
        cap = config.player_stacks[HERO]
        options: list[tuple[str, float]] = []
        for pct in config.raise_sizes_pct or ():
            amount = min(round(target * pct / 100.0, 4), cap)
            if amount > target + 1e-12:
                options.append((f"{RAISE}@{pct:g}", amount))
        if config.hero_allin and cap > target + 1e-12:
            options.append((ALLIN, cap))
        seen, kept = set(), []
        for name, amount in options:
            if amount in seen:
                continue
            seen.add(amount)
            kept.append((name, amount))
        return tuple(kept)

    def _root(self, weight: float) -> None:
        config = self.config
        reach = np.ones((self.deals, self.players))
        if config.scenario == "facing_wager":
            self._facing_root(reach, weight)
            return
        menu = _bet_menu(config)
        table = self._table(HERO, (), (CHECK, *(n for n, _a in menu)))
        strategy = self._strategy(table, HERO)
        values = [self._terminal(tuple(range(self.players)), (0.0,) * self.players)]
        for slot, (name, amount) in enumerate(menu, start=1):
            commit = [0.0] * self.players
            commit[HERO] = amount
            nxt = reach.copy()
            nxt[:, HERO] *= strategy[:, slot]
            values.append(self._responses(
                tuple(range(1, self.players)), 0, ("lead", name), (HERO,),
                tuple(commit), amount, nxt, weight,
                allow_raise=(bool(config.opponent_raise_sizes_pct)
                             or config.opponent_raise_to_bb > amount),
                allow_allin=config.opponent_allin, rounds=config.max_raise_rounds,
            ))
        expected = self._mix(strategy, values)
        self._root_values = values
        self._update(table, HERO, strategy, values, expected, reach, weight)

    def _facing_root(self, reach: "np.ndarray", weight: float) -> None:
        config = self.config
        aggressor = self.players - 1
        menu = _raise_menu(config)
        table = self._table(HERO, ("facing",), (FOLD, CALL, *(n for n, _a in menu)))
        strategy = self._strategy(table, HERO)

        fold_commit = [0.0] * self.players
        fold_commit[aggressor] = config.facing_bet_bb
        values = [self._terminal((aggressor,), tuple(fold_commit))]

        call_commit = list(fold_commit)
        call_commit[HERO] = min(config.facing_bet_bb, config.player_stacks[HERO])
        nxt = reach.copy()
        nxt[:, HERO] *= strategy[:, 1]
        values.append(self._responses(
            tuple(range(1, aggressor)), 0, ("facing", CALL), (HERO, aggressor),
            tuple(call_commit), config.facing_bet_bb, nxt, weight,
            allow_raise=(bool(config.opponent_raise_sizes_pct)
                         or config.opponent_raise_to_bb > config.facing_bet_bb),
            allow_allin=config.opponent_allin, rounds=config.max_raise_rounds,
        ))
        for slot, (name, amount) in enumerate(menu, start=2):
            raise_commit = list(fold_commit)
            raise_commit[HERO] = amount
            nxt = reach.copy()
            nxt[:, HERO] *= strategy[:, slot]
            values.append(self._responses(
                tuple(range(1, aggressor)) + (aggressor,), 0, ("facing", name),
                (HERO,), tuple(raise_commit), amount, nxt, weight,
                allow_raise=(bool(config.opponent_raise_sizes_pct)
                             or config.opponent_raise_to_bb > amount),
                allow_allin=config.opponent_allin,
                rounds=config.max_raise_rounds - 1,
            ))
        expected = self._mix(strategy, values)
        self._root_values = values
        self._update(table, HERO, strategy, values, expected, reach, weight)

    # -- public API --------------------------------------------------------
    def solve(self, iterations: int = 60, *, deadline: float | None = None,
              batch: int | None = None,
              cancelled: "Callable[[], bool] | None" = None,
              progress: "Callable[[int, int, str], None] | None" = None) -> None:
        """Train for ``iterations`` passes, optionally in mini-batches.

        ``batch`` takes a random subset of the deal pool each pass. That makes a
        pass cheaper and resamples chance between passes, so the strategy does
        not overfit one particular sample — a measured trap: with 5000 fixed
        deals, two seeds gave 43% and 99.8% check.

        ``deadline`` is soft, as in the reference: the loop stops and the average
        strategy so far stays usable. ``cancelled`` is hard — there nobody wants
        the result.
        """
        picker = random.Random(self.config.seed ^ 0x5EED)
        pool = self.deals
        for step in range(1, iterations + 1):
            if cancelled and cancelled():
                self._use_batch(None)
                raise InterruptedError("multiway solve cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                break
            if batch is not None and batch < pool:
                self._use_batch(np.asarray(picker.sample(range(pool), batch),
                                           dtype=np.int64))
            self._root(float(step))
            self.iterations = step
            if progress and (step == 1 or step % 5 == 0):
                progress(step, iterations, "solving")
        self._use_batch(None)

    def _use_batch(self, index: "np.ndarray | None") -> None:
        """Switch the active subset of deals; payoffs have to be recomputed."""
        if index is None:
            if self._batch is None:
                return
            self._batch = None
            self.deals = self._pool_size
            self._deal_combo = list(self._pool_combo)
            self._scores = self._pool_scores
        else:
            self._batch = index
            self.deals = len(index)
            self._deal_combo = [c[index] for c in self._pool_combo]
            self._scores = self._pool_scores[index]
        self._terminals.clear()

    def hero_evs(self, combo: tuple[str, str], *, deals: int = 20_000,
                 deadline: float | None = None,
                 nonce: int = 0) -> dict[str, float]:
        """EV of each root action in bb, with the hero holding ``combo``.

        Evaluated on its own pool where the hero's hand is **fixed** and only
        the opponents and the runout vary. Slicing the training pool instead
        would leave a few hundred deals for one specific combo and the EV would
        be noise — that pool is spread across the whole hero range.
        """
        index = self._combo_index[HERO].get(_canonical(combo))
        if index is None:
            raise ValueError(f"combo {''.join(combo)} is not in the hero range")
        actions = self.hero_actions()

        saved = (self._deal_combo, self._scores, self.deals, self._batch)
        self._build_eval_deals(index, deals, nonce=nonce)
        self._eval = True
        self.eval_unvisited = 0
        try:
            if deadline is not None and time.monotonic() >= deadline:
                return {}
            # Equity comes from THIS pool, not the training one: only here is
            # the hero's hand fixed, so the number belongs to the combo rather
            # than to the whole range.
            self.eval_equity = _showdown_share(self._scores)
            self.eval_samples = self.deals
            self._root(1.0)
            values = [float(value[:, HERO].mean()) for value in self._root_values]
        finally:
            self._eval = False
            self._deal_combo, self._scores, self.deals, self._batch = saved
            self._terminals.clear()
        return dict(zip(actions, values))

    def range_equity(self) -> float:
        """Equity of the whole hero range over the training pool."""
        return _showdown_share(self._pool_scores)

    def hero_actions(self) -> tuple[str, ...]:
        history = ("facing",) if self.config.scenario == "facing_wager" else ()
        table = self._tables.get((HERO, history))
        if table is None:
            raise ValueError("the hero's root node was never visited — solve() did not run")
        return table.actions

    def _build_eval_deals(self, hero_index: int, count: int, *,
                          nonce: int = 0) -> None:
        """A pool with the hero's hand fixed; only opponents and the board vary.

        ``nonce`` changes the draw without touching the trained strategy, so two
        evaluations with different nonces show how noisy the EV is.
        """
        config = self.config
        generator = np.random.default_rng((self.config.seed ^ 0xE7A1) + nonce)
        board_index = np.asarray([CARD_INDEX[c] for c in config.board],
                                 dtype=np.int64)
        board_mask = int(np.bitwise_or.reduce(np.int64(1) << board_index)) \
            if len(board_index) else 0
        missing = 5 - len(config.board)
        hero_cards = np.asarray([CARD_INDEX[c] for c in self._combos[HERO][hero_index]],
                                dtype=np.int64)

        combo_cards = [
            np.asarray([[CARD_INDEX[a], CARD_INDEX[b]] for a, b in combos],
                       dtype=np.int64)
            for combos in self._combos
        ]
        probabilities = []
        for entries in config.ranges:
            weight = np.asarray([w for _c, w in entries], dtype=np.float64)
            probabilities.append(weight / weight.sum())

        picks: list[list["np.ndarray"]] = [[] for _p in range(self.players)]
        scores: list["np.ndarray"] = []
        have, rounds = 0, 0
        while have < count and rounds < 40:
            rounds += 1
            draw = max(1024, int((count - have) * 1.6))
            choice = [np.full(draw, hero_index, dtype=np.int64)]
            choice += [generator.choice(len(self._combos[p]), size=draw,
                                        p=probabilities[p])
                       for p in range(1, self.players)]
            holes = [combo_cards[p][choice[p]] for p in range(self.players)]
            used = np.full(draw, board_mask, dtype=np.int64)
            for hole in holes:
                used |= (np.int64(1) << hole[:, 0]) | (np.int64(1) << hole[:, 1])
            ok = _popcount(used) == len(config.board) + 2 * self.players
            if not ok.any():
                continue
            used = used[ok]
            choice = [c[ok] for c in choice]
            holes = [h[ok] for h in holes]
            live = int(ok.sum())

            keys = generator.random((live, 52))
            keys[((used[:, None] >> np.arange(52)[None, :]) & 1).astype(bool)] = np.inf
            runout = np.argsort(keys, axis=1)[:, :missing] if missing else \
                np.zeros((live, 0), dtype=np.int64)
            full = np.concatenate([np.tile(board_index, (live, 1)), runout], axis=1)
            scores.append(np.stack([
                score_batch(np.concatenate([holes[p], full], axis=1))
                for p in range(self.players)
            ], axis=1))
            for p in range(self.players):
                picks[p].append(choice[p])
            have += live

        if not scores:
            raise ValueError("cannot draw any deal for this hero combo")
        assert hero_cards.shape == (2,)
        self._deal_combo = [np.concatenate(picks[p])[:count]
                            for p in range(self.players)]
        self._scores = np.concatenate(scores)[:count]
        self.deals = len(self._scores)
        self._batch = None
        self._terminals.clear()

    def hero_strategy(self, combo: tuple[str, str]) -> dict[str, float]:
        history = ("facing",) if self.config.scenario == "facing_wager" else ()
        table = self._tables.get((HERO, history))
        if table is None:
            raise ValueError("the hero's root node was never visited")
        index = self._combo_index[HERO].get(_canonical(combo))
        if index is None:
            raise ValueError(
                f"combo {''.join(combo)} is not in the hero range",
            )
        return dict(zip(table.actions, table.average()[index]))

    # -- souhrn pro UI ---------------------------------------------------
    def _deal_counts(self, player: int) -> "np.ndarray":
        """How often each of a player's combos came up in the training pool."""
        return np.bincount(self._pool_combo[player],
                           minlength=len(self._combos[player]))

    def hero_class_matrix(self) -> dict[str, dict]:
        """The whole hero range's strategy, by hand class.

        Weighted as in the reference — by accumulated strategy mass, that is
        ``strategy_sum``. A plain average would weight a combo seen five times
        the same as one seen a hundred.
        """
        from pokersolver.ranges.hand_grid import hand_class

        history = ("facing",) if self.config.scenario == "facing_wager" else ()
        table = self._tables.get((HERO, history))
        if table is None:
            return {}
        average = table.average()
        mass = table.strategy_sum.sum(axis=1)
        counts = self._deal_counts(HERO)
        buckets: dict[str, dict] = {}
        for index, combo in enumerate(self._combos[HERO]):
            if mass[index] <= 0:
                continue
            name = hand_class(*combo)
            if name is None:
                continue
            slot = buckets.setdefault(
                name, {"weight": 0.0, "sums": np.zeros(len(table.actions)),
                       "combos": 0, "deals": 0},
            )
            slot["weight"] += float(mass[index])
            slot["combos"] += 1
            slot["deals"] += int(counts[index])
            slot["sums"] += float(mass[index]) * average[index]
        return {
            name: {
                "frequencies": [float(value) / slot["weight"] for value in slot["sums"]],
                "combos": slot["combos"],
                # ``visits`` keeps the reference's meaning: how many samples
                # stand behind that combo. Here it is deals in the pool times
                # passes, because every pass walks the whole pool.
                "visits": int(slot["deals"] * max(1, self.iterations)),
            }
            for name, slot in buckets.items() if slot["weight"] > 0
        }

    def _aggregate_player(self, player: int) -> dict:
        """One opponent's actions summarised across all their nodes."""
        tables = [table for (owner, _history), table in self._tables.items()
                  if owner == player]
        entries = []
        for table in tables:
            mass = table.strategy_sum.sum(axis=1)
            live = mass > 0
            if not live.any():
                continue
            entries.append((table, mass, live))
        total = sum(float(mass[live].sum()) for _table, mass, live in entries)
        if total <= 0:
            return {"actions": (FOLD, CALL), "frequencies": (None, None),
                    "nodes": 0, "samples": 0.0}
        width = max(len(table.actions) for table, _mass, _live in entries)
        sums = np.zeros(width)
        for table, mass, live in entries:
            weighted = (table.average()[live] * mass[live, None]).sum(axis=0)
            sums[:len(table.actions)] += weighted
        names = next((table.actions for table, _mass, _live in entries
                      if len(table.actions) == width), ())
        return {
            "actions": names or (FOLD, CALL, RAISE, ALLIN)[:width],
            "frequencies": tuple(float(value) / total for value in sums),
            "nodes": sum(int(live.sum()) for _table, _mass, live in entries),
            "samples": total,
        }

    def result(self, hero_combo: tuple[str, str], evs: dict[str, float], *,
               elapsed_s: float | None = None) -> dict:
        """A payload shaped exactly like the reference's.

        Sharing the shape is deliberate: callers and tests then need not know
        which engine solved the spot. What this engine cannot produce in
        principle — opponent action EVs, best-response convergence — is ``None``
        rather than an invented number.
        """
        history = ("facing",) if self.config.scenario == "facing_wager" else ()
        table = self._tables.get((HERO, history))
        if table is None:
            raise ValueError("the hero's root node was never visited — solve() did not run")
        index = self._combo_index[HERO].get(_canonical(hero_combo))
        if index is None:
            raise ValueError(
                f"combo {''.join(hero_combo)} carries no weight in the hero range "
                f"(nebo ji blokuje board {' '.join(self.config.board)}), "
                "so the solver has no strategy for it",
            )
        actions = table.actions
        policy = [float(value) for value in table.average()[index]]
        ev_bb = [evs.get(action) for action in actions]
        deals_behind = int(self._deal_counts(HERO)[index]) * max(1, self.iterations)
        hero: dict[str, object] = {
            "actions": list(actions),
            "frequencies": policy,
            "ev_bb": ev_bb,
            "information_sets": sum(
                int((t.strategy_sum.sum(axis=1) > 0).sum())
                for (owner, _h), t in self._tables.items() if owner == HERO
            ),
            "visits": deals_behind,
            "samples_per_action": round(deals_behind / max(1, len(policy)), 1),
        }
        known = [(freq, ev) for freq, ev in zip(policy, ev_bb) if ev is not None]
        if known:
            mix_ev = sum(freq * ev for freq, ev in known)
            best_ev = max(ev for _freq, ev in known)
            hero["mix_ev_bb"] = round(mix_ev, 4)
            hero["best_ev_bb"] = round(best_ev, 4)
            hero["mix_ev_loss_bb"] = round(max(0.0, best_ev - mix_ev), 4)
        for position, action in enumerate(actions):
            hero[action] = policy[position]
            hero[f"ev_{action}_bb"] = ev_bb[position]

        opponents = []
        for player in range(1, self.players):
            aggregate = self._aggregate_player(player)
            frequencies = aggregate["frequencies"]
            names = aggregate["actions"]
            opponent = {
                "name": self.config.names[player],
                "actions": list(names),
                "frequencies": frequencies,
                # This engine does not hold opponent action EVs: its nodes are
                # arrays per combo, not trees with a stored action value.
                "ev_bb": tuple(None for _ in frequencies),
                "fold": frequencies[0] if frequencies else None,
                "call": frequencies[1] if len(frequencies) > 1 else None,
                "ev_fold_bb": None,
                "ev_call_bb": None,
                "information_sets": aggregate["nodes"],
                "samples": aggregate["samples"],
            }
            for label in (RAISE, ALLIN):
                if label in names:
                    opponent[label] = frequencies[list(names).index(label)]
                    opponent[f"ev_{label}_bb"] = None
            opponents.append(opponent)

        return {
            "format": "pokersolver-multiway-cfr-v1",
            "engine": "vector",
            "method": "vectorised sampled-chance regret-minimisation self-play",
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
            "players": self.players,
            "names": self.config.names,
            "board": self.config.board,
            "pot_bb": self.config.pot_bb,
            "bet_bb": min(self.config.bet_bb, self.config.player_stacks[HERO]),
            "requested_bet_bb": self.config.bet_bb,
            "scenario": self.config.scenario,
            "facing_bet_bb": self.config.facing_bet_bb,
            "raise_to_bb": self.config.raise_to_bb,
            "stacks_bb": self.config.player_stacks,
            "iterations": self.iterations,
            # The pool bounds the sampling error; the passes bound the
            # optimisation error. One combined number would hide both.
            "valid_deals": self._pool_size,
            "deal_pool": self._pool_size,
            "deals_processed": self._pool_size * self.iterations,
            "seed": self.config.seed,
            "elapsed_s": elapsed_s,
            "hero_combo": hero_combo,
            "hero_equity": self.eval_equity,
            "hero_range_equity": self.range_equity(),
            "evaluation_samples": self.eval_samples,
            "evaluation_unvisited": self.eval_unvisited,
            "hero_matrix": self.hero_class_matrix(),
            "hero_actions": list(actions),
            "hero": hero,
            "opponents": opponents,
            "information_sets": sum(
                int((table.strategy_sum.sum(axis=1) > 0).sum())
                for table in self._tables.values()
            ),
            "convergence_available": False,
            "limitations": [
                (
                    "one fixed facing-bet and raise-to sizing"
                    if self.config.scenario == "facing_wager"
                    else "one fixed bet sizing"
                ),
                ("at most one re-raise per betting round"
                 if self.config.opponent_raise_to_bb > 0 else "no further re-raises"),
                "once the action closes, the board runs out to showdown",
                f"a fixed pool of {self._pool_size} deals — iterating does not "
                "reduce the sampling error",
                "opponent action EVs and best-response gain come only from the "
                "reference engine",
            ],
        }

    def average_regret_bb(self) -> tuple[float, ...]:
        """Average external regret per player in bb — this must fall to zero.

        The same quantity the reference engine reports: the largest positive
        regret summed over a player's infosets, divided by passes. It is divided
        by pool size as well, because one vectorised pass accumulates regret
        across every deal at once, and without that the number would not read
        in bb.
        """
        totals = [0.0] * self.players
        for (player, _history), table in self._tables.items():
            totals[player] += float(np.maximum(table.regrets, 0.0).max(axis=1).sum())
        scale = max(1, self.iterations) * max(1, self._pool_size)
        return tuple(total / scale for total in totals)

    def decision_confidence(self, hero_combo: tuple[str, str],
                            evs: dict[str, float], *,
                            deals: int = 20_000) -> dict:
        """Can the choice of action be relied on?

        Frequency is deliberately not judged here: between near-indifferent
        actions it is not pinned down by equilibrium and can swing by tens of
        percent while the EV does not move. What matters is the **margin** of
        the best action over the second, and whether it survives an independent
        sample: a second evaluation gives the same EVs from different cards, and
        if the margin moves between samples by as much as it is worth, that is
        noise rather than a decision.
        """
        if not evs:
            return {"verdict": "no_ev"}
        control = self.hero_evs(hero_combo, deals=deals, nonce=1)
        best = max(evs, key=evs.__getitem__)
        ranked = sorted(evs.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        if control:
            ranked_control = sorted(control.values(), reverse=True)
            control_margin = (ranked_control[0] - ranked_control[1]
                              if len(ranked_control) > 1 else ranked_control[0])
            noise = abs(margin - control_margin)
            agrees = max(control, key=control.__getitem__) == best
        else:
            noise, agrees = float("inf"), False

        if not agrees:
            verdict = "unclear"
        elif margin > max(3.0 * noise, 0.05):
            verdict = "decided"
        elif margin > noise:
            verdict = "close"
        else:
            verdict = "unclear"
        return {
            "best_action": best,
            "margin_bb": round(margin, 4),
            "margin_noise_bb": round(noise, 4) if noise != float("inf") else None,
            "second_sample_agrees": agrees,
            "verdict": verdict,
            "avg_regret_bb": [round(value, 6) for value in self.average_regret_bb()],
        }

    def solve_and_result(
        self,
        hero_combo: tuple[str, str],
        *,
        iterations: int = 1_000_000,
        budget_s: float = 5.0,
        eval_deals: int = 20_000,
        batch: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Train to a budget and return a payload shaped like the reference's.

        This engine is bounded by **time**, not iterations: one pass walks the
        whole pool, so "10,000 iterations" means something different here than
        in the reference and converting between them would mislead.
        ``iterations`` is only a ceiling.
        """
        started = time.perf_counter()
        deadline = time.monotonic() + budget_s
        # Progress has to measure TIME, not passes: the iteration ceiling is
        # deliberately out of reach, so step/iterations would sit at zero all
        # run. What is reported is the real number of passes plus an ESTIMATE of
        # the total — their ratio tracks elapsed time, and the caller gets a
        # number that means something.
        def clock(step: int, _limit: int, stage: str) -> None:
            left = max(0.0, deadline - time.monotonic())
            done = (budget_s - left) / budget_s
            estimate = max(step, round(step / done)) if done > 0 else step
            progress(step, estimate, stage)

        self.solve(iterations=iterations, deadline=deadline, batch=batch,
                   cancelled=cancelled, progress=clock if progress else None)
        if self.iterations == 0:
            raise ValueError(
                "the time budget did not cover a single pass — allow more time "
                "or shrink the pool of deals",
            )
        if progress:
            progress(self.iterations, self.iterations, "evaluating")
        solve_s = round(time.perf_counter() - started, 3)
        evs = self.hero_evs(hero_combo, deals=eval_deals)
        payload = self.result(hero_combo, evs, elapsed_s=solve_s)
        # Confidence comes from EV, not frequencies — see decision_confidence.
        payload["decision"] = self.decision_confidence(
            hero_combo, evs, deals=eval_deals)
        payload["solve_s"] = solve_s
        payload["convergence_s"] = 0.0
        payload["elapsed_s"] = round(time.perf_counter() - started, 3)
        payload["budget_s"] = budget_s
        return payload


def _showdown_share(scores: "np.ndarray") -> float:
    """The share of the pot the hero takes at showdown; ties split."""
    if scores is None or not len(scores):
        return 0.0
    best = scores.max(axis=1)
    winners = (scores == best[:, None]).sum(axis=1)
    hero_wins = scores[:, HERO] == best
    return float((hero_wins / winners).mean())


def _bet_menu(config: MultiwayCfrConfig) -> tuple[tuple[str, float], ...]:
    cap = config.player_stacks[HERO]
    options: list[tuple[str, float]] = []
    if config.bet_sizes_pct:
        for pct in config.bet_sizes_pct:
            options.append((f"bet@{pct:g}", min(round(config.pot_bb * pct / 100.0, 4), cap)))
    else:
        options.append(("bet", min(config.bet_bb, cap)))
    if config.hero_allin:
        options.append((ALLIN, cap))
    seen, kept = set(), []
    for name, amount in options:
        if amount <= 1e-9 or amount in seen:
            continue
        seen.add(amount)
        kept.append((name, amount))
    return tuple(kept)


def _raise_menu(config: MultiwayCfrConfig) -> tuple[tuple[str, float], ...]:
    cap = config.player_stacks[HERO]
    options: list[tuple[str, float]] = []
    if config.raise_sizes_pct:
        for pct in config.raise_sizes_pct:
            amount = min(round(config.facing_bet_bb * pct / 100.0, 4), cap)
            if amount > config.facing_bet_bb + 1e-12:
                options.append((f"{RAISE}@{pct:g}", amount))
    elif config.raise_to_bb > config.facing_bet_bb:
        amount = min(config.raise_to_bb, cap)
        if amount > config.facing_bet_bb + 1e-12:
            options.append((RAISE, amount))
    if config.hero_allin and cap > config.facing_bet_bb + 1e-12:
        options.append((ALLIN, cap))
    seen, kept = set(), []
    for name, amount in options:
        if amount in seen:
            continue
        seen.add(amount)
        kept.append((name, amount))
    return tuple(kept)
