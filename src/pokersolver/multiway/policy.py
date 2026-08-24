"""Choosing bet sizes for the spot at hand, in two phases.

A fixed menu of one size won every broad comparison, but only because 66% of
pot is near optimal *averaged across spots*. Within a spot, EV between sizes
spreads by 0.80bb at the median and 2.07bb at worst — so size does matter; it
just cannot be paid for by keeping every branch in the tree the whole time, as
a rich menu costs two thirds of the passes.

Two phases resolve that. A short coarse phase gets the rich menu of candidates,
and its EVs pick the best bet size and raise size; the rest of the budget is
spent on those alone. Pruning happens STRICTLY within a family — a bet stays a
bet. Choosing between families (check/bet, fold/call/raise) is left to the
focused phase, so an under-trained coarse phase can only offer a worse size,
never discard the right action.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

from .fast import VectorSolver
from .parallel import DEFAULT_WORKERS, solve_parallel
from .solver import MultiwayCfrConfig

# The shared constants are defined here rather than imported: the replay tool
# imports this module, so the other direction would be a cycle.
OPPONENT_RAISES = (250.0,)
RAKE_PERCENT = 5.0     # from hand history: median 5.0-6.8% of pot over 1557 hands
RAKE_CAP_BB = 17.0     # the largest rake seen there
ALLIN_SPR = 3.0        # the solver only ever chose all-in below SPR 3

# Candidates by player count. The cap is the cost of the tree, not taste: a
# pass takes 5ms with two players and 72ms with four, so what heads-up can
# afford, a four-way spot cannot.
_CANDIDATE_BETS = {2: (33.0, 66.0, 125.0), 3: (66.0, 125.0), 4: (66.0,)}
_CANDIDATE_RAISES = {2: (200.0, 300.0), 3: (300.0,), 4: (300.0,)}

# The coarse phase's share of the budget. The rest goes to the focused phase,
# which is the one that decides.
COARSE_SHARE = 0.25


@dataclass(frozen=True)
class TwoPhaseMeta:
    """What the coarse phase did — callers log this for audit."""

    coarse_skipped: bool
    coarse_iterations: int
    coarse_evs: dict[str, float]
    chosen_bets: tuple[float, ...]
    chosen_raises: tuple[float, ...]


def spr_of(config: MultiwayCfrConfig) -> float:
    """The hero's stack-to-pot ratio at the moment of the decision."""
    return config.player_stacks[0] / max(config.pot_bb + config.facing_bet_bb, 0.01)


def candidate_config(config: MultiwayCfrConfig) -> MultiwayCfrConfig:
    """The same spot, with the candidate sizes the coarse phase should try."""
    players = len(config.ranges)
    return dataclasses.replace(
        config,
        bet_sizes_pct=_CANDIDATE_BETS.get(players, _CANDIDATE_BETS[4]),
        raise_sizes_pct=_CANDIDATE_RAISES.get(players, _CANDIDATE_RAISES[4]),
        opponent_raise_sizes_pct=OPPONENT_RAISES,
        hero_allin=spr_of(config) < ALLIN_SPR,
        opponent_allin=True,
        rake_percent=RAKE_PERCENT,
        rake_cap_bb=RAKE_CAP_BB,
    )


def _best_size(evs: dict[str, float], prefix: str,
               candidates: tuple[float, ...]) -> tuple[float, ...]:
    """The best size in a family by EV; with no data, the first candidate."""
    scored = [(evs[f"{prefix}@{pct:g}"], pct) for pct in candidates
              if f"{prefix}@{pct:g}" in evs]
    if not scored:
        return candidates[:1]
    return (max(scored)[1],)


def narrow_config(config: MultiwayCfrConfig,
                  evs: dict[str, float]) -> MultiwayCfrConfig:
    """Keep one size per family — the one with the best EV.

    The families themselves NEVER change: check, call, fold and all-in all
    survive whatever the coarse phase says. Only sizes of the same action are
    pruned against each other.
    """
    return dataclasses.replace(
        config,
        bet_sizes_pct=_best_size(evs, "bet", config.bet_sizes_pct),
        raise_sizes_pct=_best_size(evs, "raise", config.raise_sizes_pct),
    )


def solve_two_phase(
    config: MultiwayCfrConfig,
    hero_combo: tuple[str, str],
    *,
    budget_s: float = 8.0,
    workers: int = DEFAULT_WORKERS,
    deals: int = 20_000,
) -> tuple[VectorSolver, dict[str, float], dict, TwoPhaseMeta]:
    """The coarse phase picks the sizes; the focused phase solves with them.

    Returns ``(solver, evs, decision, meta)``. ``config`` carries the spot; its
    sizes are replaced by the candidates, so the ones passed in do not matter.
    """
    started = time.monotonic()
    candidates = candidate_config(config)
    single_bet = len(candidates.bet_sizes_pct) <= 1
    single_raise = len(candidates.raise_sizes_pct) <= 1

    if single_bet and single_raise:
        # One candidate in both families (four players): the coarse phase would
        # have nothing to choose between, so the focused phase takes it all.
        solver = solve_parallel(candidates, deals=deals, workers=workers,
                                budget_s=budget_s * 0.85)
        evs = solver.hero_evs(hero_combo, deals=deals)
        decision = solver.decision_confidence(hero_combo, evs, deals=deals)
        meta = TwoPhaseMeta(True, 0, {}, candidates.bet_sizes_pct,
                            candidates.raise_sizes_pct)
        return solver, evs, decision, meta

    coarse = solve_parallel(candidates, deals=deals, workers=workers,
                            budget_s=budget_s * COARSE_SHARE)
    try:
        coarse_evs = coarse.hero_evs(hero_combo, deals=deals)
    except ValueError:
        # The coarse phase produced no EV — the combo is outside the range, or
        # similar. Nothing to choose from, so take the first of each family.
        coarse_evs = {}
    narrowed = narrow_config(candidates, coarse_evs)

    remaining = budget_s - (time.monotonic() - started)
    solver = solve_parallel(narrowed, deals=deals, workers=workers,
                            budget_s=max(0.5, remaining - 0.4))
    evs = solver.hero_evs(hero_combo, deals=deals)
    decision = solver.decision_confidence(hero_combo, evs, deals=deals)
    meta = TwoPhaseMeta(False, coarse.iterations, coarse_evs,
                        narrowed.bet_sizes_pct, narrowed.raise_sizes_pct)
    return solver, evs, decision, meta
