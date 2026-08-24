"""Monte Carlo equity for Texas Hold'em, standard library only."""
from __future__ import annotations

import random

from .cards import FULL_DECK, best_hand


def equity(
    hole: list[str],
    board: list[str] | None = None,
    n_opponents: int = 1,
    iterations: int = 5000,
    rng: random.Random | None = None,
) -> float:
    """Odhad equity (0..1) pro danou ruku.

    hole:        your two cards, e.g. ["As", "Ah"]
    board:       0-5 community cards
    n_opponents: opponents holding random cards
    iterations:  how many simulations to run
    Split pots count as 1/k of a win.
    """
    rng = rng or random.Random()
    board = list(board or [])
    known = set(hole) | set(board)
    if len(known) != len(hole) + len(board):
        raise ValueError("the same card appears twice")
    deck = [c for c in FULL_DECK if c not in known]

    needed_board = 5 - len(board)
    draw_count = needed_board + 2 * n_opponents

    wins = 0.0
    for _ in range(iterations):
        drawn = rng.sample(deck, draw_count)
        full_board = board + drawn[:needed_board]
        my_score = best_hand(hole + full_board)

        idx = needed_board
        best_opp = None
        for _ in range(n_opponents):
            opp_hole = drawn[idx:idx + 2]
            idx += 2
            opp_score = best_hand(opp_hole + full_board)
            if best_opp is None or opp_score > best_opp:
                best_opp = opp_score

        if my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 0.5  # split approximated; close enough to decide on
    return wins / iterations


def equity_vs_range(
    hole: list[str],
    board: list[str] | None,
    villain_combos: list[tuple[str, str]],
    n_villains: int = 1,
    iterations: int = 4000,
    rng: random.Random | None = None,
) -> float:
    """Hero's equity against opponents drawing from a given range.

    villain_combos: the actual two-card combos in their range, blockers
    already removed. More realistic than equity() against random cards; falls
    back to uniform when the range comes out empty.
    """
    rng = rng or random.Random()
    board = list(board or [])
    known = set(hole) | set(board)

    # Combos still available — not blocked by a known card.
    combos = [c for c in villain_combos if c[0] not in known and c[1] not in known]
    if not combos:
        return equity(hole, board, n_villains, iterations, rng)

    needed_board = 5 - len(board)
    wins = 0.0
    done = 0
    for _ in range(iterations):
        used = set(known)
        # pick n_villains non-conflicting hands from the range
        villains = []
        ok = True
        for _ in range(n_villains):
            pick = None
            for _try in range(20):
                cand = combos[rng.randrange(len(combos))]
                if cand[0] not in used and cand[1] not in used:
                    pick = cand
                    break
            if pick is None:
                ok = False
                break
            used.add(pick[0]); used.add(pick[1])
            villains.append(pick)
        if not ok:
            continue

        deck = [c for c in FULL_DECK if c not in used]
        drawn = rng.sample(deck, needed_board)
        full_board = board + drawn
        my_score = best_hand(hole + full_board)
        best_opp = None
        for v in villains:
            s = best_hand(list(v) + full_board)
            if best_opp is None or s > best_opp:
                best_opp = s
        if my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 0.5
        done += 1
    return wins / done if done else equity(hole, board, n_villains, iterations, rng)


def equity_vs_ranges(
    hole: list[str],
    board: list[str] | None,
    villain_ranges: list[list[tuple[str, str]]],
    iterations: int = 4000,
    rng: random.Random | None = None,
) -> float:
    """Equity against several opponents, each with their own range of combos.

    Every iteration draws one non-conflicting combo per range. A split pot is
    divided by how many hands actually tie, not by the heads-up approximation.
    """
    if not villain_ranges:
        raise ValueError("equity_vs_ranges needs at least one opponent range")
    if any(not combos for combos in villain_ranges):
        raise ValueError("an opponent range is empty")
    rng = rng or random.Random()
    board = list(board or [])
    known = set(hole) | set(board)
    ranges = [[c for c in combos if c[0] not in known and c[1] not in known]
              for combos in villain_ranges]
    if any(not combos for combos in ranges):
        raise ValueError("an opponent range is empty once dead cards are removed")

    needed_board = 5 - len(board)
    wins = 0.0
    done = 0
    for _ in range(iterations):
        used = set(known)
        villains: list[tuple[str, str]] = []
        for combos in ranges:
            available = [c for c in combos if c[0] not in used and c[1] not in used]
            if not available:
                villains = []
                break
            pick = available[rng.randrange(len(available))]
            used.update(pick)
            villains.append(pick)
        if len(villains) != len(ranges):
            continue

        deck = [c for c in FULL_DECK if c not in used]
        full_board = board + rng.sample(deck, needed_board)
        hero_score = best_hand(hole + full_board)
        villain_scores = [best_hand(list(v) + full_board) for v in villains]
        best_villain = max(villain_scores)
        if hero_score > best_villain:
            wins += 1.0
        elif hero_score == best_villain:
            tied_villains = sum(1 for score in villain_scores if score == hero_score)
            wins += 1.0 / (1 + tied_villains)
        done += 1
    if not done:
        raise ValueError("could not draw non-conflicting hands from these ranges")
    return wins / done


def equity_vs_weighted_ranges(
    hole: list[str],
    board: list[str] | None,
    villain_ranges: list[list[tuple[tuple[str, str], float]]],
    iterations: int = 4000,
    rng: random.Random | None = None,
) -> float:
    """Multiway equity against weighted range buckets.

    Each entry is ``((card_a, card_b), weight)``. The weights come from the
    caller — this function samples by them and never adjusts them.
    """
    if not villain_ranges:
        raise ValueError("equity_vs_weighted_ranges needs at least one range")
    rng = rng or random.Random()
    board = list(board or [])
    known = set(hole) | set(board)
    raw_ranges = [entries.entries() if hasattr(entries, "entries") else entries
                  for entries in villain_ranges]
    ranges = [
        [(combo, weight) for combo, weight in entries
         if combo[0] not in known and combo[1] not in known and weight > 0]
        for entries in raw_ranges
    ]
    if any(not entries for entries in ranges):
        raise ValueError("a weighted range is empty once dead cards are removed")

    def choose(entries, used: set[str]):
        available = [(combo, weight) for combo, weight in entries
                     if combo[0] not in used and combo[1] not in used]
        total = sum(weight for _combo, weight in available)
        if total <= 0:
            return None
        needle = rng.random() * total
        for combo, weight in available:
            needle -= weight
            if needle <= 0:
                return combo
        return available[-1][0]  # floating-point edge

    needed_board = 5 - len(board)
    wins = 0.0
    done = 0
    for _ in range(iterations):
        used = set(known)
        villains: list[tuple[str, str]] = []
        for entries in ranges:
            pick = choose(entries, used)
            if pick is None:
                villains = []
                break
            used.update(pick)
            villains.append(pick)
        if len(villains) != len(ranges):
            continue
        full_board = board + rng.sample([c for c in FULL_DECK if c not in used], needed_board)
        hero_score = best_hand(hole + full_board)
        villain_scores = [best_hand(list(v) + full_board) for v in villains]
        best_villain = max(villain_scores)
        if hero_score > best_villain:
            wins += 1.0
        elif hero_score == best_villain:
            wins += 1.0 / (1 + sum(score == hero_score for score in villain_scores))
        done += 1
    if not done:
        raise ValueError("could not draw non-conflicting hands from these weighted ranges")
    return wins / done
