"""Fast analytical Push/Fold EV calculator and range exploiter for 4max AoF cash game.

Provides sub-10ms instantaneous EV calculations against customizable opponent ranges,
allowing real-time slider manipulation and exploit discovery in the Web Studio.
"""
from __future__ import annotations

import random
from typing import Final, Iterable

from ..cards import FULL_DECK
from ..ranges.hand_grid import all_hand_classes, combos_of_class
from .fast_eval import fast_best_hand
from .model import AofCashConfig, canonical_combo, settle_cash_actions


def calculate_hero_push_ev(
    hero_pos: str,
    hero_hand_class: str,
    config: AofCashConfig,
    villain_calling_ranges: dict[str, set[str]],
    *,
    monte_carlo_samples: int = 500,
    rng_seed: int = 42,
) -> dict[str, float]:
    """EV of a hero push against given opponent calling ranges.

    `hero_pos`: 'CO', 'BTN', 'SB' nebo 'BB'.
    `villain_calling_ranges` looks like {'BTN': {'AA', 'KK', 'AKs'}, 'SB': {...}}.
    Returns {'ev_fold', 'ev_jam', 'ev_delta', 'call_freq', 'all_fold_prob'}.
    """
    player_idx = config.positions.index(hero_pos.upper())
    blinds = config.blinds()
    ev_fold = -blinds[player_idx]

    combos = combos_of_class(hero_hand_class)
    if not combos:
        return {"ev_fold": ev_fold, "ev_jam": ev_fold, "ev_delta": 0.0, "call_freq": 0.0, "all_fold_prob": 1.0}

    rng = random.Random(rng_seed)
    deck_cards = set(FULL_DECK)

    # The players still to act
    subsequent_players = list(range(player_idx + 1, 4))
    
    total_jam_utility = 0.0
    total_samples = 0
    fold_count = 0

    for sample_idx in range(monte_carlo_samples):
        # Draw a random hero combo from the class
        hero_combo = rng.choice(combos)
        hero_c1, hero_c2 = hero_combo
        available_deck = list(deck_cards - {hero_c1, hero_c2})
        rng.shuffle(available_deck)

        # Deal the opponents and the board: four players in all
        deal_idx = 0
        holes = [hero_combo if p == player_idx else () for p in range(4)]
        
        for p in range(4):
            if p != player_idx:
                c1 = available_deck[deal_idx]
                c2 = available_deck[deal_idx + 1]
                deal_idx += 2
                holes[p] = (c1, c2)

        board = available_deck[deal_idx : deal_idx + 5]

        # Work out what each opponent does, from their calling range
        actions = ["F"] * 4
        actions[player_idx] = "J"

        for p in subsequent_players:
            pos_name = config.positions[p]
            v_range = villain_calling_ranges.get(pos_name, set())
            v_combo = holes[p]
            # Which hand class the opponent holds
            v_class = _combo_to_class(v_combo)
            if v_class in v_range:
                actions[p] = "J"

        # Showdown nebo uncontested
        caller_count = sum(actions[p] == "J" for p in subsequent_players)
        if caller_count == 0:
            fold_count += 1

        scores = tuple(fast_best_hand(list(holes[p]) + board) for p in range(4))
        payoffs = settle_cash_actions(config, tuple(actions), scores)  # type: ignore[arg-type]
        total_jam_utility += payoffs[player_idx]
        total_samples += 1

    ev_jam = total_jam_utility / total_samples if total_samples > 0 else ev_fold
    all_fold_prob = fold_count / total_samples if total_samples > 0 else 1.0

    return {
        "ev_fold": ev_fold,
        "ev_jam": ev_jam,
        "ev_delta": ev_jam - ev_fold,
        "call_freq": 1.0 - all_fold_prob,
        "all_fold_prob": all_fold_prob,
    }


def _combo_to_class(combo: tuple[str, str]) -> str:
    c1, c2 = combo
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    _ORDER = "AKQJT98765432"
    i1, i2 = _ORDER.index(r1), _ORDER.index(r2)
    if i1 == i2:
        return r1 + r2
    if i1 < i2:
        return r1 + r2 + ("s" if s1 == s2 else "o")
    return r2 + r1 + ("s" if s1 == s2 else "o")
