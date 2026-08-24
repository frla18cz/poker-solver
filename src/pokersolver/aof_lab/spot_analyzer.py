"""Spot & Hand Analyzer for 4max NLHE AoF Cash Game.

Allows players to analyze exact situations: 'I am in SB with TJs facing CO Jam and BTN Fold'.
Supports:
- Bayesian conditioning on prior jammers' ranges
- Archetype profiles (GTO 8BB baseline, Fish, Nit, Population Reg)
- Exact stack depth scaling
- Precise EV(Jam/Call) vs EV(Fold) in BB
- Multiway / HU Equity vs Villain ranges
- Required Pot Odds with 5% Rake
- Break-Even Opponent Jam Range Threshold
"""
from __future__ import annotations

import random
from typing import Final, Iterable

from ..cards import FULL_DECK
from ..ranges.hand_grid import all_hand_classes, combos_of_class
from .baselines import ARCHETYPES, get_archetype_range, get_range_for_percent
from .fast_eval import fast_best_hand
from .model import AofCashConfig, canonical_combo, parse_history, settle_cash_actions

get_top_percent_range = get_range_for_percent


def analyze_hand_in_spot(
    hero_pos: str,
    history: str | tuple[str, ...],
    hero_hand_class: str,
    config: AofCashConfig,
    villain_range_pcts: dict[str, float] | None = None,
    opponent_profile: str = "GTO",
    *,
    monte_carlo_samples: int = 1500,
    seed: int = 42,
) -> dict:
    """A full read on one spot, conditioning opponent ranges on their actions."""
    player_idx = config.positions.index(hero_pos.upper())
    parsed_hist = parse_history(history)
    if len(parsed_hist) != player_idx:
        raise ValueError(f"history {history} does not fit {hero_pos} — "
                         f"expected {player_idx} actions")

    eff_stack = config.stacks_bb[player_idx]

    # Ranges for every other position
    villain_ranges: dict[str, set[str]] = {}
    for p, pos in enumerate(config.positions):
        if p != player_idx:
            if villain_range_pcts and pos in villain_range_pcts:
                villain_ranges[pos] = get_range_for_percent(villain_range_pcts[pos])
            else:
                villain_ranges[pos] = get_archetype_range(
                    opponent_profile, pos, parsed_hist[:p], effective_stack_bb=eff_stack,
                )

    blinds = config.blinds()
    ev_fold = -blinds[player_idx]

    combos = combos_of_class(hero_hand_class)
    if not combos:
        raise ValueError(f"not a hand class: {hero_hand_class}")

    rng = random.Random(seed)
    deck = list(FULL_DECK)

    subsequent = list(range(player_idx + 1, 4))
    prior_jammers = [i for i, a in enumerate(parsed_hist) if a == "J"]

    total_jam_payoff = 0.0
    wins = 0
    splits = 0
    losses = 0
    showdown_count = 0

    valid_samples = 0
    attempts = 0
    max_attempts = monte_carlo_samples * 30

    while valid_samples < monte_carlo_samples and attempts < max_attempts:
        attempts += 1
        hero_combo = rng.choice(combos)
        h1, h2 = hero_combo
        avail = [c for c in deck if c != h1 and c != h2]
        rng.shuffle(avail)

        holes = [() for _ in range(4)]
        holes[player_idx] = hero_combo

        deal_ptr = 0
        for p in range(4):
            if p != player_idx:
                holes[p] = (avail[deal_ptr], avail[deal_ptr + 1])
                deal_ptr += 2

        # Conditioning: if an earlier player jammed, their hand MUST be in
        # their jamming range.
        prior_jam_valid = True
        for pj in prior_jammers:
            pj_pos = config.positions[pj]
            pj_range = villain_ranges[pj_pos]
            pj_cls = _combo_to_class(holes[pj])
            if pj_cls not in pj_range:
                prior_jam_valid = False
                break

        if not prior_jam_valid:
            continue

        board = avail[deal_ptr : deal_ptr + 5]

        actions = list(parsed_hist) + ["J"]

        for p in subsequent:
            pos_name = config.positions[p]
            v_range = villain_ranges[pos_name]
            v_combo = holes[p]
            v_class = _combo_to_class(v_combo)
            actions.append("J" if v_class in v_range else "F")

        scores = tuple(fast_best_hand(list(holes[p]) + board) for p in range(4))
        payoffs = settle_cash_actions(config, tuple(actions), scores)  # type: ignore[arg-type]
        total_jam_payoff += payoffs[player_idx]
        valid_samples += 1

        jammed_players = [p for p, a in enumerate(actions) if a == "J"]
        if len(jammed_players) >= 2:
            showdown_count += 1
            hero_score = scores[player_idx]
            max_opponent = max(scores[p] for p in jammed_players if p != player_idx)
            if hero_score > max_opponent:
                wins += 1
            elif hero_score == max_opponent:
                splits += 1
            else:
                losses += 1

    ev_jam = total_jam_payoff / max(1, valid_samples)
    ev_delta = ev_jam - ev_fold
    equity = (wins + 0.5 * splits) / max(1, showdown_count)

    call_cost = config.stacks_bb[player_idx] - blinds[player_idx]
    opp_committed = sum(
        config.stacks_bb[p] if p in prior_jammers else blinds[p]
        for p in range(4) if p != player_idx
    )
    gross_pot = call_cost + blinds[player_idx] + opp_committed
    net_pot = gross_pot * (1.0 - config.rake_pct)
    pot_odds_pct = (call_cost / max(1e-4, net_pot)) * 100.0

    if ev_delta >= 0.15:
        recommendation = "PURE JAM / CALL"
        rec_color = "green"
    elif ev_delta <= -0.15:
        recommendation = "PURE FOLD"
        rec_color = "red"
    else:
        recommendation = f"BORDERLINE MIXED (Δ {ev_delta:+.2f} BB)"
        rec_color = "amber"

    # Break-even curve against the jammer's range
    be_curve = []
    if prior_jammers:
        main_villain = config.positions[prior_jammers[0]]
        for test_pct in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]:
            test_pcts = {pos: (float(test_pct) if pos == main_villain else 15.0) for pos in config.positions}
            sub_res = _quick_ev_point(hero_pos, parsed_hist, hero_hand_class, config, test_pcts, combos, rng)
            be_curve.append({"villain_pct": test_pct, "ev_delta": sub_res})

    return {
        "hero_pos": hero_pos,
        "history": "".join(parsed_hist),
        "hand": hero_hand_class,
        "profile": opponent_profile,
        "effective_stack_bb": eff_stack,
        "ev_jam": round(ev_jam, 2),
        "ev_fold": round(ev_fold, 2),
        "ev_delta": round(ev_delta, 2),
        "equity_pct": round(equity * 100.0, 1),
        "pot_odds_req_pct": round(min(100.0, max(0.0, pot_odds_pct)), 1),
        "safety_margin_pct": round(equity * 100.0 - pot_odds_pct, 1),
        "recommendation": recommendation,
        "rec_color": rec_color,
        "break_even_curve": be_curve,
    }


def _quick_ev_point(hero_pos, parsed_hist, hand_class, config, pcts, combos, rng):
    player_idx = config.positions.index(hero_pos.upper())
    blinds = config.blinds()
    ev_fold = -blinds[player_idx]
    v_ranges = {pos: get_range_for_percent(pct) for pos, pct in pcts.items()}
    prior_jammers = [i for i, a in enumerate(parsed_hist) if a == "J"]
    subsequent = list(range(player_idx + 1, 4))
    deck = list(FULL_DECK)

    total_payoff = 0.0
    valid_samples = 0
    attempts = 0

    while valid_samples < 250 and attempts < 3000:
        attempts += 1
        hero_combo = rng.choice(combos)
        h1, h2 = hero_combo
        avail = [c for c in deck if c != h1 and c != h2]
        rng.shuffle(avail)
        holes = [() for _ in range(4)]
        holes[player_idx] = hero_combo
        ptr = 0
        for p in range(4):
            if p != player_idx:
                holes[p] = (avail[ptr], avail[ptr + 1])
                ptr += 2

        prior_jam_valid = True
        for pj in prior_jammers:
            pj_pos = config.positions[pj]
            pj_range = v_ranges[pj_pos]
            pj_cls = _combo_to_class(holes[pj])
            if pj_cls not in pj_range:
                prior_jam_valid = False
                break
        if not prior_jam_valid:
            continue

        board = avail[ptr : ptr + 5]
        actions = list(parsed_hist) + ["J"]
        for p in subsequent:
            v_combo = holes[p]
            v_class = _combo_to_class(v_combo)
            actions.append("J" if v_class in v_ranges[config.positions[p]] else "F")
        scores = tuple(fast_best_hand(list(holes[p]) + board) for p in range(4))
        payoffs = settle_cash_actions(config, tuple(actions), scores)  # type: ignore[arg-type]
        total_payoff += payoffs[player_idx]
        valid_samples += 1

    return round((total_payoff / max(1, valid_samples)) - ev_fold, 2)


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
