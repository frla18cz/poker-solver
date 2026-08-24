"""Pre-calculated GTO baseline ranges and opponent archetypes for 4max AoF cash game.

Includes:
- 8 BB Standard Buy-in GTO Baselines (CO ~28%, BTN ~36%, SB ~52%)
- Deep stack scaling (12 BB, 16 BB, 24 BB, 50 BB)
- Player Archetypes: GTO (8BB), FISH (Loose), NIT (Rock), POPULATION (Typical Reg)
"""
from __future__ import annotations

from typing import Final

# Hand classes ordered by preflop push/fold strength
_HAND_STRENGTH_ORDER: Final[list[str]] = [
    # Top Pairs & Premium Broadway
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o",
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s",
    "QJo", "QTo", "Q9o", "Q8o",
    "JTs", "J9s", "J8s", "J7s", "J6s",
    "JTo", "J9o", "J8o",
    "T9s", "T8s", "T7s", "T6s",
    "T9o", "T8o",
    "98s", "97s", "96s",
    "98o", "97o",
    "87s", "86s", "85s",
    "87o", "86o",
    "76s", "75s", "74s",
    "76o",
    "65s", "64s", "63s",
    "65o",
    "54s", "53s", "52s",
    "54o",
    "43s", "42s",
    "32s",
    "75o", "74o", "73o", "72o",
    "64o", "63o", "62o",
    "53o", "52o",
    "43o", "42o",
    "32o"
]

# Opponent Archetype Jam & Call percentages
ARCHETYPES: Final[dict[str, dict[str, float]]] = {
    "GTO": {
        "CO_jam": 28.0,
        "BTN_jam": 36.0,
        "SB_jam": 52.0,
        "call_vs_1jam": 16.0,
        "call_vs_2jams": 8.0,
    },
    "FISH": {
        "CO_jam": 48.0,
        "BTN_jam": 58.0,
        "SB_jam": 70.0,
        "call_vs_1jam": 30.0,
        "call_vs_2jams": 18.0,
    },
    "NIT": {
        "CO_jam": 12.0,
        "BTN_jam": 16.0,
        "SB_jam": 25.0,
        "call_vs_1jam": 6.0,
        "call_vs_2jams": 3.0,
    },
    "POPULATION": {
        "CO_jam": 22.0,
        "BTN_jam": 30.0,
        "SB_jam": 42.0,
        "call_vs_1jam": 13.0,
        "call_vs_2jams": 6.0,
    },
}

# Stack Depth Scaling Multipliers (Relative to 8 BB baseline)
# When stack increases, jam frequency decreases proportionally to risk/reward
STACK_SCALING: Final[dict[int, float]] = {
    8: 1.0,    # 100% of 8BB range
    12: 0.75,  # 75% of 8BB range
    16: 0.58,  # 58% of 8BB range
    24: 0.42,  # 42% of 8BB range
    40: 0.28,  # 28% of 8BB range
    80: 0.18,  # 18% of 8BB range
}


def get_range_for_percent(percent: float) -> set[str]:
    """The hand classes making up a given percentage of hands (0-100)."""
    count = max(1, min(169, int(round(169 * (percent / 100.0)))))
    return set(_HAND_STRENGTH_ORDER[:count])


def get_archetype_range(
    archetype: str,
    position: str,
    history: str | tuple[str, ...],
    effective_stack_bb: float = 8.0,
) -> set[str]:
    """An opponent's range, from their archetype and the effective stack."""
    profile = ARCHETYPES.get(archetype.upper(), ARCHETYPES["GTO"])
    hist_str = "".join(history).upper()
    prior_jams = hist_str.count("J")

    # Pick the nearest stack-depth multiplier
    stack_keys = sorted(STACK_SCALING.keys())
    closest_stack = min(stack_keys, key=lambda s: abs(s - effective_stack_bb))
    scale = STACK_SCALING[closest_stack]

    if prior_jams == 0:
        # Open jam spot
        if position.upper() == "CO":
            base_pct = profile["CO_jam"]
        elif position.upper() == "BTN":
            base_pct = profile["BTN_jam"]
        else:
            base_pct = profile["SB_jam"]
    elif prior_jams == 1:
        base_pct = profile["call_vs_1jam"]
    else:
        base_pct = profile["call_vs_2jams"]

    final_pct = max(2.0, min(100.0, base_pct * scale))
    return get_range_for_percent(final_pct)
