"""Test fast 7-card evaluator against baseline cards.best_hand."""
import random
import pytest

from pokersolver.cards import FULL_DECK, best_hand
from pokersolver.aof_lab.fast_eval import fast_best_hand, evaluate_7cards_int, _CARD_TO_INT


def test_fast_eval_order_matches_baseline():
    rng = random.Random(42)
    # Test 100 random showdown pairs to ensure fast_best_hand compares identically
    for _ in range(100):
        drawn = rng.sample(FULL_DECK, 14)
        h1 = drawn[:7]
        h2 = drawn[7:14]

        baseline_score1 = best_hand(h1)
        baseline_score2 = best_hand(h2)

        fast_score1 = fast_best_hand(h1)
        fast_score2 = fast_best_hand(h2)

        if baseline_score1 > baseline_score2:
            assert fast_score1 > fast_score2
        elif baseline_score1 < baseline_score2:
            assert fast_score1 < fast_score2
        else:
            assert fast_score1 == fast_score2
