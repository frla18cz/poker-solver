<div align="center">

# pokersolver

**Deterministic poker math as a library — equity, ranges, preflop matrices, CFR.**

<p><a href="https://frla18cz.github.io/poker-solver/">Docs</a> · <a href="#use">Quickstart</a> · <a href="#layout">API</a> · <a href="https://github.com/frla18cz/poker-arena">A table for it</a></p>

<p><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-blue"> <img alt="Python 3.14+" src="https://img.shields.io/badge/python-3.14%2B-blue"> <img alt="Core is stdlib-only" src="https://img.shields.io/badge/core-stdlib--only-brightgreen"> <img alt="88 tests" src="https://img.shields.io/badge/tests-88-brightgreen"> <img alt="825 precomputed spots" src="https://img.shields.io/badge/preflop%20spots-825-blue"></p>

</div>

Card evaluation, Monte Carlo equity against weighted ranges, the 169-hand grid,
precomputed preflop matrices, and CFR solvers for all-in-or-fold and multiway
postflop spots.

The library has no notion of "a player to act" and no game-state type of its
own. It takes primitives — cards, stacks in big blinds, positions as strings —
and returns numbers. Wiring it to an engine is the caller's job.

## Install

```bash
pip install -e '.[dev]'          # core is stdlib-only; numpy only for multiway CFR
```

Requires Python 3.14+.

## Use

```python
from pokersolver.cards import best_hand
from pokersolver.equity import equity_vs_range
from pokersolver.ranges.hand_grid import hand_class
from pokersolver.ranges.gto import default_multi_solutions

hand_class("As", "Kh")                  # 'AKo'
solutions = default_multi_solutions()   # precomputed preflop matrices
spot = solutions.for_stack(100).for_open(2.5).get_or_none("btn_vs_co")
```

Preflop matrices ship in `src/pokersolver/ranges/gto/data/`. Point
`PF_GTO_DATA_DIR` at another directory to swap the dataset without reinstalling.

## Layout

| Module | What it does |
|---|---|
| `cards` | deck, 7-card hand scoring, `best_hand` |
| `equity` | Monte Carlo equity, incl. against weighted ranges |
| `ranges/hand_grid`, `ranges/range` | the 169 hand classes, range parsing |
| `ranges/gto` | loading and building precomputed preflop matrices |
| `aof`, `aof_lab` | all-in-or-fold CFR solvers |
| `multiway` | multiway postflop CFR (needs numpy) |
| `handdesc`, `board_texture`, `position` | board and position description |

## Tests

```bash
python -m pytest -q
```

## A table for it

Because the library has no game state of its own, it cannot deal a hand or take
a turn — that is the caller's job, and doing it well is a project in itself.
[pokerarena][arena] is that project: a poker table you sit at in a browser, with
friends or bots or both. Its `solver` seat is this library wired to an engine —
equity against every opponent still in the hand, played against the price.

Useful as a worked example of what "wiring it to an engine" actually means, if
you are about to do the same.

[arena]: https://github.com/frla18cz/poker-arena

## What it does not do

> [!IMPORTANT]
> The multiway CFR abstracts the game to make it tractable, and the result
> reports exactly how: one fixed bet sizing, at most one re-raise per betting
> round, and the board running out to showdown once the action closes. Read
> `result["limitations"]` before treating a number as ground truth.

- No game state, no player to act, no engine. It takes primitives and returns
  numbers; wiring it to a table is the caller's job.
- Monte Carlo equity is sampled, not exact. More iterations buy precision, and
  the sampling error does not go away.
- The shipped preflop matrices are one dataset at three stack depths. They are
  a starting point, not a solved game — point `PF_GTO_DATA_DIR` elsewhere to
  swap them.
- No postflop bet-sizing search. The abstraction above is the price of getting
  a multiway answer at all.

## Status

Extracted from a larger poker project and still settling into its own shape.
The multiway CFR and the equity code have had real use; the APIs may still move.
