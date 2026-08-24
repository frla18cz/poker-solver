# Preflop matrices

Precomputed preflop solutions. A caller that finds its spot in a matrix can play
straight from it; anything that does not map is the caller's problem to solve
another way.

> **Provenance:** how these particular files were produced is not documented
> here yet. Fill this in before relying on them for anything you publish.

## Active files (multi-depth)

Six-max, 2.5x open (SB 3.5x). The loader reads them as `MultiDepthSolutions` and
picks a matrix by **effective stack**.

- **`nl100_50bb.json`** — 50bb (short) · 182 spots
- **`nl100_100bb.json`** — 100bb (baseline) · 233 spots
- **`nl100_200bb.json`** — 200bb (deep) · 197 spots

## Choosing a matrix by stack depth

`loader.MultiDepthSolutions.for_stack(bb)` takes the nearest depth, preferring
the deeper one on a tie. Effective stack is the smallest stack still in the
hand, in big blinds. Roughly: **≤75bb → 50bb**, **75-150bb → 100bb**,
**≥150bb → 200bb**.

Depth genuinely changes the answer. In `sb_vs_3b_bb` with KTs: **50bb jams
~99%** — short stacks are jam-or-fold — **100bb jams 59%**, and **200bb calls
or 4-bets and never jams**.

## The thing to know: rake changes the strategy

**KTs jamming 59% in `sb_vs_3b_bb` is not a bug. It is the rake.** The same spot
under two rake structures:

| Rake | Fold | Call | 4-bet 22 | Jam |
|------|------|------|----------|-----|
| Micro, high rake | 0% | 86.7% | 13.3% | **0%** |
| NL100, low rake | 0% | 41% | 0% | **59%** |

Lower rake means more aggression; higher rake tightens everything toward
call/fold. **This dataset is the low-rake one.** At micro stakes it is
systematically too loose and will over-jam — a high-rake dataset belongs there
instead.

## Open-size families (2.0 min-raise vs 2.5 baseline)

Depth is not the only axis. `loader.MultiSizeSolutions.for_open(open_bb)` also
picks a family by the **actual size of the open being faced**, and
`facing_open_sizing_mismatch` then checks that the real open is within
`OPEN_SIZE_TOLERANCE_BB` (0.25bb) of the family's nominal size. Within tolerance
the matrix applies; outside it, the caller should decide for itself with the
real price — blind defence is the spot in poker most sensitive to open size.

- **2.5** (`nl100_{50,100,200}bb.json`) — the baseline, complete.
- **2.0** (`nl100_2bbopen_{50,100,200}bb.json`) — min-raise, **defence spots only.**

### Why the 2bb family is smaller, not a full copy

Open size only changes the strategy where a 2bb bet actually enters the
sequence, and the intended caller opens 2.5 itself. So the 2bb family covers
only the spots where **an opponent opens 2bb and the hero responds** — blind
defence plus overcalls:

| Category | Spots per depth | In the 2bb family | Why |
|---|---|---|---|
| `*_vs_*` heads-up (blind defence) | 16 | yes | the hero faces a 2bb open |
| `*_vs_*_*` with a caller (multiway) | 20 | yes | same, multiway |
| `*_mw_*` blinds | 15 | yes | hero in the blinds against an open plus calls |
| `*_oc_*` overcall | 20 | yes | hero behind a limper or caller |
| 3bet and beyond | 106–157 | no | a different open size shifts the whole reraise tree, so the 2.5 matrix would not fit anyway |
| `rfi_*` | 5 | no | the hero opens 2.5, not 2bb |

That is roughly 71 defence spots per depth across three depths — about 210
matrices, against roughly 612 for the complete 2.5 set.

### A 3bb family: not yet

3bb sits 0.5bb from 2.5 — exactly as far as 2.0 does. It is not "close enough to
2.5"; for price-sensitive blind defence it is a different spot. It is left out
for now because the sizing guard already handles it safely: 3bb falls outside
the tolerance of both families, so `facing_open_sizing_mismatch` flags it and
the caller decides with the real price rather than against the wrong matrix.
Adding a 3.0 family later needs no code change, only files.

## What is not in the data

- **Ten self-inconsistent nodes were removed** (`bb_sqz_vs_*_call/fold_*_4b`).
  Their sequences were truncated in a way that decoded to the wrong hero and
  collided with `*_vs_sqz_*_bb`.
- **Thirty-four deep nodes are absent**: `*_vs_3b_cc_*` (hero against a 3-bet
  with a cold-caller) and `*_sqz_vs_*_4b_*` (squeezer against a 4-bet). These
  are rare four-way spots, and a caller has to handle them itself.

## Building

`python -m pokersolver.ranges.gto.build <source_dir> [out_path]` reads raw
per-node JSON and writes the consolidated file. It flattens faithfully and does
not rearrange keys. The source directory comes from `PF_GTO_SOURCE_DIR` or
`./raw_nodes`; pass `out_path` explicitly for another family or depth, or the
baseline will be overwritten.

Swap the whole dataset without reinstalling by pointing `PF_GTO_DATA_DIR` at
another directory.
