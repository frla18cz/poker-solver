"""Command line for the offline 4-max NLHE all-in-or-fold study solver."""
from __future__ import annotations

import argparse
from pathlib import Path

from .solver import AofCfrSolver, AofConfig, render_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline 4max NLHE All-In or Fold regret-minimisation lab.",
    )
    parser.add_argument("--iterations", type=int, default=20_000,
                        help="how many deals to sample (default: 20000)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--stack-bb", type=float, default=8.0, help="effective stack in big blinds")
    parser.add_argument("--rake-bb", type=float, default=0.12, help="flat fee na handu v bb")
    parser.add_argument("--no-rake-on-uncontested", action="store_true",
                        help="skip the fee on a walk or an uncontested jam")
    parser.add_argument("--position", default="CO", choices=("CO", "BTN", "SB", "BB"),
                        help="which position's matrix to print")
    parser.add_argument("--history", default="",
                        help="actions so far, e.g. FJ for the SB")
    parser.add_argument("--metric", default="jam", choices=("jam", "delta"),
                        help="jam frekvence nebo EV(jam)-EV(fold)")
    parser.add_argument("--out", type=Path, help="where to write the full JSON export")
    parser.add_argument("--evaluate", type=int, default=0,
                        help="independent Monte Carlo hands for position EV in the export")
    parser.add_argument("--progress-every", type=int, default=10_000,
                        help="how often to report progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AofConfig(
        stack_bb=args.stack_bb,
        flat_rake_bb=args.rake_bb,
        rake_on_uncontested=not args.no_rake_on_uncontested,
    )
    solver = AofCfrSolver(config, seed=args.seed)

    def progress(done: int, total: int) -> None:
        print(f"AoF solve: {done:,}/{total:,} deals  ({len(solver.nodes):,} information sets)")

    solver.solve(args.iterations, progress=progress, progress_every=args.progress_every)
    matrix = solver.matrix(args.position, args.history)
    label = args.history.upper() or "root"
    print(f"\n{args.position.upper()} after {label}; metric={args.metric}; "
          f"{solver.iterations:,} deals")
    print("\n".join(render_matrix(matrix, metric=args.metric)))
    if args.out:
        output = solver.write_json(args.out, evaluation_samples=args.evaluate)
        print(f"\nJSON export: {output}")
    elif args.evaluate:
        print("\nAverage EV (bb/hand):")
        for position, ev in solver.evaluate_average_strategy(args.evaluate).items():
            print(f"  {position}: {ev:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
