"""One spot across several cores, by splitting the deal pool between processes.

This is exact rather than approximate because regret is **a sum over deals**.
Splitting the pool into W parts, computing partial regrets and adding them up
gives precisely what computing all deals at once would. It is not an
approximation and not a different algorithm.

Processes rather than threads, because threads measured slower — 8 threads came
out at 0.33x. An operation over a 20,000-element array takes about 0.2ms, which
is the same order as the cost of handing out the work. float32 was tried and
rejected (this is not memory-bound), as was reformulating it as matrix
multiplication for BLAS (Accelerate uses two cores even on a 2048x2048 matrix).

The price of exactness is synchronising after EVERY pass: two barriers per
iteration. Merging regrets only occasionally would leave each process training
against its own stale copy of the opponents, and the answer would stop being
the same one.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from multiprocessing.sharedctypes import RawArray
from dataclasses import dataclass, replace

import numpy as np

from .fast import VectorSolver
from .solver import MultiwayCfrConfig

# Six, not "however many cores exist": beyond six, passes grow by a few percent
# because the barriers eat the gain — 6 cores gave 2332 passes, 8 gave 2416.
# The rest is left to the system.
DEFAULT_WORKERS = min(6, max(1, (os.cpu_count() or 1)))


@dataclass(frozen=True)
class _Layout:
    """Where each table sits in the flat deltas buffer."""

    keys: tuple
    shapes: tuple
    offsets: tuple
    size: int


def _layout_of(solver: VectorSolver) -> _Layout:
    keys, shapes, offsets, offset = [], [], [], 0
    for key in sorted(solver._tables, key=lambda k: (k[0], k[1])):
        table = solver._tables[key]
        keys.append(key)
        shapes.append(table.regrets.shape)
        offsets.append(offset)
        offset += table.regrets.size
    return _Layout(tuple(keys), tuple(shapes), tuple(offsets), offset)


def _probe(config: MultiwayCfrConfig) -> tuple[VectorSolver, _Layout]:
    """One pass over a small pool, to create the tables and learn their shape."""
    probe = VectorSolver(config, deals=256)
    probe.solve(iterations=1)
    return probe, _layout_of(probe)


def _flatten(solver: VectorSolver, layout: _Layout, out: "np.ndarray") -> None:
    for key, offset, shape in zip(layout.keys, layout.offsets, layout.shapes):
        table = solver._tables[key]
        count = table.regrets.size
        out[offset:offset + count] = table.regrets.ravel()
        out[layout.size + offset:layout.size + offset + count] = \
            table.strategy_sum.ravel()


def _worker(rank: int, workers: int, config: MultiwayCfrConfig, deals: int,
            deadline: float, layout: _Layout, shared, barrier, stop, result,
            counter, cancel) -> None:
    buffer = np.frombuffer(shared, dtype=np.float64).reshape(workers, 2 * layout.size)
    # The seed shifts per worker; otherwise every process would sample the SAME
    # deals and eight of them would compute the same thing eight times.
    solver = VectorSolver(replace(config, seed=config.seed + rank), deals=deals)
    # The tables only come into being while walking the tree, so one pass runs
    # and is immediately zeroed — an empty dict has nothing to read.
    solver._root(1.0)
    for table in solver._tables.values():
        table.regrets[...] = 0.0
        table.strategy_sum[...] = 0.0
    mine = buffer[rank]
    snapshot = np.zeros(2 * layout.size)
    step = 0
    while True:
        if rank == 0:
            # One process decides and the others read it after the barrier;
            # if each timed itself they could disagree, and the barrier
            # by zatuhla.
            stop.value = 1 if (time.monotonic() >= deadline or cancel.value) else 0
        barrier.wait()
        if stop.value:
            break
        step += 1
        if rank == 0:
            counter.value = step
        _flatten(solver, layout, snapshot)
        solver._root(float(step))
        _flatten(solver, layout, mine)
        mine -= snapshot
        barrier.wait()
        # Each worker sums all the parts itself — a few hundred kB, and it saves
        # a round trip through the parent.
        total = buffer.sum(axis=0)
        combined = snapshot + total
        for key, offset, shape in zip(layout.keys, layout.offsets, layout.shapes):
            table = solver._tables[key]
            count = table.regrets.size
            table.regrets[...] = combined[offset:offset + count].reshape(shape)
            table.strategy_sum[...] = combined[
                layout.size + offset:layout.size + offset + count].reshape(shape)
        barrier.wait()
    if rank == 0:
        payload = np.zeros(2 * layout.size)
        _flatten(solver, layout, payload)
        result.put((step, payload))


def solve_parallel(config: MultiwayCfrConfig, *, deals: int = 20_000,
                   workers: int = DEFAULT_WORKERS, budget_s: float = 8.0,
                   progress=None, cancelled=None) -> VectorSolver:
    """Train across ``workers`` processes and return a solver with its tables.

    The returned solver is ready for ``hero_evs`` and ``decision_confidence``;
    evaluation runs in a single process, because it takes a fraction of a second.
    """
    if workers < 2:
        solver = VectorSolver(config, deals=deals)
        solver.solve(iterations=10_000_000,
                     deadline=time.monotonic() + budget_s,
                     progress=progress, cancelled=cancelled)
        return solver

    parent, layout = _probe(config)
    shard = max(256, deals // workers)
    deadline = time.monotonic() + budget_s
    # Spawn rather than fork: numpy/Accelerate hold threads, and forking from a
    # threaded process can hang — Python warns about it itself. Spawn costs a
    # startup, which is worth paying at second-long budgets.
    context = mp.get_context("spawn")
    shared = RawArray("d", workers * 2 * layout.size)
    barrier = context.Barrier(workers)
    stop = context.Value("i", 0)
    counter = context.Value("q", 0)
    cancel = context.Value("i", 0)
    result = context.Queue()

    processes = [
        context.Process(target=_worker,
                        args=(rank, workers, config, shard, deadline, layout,
                              shared, barrier, stop, result, counter, cancel),
                        daemon=True)
        for rank in range(workers)
    ]
    for process in processes:
        process.start()
    # The parent cannot just wait: it has to report progress and be able to
    # cancel. Rank 0 fills the counter, and the projected total comes from
    # elapsed time, because the run is bounded by time and not by passes.
    steps = payload = None
    while steps is None:
        try:
            steps, payload = result.get(timeout=0.2)
        except queue.Empty:
            if cancelled is not None and cancelled():
                cancel.value = 1
            if progress is not None:
                done = min(0.999, max(1e-6,
                                      1.0 - max(0.0, deadline - time.monotonic()) / budget_s))
                current = int(counter.value)
                progress(current, max(current, int(current / done)), "solving")
    if progress is not None:
        # The final report has to match the real number of passes. The counter is
        # read every 0.2s, so otherwise it lags and the display claims fewer
        # passes than actually happened.
        progress(steps, steps, "solving")
    for process in processes:
        process.join(timeout=10)

    for key, offset, shape in zip(layout.keys, layout.offsets, layout.shapes):
        table = parent._tables[key]
        count = table.regrets.size
        table.regrets[...] = payload[offset:offset + count].reshape(shape)
        table.strategy_sum[...] = payload[
            layout.size + offset:layout.size + offset + count].reshape(shape)
    parent.iterations = steps
    return parent
