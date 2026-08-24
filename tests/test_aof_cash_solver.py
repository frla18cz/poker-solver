"""Unit tests for AoF Cash Range-Based GTO Solver."""
import pytest
from pokersolver.aof_lab.model import AofCashConfig
from pokersolver.aof_lab.solver import AofCashCfrSolver


def test_solver_deterministic_pure_actions():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05)
    solver = AofCashCfrSolver(cfg, seed=42)
    solver.solve(30_000, workers=1)

    matrix_sb = solver.matrix("SB", "JF")
    # AA and KK must be 100% Pure Jam in SB vs CO Jam
    assert matrix_sb["AA"].jam == 1.0
    assert matrix_sb["KK"].jam == 1.0
    # 72o must be 0% Pure Fold
    assert matrix_sb["72o"].jam == 0.0
    assert matrix_sb["83o"].jam == 0.0


def test_all_15_matrices():
    cfg = AofCashConfig(stacks_bb=(20.0, 20.0, 20.0, 20.0), rake_pct=0.05)
    solver = AofCashCfrSolver(cfg, seed=42)
    solver.solve(10_000, workers=1)

    all_m = solver.all_matrices()
    assert len(all_m) == 15
    assert "CO:root" in all_m
    assert "SB:JF" in all_m
    assert "BB:JJJ" in all_m
