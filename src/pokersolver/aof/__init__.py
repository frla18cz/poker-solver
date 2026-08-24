"""Offline solver pro 4max NLHE All-In or Fold.

Deliberately separate from any live client or cash-game strategy: it sends no
actions, it only computes and exports study matrices.
"""

from .solver import AofConfig, AofCfrSolver, AofInfoSet, MatrixCell

__all__ = ["AofConfig", "AofCfrSolver", "AofInfoSet", "MatrixCell"]
