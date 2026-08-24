"""Offline multiway poker study tools."""

from .solver import MultiwayCfrConfig, MultiwayCfrSolver, WeightedRange

__all__ = ["MultiwayCfrConfig", "MultiwayCfrSolver", "VectorSolver", "WeightedRange"]


def __getattr__(name: str):
    """``VectorSolver`` on demand, because ``fast`` imports the optional numpy.

    Imported at the top it would break ``multiway`` entirely wherever the
    ``[cfr]`` extra is not installed, and most callers never need it.
    """
    if name == "VectorSolver":
        from .fast import VectorSolver

        return VectorSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
