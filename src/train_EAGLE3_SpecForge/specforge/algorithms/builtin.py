"""Explicit immutable catalog of SpecForge's built-in algorithms."""

from __future__ import annotations

from specforge.algorithms.eagle3.providers import create_registration as eagle3
from specforge.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry((eagle3(),))


__all__ = ["builtin_algorithm_registry"]
