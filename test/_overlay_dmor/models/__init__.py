# models/__init__.py
"""
Public API for the `models` package.

This file intentionally exposes a small, stable surface:
- DMOR: core routing + fusion block
- DMOREdgeNet: minimal end-to-end edge net
- build_operator_pool: operator pool factory
"""

from .dmor import DMOR
from .net import DMOREdgeNet
from .operators import build_operator_pool

__all__ = ["DMOR", "DMOREdgeNet", "build_operator_pool"]

# Optional version tag (useful for releases / papers)
__version__ = "0.2.0"
