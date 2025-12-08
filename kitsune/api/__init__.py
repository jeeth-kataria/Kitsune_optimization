"""
Kitsune API - User-facing API layer

Contains drop-in replacements for PyTorch components:
- KitsuneOptimizer: High-level training optimizer
- OptimizationConfig: Configuration for optimizations
- optimize_model: Quick setup helper
"""

from .optimizer import (
    KitsuneOptimizer,
    OptimizationConfig,
    OptimizationStats,
    optimize_model,
)

__all__ = [
    "KitsuneOptimizer",
    "OptimizationConfig",
    "OptimizationStats",
    "optimize_model",
]
