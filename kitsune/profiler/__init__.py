"""
Kitsune Profiler - Performance monitoring and metrics collection

Provides CUDA-aware timing, memory tracking, and performance metrics
for benchmarking and optimization.
"""

import logging

from .cuda_timer import CUDATimer
from .memory_tracker import MemoryTracker
from .metrics import Metrics, MetricsCollector
from .reporter import Profiler, ProfileResult


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = [
    "CUDATimer",
    "MemoryTracker",
    "Metrics",
    "MetricsCollector",
    "Profiler",
    "ProfileResult",
    "get_logger",
]
