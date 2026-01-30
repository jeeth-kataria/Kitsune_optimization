"""
Kitsune Memory - Memory optimization layer

Contains memory management optimizations:
- Zero-allocation memory pool with size-class binning
- Double/triple buffering for transfer overlap
- Async data prefetching for DataLoaders
- Tensor lifetime analysis for memory reuse
"""

from .double_buffer import BufferState, DoubleBuffer, H2DOverlap, PrefetchBuffer, TripleBuffer
from .lifetime import LifetimeAnalyzer, MemoryEvent, TensorLifeEvent, TensorLifetime
from .pool import (
    AllocationStats,
    MemoryPool,
    SizeClass,
    TensorCache,
    get_memory_pool,
    reset_memory_pool,
)
from .prefetch import AsyncPrefetcher, CUDAPrefetcher, PinnedDataLoader, create_prefetched_loader

__all__ = [
    # Memory Pool
    "MemoryPool",
    "TensorCache",
    "SizeClass",
    "AllocationStats",
    "get_memory_pool",
    "reset_memory_pool",
    # Double Buffer
    "DoubleBuffer",
    "TripleBuffer",
    "PrefetchBuffer",
    "H2DOverlap",
    "BufferState",
    # Prefetch
    "AsyncPrefetcher",
    "CUDAPrefetcher",
    "PinnedDataLoader",
    "create_prefetched_loader",
    # Lifetime
    "LifetimeAnalyzer",
    "TensorLifetime",
    "TensorLifeEvent",
    "MemoryEvent",
]
