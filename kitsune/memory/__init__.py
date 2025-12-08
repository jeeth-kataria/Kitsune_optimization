"""
Kitsune Memory - Memory optimization layer

Contains memory management optimizations:
- Zero-allocation memory pool with size-class binning
- Double/triple buffering for transfer overlap
- Async data prefetching for DataLoaders
- Tensor lifetime analysis for memory reuse
"""

from .pool import (
    MemoryPool,
    TensorCache,
    SizeClass,
    AllocationStats,
    get_memory_pool,
    reset_memory_pool,
)
from .double_buffer import (
    DoubleBuffer,
    TripleBuffer,
    PrefetchBuffer,
    H2DOverlap,
    BufferState,
)
from .prefetch import (
    AsyncPrefetcher,
    CUDAPrefetcher,
    PinnedDataLoader,
    create_prefetched_loader,
)
from .lifetime import (
    LifetimeAnalyzer,
    TensorLifetime,
    TensorLifeEvent,
    MemoryEvent,
)

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
