"""
Memory Pool - Zero-allocation memory management for CUDA tensors.

Implements size-class binning to reuse memory allocations and reduce
the overhead of frequent malloc/free calls during training.
"""

from __future__ import annotations

import threading
import math
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import torch

from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class AllocationStats:
    """Statistics for memory pool allocations."""
    total_allocations: int = 0
    total_deallocations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_allocated: int = 0
    bytes_cached: int = 0
    peak_bytes_allocated: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class SizeClass:
    """
    Size class for memory binning.

    Groups allocations into size classes to improve cache hit rate.
    Uses power-of-2 sizing with minimum granularity.
    """

    # Minimum allocation size (512 bytes)
    MIN_SIZE = 512

    # Size classes: 512B, 1KB, 2KB, 4KB, ..., up to 1GB
    # After 1GB, use exact sizes
    MAX_BINNED_SIZE = 1 << 30  # 1GB

    @classmethod
    def get_size_class(cls, size: int) -> int:
        """
        Get the size class for a given allocation size.

        Args:
            size: Requested allocation size in bytes

        Returns:
            Size class (rounded up to nearest bin)
        """
        if size <= cls.MIN_SIZE:
            return cls.MIN_SIZE

        if size > cls.MAX_BINNED_SIZE:
            # For very large allocations, use exact size
            return size

        # Round up to next power of 2
        return 1 << (size - 1).bit_length()

    @classmethod
    def get_all_classes(cls) -> List[int]:
        """Get all standard size classes."""
        classes = []
        size = cls.MIN_SIZE
        while size <= cls.MAX_BINNED_SIZE:
            classes.append(size)
            size *= 2
        return classes


@dataclass
class CachedBlock:
    """A cached memory block."""
    data: torch.Tensor
    size_class: int
    actual_size: int
    device: torch.device
    in_use: bool = False
    allocation_count: int = 0


class MemoryPool:
    """
    Memory pool with size-class binning for efficient tensor allocation.

    Maintains separate free lists for each size class, allowing O(1)
    allocation when a suitable block is available.

    Usage:
        pool = MemoryPool(device="cuda")

        # Allocate tensor
        tensor = pool.allocate(shape=(64, 256), dtype=torch.float32)

        # Use tensor...

        # Return to pool
        pool.deallocate(tensor)

        # Or use context manager
        with pool.allocate_temp(shape=(64, 256)) as tensor:
            # tensor is automatically returned when done
            pass
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_cached_bytes: int = 1 << 30,  # 1GB default
        enable_defrag: bool = True,
    ):
        """
        Initialize the memory pool.

        Args:
            device: CUDA device to allocate on
            max_cached_bytes: Maximum bytes to keep cached
            enable_defrag: Whether to enable defragmentation
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._max_cached_bytes = max_cached_bytes
        self._enable_defrag = enable_defrag

        # Free lists by size class
        self._free_blocks: Dict[int, List[CachedBlock]] = defaultdict(list)

        # Active allocations (tensor id -> block)
        self._active_blocks: Dict[int, CachedBlock] = {}

        # Statistics
        self.stats = AllocationStats()

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"MemoryPool initialized on {device}, max cache: {max_cached_bytes / 1e9:.2f}GB")

    @property
    def device(self) -> torch.device:
        """Get the device for this pool."""
        return self._device

    @property
    def cached_bytes(self) -> int:
        """Total bytes currently cached."""
        return self.stats.bytes_cached

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Allocate a tensor from the pool.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Allocated tensor (may be from cache)
        """
        # Calculate required size
        numel = 1
        for dim in shape:
            numel *= dim
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_bytes = numel * element_size

        # Get size class
        size_class = SizeClass.get_size_class(required_bytes)

        with self._lock:
            self.stats.total_allocations += 1

            # Check free list for this size class
            if self._free_blocks[size_class]:
                block = self._free_blocks[size_class].pop()
                self.stats.cache_hits += 1
                self.stats.bytes_cached -= block.actual_size

                # Reshape/view the cached tensor
                block.in_use = True
                block.allocation_count += 1

                # Create view with requested shape
                tensor = block.data.view(-1)[:numel].view(shape)

                # Track active allocation
                self._active_blocks[id(tensor)] = block

                return tensor

            # Cache miss - need to allocate new
            self.stats.cache_misses += 1

        # Allocate new tensor (outside lock for performance)
        # Allocate with size class to enable reuse
        padded_numel = size_class // element_size
        storage = torch.empty(padded_numel, dtype=dtype, device=self._device)

        # Create block
        block = CachedBlock(
            data=storage,
            size_class=size_class,
            actual_size=size_class,
            device=self._device,
            in_use=True,
            allocation_count=1,
        )

        # Create view with requested shape
        tensor = storage[:numel].view(shape)

        with self._lock:
            self._active_blocks[id(tensor)] = block
            self.stats.bytes_allocated += size_class
            self.stats.peak_bytes_allocated = max(
                self.stats.peak_bytes_allocated,
                self.stats.bytes_allocated
            )

        return tensor

    def deallocate(self, tensor: torch.Tensor) -> bool:
        """
        Return a tensor to the pool.

        Args:
            tensor: Tensor to deallocate

        Returns:
            True if tensor was from this pool and deallocated
        """
        tensor_id = id(tensor)

        with self._lock:
            if tensor_id not in self._active_blocks:
                return False

            block = self._active_blocks.pop(tensor_id)
            block.in_use = False

            self.stats.total_deallocations += 1

            # Check if we should cache this block
            if self.stats.bytes_cached + block.actual_size <= self._max_cached_bytes:
                self._free_blocks[block.size_class].append(block)
                self.stats.bytes_cached += block.actual_size
            else:
                # Release memory
                self.stats.bytes_allocated -= block.actual_size
                # Block will be garbage collected

            return True

    def allocate_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Allocate a tensor with the same shape and dtype.

        Args:
            tensor: Template tensor

        Returns:
            New tensor from pool
        """
        return self.allocate(tensor.shape, tensor.dtype)

    def allocate_temp(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """
        Context manager for temporary tensor allocation.

        Args:
            shape: Tensor shape
            dtype: Data type

        Yields:
            Allocated tensor (automatically deallocated on exit)
        """
        class TempAllocation:
            def __init__(self, pool: MemoryPool, shape: Tuple[int, ...], dtype: torch.dtype):
                self.pool = pool
                self.shape = shape
                self.dtype = dtype
                self.tensor = None

            def __enter__(self):
                self.tensor = self.pool.allocate(self.shape, self.dtype)
                return self.tensor

            def __exit__(self, *args):
                if self.tensor is not None:
                    self.pool.deallocate(self.tensor)

        return TempAllocation(self, shape, dtype)

    def clear(self) -> None:
        """Clear all cached blocks."""
        with self._lock:
            self._free_blocks.clear()
            self.stats.bytes_cached = 0

    def defragment(self) -> int:
        """
        Defragment the pool by releasing unused blocks.

        Returns:
            Number of bytes released
        """
        if not self._enable_defrag:
            return 0

        released = 0
        with self._lock:
            for size_class in list(self._free_blocks.keys()):
                blocks = self._free_blocks[size_class]
                # Keep only half the blocks in each class
                keep = len(blocks) // 2
                released_blocks = blocks[keep:]
                self._free_blocks[size_class] = blocks[:keep]

                for block in released_blocks:
                    released += block.actual_size
                    self.stats.bytes_cached -= block.actual_size
                    self.stats.bytes_allocated -= block.actual_size

        if released > 0:
            logger.debug(f"Defragmented {released / 1e6:.2f}MB")

        return released

    def get_stats(self) -> Dict[str, any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "total_allocations": self.stats.total_allocations,
                "total_deallocations": self.stats.total_deallocations,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "hit_rate": self.stats.hit_rate,
                "bytes_allocated": self.stats.bytes_allocated,
                "bytes_cached": self.stats.bytes_cached,
                "peak_bytes_allocated": self.stats.peak_bytes_allocated,
                "num_size_classes": len(self._free_blocks),
                "active_tensors": len(self._active_blocks),
            }

    def __repr__(self) -> str:
        return (
            f"MemoryPool(device={self._device}, "
            f"cached={self.stats.bytes_cached / 1e6:.1f}MB, "
            f"hit_rate={self.stats.hit_rate:.1%})"
        )


class TensorCache:
    """
    High-level cache for frequently used tensor shapes.

    Maintains pre-allocated tensors for common shapes to avoid
    allocation overhead entirely.
    """

    def __init__(
        self,
        pool: Optional[MemoryPool] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize tensor cache.

        Args:
            pool: Memory pool to use (creates one if None)
            device: Device for allocations
        """
        self._pool = pool or MemoryPool(device=device)
        self._cache: Dict[Tuple, List[torch.Tensor]] = defaultdict(list)
        self._lock = threading.Lock()

    def get(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        zero: bool = False,
    ) -> torch.Tensor:
        """
        Get a tensor from the cache or allocate new.

        Args:
            shape: Tensor shape
            dtype: Data type
            zero: Whether to zero the tensor

        Returns:
            Tensor from cache or newly allocated
        """
        key = (shape, dtype)

        with self._lock:
            if self._cache[key]:
                tensor = self._cache[key].pop()
                if zero:
                    tensor.zero_()
                return tensor

        # Allocate new
        tensor = self._pool.allocate(shape, dtype)
        if zero:
            tensor.zero_()
        return tensor

    def put(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the cache.

        Args:
            tensor: Tensor to cache
        """
        key = (tuple(tensor.shape), tensor.dtype)

        with self._lock:
            self._cache[key].append(tensor)

    def prealloc(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        count: int = 4,
    ) -> None:
        """
        Pre-allocate tensors for a given shape.

        Args:
            shape: Tensor shape
            dtype: Data type
            count: Number of tensors to pre-allocate
        """
        key = (shape, dtype)
        tensors = [self._pool.allocate(shape, dtype) for _ in range(count)]

        with self._lock:
            self._cache[key].extend(tensors)

    def clear(self) -> None:
        """Clear all cached tensors."""
        with self._lock:
            for tensors in self._cache.values():
                for tensor in tensors:
                    self._pool.deallocate(tensor)
            self._cache.clear()


# Global memory pool singleton
_global_pool: Optional[MemoryPool] = None


def get_memory_pool(device: Optional[torch.device] = None) -> MemoryPool:
    """
    Get or create the global memory pool.

    Args:
        device: Device for the pool (only used on first call)

    Returns:
        The global memory pool
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = MemoryPool(device=device)
    return _global_pool


def reset_memory_pool() -> None:
    """Reset the global memory pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.clear()
    _global_pool = None
