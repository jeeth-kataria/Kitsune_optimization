"""
Memory Tracker - GPU memory monitoring and analysis

Tracks GPU memory allocation, peak usage, and provides
utilities for memory profiling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state."""

    name: str
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in MB."""
        return self.allocated_bytes / (1024**2)

    @property
    def reserved_mb(self) -> float:
        """Reserved memory in MB."""
        return self.reserved_bytes / (1024**2)

    @property
    def max_allocated_mb(self) -> float:
        """Peak allocated memory in MB."""
        return self.max_allocated_bytes / (1024**2)

    @property
    def max_reserved_mb(self) -> float:
        """Peak reserved memory in MB."""
        return self.max_reserved_bytes / (1024**2)

    def __repr__(self) -> str:
        return (
            f"MemorySnapshot(name='{self.name}', "
            f"allocated={self.allocated_mb:.2f}MB, "
            f"peak={self.max_allocated_mb:.2f}MB)"
        )


@dataclass
class MemoryDelta:
    """Change in memory between two snapshots."""

    name: str
    start_allocated_bytes: int
    end_allocated_bytes: int
    peak_allocated_bytes: int
    delta_bytes: int

    @property
    def delta_mb(self) -> float:
        """Memory change in MB."""
        return self.delta_bytes / (1024**2)

    @property
    def peak_mb(self) -> float:
        """Peak memory during operation in MB."""
        return self.peak_allocated_bytes / (1024**2)

    def __repr__(self) -> str:
        sign = "+" if self.delta_bytes >= 0 else ""
        return (
            f"MemoryDelta(name='{self.name}', "
            f"delta={sign}{self.delta_mb:.2f}MB, "
            f"peak={self.peak_mb:.2f}MB)"
        )


@dataclass
class MemoryTracker:
    """
    Track GPU memory usage for profiling and optimization.

    Provides utilities to monitor memory allocation, track peak usage,
    and analyze memory patterns during training.

    Example:
        >>> tracker = MemoryTracker()
        >>> with tracker.track("forward_pass"):
        ...     output = model(input)
        >>> print(tracker.get_delta("forward_pass"))

        >>> # Manual tracking
        >>> tracker.snapshot("before_backward")
        >>> loss.backward()
        >>> tracker.snapshot("after_backward")
        >>> print(tracker.compare("before_backward", "after_backward"))
    """

    device: int = 0
    enabled: bool = True
    _snapshots: dict[str, MemorySnapshot] = field(default_factory=dict)
    _deltas: dict[str, MemoryDelta] = field(default_factory=dict)
    _tracking_stack: list[tuple[str, int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize CUDA context if needed."""
        if self.enabled and torch.cuda.is_available():
            torch.cuda.set_device(self.device)

    def _get_memory_stats(self) -> tuple[int, int, int, int]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return 0, 0, 0, 0

        return (
            torch.cuda.memory_allocated(self.device),
            torch.cuda.memory_reserved(self.device),
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
        )

    def snapshot(self, name: str) -> MemorySnapshot:
        """
        Take a snapshot of current memory state.

        Args:
            name: Identifier for this snapshot

        Returns:
            MemorySnapshot with current memory stats
        """
        allocated, reserved, max_allocated, max_reserved = self._get_memory_stats()
        snapshot = MemorySnapshot(
            name=name,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            max_allocated_bytes=max_allocated,
            max_reserved_bytes=max_reserved,
        )
        self._snapshots[name] = snapshot
        return snapshot

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def compare(self, start_name: str, end_name: str) -> Optional[MemoryDelta]:
        """
        Compare two snapshots.

        Args:
            start_name: Name of the starting snapshot
            end_name: Name of the ending snapshot

        Returns:
            MemoryDelta showing the change, or None if snapshots not found
        """
        start = self._snapshots.get(start_name)
        end = self._snapshots.get(end_name)

        if start is None or end is None:
            return None

        return MemoryDelta(
            name=f"{start_name} -> {end_name}",
            start_allocated_bytes=start.allocated_bytes,
            end_allocated_bytes=end.allocated_bytes,
            peak_allocated_bytes=end.max_allocated_bytes,
            delta_bytes=end.allocated_bytes - start.allocated_bytes,
        )

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for tracking memory during a block.

        Args:
            name: Identifier for this tracking session

        Example:
            >>> with tracker.track("model_forward"):
            ...     output = model(input)
        """
        # Reset peak stats for accurate peak measurement
        self.reset_peak_stats()

        start_allocated = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
        self._tracking_stack.append((name, start_allocated, 0))

        try:
            yield
        finally:
            end_allocated = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
            peak_allocated = torch.cuda.max_memory_allocated(self.device) if torch.cuda.is_available() else 0

            name, start_allocated, _ = self._tracking_stack.pop()

            delta = MemoryDelta(
                name=name,
                start_allocated_bytes=start_allocated,
                end_allocated_bytes=end_allocated,
                peak_allocated_bytes=peak_allocated,
                delta_bytes=end_allocated - start_allocated,
            )
            self._deltas[name] = delta

    def get_delta(self, name: str) -> Optional[MemoryDelta]:
        """Get the memory delta for a tracked operation."""
        return self._deltas.get(name)

    def get_snapshot(self, name: str) -> Optional[MemorySnapshot]:
        """Get a specific snapshot by name."""
        return self._snapshots.get(name)

    @property
    def current_allocated_mb(self) -> float:
        """Current allocated memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024**2)

    @property
    def current_reserved_mb(self) -> float:
        """Current reserved memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved(self.device) / (1024**2)

    @property
    def peak_allocated_mb(self) -> float:
        """Peak allocated memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024**2)

    @property
    def total_memory_mb(self) -> float:
        """Total GPU memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(self.device).total_memory / (1024**2)

    @property
    def free_memory_mb(self) -> float:
        """Approximately free GPU memory in MB."""
        return self.total_memory_mb - self.current_reserved_mb

    def clear(self) -> None:
        """Clear all snapshots and deltas."""
        self._snapshots.clear()
        self._deltas.clear()
        self._tracking_stack.clear()

    def summary(self) -> str:
        """Generate a summary of memory tracking results."""
        lines = ["=" * 60, "Memory Tracking Summary", "=" * 60]

        # Current state
        lines.append(f"\nCurrent State:")
        lines.append(f"  Allocated: {self.current_allocated_mb:.2f} MB")
        lines.append(f"  Reserved:  {self.current_reserved_mb:.2f} MB")
        lines.append(f"  Peak:      {self.peak_allocated_mb:.2f} MB")
        lines.append(f"  Total:     {self.total_memory_mb:.2f} MB")
        lines.append(f"  Free:      {self.free_memory_mb:.2f} MB")

        # Tracked operations
        if self._deltas:
            lines.append(f"\nTracked Operations:")
            for delta in self._deltas.values():
                sign = "+" if delta.delta_bytes >= 0 else ""
                lines.append(
                    f"  {delta.name:30s} | "
                    f"Delta: {sign}{delta.delta_mb:8.2f} MB | "
                    f"Peak: {delta.peak_mb:8.2f} MB"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def get_gpu_memory_info(device: int = 0) -> dict:
    """
    Get comprehensive GPU memory information.

    Args:
        device: CUDA device index

    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    props = torch.cuda.get_device_properties(device)

    return {
        "cuda_available": True,
        "device_name": props.name,
        "total_memory_mb": props.total_memory / (1024**2),
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        "utilization_pct": (
            torch.cuda.memory_allocated(device) / props.total_memory * 100
        ),
    }
