"""
CUDA Timer - High-precision timing using CUDA events

Uses CUDA events for accurate GPU timing that accounts for
asynchronous kernel execution.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch


@dataclass
class TimingResult:
    """Result from a timing measurement."""

    name: str
    cuda_time_ms: float
    wall_time_ms: float
    iterations: int = 1

    @property
    def cuda_time_per_iter_ms(self) -> float:
        """CUDA time per iteration in milliseconds."""
        return self.cuda_time_ms / self.iterations

    @property
    def wall_time_per_iter_ms(self) -> float:
        """Wall time per iteration in milliseconds."""
        return self.wall_time_ms / self.iterations

    def __repr__(self) -> str:
        if self.iterations > 1:
            return (
                f"TimingResult(name='{self.name}', "
                f"cuda={self.cuda_time_per_iter_ms:.3f}ms/iter, "
                f"wall={self.wall_time_per_iter_ms:.3f}ms/iter, "
                f"iters={self.iterations})"
            )
        return (
            f"TimingResult(name='{self.name}', "
            f"cuda={self.cuda_time_ms:.3f}ms, "
            f"wall={self.wall_time_ms:.3f}ms)"
        )


@dataclass
class CUDATimer:
    """
    High-precision timer using CUDA events.

    Provides accurate timing for GPU operations by using CUDA events
    which properly account for asynchronous kernel execution.

    Example:
        >>> timer = CUDATimer()
        >>> with timer.time("forward_pass"):
        ...     output = model(input)
        >>> print(timer.get_result("forward_pass"))

        >>> # Or manual timing
        >>> timer.start("backward")
        >>> loss.backward()
        >>> timer.stop("backward")
        >>> print(timer.results)
    """

    device: int = 0
    enabled: bool = True
    _results: dict[str, TimingResult] = field(default_factory=dict)
    _active_timers: dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize CUDA context if needed."""
        if self.enabled and torch.cuda.is_available():
            # Ensure CUDA is initialized
            torch.cuda.set_device(self.device)

    def _create_events(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create a pair of CUDA events for timing."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return start, end

    def start(self, name: str) -> None:
        """
        Start timing a named operation.

        Args:
            name: Identifier for this timing measurement
        """
        if not self.enabled or not torch.cuda.is_available():
            self._active_timers[name] = (None, None, time.perf_counter())
            return

        torch.cuda.synchronize(self.device)
        start_event, end_event = self._create_events()
        start_event.record()
        wall_start = time.perf_counter()
        self._active_timers[name] = (start_event, end_event, wall_start)

    def stop(self, name: str) -> TimingResult:
        """
        Stop timing a named operation.

        Args:
            name: Identifier for the timing measurement to stop

        Returns:
            TimingResult with measured times
        """
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        start_event, end_event, wall_start = self._active_timers.pop(name)
        wall_end = time.perf_counter()
        wall_time_ms = (wall_end - wall_start) * 1000

        if start_event is None or not torch.cuda.is_available():
            # CUDA not available, use wall time only
            result = TimingResult(
                name=name,
                cuda_time_ms=wall_time_ms,
                wall_time_ms=wall_time_ms,
            )
        else:
            end_event.record()
            torch.cuda.synchronize(self.device)
            cuda_time_ms = start_event.elapsed_time(end_event)
            result = TimingResult(
                name=name,
                cuda_time_ms=cuda_time_ms,
                wall_time_ms=wall_time_ms,
            )

        self._results[name] = result
        return result

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for timing a block of code.

        Args:
            name: Identifier for this timing measurement

        Example:
            >>> with timer.time("forward"):
            ...     output = model(input)
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def time_iterations(
        self,
        name: str,
        func: callable,
        iterations: int = 100,
        warmup: int = 10,
    ) -> TimingResult:
        """
        Time a function over multiple iterations.

        Args:
            name: Identifier for this timing measurement
            func: Function to time (called with no arguments)
            iterations: Number of iterations to time
            warmup: Number of warmup iterations (not timed)

        Returns:
            TimingResult with total and per-iteration times
        """
        # Warmup
        for _ in range(warmup):
            func()

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # Time iterations
        self.start(name)
        for _ in range(iterations):
            func()
        result = self.stop(name)

        # Update with iteration count
        result.iterations = iterations
        self._results[name] = result
        return result

    def get_result(self, name: str) -> Optional[TimingResult]:
        """Get the timing result for a named operation."""
        return self._results.get(name)

    @property
    def results(self) -> dict[str, TimingResult]:
        """Get all timing results."""
        return dict(self._results)

    def clear(self) -> None:
        """Clear all timing results."""
        self._results.clear()
        self._active_timers.clear()

    def summary(self) -> str:
        """Generate a summary of all timing results."""
        if not self._results:
            return "No timing results recorded."

        lines = ["=" * 60, "CUDA Timing Summary", "=" * 60]

        # Sort by CUDA time descending
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: r.cuda_time_ms,
            reverse=True,
        )

        for result in sorted_results:
            if result.iterations > 1:
                lines.append(
                    f"  {result.name:30s} | "
                    f"CUDA: {result.cuda_time_per_iter_ms:8.3f} ms/iter | "
                    f"Wall: {result.wall_time_per_iter_ms:8.3f} ms/iter | "
                    f"Iters: {result.iterations}"
                )
            else:
                lines.append(
                    f"  {result.name:30s} | "
                    f"CUDA: {result.cuda_time_ms:8.3f} ms | "
                    f"Wall: {result.wall_time_ms:8.3f} ms"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def synchronize_and_time(func: callable, device: int = 0) -> float:
    """
    Time a function with CUDA synchronization.

    Simple utility for one-off timing measurements.

    Args:
        func: Function to time
        device: CUDA device index

    Returns:
        Elapsed time in milliseconds
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)
    else:
        start = time.perf_counter()
        func()
        return (time.perf_counter() - start) * 1000
