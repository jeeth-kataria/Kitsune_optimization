"""
Profiler and Reporter - High-level profiling API

Provides a unified interface for profiling PyTorch training loops
with timing, memory tracking, and metrics collection.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import torch

from .cuda_timer import CUDATimer, TimingResult
from .memory_tracker import MemoryTracker, MemoryDelta
from .metrics import Metrics, MetricsCollector, AggregatedMetrics


@dataclass
class ProfileResult:
    """Complete profiling result for an operation."""

    name: str
    timing: Optional[TimingResult] = None
    memory: Optional[MemoryDelta] = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"ProfileResult(name='{self.name}'"]
        if self.timing:
            parts.append(f"cuda={self.timing.cuda_time_ms:.2f}ms")
        if self.memory:
            parts.append(f"peak_mem={self.memory.peak_mb:.1f}MB")
        return ", ".join(parts) + ")"


@dataclass
class Profiler:
    """
    High-level profiler for PyTorch training loops.

    Combines CUDA timing, memory tracking, and metrics collection
    into a unified profiling interface.

    Example:
        >>> profiler = Profiler()
        >>>
        >>> # Profile individual operations
        >>> with profiler.profile("forward"):
        ...     output = model(input)
        >>>
        >>> # Or profile full training steps
        >>> for i, batch in enumerate(dataloader):
        ...     profiler.start_iteration(i, batch_size=32)
        ...     with profiler.profile("forward"):
        ...         output = model(batch)
        ...     with profiler.profile("backward"):
        ...         loss.backward()
        ...     with profiler.profile("optimizer"):
        ...         optimizer.step()
        ...     profiler.end_iteration()
        >>>
        >>> print(profiler.summary())
    """

    device: int = 0
    enabled: bool = True
    warmup_iterations: int = 10

    _timer: CUDATimer = field(default_factory=lambda: CUDATimer())
    _memory: MemoryTracker = field(default_factory=lambda: MemoryTracker())
    _metrics: MetricsCollector = field(default_factory=lambda: MetricsCollector())
    _results: dict[str, ProfileResult] = field(default_factory=dict)

    # Current iteration state
    _current_iteration: int = -1
    _current_batch_size: int = 0
    _iteration_start_time: float = 0.0
    _phase_times: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize components with device."""
        self._timer = CUDATimer(device=self.device, enabled=self.enabled)
        self._memory = MemoryTracker(device=self.device, enabled=self.enabled)
        self._metrics = MetricsCollector(warmup_iterations=self.warmup_iterations)

    @contextmanager
    def profile(self, name: str) -> Generator[ProfileResult, None, None]:
        """
        Context manager for profiling a block of code.

        Args:
            name: Identifier for this profiling session

        Yields:
            ProfileResult that will be populated on exit
        """
        result = ProfileResult(name=name)

        if not self.enabled:
            yield result
            return

        # Start timing and memory tracking
        self._memory.reset_peak_stats()
        self._timer.start(name)

        # Track memory
        start_mem = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0

        try:
            yield result
        finally:
            # Stop timing
            timing = self._timer.stop(name)
            result.timing = timing

            # Get memory delta
            end_mem = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
            peak_mem = torch.cuda.max_memory_allocated(self.device) if torch.cuda.is_available() else 0

            result.memory = MemoryDelta(
                name=name,
                start_allocated_bytes=start_mem,
                end_allocated_bytes=end_mem,
                peak_allocated_bytes=peak_mem,
                delta_bytes=end_mem - start_mem,
            )

            self._results[name] = result

            # Record phase time if in iteration
            if self._current_iteration >= 0:
                self._phase_times[name] = timing.cuda_time_ms

    def start_iteration(self, iteration: int, batch_size: int) -> None:
        """
        Mark the start of a training iteration.

        Args:
            iteration: Iteration number
            batch_size: Batch size for this iteration
        """
        self._current_iteration = iteration
        self._current_batch_size = batch_size
        self._phase_times.clear()
        self._iteration_start_time = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self._memory.reset_peak_stats()

    def end_iteration(self, loss: Optional[float] = None) -> Optional[Metrics]:
        """
        Mark the end of a training iteration and record metrics.

        Args:
            loss: Optional loss value for this iteration

        Returns:
            Metrics for this iteration
        """
        if self._current_iteration < 0:
            return None

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        total_time = (time.perf_counter() - self._iteration_start_time) * 1000

        metrics = Metrics(
            iteration=self._current_iteration,
            batch_size=self._current_batch_size,
            forward_time_ms=self._phase_times.get("forward", 0.0),
            backward_time_ms=self._phase_times.get("backward", 0.0),
            optimizer_time_ms=self._phase_times.get("optimizer", 0.0),
            total_time_ms=total_time,
            memory_allocated_mb=self._memory.current_allocated_mb,
            memory_peak_mb=self._memory.peak_allocated_mb,
            loss=loss,
        )

        self._metrics.add(metrics)
        self._current_iteration = -1

        return metrics

    def time_function(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
    ) -> TimingResult:
        """
        Time a function over multiple iterations.

        Args:
            name: Name for this timing measurement
            func: Function to time
            iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            TimingResult with timing statistics
        """
        return self._timer.time_iterations(name, func, iterations, warmup)

    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get profiling result by name."""
        return self._results.get(name)

    def get_aggregated_metrics(self) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics from collected iterations."""
        return self._metrics.aggregate()

    def clear(self) -> None:
        """Clear all profiling data."""
        self._timer.clear()
        self._memory.clear()
        self._metrics.clear()
        self._results.clear()
        self._phase_times.clear()
        self._current_iteration = -1

    def summary(self) -> str:
        """Generate comprehensive profiling summary."""
        lines = [
            "=" * 70,
            "Kitsune Profiler Summary",
            "=" * 70,
        ]

        # Device info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            lines.append(f"\nDevice: {props.name}")
            lines.append(f"Total Memory: {props.total_memory / (1024**3):.1f} GB")

        # Aggregated metrics
        agg = self._metrics.aggregate()
        if agg:
            lines.append(f"\n--- Training Metrics ({agg.count} iterations) ---")
            lines.append(f"Average time per iteration: {agg.mean_time_ms:.2f} ± {agg.std_time_ms:.2f} ms")
            lines.append(f"Throughput: {agg.mean_throughput:.1f} samples/second")
            lines.append(f"Peak memory: {agg.peak_memory_mb:.1f} MB")

            # Phase breakdown
            breakdown = self._metrics.get_phase_breakdown()
            if breakdown:
                lines.append("\n--- Phase Breakdown ---")
                total = breakdown.get("total_ms", 1)
                for phase in ["forward", "backward", "optimizer"]:
                    key = f"{phase}_ms"
                    if key in breakdown:
                        pct = breakdown[key] / total * 100
                        lines.append(f"  {phase:12s}: {breakdown[key]:8.2f} ms ({pct:5.1f}%)")

        # Individual profiled operations
        if self._results:
            lines.append("\n--- Profiled Operations ---")
            for name, result in sorted(self._results.items()):
                if result.timing:
                    mem_str = ""
                    if result.memory:
                        mem_str = f" | Peak: {result.memory.peak_mb:.1f} MB"
                    lines.append(
                        f"  {name:30s}: {result.timing.cuda_time_ms:8.2f} ms{mem_str}"
                    )

        lines.append("=" * 70)
        return "\n".join(lines)

    def compare_with_baseline(
        self,
        baseline_time_ms: float,
        name: str = "optimized",
    ) -> str:
        """
        Generate comparison with baseline timing.

        Args:
            baseline_time_ms: Baseline time in milliseconds
            name: Name of the optimized implementation

        Returns:
            Formatted comparison string
        """
        agg = self._metrics.aggregate()
        if agg is None:
            return "No metrics collected for comparison."

        speedup = baseline_time_ms / agg.mean_time_ms
        improvement_pct = (speedup - 1) * 100

        lines = [
            "=" * 50,
            "Speedup Comparison",
            "=" * 50,
            f"Baseline:   {baseline_time_ms:.2f} ms",
            f"{name:10s}: {agg.mean_time_ms:.2f} ms",
            "",
            f"Speedup:    {speedup:.2f}x",
            f"Improvement: {improvement_pct:+.1f}%",
            "=" * 50,
        ]

        return "\n".join(lines)
