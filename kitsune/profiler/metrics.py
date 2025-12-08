"""
Metrics - Performance metrics collection and aggregation

Provides utilities for collecting, aggregating, and analyzing
training performance metrics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Metrics:
    """
    Container for a single metric measurement.

    Stores timing, throughput, and memory metrics for a single
    training iteration or batch.
    """

    iteration: int
    batch_size: int
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    total_time_ms: float
    memory_allocated_mb: float
    memory_peak_mb: float
    loss: Optional[float] = None

    @property
    def throughput(self) -> float:
        """Samples per second."""
        if self.total_time_ms <= 0:
            return 0.0
        return self.batch_size / (self.total_time_ms / 1000)

    @property
    def forward_pct(self) -> float:
        """Percentage of time spent in forward pass."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.forward_time_ms / self.total_time_ms) * 100

    @property
    def backward_pct(self) -> float:
        """Percentage of time spent in backward pass."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.backward_time_ms / self.total_time_ms) * 100

    @property
    def optimizer_pct(self) -> float:
        """Percentage of time spent in optimizer step."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.optimizer_time_ms / self.total_time_ms) * 100


@dataclass
class AggregatedMetrics:
    """Aggregated statistics over multiple measurements."""

    count: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    mean_throughput: float
    mean_memory_mb: float
    peak_memory_mb: float

    def __repr__(self) -> str:
        return (
            f"AggregatedMetrics(n={self.count}, "
            f"time={self.mean_time_ms:.2f}±{self.std_time_ms:.2f}ms, "
            f"throughput={self.mean_throughput:.1f} samples/s, "
            f"peak_mem={self.peak_memory_mb:.1f}MB)"
        )


@dataclass
class MetricsCollector:
    """
    Collect and aggregate performance metrics over multiple iterations.

    Example:
        >>> collector = MetricsCollector()
        >>>
        >>> for i, batch in enumerate(dataloader):
        ...     with collector.measure(i, batch_size=32) as m:
        ...         m.time_forward(lambda: model(batch))
        ...         m.time_backward(lambda: loss.backward())
        ...         m.time_optimizer(lambda: optimizer.step())
        >>>
        >>> print(collector.summary())
    """

    warmup_iterations: int = 10
    _metrics: list[Metrics] = field(default_factory=list)
    _current_iteration: int = 0

    def add(self, metrics: Metrics) -> None:
        """Add a metrics measurement."""
        self._metrics.append(metrics)
        self._current_iteration = metrics.iteration

    def get_metrics(self, skip_warmup: bool = True) -> list[Metrics]:
        """
        Get collected metrics.

        Args:
            skip_warmup: If True, exclude warmup iterations

        Returns:
            List of Metrics objects
        """
        if skip_warmup:
            return [m for m in self._metrics if m.iteration >= self.warmup_iterations]
        return list(self._metrics)

    def aggregate(self, skip_warmup: bool = True) -> Optional[AggregatedMetrics]:
        """
        Aggregate metrics into summary statistics.

        Args:
            skip_warmup: If True, exclude warmup iterations

        Returns:
            AggregatedMetrics or None if no metrics collected
        """
        metrics = self.get_metrics(skip_warmup)
        if not metrics:
            return None

        times = [m.total_time_ms for m in metrics]
        throughputs = [m.throughput for m in metrics]
        memories = [m.memory_allocated_mb for m in metrics]
        peak_memories = [m.memory_peak_mb for m in metrics]

        return AggregatedMetrics(
            count=len(metrics),
            mean_time_ms=statistics.mean(times),
            std_time_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time_ms=min(times),
            max_time_ms=max(times),
            mean_throughput=statistics.mean(throughputs),
            mean_memory_mb=statistics.mean(memories),
            peak_memory_mb=max(peak_memories),
        )

    def get_phase_breakdown(self, skip_warmup: bool = True) -> dict[str, float]:
        """
        Get average time breakdown by phase.

        Returns:
            Dictionary with average time per phase in ms
        """
        metrics = self.get_metrics(skip_warmup)
        if not metrics:
            return {}

        return {
            "forward_ms": statistics.mean(m.forward_time_ms for m in metrics),
            "backward_ms": statistics.mean(m.backward_time_ms for m in metrics),
            "optimizer_ms": statistics.mean(m.optimizer_time_ms for m in metrics),
            "total_ms": statistics.mean(m.total_time_ms for m in metrics),
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._current_iteration = 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        agg = self.aggregate()
        if agg is None:
            return "No metrics collected."

        breakdown = self.get_phase_breakdown()

        lines = [
            "=" * 60,
            "Performance Metrics Summary",
            "=" * 60,
            f"Iterations: {agg.count} (excluding {self.warmup_iterations} warmup)",
            "",
            "Timing:",
            f"  Total:     {agg.mean_time_ms:8.2f} ± {agg.std_time_ms:.2f} ms",
            f"  Forward:   {breakdown['forward_ms']:8.2f} ms ({breakdown['forward_ms']/breakdown['total_ms']*100:.1f}%)",
            f"  Backward:  {breakdown['backward_ms']:8.2f} ms ({breakdown['backward_ms']/breakdown['total_ms']*100:.1f}%)",
            f"  Optimizer: {breakdown['optimizer_ms']:8.2f} ms ({breakdown['optimizer_ms']/breakdown['total_ms']*100:.1f}%)",
            f"  Min/Max:   {agg.min_time_ms:.2f} / {agg.max_time_ms:.2f} ms",
            "",
            "Throughput:",
            f"  {agg.mean_throughput:.1f} samples/second",
            "",
            "Memory:",
            f"  Average:   {agg.mean_memory_mb:.1f} MB",
            f"  Peak:      {agg.peak_memory_mb:.1f} MB",
            "=" * 60,
        ]

        return "\n".join(lines)


def calculate_speedup(baseline_time_ms: float, optimized_time_ms: float) -> float:
    """
    Calculate speedup ratio.

    Args:
        baseline_time_ms: Time for baseline implementation
        optimized_time_ms: Time for optimized implementation

    Returns:
        Speedup ratio (e.g., 2.0 means 2x faster)
    """
    if optimized_time_ms <= 0:
        return float("inf")
    return baseline_time_ms / optimized_time_ms


def calculate_speedup_percent(baseline_time_ms: float, optimized_time_ms: float) -> float:
    """
    Calculate speedup as a percentage improvement.

    Args:
        baseline_time_ms: Time for baseline implementation
        optimized_time_ms: Time for optimized implementation

    Returns:
        Percentage improvement (e.g., 100 means 100% faster / 2x speedup)
    """
    speedup = calculate_speedup(baseline_time_ms, optimized_time_ms)
    return (speedup - 1) * 100
