"""
Unit tests for the Kitsune profiler module.
"""

import pytest
import torch
import torch.nn as nn

from kitsune.profiler import CUDATimer, MemoryTracker, Profiler
from kitsune.profiler.metrics import Metrics, MetricsCollector, calculate_speedup


class TestCUDATimer:
    """Tests for CUDATimer."""

    def test_timer_creation(self):
        """Test timer can be created."""
        timer = CUDATimer()
        assert timer.enabled
        assert timer.device == 0

    def test_start_stop(self):
        """Test basic start/stop timing."""
        timer = CUDATimer()
        timer.start("test_op")
        result = timer.stop("test_op")

        assert result.name == "test_op"
        assert result.wall_time_ms >= 0
        assert result.cuda_time_ms >= 0

    def test_context_manager(self):
        """Test timing via context manager."""
        timer = CUDATimer()

        with timer.time("context_test"):
            # Simulate some work
            x = torch.randn(100, 100)
            _ = x @ x.T

        result = timer.get_result("context_test")
        assert result is not None
        assert result.cuda_time_ms >= 0

    def test_multiple_timings(self):
        """Test timing multiple operations."""
        timer = CUDATimer()

        timer.start("op1")
        timer.stop("op1")

        timer.start("op2")
        timer.stop("op2")

        assert "op1" in timer.results
        assert "op2" in timer.results

    def test_stop_without_start_raises(self):
        """Test that stopping without starting raises error."""
        timer = CUDATimer()

        with pytest.raises(ValueError, match="was not started"):
            timer.stop("nonexistent")

    def test_clear(self):
        """Test clearing results."""
        timer = CUDATimer()

        with timer.time("test"):
            pass

        assert len(timer.results) > 0
        timer.clear()
        assert len(timer.results) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_timing_accuracy(self):
        """Test CUDA timing is reasonably accurate."""
        timer = CUDATimer()

        def work():
            x = torch.randn(1000, 1000, device="cuda")
            for _ in range(10):
                x = x @ x.T

        result = timer.time_iterations("cuda_work", work, iterations=5, warmup=2)
        assert result.iterations == 5
        assert result.cuda_time_ms > 0


class TestMemoryTracker:
    """Tests for MemoryTracker."""

    def test_tracker_creation(self):
        """Test tracker can be created."""
        tracker = MemoryTracker()
        assert tracker.enabled
        assert tracker.device == 0

    def test_snapshot(self):
        """Test taking memory snapshot."""
        tracker = MemoryTracker()
        snapshot = tracker.snapshot("test")

        assert snapshot.name == "test"
        assert snapshot.allocated_bytes >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_track_context_manager(self):
        """Test tracking via context manager."""
        tracker = MemoryTracker()

        with tracker.track("allocation"):
            x = torch.randn(1000, 1000, device="cuda")

        delta = tracker.get_delta("allocation")
        assert delta is not None
        assert delta.peak_allocated_bytes > 0

    def test_compare_snapshots(self):
        """Test comparing two snapshots."""
        tracker = MemoryTracker()

        tracker.snapshot("before")
        tracker.snapshot("after")

        delta = tracker.compare("before", "after")
        assert delta is not None

    def test_clear(self):
        """Test clearing tracker data."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        assert tracker.get_snapshot("test") is not None
        tracker.clear()
        assert tracker.get_snapshot("test") is None


class TestMetrics:
    """Tests for Metrics and MetricsCollector."""

    def test_metrics_creation(self):
        """Test creating a Metrics object."""
        m = Metrics(
            iteration=0,
            batch_size=32,
            forward_time_ms=10.0,
            backward_time_ms=15.0,
            optimizer_time_ms=5.0,
            total_time_ms=30.0,
            memory_allocated_mb=100.0,
            memory_peak_mb=150.0,
        )

        assert m.throughput == pytest.approx(32 / 0.030, rel=0.01)
        assert m.forward_pct == pytest.approx(33.33, rel=0.01)

    def test_collector_add_and_aggregate(self):
        """Test adding metrics and aggregating."""
        collector = MetricsCollector(warmup_iterations=2)

        for i in range(5):
            m = Metrics(
                iteration=i,
                batch_size=32,
                forward_time_ms=10.0,
                backward_time_ms=15.0,
                optimizer_time_ms=5.0,
                total_time_ms=30.0,
                memory_allocated_mb=100.0,
                memory_peak_mb=150.0,
            )
            collector.add(m)

        # Should skip 2 warmup iterations
        agg = collector.aggregate(skip_warmup=True)
        assert agg is not None
        assert agg.count == 3  # 5 - 2 warmup

    def test_speedup_calculation(self):
        """Test speedup calculation."""
        baseline = 100.0
        optimized = 50.0

        speedup = calculate_speedup(baseline, optimized)
        assert speedup == pytest.approx(2.0)


class TestProfiler:
    """Tests for the high-level Profiler."""

    def test_profiler_creation(self):
        """Test profiler can be created."""
        profiler = Profiler()
        assert profiler.enabled

    def test_profile_context_manager(self):
        """Test profiling via context manager."""
        profiler = Profiler()

        with profiler.profile("test_op") as result:
            x = torch.randn(100, 100)
            _ = x @ x.T

        assert result.name == "test_op"
        assert result.timing is not None

    def test_iteration_tracking(self):
        """Test iteration-level tracking."""
        profiler = Profiler(warmup_iterations=2)

        for i in range(5):
            profiler.start_iteration(i, batch_size=32)

            with profiler.profile("forward"):
                pass

            with profiler.profile("backward"):
                pass

            profiler.end_iteration()

        agg = profiler.get_aggregated_metrics()
        assert agg is not None
        assert agg.count == 3  # 5 - 2 warmup

    def test_clear(self):
        """Test clearing profiler data."""
        profiler = Profiler()

        with profiler.profile("test"):
            pass

        assert profiler.get_result("test") is not None
        profiler.clear()
        assert profiler.get_result("test") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
