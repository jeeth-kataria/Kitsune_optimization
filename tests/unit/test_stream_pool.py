"""
Unit tests for CUDA stream pool.
"""

import pytest
import torch

from kitsune.cuda.stream_pool import (
    CUDAStream,
    StreamPool,
    StreamScheduler,
    get_stream_pool,
    reset_stream_pool,
)


class TestCUDAStream:
    """Tests for CUDAStream wrapper."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stream_creation(self):
        """Test stream wrapper creation."""
        stream = CUDAStream(stream_id=0)
        assert stream.stream_id == 0
        assert stream.stream is not None
        assert stream.stats.tasks_executed == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stream_context(self):
        """Test stream context manager."""
        stream = CUDAStream(stream_id=0)

        with stream.context():
            # Operations in this block run on the stream
            x = torch.zeros(100, device="cuda")
            y = x + 1

        # Should complete without error
        stream.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_event_recording(self):
        """Test event recording on stream."""
        stream = CUDAStream(stream_id=0)

        with stream.context():
            x = torch.zeros(100, device="cuda")
            event = stream.record_event()

        assert event is not None
        event.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stream_wait(self):
        """Test stream waiting for another stream."""
        stream1 = CUDAStream(stream_id=0)
        stream2 = CUDAStream(stream_id=1)

        # Do work on stream1
        with stream1.context():
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            stream1.record_event()

        # Make stream2 wait for stream1
        with stream2.context():
            stream2.wait_stream(stream1)
            # This should only execute after stream1 completes
            z = torch.zeros(100, device="cuda")

        stream2.synchronize()


class TestStreamPool:
    """Tests for StreamPool."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pool_creation(self):
        """Test pool creation."""
        pool = StreamPool(num_streams=4)
        assert pool.num_streams == 4
        assert pool.enabled

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_stream(self):
        """Test getting stream by ID."""
        pool = StreamPool(num_streams=4)

        stream = pool.get_stream(0)
        assert stream.stream_id == 0

        stream = pool.get_stream(2)
        assert stream.stream_id == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_round_robin(self):
        """Test round-robin stream assignment."""
        pool = StreamPool(num_streams=4, include_default=False)

        ids = [pool.next_stream().stream_id for _ in range(8)]

        # Should cycle through 0, 1, 2, 3, 0, 1, 2, 3
        assert ids == [0, 1, 2, 3, 0, 1, 2, 3]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_least_loaded(self):
        """Test getting least loaded stream."""
        pool = StreamPool(num_streams=4)

        # Initially all have 0 tasks
        stream = pool.get_least_loaded()
        assert stream is not None

        # Increment one stream's count
        pool.streams[0].stats.tasks_executed = 10

        # Should get a different stream
        stream = pool.get_least_loaded()
        assert stream.stream_id != 0 or pool.num_streams == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_synchronize_all(self):
        """Test synchronizing all streams."""
        pool = StreamPool(num_streams=4)

        # Launch work on all streams
        for i in range(4):
            stream = pool.get_stream(i)
            with stream.context():
                x = torch.randn(100, 100, device="cuda")
                y = torch.matmul(x, x)

        # Should block until all complete
        pool.synchronize_all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_on_stream(self):
        """Test executing function on stream."""
        pool = StreamPool(num_streams=4)

        result = []

        def my_kernel():
            x = torch.randn(100, device="cuda")
            result.append(x.sum().item())

        event = pool.execute_on_stream(
            stream_id=0,
            func=my_kernel,
        )

        event.synchronize()
        assert len(result) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stats_tracking(self):
        """Test statistics tracking."""
        pool = StreamPool(num_streams=2)

        # Execute some work
        for _ in range(5):
            pool.execute_on_stream(0, lambda: None)
        for _ in range(3):
            pool.execute_on_stream(1, lambda: None)

        pool.synchronize_all()

        stats = pool.get_stats()
        assert stats[0].tasks_executed == 5
        assert stats[1].tasks_executed == 3


class TestStreamScheduler:
    """Tests for StreamScheduler."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        pool = StreamPool(num_streams=4)
        scheduler = StreamScheduler(pool)
        assert scheduler.pool is pool

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_assign_stream_no_deps(self):
        """Test stream assignment for task without dependencies."""
        pool = StreamPool(num_streams=4, include_default=False)
        scheduler = StreamScheduler(pool)

        # Tasks without dependencies get round-robin assignment
        stream_id = scheduler.assign_stream(task_id=0, dependencies=[])
        assert 0 <= stream_id < 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_assign_stream_preferred(self):
        """Test stream assignment with preference."""
        pool = StreamPool(num_streams=4)
        scheduler = StreamScheduler(pool)

        stream_id = scheduler.assign_stream(
            task_id=0,
            dependencies=[],
            preferred_stream=2,
        )
        assert stream_id == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_task(self):
        """Test task execution on scheduler."""
        pool = StreamPool(num_streams=4)
        scheduler = StreamScheduler(pool)

        results = []

        def kernel():
            results.append("executed")

        event = scheduler.execute_task(
            task_id=0,
            stream_id=0,
            func=kernel,
            dependencies=[],
        )

        event.synchronize()
        assert "executed" in results

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_with_dependencies(self):
        """Test task execution with dependencies."""
        pool = StreamPool(num_streams=4)
        scheduler = StreamScheduler(pool)

        execution_order = []

        def task1():
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)  # Significant work
            execution_order.append(1)

        def task2():
            execution_order.append(2)

        # Execute task1 first
        scheduler.execute_task(
            task_id=1,
            stream_id=0,
            func=task1,
            dependencies=[],
        )

        # Execute task2 with dependency on task1
        scheduler.execute_task(
            task_id=2,
            stream_id=1,
            func=task2,
            dependencies=[1],
        )

        scheduler.synchronize()

        # Task 2 should complete after task 1
        assert 1 in execution_order
        assert 2 in execution_order


class TestGlobalStreamPool:
    """Tests for global stream pool singleton."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_global_pool(self):
        """Test getting global pool."""
        reset_stream_pool()
        pool = get_stream_pool(num_streams=4)
        assert pool is not None

        # Same pool returned on subsequent calls
        pool2 = get_stream_pool()
        assert pool2 is pool

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_global_pool(self):
        """Test resetting global pool."""
        pool1 = get_stream_pool()
        reset_stream_pool()
        pool2 = get_stream_pool()

        # Should be different pools
        assert pool1 is not pool2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
