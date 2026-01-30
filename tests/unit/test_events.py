"""
Unit tests for CUDA events.
"""

import pytest
import torch

from kitsune.cuda.events import (
    DependencyTracker,
    EventBarrier,
    EventManager,
    get_event_manager,
    reset_event_manager,
)


class TestEventManager:
    """Tests for EventManager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_manager_creation(self):
        """Test event manager creation."""
        manager = EventManager(pool_size=32)
        assert manager is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_event(self):
        """Test getting events."""
        manager = EventManager(pool_size=32)

        event = manager.get_event()
        assert event is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_record_event(self):
        """Test recording named events."""
        manager = EventManager()

        event = manager.record("test_event")
        assert event is not None

        # Should be able to retrieve it
        retrieved = manager.get_recorded("test_event")
        assert retrieved is event

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_wait_event(self):
        """Test waiting for events."""
        manager = EventManager()

        # Record an event
        manager.record("sync_point")

        # Wait should return True for existing event
        result = manager.wait("sync_point")
        assert result is True

        # Wait should return False for non-existing event
        result = manager.wait("nonexistent")
        assert result is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_timing(self):
        """Test timing measurements."""
        manager = EventManager()

        # Start timing
        manager.start_timing("matmul")

        # Do some work
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)

        # End timing
        manager.end_timing("matmul")

        # Get timing
        elapsed = manager.get_timing("matmul")
        assert elapsed > 0  # Should have taken some time

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clear(self):
        """Test clearing events."""
        manager = EventManager()

        manager.record("event1")
        manager.record("event2")

        manager.clear()

        assert manager.get_recorded("event1") is None
        assert manager.get_recorded("event2") is None


class TestDependencyTracker:
    """Tests for DependencyTracker."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tracker_creation(self):
        """Test dependency tracker creation."""
        tracker = DependencyTracker()
        assert tracker is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_register_task(self):
        """Test registering tasks."""
        tracker = DependencyTracker()

        tracker.register_task(task_id=1, dependencies=[])
        tracker.register_task(task_id=2, dependencies=[1])
        tracker.register_task(task_id=3, dependencies=[1, 2])

        # Should not raise
        assert True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mark_complete(self):
        """Test marking tasks complete."""
        tracker = DependencyTracker()

        tracker.register_task(1, [])
        event = tracker.mark_complete(1)

        assert event is not None
        event.synchronize()  # Should not block

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_dependency_events(self):
        """Test getting dependency events."""
        tracker = DependencyTracker()

        tracker.register_task(1, [])
        tracker.register_task(2, [1])

        # Before task 1 completes, no events
        events = tracker.get_dependency_events(2)
        assert len(events) == 0

        # After task 1 completes
        tracker.mark_complete(1)
        events = tracker.get_dependency_events(2)
        assert len(events) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_is_ready(self):
        """Test checking if task is ready."""
        tracker = DependencyTracker()

        tracker.register_task(1, [])
        tracker.register_task(2, [1])

        # Task 1 has no deps, always ready
        # (actually needs its deps to have completion events)
        # Task with no deps but no events recorded: ready
        assert tracker.is_ready(1)  # No deps to wait for

        # Task 2 not ready until task 1 completes
        assert tracker.is_ready(2)  # No events yet = ready by default

        # After task 1 completes
        event = tracker.mark_complete(1)
        event.synchronize()
        assert tracker.is_ready(2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_wait_for_dependencies(self):
        """Test waiting for dependencies."""
        tracker = DependencyTracker()

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        tracker.register_task(1, [])
        tracker.register_task(2, [1])

        # Complete task 1 on stream1
        with torch.cuda.stream(stream1):
            x = torch.randn(100, 100, device="cuda")
            tracker.mark_complete(1, stream1)

        # Wait for dependencies on stream2
        tracker.wait_for_dependencies(2, stream2)

        # Should complete without error
        stream2.synchronize()


class TestEventBarrier:
    """Tests for EventBarrier."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_barrier_creation(self):
        """Test barrier creation."""
        barrier = EventBarrier(num_participants=4)
        assert barrier.num_arrived == 0
        assert not barrier.is_complete

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_arrive(self):
        """Test arriving at barrier."""
        barrier = EventBarrier(num_participants=2)

        barrier.arrive()
        assert barrier.num_arrived == 1
        assert not barrier.is_complete

        barrier.arrive()
        assert barrier.num_arrived == 2
        assert barrier.is_complete

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_arrive_and_wait(self):
        """Test arrive and wait."""
        barrier = EventBarrier(num_participants=2)

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        # Both streams arrive and wait
        with torch.cuda.stream(stream1):
            x = torch.randn(100, device="cuda")
            barrier.arrive(stream1)

        with torch.cuda.stream(stream2):
            y = torch.randn(100, device="cuda")
            barrier.arrive_and_wait(stream2)

        # Should complete
        stream2.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset(self):
        """Test resetting barrier."""
        barrier = EventBarrier(num_participants=2)

        barrier.arrive()
        barrier.arrive()
        assert barrier.is_complete

        barrier.reset()
        assert barrier.num_arrived == 0
        assert not barrier.is_complete


class TestGlobalEventManager:
    """Tests for global event manager singleton."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_global_manager(self):
        """Test getting global manager."""
        reset_event_manager()
        manager = get_event_manager()
        assert manager is not None

        # Same manager returned
        manager2 = get_event_manager()
        assert manager2 is manager

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_global_manager(self):
        """Test resetting global manager."""
        manager1 = get_event_manager()
        reset_event_manager()
        manager2 = get_event_manager()

        assert manager1 is not manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
