"""
CUDA Stream Pool - Manage multiple CUDA streams for parallel execution.

Enables concurrent kernel execution on GPU by distributing independent
operations across multiple streams.
"""

from __future__ import annotations

import threading
from typing import Optional, List, Dict, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch

from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class StreamStats:
    """Statistics for a single CUDA stream."""
    stream_id: int
    tasks_executed: int = 0
    total_time_ms: float = 0.0
    active: bool = False

    def record_task(self, time_ms: float) -> None:
        """Record a task execution."""
        self.tasks_executed += 1
        self.total_time_ms += time_ms


class CUDAStream:
    """
    Wrapper around a CUDA stream with tracking capabilities.

    Provides:
    - Stream lifecycle management
    - Event recording for synchronization
    - Statistics tracking
    """

    def __init__(self, stream_id: int, stream: Optional[torch.cuda.Stream] = None):
        """
        Initialize a CUDA stream wrapper.

        Args:
            stream_id: Unique identifier for this stream
            stream: Optional existing stream, creates new if None
        """
        self.stream_id = stream_id
        self._stream = stream if stream is not None else torch.cuda.Stream()
        self.stats = StreamStats(stream_id=stream_id)
        self._last_event: Optional[torch.cuda.Event] = None

    @property
    def stream(self) -> torch.cuda.Stream:
        """Get the underlying CUDA stream."""
        return self._stream

    @property
    def is_default(self) -> bool:
        """Check if this is the default stream."""
        return self._stream == torch.cuda.default_stream()

    def record_event(self, enable_timing: bool = False) -> torch.cuda.Event:
        """
        Record an event on this stream.

        Args:
            enable_timing: Whether to enable timing for this event

        Returns:
            The recorded CUDA event
        """
        event = torch.cuda.Event(enable_timing=enable_timing)
        event.record(self._stream)
        self._last_event = event
        return event

    def wait_event(self, event: torch.cuda.Event) -> None:
        """
        Make this stream wait for an event.

        Args:
            event: The event to wait for
        """
        self._stream.wait_event(event)

    def wait_stream(self, other: "CUDAStream") -> None:
        """
        Make this stream wait for another stream's last event.

        Args:
            other: The stream to wait for
        """
        if other._last_event is not None:
            self.wait_event(other._last_event)

    def synchronize(self) -> None:
        """Block until all operations on this stream complete."""
        self._stream.synchronize()

    @contextmanager
    def context(self):
        """Context manager to execute operations on this stream."""
        with torch.cuda.stream(self._stream):
            self.stats.active = True
            try:
                yield self
            finally:
                self.stats.active = False

    def __repr__(self) -> str:
        return f"CUDAStream(id={self.stream_id}, tasks={self.stats.tasks_executed})"


class StreamPool:
    """
    Pool of CUDA streams for parallel kernel execution.

    Manages a fixed number of streams and distributes work across them
    using various assignment strategies.

    Usage:
        pool = StreamPool(num_streams=4)

        # Get stream by index
        stream = pool.get_stream(0)

        # Round-robin assignment
        stream = pool.next_stream()

        # Automatic assignment based on load
        stream = pool.get_least_loaded()
    """

    def __init__(
        self,
        num_streams: int = 4,
        include_default: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the stream pool.

        Args:
            num_streams: Number of streams to create (1-16)
            include_default: Whether to include the default stream in the pool
            device: CUDA device to create streams on
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, StreamPool will use CPU fallback")
            self._streams: List[CUDAStream] = []
            self._num_streams = 0
            self._enabled = False
            return

        num_streams = max(1, min(16, num_streams))  # Clamp to 1-16
        self._device = device or torch.cuda.current_device()
        self._enabled = True

        with torch.cuda.device(self._device):
            self._streams = []

            # Optionally include default stream as stream 0
            if include_default:
                default_stream = CUDAStream(
                    stream_id=0,
                    stream=torch.cuda.default_stream()
                )
                self._streams.append(default_stream)
                num_streams -= 1

            # Create additional streams
            for i in range(num_streams):
                stream_id = len(self._streams)
                self._streams.append(CUDAStream(stream_id=stream_id))

        self._num_streams = len(self._streams)
        self._round_robin_idx = 0
        self._lock = threading.Lock()

        logger.info(f"StreamPool initialized with {self._num_streams} streams")

    @property
    def num_streams(self) -> int:
        """Number of streams in the pool."""
        return self._num_streams

    @property
    def enabled(self) -> bool:
        """Whether the pool is enabled (CUDA available)."""
        return self._enabled

    @property
    def streams(self) -> List[CUDAStream]:
        """Get all streams in the pool."""
        return self._streams

    def get_stream(self, stream_id: int) -> CUDAStream:
        """
        Get a specific stream by ID.

        Args:
            stream_id: Stream identifier (0-indexed)

        Returns:
            The requested CUDA stream

        Raises:
            IndexError: If stream_id is out of range
        """
        if not self._enabled:
            raise RuntimeError("StreamPool not enabled (CUDA not available)")
        return self._streams[stream_id % self._num_streams]

    def next_stream(self) -> CUDAStream:
        """
        Get the next stream in round-robin fashion.

        Returns:
            The next CUDA stream
        """
        if not self._enabled:
            raise RuntimeError("StreamPool not enabled (CUDA not available)")

        with self._lock:
            stream = self._streams[self._round_robin_idx]
            self._round_robin_idx = (self._round_robin_idx + 1) % self._num_streams
        return stream

    def get_least_loaded(self) -> CUDAStream:
        """
        Get the stream with the least queued work.

        This is a simple heuristic based on task count. For more accurate
        load balancing, use event-based tracking.

        Returns:
            The least loaded CUDA stream
        """
        if not self._enabled:
            raise RuntimeError("StreamPool not enabled (CUDA not available)")

        return min(self._streams, key=lambda s: s.stats.tasks_executed)

    def synchronize_all(self) -> None:
        """Synchronize all streams in the pool."""
        if not self._enabled:
            return
        for stream in self._streams:
            stream.synchronize()

    def reset_stats(self) -> None:
        """Reset statistics for all streams."""
        for stream in self._streams:
            stream.stats = StreamStats(stream_id=stream.stream_id)

    def get_stats(self) -> Dict[int, StreamStats]:
        """
        Get statistics for all streams.

        Returns:
            Dictionary mapping stream ID to stats
        """
        return {s.stream_id: s.stats for s in self._streams}

    def execute_on_stream(
        self,
        stream_id: int,
        func: Callable[[], Any],
        wait_for: Optional[List[torch.cuda.Event]] = None,
        record_event: bool = True,
    ) -> Optional[torch.cuda.Event]:
        """
        Execute a function on a specific stream.

        Args:
            stream_id: Stream to execute on
            func: Function to execute
            wait_for: Events to wait for before execution
            record_event: Whether to record an event after execution

        Returns:
            Event recorded after execution, or None
        """
        if not self._enabled:
            func()
            return None

        stream = self.get_stream(stream_id)

        with stream.context():
            # Wait for dependencies
            if wait_for:
                for event in wait_for:
                    stream.wait_event(event)

            # Execute
            result = func()
            stream.stats.tasks_executed += 1

            # Record completion event
            if record_event:
                return stream.record_event()

        return None

    def __repr__(self) -> str:
        if not self._enabled:
            return "StreamPool(disabled)"
        return f"StreamPool(streams={self._num_streams})"


class StreamScheduler:
    """
    Scheduler that assigns tasks to streams based on dependencies.

    Uses wavefront parallelism: tasks at the same level (no inter-dependencies)
    are distributed across streams for parallel execution.
    """

    def __init__(self, pool: StreamPool):
        """
        Initialize the stream scheduler.

        Args:
            pool: Stream pool to use for execution
        """
        self.pool = pool
        self._task_events: Dict[int, torch.cuda.Event] = {}

    def assign_stream(
        self,
        task_id: int,
        dependencies: List[int],
        preferred_stream: Optional[int] = None,
    ) -> int:
        """
        Assign a stream to a task based on dependencies.

        Strategy:
        - If task has no dependencies, use round-robin
        - If task has dependencies, prefer stream of largest dependency
        - If preferred_stream is set, use that

        Args:
            task_id: Task identifier
            dependencies: List of task IDs this task depends on
            preferred_stream: Optional preferred stream ID

        Returns:
            Assigned stream ID
        """
        if not self.pool.enabled:
            return 0

        if preferred_stream is not None:
            return preferred_stream % self.pool.num_streams

        if not dependencies:
            return self.pool.next_stream().stream_id

        # Find stream of largest dependency (to minimize synchronization)
        # For now, just use round-robin
        return self.pool.next_stream().stream_id

    def execute_task(
        self,
        task_id: int,
        stream_id: int,
        func: Callable[[], Any],
        dependencies: List[int],
    ) -> torch.cuda.Event:
        """
        Execute a task on its assigned stream.

        Args:
            task_id: Task identifier
            stream_id: Stream to execute on
            func: Function to execute
            dependencies: Task IDs this task depends on

        Returns:
            Event recorded after execution
        """
        # Gather dependency events
        wait_events = []
        for dep_id in dependencies:
            if dep_id in self._task_events:
                wait_events.append(self._task_events[dep_id])

        # Execute on stream
        event = self.pool.execute_on_stream(
            stream_id=stream_id,
            func=func,
            wait_for=wait_events,
            record_event=True,
        )

        # Store event for dependents
        if event is not None:
            self._task_events[task_id] = event

        return event

    def clear(self) -> None:
        """Clear task events."""
        self._task_events.clear()

    def synchronize(self) -> None:
        """Synchronize all streams."""
        self.pool.synchronize_all()


# Global stream pool singleton
_global_pool: Optional[StreamPool] = None


def get_stream_pool(num_streams: int = 4) -> StreamPool:
    """
    Get or create the global stream pool.

    Args:
        num_streams: Number of streams (only used on first call)

    Returns:
        The global stream pool
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = StreamPool(num_streams=num_streams)
    return _global_pool


def reset_stream_pool() -> None:
    """Reset the global stream pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.synchronize_all()
    _global_pool = None
