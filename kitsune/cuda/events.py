"""
CUDA Events - Synchronization primitives for stream coordination.

Provides event-based synchronization for dependency management
between operations on different CUDA streams.
"""

from __future__ import annotations

import threading
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
import torch

from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class EventTiming:
    """Timing information from an event pair."""
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    name: str = ""

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if not (self.start_event.query() and self.end_event.query()):
            return -1.0  # Not yet complete
        return self.start_event.elapsed_time(self.end_event)


class EventManager:
    """
    Manages CUDA events for synchronization and timing.

    Provides:
    - Event creation and pooling (to avoid allocation overhead)
    - Timing measurement between event pairs
    - Dependency tracking for multi-stream execution
    """

    def __init__(self, pool_size: int = 64, enable_timing: bool = False):
        """
        Initialize the event manager.

        Args:
            pool_size: Number of events to pre-allocate
            enable_timing: Whether to enable timing on events
        """
        self._enable_timing = enable_timing
        self._pool: List[torch.cuda.Event] = []
        self._pool_size = pool_size
        self._active_events: Dict[str, torch.cuda.Event] = {}
        self._timing_pairs: Dict[str, EventTiming] = {}
        self._lock = threading.Lock()

        if torch.cuda.is_available():
            self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Pre-allocate event pool."""
        for _ in range(self._pool_size):
            event = torch.cuda.Event(enable_timing=self._enable_timing)
            self._pool.append(event)
        logger.debug(f"EventManager: pre-allocated {self._pool_size} events")

    def get_event(self, enable_timing: Optional[bool] = None) -> torch.cuda.Event:
        """
        Get an event from the pool or create a new one.

        Args:
            enable_timing: Override timing setting for this event

        Returns:
            A CUDA event
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        timing = enable_timing if enable_timing is not None else self._enable_timing

        with self._lock:
            if self._pool and not timing:  # Only reuse non-timing events
                return self._pool.pop()
            return torch.cuda.Event(enable_timing=timing)

    def return_event(self, event: torch.cuda.Event) -> None:
        """
        Return an event to the pool for reuse.

        Args:
            event: Event to return
        """
        with self._lock:
            if len(self._pool) < self._pool_size * 2:  # Don't let pool grow unbounded
                self._pool.append(event)

    def record(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
        enable_timing: bool = False,
    ) -> torch.cuda.Event:
        """
        Record a named event.

        Args:
            name: Unique name for this event
            stream: Stream to record on (None for current)
            enable_timing: Whether to enable timing

        Returns:
            The recorded event
        """
        event = self.get_event(enable_timing=enable_timing)
        event.record(stream)
        self._active_events[name] = event
        return event

    def get_recorded(self, name: str) -> Optional[torch.cuda.Event]:
        """
        Get a previously recorded event by name.

        Args:
            name: Event name

        Returns:
            The event, or None if not found
        """
        return self._active_events.get(name)

    def wait(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> bool:
        """
        Make a stream wait for a named event.

        Args:
            name: Event name to wait for
            stream: Stream that should wait (None for current)

        Returns:
            True if event was found and waited on
        """
        event = self._active_events.get(name)
        if event is None:
            return False

        if stream is not None:
            stream.wait_event(event)
        else:
            torch.cuda.current_stream().wait_event(event)
        return True

    def synchronize(self, name: str) -> bool:
        """
        Synchronize on a named event (blocking).

        Args:
            name: Event name

        Returns:
            True if event was found and synchronized
        """
        event = self._active_events.get(name)
        if event is None:
            return False
        event.synchronize()
        return True

    def start_timing(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.cuda.Event:
        """
        Start timing measurement.

        Args:
            name: Name for this timing
            stream: Stream to record on

        Returns:
            The start event
        """
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)

        # Create timing pair with placeholder end event
        end_event = torch.cuda.Event(enable_timing=True)
        self._timing_pairs[name] = EventTiming(
            start_event=start_event,
            end_event=end_event,
            name=name,
        )

        return start_event

    def end_timing(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.cuda.Event]:
        """
        End timing measurement.

        Args:
            name: Name of timing to end
            stream: Stream to record on

        Returns:
            The end event, or None if timing not found
        """
        timing = self._timing_pairs.get(name)
        if timing is None:
            return None

        timing.end_event.record(stream)
        return timing.end_event

    def get_timing(self, name: str, synchronize: bool = True) -> float:
        """
        Get elapsed time for a timing measurement.

        Args:
            name: Timing name
            synchronize: Whether to synchronize before reading

        Returns:
            Elapsed time in milliseconds, or -1 if not found/complete
        """
        timing = self._timing_pairs.get(name)
        if timing is None:
            return -1.0

        if synchronize:
            timing.end_event.synchronize()

        return timing.elapsed_ms

    def clear(self) -> None:
        """Clear all active events and timings."""
        self._active_events.clear()
        self._timing_pairs.clear()


class DependencyTracker:
    """
    Track dependencies between tasks using events.

    Maps task IDs to completion events and provides methods
    for synchronization based on dependency graphs.
    """

    def __init__(self):
        """Initialize the dependency tracker."""
        self._task_events: Dict[int, torch.cuda.Event] = {}
        self._dependencies: Dict[int, Set[int]] = {}
        self._lock = threading.Lock()

    def register_task(self, task_id: int, dependencies: List[int]) -> None:
        """
        Register a task with its dependencies.

        Args:
            task_id: Task identifier
            dependencies: List of task IDs this task depends on
        """
        with self._lock:
            self._dependencies[task_id] = set(dependencies)

    def mark_complete(
        self,
        task_id: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.cuda.Event:
        """
        Mark a task as complete by recording an event.

        Args:
            task_id: Task identifier
            stream: Stream the task completed on

        Returns:
            Completion event
        """
        event = torch.cuda.Event()
        event.record(stream)

        with self._lock:
            self._task_events[task_id] = event

        return event

    def get_dependency_events(self, task_id: int) -> List[torch.cuda.Event]:
        """
        Get completion events for all dependencies of a task.

        Args:
            task_id: Task identifier

        Returns:
            List of dependency completion events
        """
        with self._lock:
            deps = self._dependencies.get(task_id, set())
            events = []
            for dep_id in deps:
                if dep_id in self._task_events:
                    events.append(self._task_events[dep_id])
            return events

    def wait_for_dependencies(
        self,
        task_id: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Make a stream wait for all dependencies of a task.

        Args:
            task_id: Task identifier
            stream: Stream that should wait
        """
        events = self.get_dependency_events(task_id)
        target_stream = stream or torch.cuda.current_stream()

        for event in events:
            target_stream.wait_event(event)

    def is_ready(self, task_id: int) -> bool:
        """
        Check if all dependencies of a task are complete.

        Args:
            task_id: Task identifier

        Returns:
            True if all dependencies have completed
        """
        events = self.get_dependency_events(task_id)
        return all(event.query() for event in events)

    def clear(self) -> None:
        """Clear all tracked tasks and dependencies."""
        with self._lock:
            self._task_events.clear()
            self._dependencies.clear()


class EventBarrier:
    """
    Barrier synchronization using CUDA events.

    Allows multiple streams to synchronize at a barrier point.
    """

    def __init__(self, num_participants: int):
        """
        Initialize the barrier.

        Args:
            num_participants: Number of streams that must reach the barrier
        """
        self._num_participants = num_participants
        self._events: List[torch.cuda.Event] = []
        self._lock = threading.Lock()

    def arrive(self, stream: Optional[torch.cuda.Stream] = None) -> torch.cuda.Event:
        """
        Signal arrival at the barrier.

        Args:
            stream: Stream arriving at the barrier

        Returns:
            The arrival event
        """
        event = torch.cuda.Event()
        event.record(stream)

        with self._lock:
            self._events.append(event)

        return event

    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Wait for all participants to arrive.

        Args:
            stream: Stream that should wait
        """
        with self._lock:
            events = list(self._events)

        target_stream = stream or torch.cuda.current_stream()
        for event in events:
            target_stream.wait_event(event)

    def arrive_and_wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Arrive at barrier and wait for all participants.

        Args:
            stream: Stream participating in the barrier
        """
        self.arrive(stream)
        self.wait(stream)

    def reset(self) -> None:
        """Reset the barrier for reuse."""
        with self._lock:
            self._events.clear()

    @property
    def num_arrived(self) -> int:
        """Number of participants that have arrived."""
        with self._lock:
            return len(self._events)

    @property
    def is_complete(self) -> bool:
        """Whether all participants have arrived."""
        return self.num_arrived >= self._num_participants


# Global event manager singleton
_global_event_manager: Optional[EventManager] = None


def get_event_manager(enable_timing: bool = False) -> EventManager:
    """
    Get or create the global event manager.

    Args:
        enable_timing: Whether to enable timing (only used on first call)

    Returns:
        The global event manager
    """
    global _global_event_manager
    if _global_event_manager is None:
        _global_event_manager = EventManager(enable_timing=enable_timing)
    return _global_event_manager


def reset_event_manager() -> None:
    """Reset the global event manager."""
    global _global_event_manager
    if _global_event_manager is not None:
        _global_event_manager.clear()
    _global_event_manager = None
