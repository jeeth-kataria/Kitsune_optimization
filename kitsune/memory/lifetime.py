"""
Tensor Lifetime Analysis - Track tensor usage for memory optimization.

Analyzes when tensors are created, used, and no longer needed to enable
more aggressive memory reuse and reduce peak memory.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import torch

from ..profiler import get_logger
from ..core.graph import ComputationGraph
from ..core.task import Task

logger = get_logger(__name__)


class TensorLifeEvent(Enum):
    """Events in a tensor's lifetime."""
    CREATE = auto()     # Tensor created
    READ = auto()       # Tensor read (input to op)
    WRITE = auto()      # Tensor written (output of op)
    LAST_USE = auto()   # Last use of tensor
    FREE = auto()       # Tensor can be freed


@dataclass
class TensorLifetime:
    """Lifetime information for a single tensor."""
    tensor_id: int
    name: str = ""
    size_bytes: int = 0
    dtype: torch.dtype = torch.float32
    shape: Tuple[int, ...] = ()

    # Lifetime bounds (task IDs)
    created_at: int = -1        # Task that creates this tensor
    last_used_at: int = -1      # Last task that uses this tensor
    freed_at: int = -1          # Task after which tensor can be freed

    # Usage tracking
    read_by: Set[int] = field(default_factory=set)   # Tasks that read this tensor
    written_by: Set[int] = field(default_factory=set)  # Tasks that write this tensor

    @property
    def lifetime_span(self) -> int:
        """Number of tasks this tensor lives across."""
        if self.created_at < 0 or self.last_used_at < 0:
            return 0
        return self.last_used_at - self.created_at + 1

    @property
    def is_temporary(self) -> bool:
        """Check if tensor is temporary (short lifetime)."""
        return self.lifetime_span <= 2


@dataclass
class MemoryEvent:
    """A memory event at a specific point in execution."""
    task_id: int
    event_type: TensorLifeEvent
    tensor_id: int
    delta_bytes: int  # Positive for alloc, negative for free


class LifetimeAnalyzer:
    """
    Analyze tensor lifetimes in a computation graph.

    Determines when tensors are created and when they can be safely freed,
    enabling memory reuse optimizations.

    Usage:
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(graph)

        # Get memory timeline
        timeline = analyzer.get_memory_timeline()

        # Find tensors that can be freed after a task
        freeable = analyzer.get_freeable_after(task_id)
    """

    def __init__(self):
        """Initialize the lifetime analyzer."""
        self._lifetimes: Dict[int, TensorLifetime] = {}
        self._memory_timeline: List[MemoryEvent] = []
        self._graph: Optional[ComputationGraph] = None

    def analyze(self, graph: ComputationGraph) -> Dict[int, TensorLifetime]:
        """
        Analyze tensor lifetimes in a computation graph.

        Args:
            graph: Computation graph to analyze

        Returns:
            Dictionary mapping tensor ID to lifetime info
        """
        self._graph = graph
        self._lifetimes.clear()
        self._memory_timeline.clear()

        # Get topological order
        topo_order = graph.topological_order()
        task_order = {task.id: idx for idx, task in enumerate(topo_order)}

        # First pass: identify all tensors and their creation points
        tensor_id = 0
        for task in topo_order:
            # Output tensors are created by this task
            for output_shape in task.output_shapes:
                if output_shape:  # Skip empty shapes
                    lifetime = TensorLifetime(
                        tensor_id=tensor_id,
                        name=f"{task.name}_out_{tensor_id}",
                        shape=output_shape,
                        dtype=task.dtype,
                        size_bytes=self._calculate_size(output_shape, task.dtype),
                        created_at=task.id,
                        written_by={task.id},
                    )
                    self._lifetimes[tensor_id] = lifetime
                    tensor_id += 1

        # Second pass: track tensor usage
        for task in topo_order:
            # Find tensors read by this task
            for dep_id in task.inputs:
                dep_task = graph.get_task(dep_id)
                if dep_task is None:
                    continue

                # Find tensors created by dependency
                for tid, lifetime in self._lifetimes.items():
                    if lifetime.created_at == dep_id:
                        lifetime.read_by.add(task.id)
                        lifetime.last_used_at = max(lifetime.last_used_at, task.id)

        # Third pass: determine when tensors can be freed
        for tid, lifetime in self._lifetimes.items():
            if lifetime.last_used_at >= 0:
                lifetime.freed_at = lifetime.last_used_at
            else:
                # Never used after creation - can free immediately
                lifetime.freed_at = lifetime.created_at

        # Build memory timeline
        self._build_memory_timeline(topo_order, task_order)

        logger.debug(f"Analyzed {len(self._lifetimes)} tensors")
        return self._lifetimes

    def _calculate_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Calculate tensor size in bytes."""
        numel = 1
        for dim in shape:
            numel *= dim
        element_size = torch.tensor([], dtype=dtype).element_size()
        return numel * element_size

    def _build_memory_timeline(
        self,
        topo_order: List[Task],
        task_order: Dict[int, int],
    ) -> None:
        """Build memory event timeline."""
        events = []

        for tid, lifetime in self._lifetimes.items():
            # Allocation event
            events.append(MemoryEvent(
                task_id=lifetime.created_at,
                event_type=TensorLifeEvent.CREATE,
                tensor_id=tid,
                delta_bytes=lifetime.size_bytes,
            ))

            # Free event
            if lifetime.freed_at >= 0:
                events.append(MemoryEvent(
                    task_id=lifetime.freed_at,
                    event_type=TensorLifeEvent.FREE,
                    tensor_id=tid,
                    delta_bytes=-lifetime.size_bytes,
                ))

        # Sort by task order
        events.sort(key=lambda e: (task_order.get(e.task_id, 0), e.event_type.value))
        self._memory_timeline = events

    def get_memory_timeline(self) -> List[MemoryEvent]:
        """Get the memory event timeline."""
        return self._memory_timeline

    def get_peak_memory(self) -> int:
        """Calculate peak memory usage."""
        current = 0
        peak = 0

        for event in self._memory_timeline:
            current += event.delta_bytes
            peak = max(peak, current)

        return peak

    def get_memory_at_task(self, task_id: int) -> int:
        """Get memory usage at a specific task."""
        current = 0

        for event in self._memory_timeline:
            if event.task_id > task_id:
                break
            current += event.delta_bytes

        return current

    def get_freeable_after(self, task_id: int) -> List[TensorLifetime]:
        """
        Get tensors that can be freed after a task completes.

        Args:
            task_id: Task ID

        Returns:
            List of tensor lifetimes that can be freed
        """
        return [
            lifetime for lifetime in self._lifetimes.values()
            if lifetime.freed_at == task_id
        ]

    def get_live_tensors_at(self, task_id: int) -> List[TensorLifetime]:
        """
        Get tensors that are live at a specific task.

        Args:
            task_id: Task ID

        Returns:
            List of live tensor lifetimes
        """
        return [
            lifetime for lifetime in self._lifetimes.values()
            if lifetime.created_at <= task_id <= lifetime.freed_at
        ]

    def get_temporary_tensors(self) -> List[TensorLifetime]:
        """Get tensors with short lifetimes (candidates for pooling)."""
        return [lt for lt in self._lifetimes.values() if lt.is_temporary]

    def get_long_lived_tensors(self, min_span: int = 5) -> List[TensorLifetime]:
        """Get tensors with long lifetimes."""
        return [lt for lt in self._lifetimes.values() if lt.lifetime_span >= min_span]

    def optimize_memory_layout(self) -> Dict[int, int]:
        """
        Compute optimized memory offsets for tensors.

        Uses a simple greedy algorithm to pack tensors with
        non-overlapping lifetimes into the same memory region.

        Returns:
            Dictionary mapping tensor ID to memory offset
        """
        if not self._lifetimes:
            return {}

        # Sort tensors by creation time
        sorted_tensors = sorted(
            self._lifetimes.values(),
            key=lambda lt: lt.created_at
        )

        # Greedy allocation
        offsets: Dict[int, int] = {}
        memory_slots: List[Tuple[int, int, int]] = []  # (end_time, offset, size)

        for lifetime in sorted_tensors:
            # Find a slot that's free by the time we need it
            best_offset = None
            best_slot_idx = None

            for idx, (end_time, offset, size) in enumerate(memory_slots):
                if end_time <= lifetime.created_at and size >= lifetime.size_bytes:
                    if best_offset is None or offset < best_offset:
                        best_offset = offset
                        best_slot_idx = idx

            if best_offset is not None:
                # Reuse existing slot
                offsets[lifetime.tensor_id] = best_offset
                memory_slots[best_slot_idx] = (lifetime.freed_at, best_offset, lifetime.size_bytes)
            else:
                # Allocate new slot
                new_offset = sum(s[2] for s in memory_slots if s[0] > lifetime.created_at)
                offsets[lifetime.tensor_id] = new_offset
                memory_slots.append((lifetime.freed_at, new_offset, lifetime.size_bytes))

        return offsets

    def summary(self) -> str:
        """Get a summary of the lifetime analysis."""
        if not self._lifetimes:
            return "No tensors analyzed"

        total_bytes = sum(lt.size_bytes for lt in self._lifetimes.values())
        peak_bytes = self.get_peak_memory()
        temp_count = len(self.get_temporary_tensors())

        lines = [
            "Tensor Lifetime Analysis",
            "=" * 40,
            f"Total tensors: {len(self._lifetimes)}",
            f"Total size: {total_bytes / 1e6:.2f} MB",
            f"Peak memory: {peak_bytes / 1e6:.2f} MB",
            f"Temporary tensors: {temp_count}",
            f"Potential savings: {(total_bytes - peak_bytes) / 1e6:.2f} MB",
        ]

        return "\n".join(lines)
