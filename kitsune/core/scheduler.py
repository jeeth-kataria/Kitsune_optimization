"""
Dataflow Scheduler - Core scheduling engine

Orchestrates task execution with dependency-aware scheduling,
preparing for CUDA stream parallelism in Week 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
import heapq
import time

import torch
import torch.nn as nn

from .graph import ComputationGraph, CycleDetectedError
from .task import Task, TaskStatus, TaskType
from ..utils.logging import get_logger

logger = get_logger("scheduler")


@dataclass
class ScheduleStep:
    """A single step in the execution schedule."""
    task: Task
    stream_id: int = 0  # CUDA stream assignment (for Week 3)
    wait_for: List[int] = field(default_factory=list)  # Task IDs to wait for


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a computation graph.

    Contains ordered steps with stream assignments and
    synchronization points.
    """
    steps: List[ScheduleStep] = field(default_factory=list)
    num_streams: int = 1
    estimated_time_us: float = 0.0

    def add_step(self, task: Task, stream_id: int = 0, wait_for: List[int] = None):
        """Add a step to the plan."""
        self.steps.append(ScheduleStep(
            task=task,
            stream_id=stream_id,
            wait_for=wait_for or [],
        ))

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def summary(self) -> str:
        """Generate execution plan summary."""
        lines = [
            "=" * 50,
            "Execution Plan Summary",
            "=" * 50,
            f"Total steps: {len(self.steps)}",
            f"Streams used: {self.num_streams}",
            f"Estimated time: {self.estimated_time_us:.2f} µs",
            "",
            "Steps:",
        ]

        for i, step in enumerate(self.steps[:20]):  # Show first 20
            wait_str = f" (wait: {step.wait_for})" if step.wait_for else ""
            lines.append(
                f"  {i:3d}. [{step.stream_id}] {step.task.name} ({step.task.op_type}){wait_str}"
            )

        if len(self.steps) > 20:
            lines.append(f"  ... and {len(self.steps) - 20} more steps")

        lines.append("=" * 50)
        return "\n".join(lines)


class BaseScheduler(ABC):
    """Abstract base class for schedulers."""

    @abstractmethod
    def schedule(self, graph: ComputationGraph) -> ExecutionPlan:
        """
        Create an execution plan from a computation graph.

        Args:
            graph: ComputationGraph to schedule

        Returns:
            ExecutionPlan with ordered steps
        """
        pass


class TopologicalScheduler(BaseScheduler):
    """
    Basic topological order scheduler.

    Executes tasks in dependency order (Kahn's algorithm).
    Foundation for more advanced schedulers.
    """

    def __init__(self, num_streams: int = 1):
        """
        Args:
            num_streams: Number of execution streams (for future use)
        """
        self.num_streams = num_streams

    def schedule(self, graph: ComputationGraph) -> ExecutionPlan:
        """
        Create execution plan in topological order.

        Args:
            graph: ComputationGraph to schedule

        Returns:
            ExecutionPlan with tasks in valid execution order
        """
        plan = ExecutionPlan(num_streams=self.num_streams)

        try:
            ordered_tasks = graph.topological_order()
        except CycleDetectedError as e:
            logger.error(f"Cannot schedule graph with cycle: {e}")
            raise

        total_time = 0.0
        for task in ordered_tasks:
            plan.add_step(task, stream_id=0)
            if task.cost:
                total_time += task.cost.estimated_time_us

        plan.estimated_time_us = total_time

        logger.debug(f"Created topological schedule with {len(plan)} steps")
        return plan


class PriorityScheduler(BaseScheduler):
    """
    Priority-based scheduler with critical path awareness.

    Prioritizes tasks based on:
    1. Critical path membership (higher priority)
    2. Number of dependents (more dependents = higher priority)
    3. Task cost (expensive tasks first to maximize parallelism)
    """

    def __init__(self, num_streams: int = 1):
        self.num_streams = num_streams

    def schedule(self, graph: ComputationGraph) -> ExecutionPlan:
        """
        Create execution plan with priority ordering.

        Tasks on the critical path are scheduled first to minimize
        total execution time.
        """
        plan = ExecutionPlan(num_streams=self.num_streams)

        # Get critical path tasks
        critical_path = set(t.id for t in graph.get_critical_path())

        # Calculate priorities
        priorities = self._calculate_priorities(graph, critical_path)

        # Schedule using modified topological sort with priorities
        completed: Set[int] = set()
        ready_heap: List[Tuple[float, int, Task]] = []  # (neg_priority, id, task)

        # Initialize with tasks that have no dependencies
        for task in graph.tasks:
            if not task.inputs:
                priority = priorities.get(task.id, 0)
                heapq.heappush(ready_heap, (-priority, task.id, task))

        total_time = 0.0

        while ready_heap:
            _, task_id, task = heapq.heappop(ready_heap)

            # Add to plan
            plan.add_step(task, stream_id=0)
            completed.add(task_id)

            if task.cost:
                total_time += task.cost.estimated_time_us

            # Update ready tasks
            for dep_id in task.outputs:
                dep_task = graph.get_task(dep_id)
                if dep_task and all(inp in completed for inp in dep_task.inputs):
                    priority = priorities.get(dep_id, 0)
                    heapq.heappush(ready_heap, (-priority, dep_id, dep_task))

        plan.estimated_time_us = total_time

        logger.debug(f"Created priority schedule with {len(plan)} steps")
        return plan

    def _calculate_priorities(
        self,
        graph: ComputationGraph,
        critical_path: Set[int],
    ) -> Dict[int, float]:
        """Calculate priority scores for tasks."""
        priorities = {}

        for task in graph.tasks:
            priority = 0.0

            # Critical path bonus (highest priority)
            if task.id in critical_path:
                priority += 1000.0

            # More dependents = higher priority (unlocks more work)
            priority += len(task.outputs) * 10.0

            # Higher cost = higher priority (better utilization)
            if task.cost:
                priority += task.cost.estimated_time_us / 1000.0  # Normalize

            priorities[task.id] = priority

        return priorities


class WavefrontScheduler(BaseScheduler):
    """
    Level-based (wavefront) scheduler.

    Groups tasks by dependency depth. Tasks at the same level
    have no dependencies on each other and can execute in parallel.

    This is the foundation for CUDA stream parallelism (Week 3).
    """

    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams

    def schedule(self, graph: ComputationGraph) -> ExecutionPlan:
        """
        Create execution plan with wavefront scheduling.

        Tasks are grouped by level and assigned to streams
        within each level for parallel execution.
        """
        plan = ExecutionPlan(num_streams=self.num_streams)

        # Get parallel levels
        levels = graph.get_parallel_levels()

        total_time = 0.0

        for level_idx, level_tasks in enumerate(levels):
            # Assign tasks to streams (round-robin within level)
            for i, task in enumerate(level_tasks):
                stream_id = i % self.num_streams

                # Tasks in this level wait for all tasks in previous level
                # (simplified - Week 3 will add proper event-based sync)
                wait_for = []
                if level_idx > 0:
                    # In full implementation, only wait for actual dependencies
                    wait_for = [t.id for t in levels[level_idx - 1] if task.id in t.outputs]

                plan.add_step(task, stream_id=stream_id, wait_for=wait_for)

            # Estimate time for this level (parallel execution)
            level_times = [t.cost.estimated_time_us for t in level_tasks if t.cost]
            if level_times:
                total_time += max(level_times)  # Parallel: take max, not sum

        plan.estimated_time_us = total_time

        logger.debug(
            f"Created wavefront schedule: {len(plan)} steps, "
            f"{len(levels)} levels, {self.num_streams} streams"
        )
        return plan

    def get_parallelism_stats(self, graph: ComputationGraph) -> Dict[str, Any]:
        """Get statistics about potential parallelism."""
        levels = graph.get_parallel_levels()

        tasks_per_level = [len(level) for level in levels]
        avg_parallelism = sum(tasks_per_level) / len(levels) if levels else 0

        return {
            "num_levels": len(levels),
            "total_tasks": sum(tasks_per_level),
            "max_parallelism": max(tasks_per_level) if tasks_per_level else 0,
            "avg_parallelism": avg_parallelism,
            "tasks_per_level": tasks_per_level,
        }


class DataflowScheduler:
    """
    Main scheduler interface for Kitsune.

    Combines graph capture, scheduling, and execution into
    a unified interface.
    """

    def __init__(
        self,
        num_streams: int = 4,
        scheduler_type: str = "wavefront",
        device: int = 0,
    ):
        """
        Args:
            num_streams: Number of CUDA streams for parallel execution
            scheduler_type: "topological", "priority", or "wavefront"
            device: CUDA device index
        """
        self.num_streams = num_streams
        self.device = device

        # Select scheduler implementation
        if scheduler_type == "topological":
            self._scheduler = TopologicalScheduler(num_streams)
        elif scheduler_type == "priority":
            self._scheduler = PriorityScheduler(num_streams)
        else:
            self._scheduler = WavefrontScheduler(num_streams)

        self._current_graph: Optional[ComputationGraph] = None
        self._current_plan: Optional[ExecutionPlan] = None

        logger.info(f"DataflowScheduler initialized: {scheduler_type}, {num_streams} streams")

    def capture_and_schedule(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> ExecutionPlan:
        """
        Capture computation graph and create execution plan.

        Args:
            model: PyTorch model
            sample_input: Representative input for tracing

        Returns:
            ExecutionPlan ready for execution
        """
        from ..pytorch.graph_capture import capture_graph

        # Capture graph
        logger.info("Capturing computation graph...")
        self._current_graph = capture_graph(model, sample_input)
        logger.info(f"Captured {self._current_graph.num_tasks} tasks")

        # Create schedule
        logger.info("Creating execution schedule...")
        self._current_plan = self._scheduler.schedule(self._current_graph)
        logger.info(f"Schedule created: {len(self._current_plan)} steps")

        return self._current_plan

    def schedule(self, graph: ComputationGraph) -> ExecutionPlan:
        """
        Create execution plan from existing graph.

        Args:
            graph: ComputationGraph to schedule

        Returns:
            ExecutionPlan
        """
        self._current_graph = graph
        self._current_plan = self._scheduler.schedule(graph)
        return self._current_plan

    @property
    def graph(self) -> Optional[ComputationGraph]:
        """Current computation graph."""
        return self._current_graph

    @property
    def plan(self) -> Optional[ExecutionPlan]:
        """Current execution plan."""
        return self._current_plan

    def summary(self) -> str:
        """Generate summary of current state."""
        lines = ["=" * 60, "DataflowScheduler Summary", "=" * 60]

        if self._current_graph:
            lines.append(f"\nGraph: {self._current_graph.num_tasks} tasks")

            # Parallelism stats
            if isinstance(self._scheduler, WavefrontScheduler):
                stats = self._scheduler.get_parallelism_stats(self._current_graph)
                lines.append(f"Levels: {stats['num_levels']}")
                lines.append(f"Max parallelism: {stats['max_parallelism']}")
                lines.append(f"Avg parallelism: {stats['avg_parallelism']:.1f}")
        else:
            lines.append("\nNo graph captured yet.")

        if self._current_plan:
            lines.append(f"\nPlan: {len(self._current_plan)} steps")
            lines.append(f"Estimated time: {self._current_plan.estimated_time_us:.2f} µs")
        else:
            lines.append("\nNo plan created yet.")

        lines.append("=" * 60)
        return "\n".join(lines)
