"""
Computation Graph - DAG representation of neural network operations

Builds and manages the dependency graph for scheduling decisions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Iterator, Tuple
import heapq

from .task import Task, TaskStatus, TaskType, TaskCost


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the task graph."""
    pass


@dataclass
class ComputationGraph:
    """
    Directed Acyclic Graph (DAG) of computation tasks.

    Provides:
    - Task management (add, remove, query)
    - Dependency tracking
    - Topological ordering
    - Ready task detection
    - Graph analysis utilities
    """

    _tasks: Dict[int, Task] = field(default_factory=dict)
    _next_id: int = 0

    # Dependency tracking
    _forward_deps: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    _backward_deps: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    # Ready queue (tasks with all dependencies satisfied)
    _ready_tasks: Set[int] = field(default_factory=set)

    # Completed tasks
    _completed_tasks: Set[int] = field(default_factory=set)

    def add_task(
        self,
        name: str,
        op_type: str,
        task_type: TaskType = TaskType.COMPUTE,
        inputs: List[int] = None,
        input_shapes: List[tuple] = None,
        output_shapes: List[tuple] = None,
        **kwargs,
    ) -> Task:
        """
        Add a new task to the graph.

        Args:
            name: Human-readable task name
            op_type: Operation type (e.g., "linear", "relu")
            task_type: Category of task
            inputs: List of task IDs this depends on
            input_shapes: Shapes of input tensors
            output_shapes: Shapes of output tensors
            **kwargs: Additional Task parameters

        Returns:
            The created Task
        """
        task_id = self._next_id
        self._next_id += 1

        inputs = set(inputs) if inputs else set()

        task = Task(
            id=task_id,
            name=name,
            op_type=op_type,
            task_type=task_type,
            inputs=inputs,
            input_shapes=input_shapes or [],
            output_shapes=output_shapes or [],
            **kwargs,
        )

        self._tasks[task_id] = task

        # Update dependency tracking
        for dep_id in inputs:
            if dep_id in self._tasks:
                self._forward_deps[dep_id].add(task_id)
                self._backward_deps[task_id].add(dep_id)
                self._tasks[dep_id].add_dependent(task_id)

        # Check if task is immediately ready
        if not inputs or all(
            self._tasks.get(dep_id, Task(0, "", "")).is_completed
            for dep_id in inputs
        ):
            self._ready_tasks.add(task_id)
            task.mark_ready()

        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def remove_task(self, task_id: int) -> bool:
        """
        Remove a task from the graph.

        Note: Also removes dependency links.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]

        # Remove from dependents' input lists
        for dep_id in self._forward_deps[task_id]:
            self._backward_deps[dep_id].discard(task_id)
            if dep_id in self._tasks:
                self._tasks[dep_id].inputs.discard(task_id)

        # Remove from dependencies' output lists
        for dep_id in self._backward_deps[task_id]:
            self._forward_deps[dep_id].discard(task_id)
            if dep_id in self._tasks:
                self._tasks[dep_id].outputs.discard(task_id)

        # Clean up tracking
        del self._forward_deps[task_id]
        del self._backward_deps[task_id]
        self._ready_tasks.discard(task_id)
        self._completed_tasks.discard(task_id)
        del self._tasks[task_id]

        return True

    def add_dependency(self, from_id: int, to_id: int) -> bool:
        """
        Add a dependency: to_id depends on from_id.

        Args:
            from_id: Task that must complete first
            to_id: Task that depends on from_id

        Returns:
            True if dependency was added
        """
        if from_id not in self._tasks or to_id not in self._tasks:
            return False

        self._forward_deps[from_id].add(to_id)
        self._backward_deps[to_id].add(from_id)
        self._tasks[from_id].add_dependent(to_id)
        self._tasks[to_id].add_dependency(from_id)

        # Update ready status
        if to_id in self._ready_tasks and from_id not in self._completed_tasks:
            self._ready_tasks.discard(to_id)
            self._tasks[to_id].status = TaskStatus.PENDING

        return True

    def mark_completed(self, task_id: int) -> List[Task]:
        """
        Mark a task as completed and update ready queue.

        Args:
            task_id: ID of completed task

        Returns:
            List of newly ready tasks
        """
        if task_id not in self._tasks:
            return []

        task = self._tasks[task_id]
        task.mark_completed()
        self._completed_tasks.add(task_id)
        self._ready_tasks.discard(task_id)

        # Check dependents for newly ready tasks
        newly_ready = []
        for dep_id in self._forward_deps[task_id]:
            dep_task = self._tasks.get(dep_id)
            if dep_task and dep_task.status == TaskStatus.PENDING:
                # Check if all dependencies are now complete
                if all(d in self._completed_tasks for d in self._backward_deps[dep_id]):
                    dep_task.mark_ready()
                    self._ready_tasks.add(dep_id)
                    newly_ready.append(dep_task)

        return newly_ready

    def get_ready_tasks(self, max_count: int = None) -> List[Task]:
        """
        Get tasks that are ready to execute.

        Args:
            max_count: Maximum number of tasks to return

        Returns:
            List of ready Task objects
        """
        ready = [self._tasks[tid] for tid in self._ready_tasks]

        # Sort by priority (higher first), then by ID (FIFO)
        ready.sort(key=lambda t: (-t.priority, t.id))

        if max_count:
            ready = ready[:max_count]

        return ready

    def topological_order(self) -> List[Task]:
        """
        Return tasks in topological order (Kahn's algorithm).

        Raises:
            CycleDetectedError: If the graph contains a cycle
        """
        # Calculate in-degrees
        in_degree = {tid: len(self._backward_deps[tid]) for tid in self._tasks}

        # Start with zero in-degree nodes
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            tid = queue.popleft()
            result.append(self._tasks[tid])

            for dep_id in self._forward_deps[tid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(result) != len(self._tasks):
            raise CycleDetectedError(
                f"Graph contains a cycle. Processed {len(result)}/{len(self._tasks)} tasks."
            )

        return result

    def get_parallel_levels(self) -> List[List[Task]]:
        """
        Group tasks into parallel execution levels.

        Tasks at the same level have no dependencies on each other
        and can execute in parallel.

        Returns:
            List of levels, each containing tasks that can run in parallel
        """
        # Calculate depth for each task (longest path from root)
        depths: Dict[int, int] = {}

        def get_depth(tid: int) -> int:
            if tid in depths:
                return depths[tid]

            deps = self._backward_deps[tid]
            if not deps:
                depths[tid] = 0
            else:
                depths[tid] = max(get_depth(d) for d in deps) + 1

            return depths[tid]

        # Calculate depths for all tasks
        for tid in self._tasks:
            get_depth(tid)

        # Group by depth
        levels: Dict[int, List[Task]] = defaultdict(list)
        for tid, depth in depths.items():
            levels[depth].append(self._tasks[tid])

        # Sort levels and tasks within levels
        return [levels[i] for i in sorted(levels.keys())]

    def get_critical_path(self) -> List[Task]:
        """
        Find the critical path (longest path through the graph).

        Uses task costs for path length calculation.
        """
        if not self._tasks:
            return []

        # Calculate longest path to each task
        longest_path_to: Dict[int, Tuple[float, List[int]]] = {}

        def get_longest_path(tid: int) -> Tuple[float, List[int]]:
            if tid in longest_path_to:
                return longest_path_to[tid]

            task = self._tasks[tid]
            task_cost = task.cost.estimated_time_us if task.cost else 1.0

            deps = self._backward_deps[tid]
            if not deps:
                longest_path_to[tid] = (task_cost, [tid])
            else:
                # Find dependency with longest path
                best_cost = 0
                best_path = []
                for dep_id in deps:
                    dep_cost, dep_path = get_longest_path(dep_id)
                    if dep_cost > best_cost:
                        best_cost = dep_cost
                        best_path = dep_path

                longest_path_to[tid] = (best_cost + task_cost, best_path + [tid])

            return longest_path_to[tid]

        # Find task with longest total path
        best_total = 0
        best_path = []
        for tid in self._tasks:
            cost, path = get_longest_path(tid)
            if cost > best_total:
                best_total = cost
                best_path = path

        return [self._tasks[tid] for tid in best_path]

    def is_valid(self) -> Tuple[bool, str]:
        """
        Validate the graph structure.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for cycles
        try:
            self.topological_order()
        except CycleDetectedError as e:
            return False, str(e)

        # Check for dangling dependencies
        for tid, task in self._tasks.items():
            for dep_id in task.inputs:
                if dep_id not in self._tasks:
                    return False, f"Task {tid} depends on non-existent task {dep_id}"

        return True, "Graph is valid"

    def __len__(self) -> int:
        """Number of tasks in the graph."""
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks in ID order."""
        for tid in sorted(self._tasks.keys()):
            yield self._tasks[tid]

    def __contains__(self, task_id: int) -> bool:
        """Check if task ID exists in graph."""
        return task_id in self._tasks

    @property
    def tasks(self) -> List[Task]:
        """All tasks in the graph."""
        return list(self._tasks.values())

    @property
    def num_tasks(self) -> int:
        """Number of tasks."""
        return len(self._tasks)

    @property
    def num_completed(self) -> int:
        """Number of completed tasks."""
        return len(self._completed_tasks)

    @property
    def num_ready(self) -> int:
        """Number of ready tasks."""
        return len(self._ready_tasks)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return len(self._completed_tasks) == len(self._tasks)

    @property
    def is_empty(self) -> bool:
        """Check if graph has no tasks."""
        return len(self._tasks) == 0

    def summary(self) -> str:
        """Generate a summary of the graph."""
        lines = [
            "=" * 50,
            "Computation Graph Summary",
            "=" * 50,
            f"Total tasks:     {self.num_tasks}",
            f"Completed:       {self.num_completed}",
            f"Ready:           {self.num_ready}",
            f"Pending:         {self.num_tasks - self.num_completed - self.num_ready}",
        ]

        # Task type breakdown
        type_counts = defaultdict(int)
        for task in self._tasks.values():
            type_counts[task.task_type.name] += 1

        if type_counts:
            lines.append("\nTask types:")
            for ttype, count in sorted(type_counts.items()):
                lines.append(f"  {ttype}: {count}")

        # Parallel levels
        try:
            levels = self.get_parallel_levels()
            lines.append(f"\nParallel levels: {len(levels)}")
            for i, level in enumerate(levels[:5]):  # Show first 5 levels
                lines.append(f"  Level {i}: {len(level)} tasks")
            if len(levels) > 5:
                lines.append(f"  ... and {len(levels) - 5} more levels")
        except Exception:
            pass

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dot(self) -> str:
        """
        Export graph to DOT format for visualization.

        Can be rendered with Graphviz: `dot -Tpng graph.dot -o graph.png`
        """
        lines = ["digraph ComputationGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")

        # Add nodes
        for task in self._tasks.values():
            color = {
                TaskStatus.COMPLETED: "green",
                TaskStatus.RUNNING: "yellow",
                TaskStatus.READY: "lightblue",
                TaskStatus.PENDING: "white",
                TaskStatus.FAILED: "red",
            }.get(task.status, "white")

            label = f"{task.name}\\n{task.op_type}"
            lines.append(f'  {task.id} [label="{label}", style=filled, fillcolor={color}];')

        # Add edges
        for from_id, to_ids in self._forward_deps.items():
            for to_id in to_ids:
                lines.append(f"  {from_id} -> {to_id};")

        lines.append("}")
        return "\n".join(lines)
