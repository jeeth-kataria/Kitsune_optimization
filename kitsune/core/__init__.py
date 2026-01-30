"""
Kitsune Core - Dataflow scheduling engine

Contains the core scheduling infrastructure:
- Task representation and dependency graph
- Dataflow scheduler
- Cost model for operation estimation
"""

from .task import Task, TaskType, TaskStatus, TaskCost
from .graph import ComputationGraph, CycleDetectedError
from .scheduler import (
    DataflowScheduler,
    ExecutionPlan,
    ScheduleStep,
    TopologicalScheduler,
    PriorityScheduler,
    WavefrontScheduler,
)
from .executor import (
    StreamExecutor,
    ModelExecutor,
    ParallelForwardExecutor,
    ExecutionResult,
)
from .optimized_wrapper import OptimizedModelWrapper, create_optimized_model

__all__ = [
    # Task
    "Task",
    "TaskType",
    "TaskStatus",
    "TaskCost",
    # Graph
    "ComputationGraph",
    "CycleDetectedError",
    # Scheduler
    "DataflowScheduler",
    "ExecutionPlan",
    "ScheduleStep",
    "TopologicalScheduler",
    "PriorityScheduler",
    "WavefrontScheduler",
    # Executor
    "StreamExecutor",
    "ModelExecutor",
    "ParallelForwardExecutor",
    "ExecutionResult",
    # Optimized wrapper
    "OptimizedModelWrapper",
    "create_optimized_model",
]
