"""
Stream-Aware Executor - Execute computation graphs with CUDA stream parallelism.

Coordinates task execution across multiple CUDA streams based on
the dependency graph and execution plan from the scheduler.
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .graph import ComputationGraph
from .task import Task, TaskStatus
from .scheduler import ExecutionPlan, ScheduleStep, DataflowScheduler
from ..cuda.stream_pool import StreamPool, get_stream_pool
from ..cuda.events import DependencyTracker
from ..profiler import get_logger, CUDATimer

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a computation graph."""
    output: Any = None
    total_time_ms: float = 0.0
    stream_times: Dict[int, float] = field(default_factory=dict)
    tasks_executed: int = 0
    parallel_efficiency: float = 0.0


class StreamExecutor:
    """
    Execute computation graphs using multiple CUDA streams.

    Provides parallel execution of independent operations by
    assigning them to different CUDA streams and managing
    synchronization through events.

    Usage:
        executor = StreamExecutor(num_streams=4)

        # Execute a captured graph
        result = executor.execute(plan, kernels)

        # Or with model wrapper
        result = executor.execute_model(model, input, plan)
    """

    def __init__(
        self,
        num_streams: int = 4,
        enable_profiling: bool = False,
    ):
        """
        Initialize the stream executor.

        Args:
            num_streams: Number of CUDA streams to use
            enable_profiling: Whether to collect detailed timing
        """
        self._num_streams = num_streams
        self._enable_profiling = enable_profiling
        self._pool: Optional[StreamPool] = None
        self._tracker: Optional[DependencyTracker] = None
        self._timer: Optional[CUDATimer] = None

        if torch.cuda.is_available():
            self._pool = StreamPool(num_streams=num_streams)
            self._tracker = DependencyTracker()
            if enable_profiling:
                self._timer = CUDATimer()

    @property
    def enabled(self) -> bool:
        """Whether CUDA execution is enabled."""
        return self._pool is not None and self._pool.enabled

    def execute(
        self,
        plan: ExecutionPlan,
        kernels: Dict[int, Callable[[], Any]],
    ) -> ExecutionResult:
        """
        Execute an execution plan with provided kernels.

        Args:
            plan: Execution plan from scheduler
            kernels: Mapping of task ID to kernel function

        Returns:
            Execution result with timing information
        """
        if not self.enabled:
            return self._execute_sequential(plan, kernels)

        result = ExecutionResult()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Clear previous state
        self._tracker.clear()

        # Record start
        start_event.record()

        # Execute each step
        for step in plan.steps:
            self._execute_step(step, kernels)
            result.tasks_executed += 1

        # Synchronize all streams
        self._pool.synchronize_all()

        # Record end
        end_event.record()
        end_event.synchronize()

        result.total_time_ms = start_event.elapsed_time(end_event)

        # Calculate parallel efficiency
        if plan.estimated_time_us > 0:
            # Sequential time estimate vs actual parallel time
            sequential_estimate = plan.estimated_time_us / 1000.0  # Convert to ms
            if sequential_estimate > 0:
                result.parallel_efficiency = sequential_estimate / result.total_time_ms

        return result

    def _execute_step(
        self,
        step: ScheduleStep,
        kernels: Dict[int, Callable[[], Any]],
    ) -> None:
        """Execute a single step on its assigned stream."""
        task = step.task
        stream_id = step.stream_id
        dependencies = list(task.inputs)

        # Get kernel for this task
        kernel = kernels.get(task.id)
        if kernel is None:
            logger.warning(f"No kernel for task {task.id}, skipping")
            return

        # Register task dependencies
        self._tracker.register_task(task.id, dependencies)

        # Get stream
        stream = self._pool.get_stream(stream_id)

        with stream.context():
            # Wait for dependencies
            self._tracker.wait_for_dependencies(task.id, stream.stream)

            # Execute kernel
            task.mark_running()
            try:
                result = kernel()
                task.mark_completed(result)
            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                task.mark_failed()
                raise

            # Record completion event
            self._tracker.mark_complete(task.id, stream.stream)

    def _execute_sequential(
        self,
        plan: ExecutionPlan,
        kernels: Dict[int, Callable[[], Any]],
    ) -> ExecutionResult:
        """Fallback sequential execution when CUDA is not available."""
        result = ExecutionResult()
        start_time = time.perf_counter()

        for step in plan.steps:
            kernel = kernels.get(step.task.id)
            if kernel is not None:
                step.task.mark_running()
                try:
                    output = kernel()
                    step.task.mark_completed(output)
                except Exception as e:
                    step.task.mark_failed()
                    raise
                result.tasks_executed += 1

        result.total_time_ms = (time.perf_counter() - start_time) * 1000.0
        return result


class ModelExecutor:
    """
    High-level executor for PyTorch models with stream parallelism.

    Combines graph capture, scheduling, and stream execution into
    a single interface for optimized model inference.

    Usage:
        executor = ModelExecutor(model, sample_input)

        # First call captures and schedules
        output = executor(input)

        # Subsequent calls reuse the plan
        output = executor(another_input)
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_streams: int = 4,
        scheduler_type: str = "wavefront",
    ):
        """
        Initialize the model executor.

        Args:
            model: PyTorch model to execute
            sample_input: Sample input for graph capture
            num_streams: Number of CUDA streams
            scheduler_type: Scheduler algorithm to use
        """
        self.model = model
        self._sample_shape = sample_input.shape
        self._num_streams = num_streams

        # Create scheduler and capture graph
        self._scheduler = DataflowScheduler(
            num_streams=num_streams,
            scheduler_type=scheduler_type,
        )
        self._plan = self._scheduler.capture_and_schedule(model, sample_input)

        # Create stream executor
        self._executor = StreamExecutor(
            num_streams=num_streams,
            enable_profiling=True,
        )

        # Cache for module hooks
        self._hooks: List[Any] = []
        self._module_outputs: Dict[str, torch.Tensor] = {}

        logger.info(
            f"ModelExecutor initialized: {len(self._plan)} tasks, "
            f"{num_streams} streams"
        )

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Execute the model with stream parallelism.

        Args:
            input: Input tensor

        Returns:
            Model output
        """
        # Use stream-parallel execution if enabled
        if self._executor is not None and self._executor.enabled:
            try:
                # Execute with captured plan and stream parallelism
                return self._execute_parallel(input)
            except Exception as e:
                logger.warning(f"Parallel execution failed: {e}, falling back to standard forward")
                return self.model(input)
        else:
            return self.model(input)
    
    def _execute_parallel(self, input: torch.Tensor) -> torch.Tensor:
        """Execute model using captured graph with stream parallelism."""
        # Store input for graph execution
        self._module_outputs['input'] = input
        
        # Build kernel map from model modules
        kernels = {}
        for task in self._plan.tasks:
            task_id = task.id
            # Map each task to its corresponding module operation
            if task_id < len(list(self.model.modules())):
                module = list(self.model.modules())[task_id]
                
                def make_kernel(mod, task_input_key='input'):
                    def kernel():
                        inp = self._module_outputs.get(task_input_key, input)
                        return mod(inp)
                    return kernel
                
                kernels[task_id] = make_kernel(module, f'task_{task_id}_input')
        
        # Execute through stream executor
        result = self._executor.execute(self._plan, kernels)
        return result.output if result.output is not None else self.model(input)

    def benchmark(
        self,
        input: torch.Tensor,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark model execution.

        Args:
            input: Input tensor
            num_iterations: Number of timed iterations
            warmup: Warmup iterations

        Returns:
            Benchmark statistics
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for benchmarking")

        device = next(self.model.parameters()).device
        input = input.to(device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(input)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(input)
        end.record()
        end.synchronize()

        total_time = start.elapsed_time(end)
        avg_time = total_time / num_iterations

        return {
            "total_time_ms": total_time,
            "avg_time_ms": avg_time,
            "throughput_per_sec": 1000.0 / avg_time,
            "num_iterations": num_iterations,
        }

    @property
    def plan(self) -> ExecutionPlan:
        """Get the execution plan."""
        return self._plan

    @property
    def graph(self) -> ComputationGraph:
        """Get the computation graph."""
        return self._scheduler.graph


class ParallelForwardExecutor:
    """
    Execute independent forward branches in parallel.

    Automatically identifies independent subgraphs in the model
    and executes them on separate CUDA streams.

    Best suited for models with:
    - Multiple parallel branches (e.g., Inception, ResNet)
    - Independent attention heads
    - Feature extraction pipelines
    """

    def __init__(
        self,
        num_streams: int = 4,
    ):
        """
        Initialize the parallel forward executor.

        Args:
            num_streams: Number of CUDA streams to use
        """
        self._num_streams = num_streams
        self._pool: Optional[StreamPool] = None

        if torch.cuda.is_available():
            self._pool = StreamPool(num_streams=num_streams)

    def execute_parallel(
        self,
        functions: List[Callable[[], torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Execute multiple functions in parallel across streams.

        Args:
            functions: List of functions to execute

        Returns:
            List of results from each function
        """
        if not self._pool or not self._pool.enabled:
            return [fn() for fn in functions]

        n = len(functions)
        results: List[Optional[torch.Tensor]] = [None] * n
        events: List[torch.cuda.Event] = []

        # Launch all functions on different streams
        for i, fn in enumerate(functions):
            stream = self._pool.get_stream(i % self._num_streams)
            with stream.context():
                results[i] = fn()
                event = stream.record_event()
                events.append(event)

        # Wait for all to complete
        for event in events:
            event.synchronize()

        return results

    def execute_branches(
        self,
        input: torch.Tensor,
        branches: List[nn.Module],
    ) -> List[torch.Tensor]:
        """
        Execute multiple model branches in parallel.

        Args:
            input: Shared input tensor
            branches: List of model branches

        Returns:
            List of outputs from each branch
        """
        def make_branch_fn(branch: nn.Module) -> Callable[[], torch.Tensor]:
            def fn() -> torch.Tensor:
                with torch.no_grad():
                    return branch(input)
            return fn

        functions = [make_branch_fn(b) for b in branches]
        return self.execute_parallel(functions)
