"""
Task - Fundamental unit of schedulable work

Represents a single operation in the computation graph with
dependency information and cost metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Set, List, Dict
import torch


class TaskType(Enum):
    """Type of task/operation."""
    COMPUTE = auto()      # GPU computation (matmul, conv, etc.)
    TRANSFER_H2D = auto() # Host to Device transfer
    TRANSFER_D2H = auto() # Device to Host transfer
    SYNC = auto()         # Synchronization point
    MEMORY = auto()       # Memory allocation/deallocation


class TaskStatus(Enum):
    """Execution status of a task."""
    PENDING = auto()      # Not yet ready to execute
    READY = auto()        # All dependencies satisfied
    RUNNING = auto()      # Currently executing
    COMPLETED = auto()    # Finished execution
    FAILED = auto()       # Execution failed


@dataclass
class TaskCost:
    """Cost estimates for scheduling decisions."""

    # Computational cost
    flops: int = 0                    # Floating point operations
    estimated_time_us: float = 0.0    # Estimated execution time in microseconds

    # Memory cost
    memory_read_bytes: int = 0        # Bytes read from memory
    memory_write_bytes: int = 0       # Bytes written to memory
    peak_memory_bytes: int = 0        # Peak memory during execution

    # Scheduling hints
    is_memory_bound: bool = False     # True if memory-bound, False if compute-bound
    parallelizable: bool = True       # Can run in parallel with other tasks

    @property
    def memory_total_bytes(self) -> int:
        """Total memory traffic."""
        return self.memory_read_bytes + self.memory_write_bytes

    @classmethod
    def estimate_from_shape(
        cls,
        op_type: str,
        input_shapes: List[tuple],
        output_shapes: List[tuple],
        dtype: torch.dtype = torch.float32,
    ) -> "TaskCost":
        """
        Estimate cost from operation type and tensor shapes.

        Args:
            op_type: Operation type (e.g., "linear", "conv2d", "relu")
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes
            dtype: Data type for memory calculations
        """
        dtype_size = torch.tensor([], dtype=dtype).element_size()

        # Simple element count for shapes
        def count_elements(shape):
            if not shape:
                return 0
            result = 1
            for dim in shape:
                result *= dim
            return result

        input_elements = sum(count_elements(s) for s in input_shapes if s)
        output_elements = sum(count_elements(s) for s in output_shapes if s)

        memory_read = input_elements * dtype_size
        memory_write = output_elements * dtype_size

        # Estimate FLOPs based on operation type
        flops = cls._estimate_flops(op_type, input_shapes, output_shapes)

        # Determine if memory-bound using simple roofline heuristic
        # RTX 3050: ~9 TFLOPS FP32, ~192 GB/s memory bandwidth
        # Arithmetic intensity threshold ≈ 47 FLOPs/byte
        arithmetic_intensity = flops / max(memory_read + memory_write, 1)
        is_memory_bound = arithmetic_intensity < 47

        return cls(
            flops=flops,
            memory_read_bytes=memory_read,
            memory_write_bytes=memory_write,
            peak_memory_bytes=memory_read + memory_write,
            is_memory_bound=is_memory_bound,
        )

    @staticmethod
    def _estimate_flops(op_type: str, input_shapes: List[tuple], output_shapes: List[tuple]) -> int:
        """Estimate FLOPs for common operations."""
        op_type = op_type.lower()

        if not input_shapes:
            return 0

        # Helper to count elements
        def numel(shape):
            result = 1
            for dim in shape:
                result *= dim
            return result

        if op_type in ("linear", "addmm", "mm", "matmul"):
            # Matrix multiply: 2 * M * N * K
            if len(input_shapes) >= 2:
                # input: (B, M, K) or (M, K), weight: (K, N)
                inp = input_shapes[0]
                if len(inp) == 2:
                    M, K = inp
                    N = input_shapes[1][-1] if len(input_shapes) > 1 else K
                    return 2 * M * K * N
                elif len(inp) >= 3:
                    B, M, K = inp[-3], inp[-2], inp[-1]
                    N = input_shapes[1][-1] if len(input_shapes) > 1 else K
                    return 2 * B * M * K * N
            return numel(output_shapes[0]) * 2 if output_shapes else 0

        elif op_type in ("conv2d", "conv1d", "conv3d"):
            # Conv: 2 * output_elements * kernel_size * in_channels
            if output_shapes:
                out_elements = numel(output_shapes[0])
                # Assume 3x3 kernel, estimate
                return out_elements * 18 * (input_shapes[0][1] if input_shapes else 1)
            return 0

        elif op_type in ("relu", "gelu", "sigmoid", "tanh", "silu"):
            # Elementwise activation: ~1-5 ops per element
            return numel(output_shapes[0]) * 2 if output_shapes else 0

        elif op_type in ("add", "sub", "mul", "div"):
            # Elementwise: 1 op per element
            return numel(output_shapes[0]) if output_shapes else 0

        elif op_type in ("layernorm", "batchnorm", "instancenorm"):
            # Normalization: ~10 ops per element (mean, var, normalize)
            return numel(output_shapes[0]) * 10 if output_shapes else 0

        elif op_type in ("softmax", "log_softmax"):
            # Softmax: ~5 ops per element
            return numel(output_shapes[0]) * 5 if output_shapes else 0

        elif op_type in ("dropout",):
            # Dropout: 2 ops per element (mask + multiply)
            return numel(output_shapes[0]) * 2 if output_shapes else 0

        else:
            # Default: assume elementwise
            return numel(output_shapes[0]) if output_shapes else 0


@dataclass
class Task:
    """
    Represents a single schedulable unit of work.

    A task wraps a PyTorch operation with dependency tracking
    and cost metadata for intelligent scheduling.
    """

    # Identity
    id: int
    name: str
    op_type: str
    task_type: TaskType = TaskType.COMPUTE

    # Dependencies (task IDs)
    inputs: Set[int] = field(default_factory=set)   # Tasks this depends on
    outputs: Set[int] = field(default_factory=set)  # Tasks that depend on this

    # Tensor information
    input_shapes: List[tuple] = field(default_factory=list)
    output_shapes: List[tuple] = field(default_factory=list)
    dtype: torch.dtype = torch.float32

    # Cost estimation
    cost: Optional[TaskCost] = None

    # Execution state
    status: TaskStatus = TaskStatus.PENDING

    # Scheduling hints
    stream_affinity: Optional[int] = None  # Preferred CUDA stream
    priority: int = 0                       # Higher = more important

    # For execution
    kernel: Optional[Callable] = None
    result: Optional[Any] = None

    def __post_init__(self):
        """Ensure sets and estimate cost if not provided."""
        if isinstance(self.inputs, (list, tuple)):
            self.inputs = set(self.inputs)
        if isinstance(self.outputs, (list, tuple)):
            self.outputs = set(self.outputs)

        # Auto-estimate cost if not provided
        if self.cost is None:
            if self.input_shapes and self.output_shapes:
                self.cost = TaskCost.estimate_from_shape(
                    self.op_type,
                    self.input_shapes,
                    self.output_shapes,
                    self.dtype,
                )
            else:
                # Default cost for tasks without shape information
                self.cost = TaskCost()

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return self.status == TaskStatus.READY

    @property
    def is_completed(self) -> bool:
        """Check if task has completed."""
        return self.status == TaskStatus.COMPLETED

    @property
    def num_dependencies(self) -> int:
        """Number of input dependencies."""
        return len(self.inputs)

    @property
    def num_dependents(self) -> int:
        """Number of tasks that depend on this."""
        return len(self.outputs)

    def add_dependency(self, task_id: int) -> None:
        """Add an input dependency."""
        self.inputs.add(task_id)

    def add_dependent(self, task_id: int) -> None:
        """Add a task that depends on this."""
        self.outputs.add(task_id)

    def mark_ready(self) -> None:
        """Mark task as ready to execute."""
        self.status = TaskStatus.READY

    def mark_running(self) -> None:
        """Mark task as currently executing."""
        self.status = TaskStatus.RUNNING

    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id}, name='{self.name}', op='{self.op_type}', "
            f"deps={len(self.inputs)}, status={self.status.name})"
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if isinstance(other, Task):
            return self.id == other.id
        return False
