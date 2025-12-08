"""
CUDA Graphs - Capture and replay CUDA operations for reduced overhead.

CUDA Graphs allow capturing a sequence of CUDA operations and replaying
them with minimal CPU overhead, which is especially beneficial for
small kernels where launch overhead dominates.
"""

from __future__ import annotations

import threading
from typing import Optional, List, Callable, Any, Dict
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch

from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class CaptureStats:
    """Statistics about a captured CUDA graph."""
    name: str
    num_nodes: int = 0
    capture_time_ms: float = 0.0
    replay_count: int = 0
    total_replay_time_ms: float = 0.0

    @property
    def avg_replay_time_ms(self) -> float:
        """Average replay time."""
        if self.replay_count == 0:
            return 0.0
        return self.total_replay_time_ms / self.replay_count


class CUDAGraphCapture:
    """
    Capture and replay CUDA operations as a graph.

    CUDA Graphs provide:
    - Reduced kernel launch overhead
    - Better optimization opportunities
    - Consistent execution patterns

    Usage:
        capture = CUDAGraphCapture("my_forward")

        # Warmup (required)
        for _ in range(3):
            output = model(input)

        # Capture
        with capture.capture():
            output = model(input)

        # Replay (fast)
        for _ in range(100):
            capture.replay()
    """

    def __init__(
        self,
        name: str = "graph",
        pool: Optional[torch.cuda.MemoryPool] = None,
    ):
        """
        Initialize CUDA graph capture.

        Args:
            name: Name for this graph (for logging)
            pool: Optional memory pool to use during capture
        """
        self.name = name
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False
        self._stream: Optional[torch.cuda.Stream] = None
        self._pool = pool
        self.stats = CaptureStats(name=name)

    @property
    def is_captured(self) -> bool:
        """Whether a graph has been captured."""
        return self._captured

    @contextmanager
    def capture(self, stream: Optional[torch.cuda.Stream] = None):
        """
        Context manager for capturing CUDA operations.

        All CUDA operations within this context will be captured
        into a graph for later replay.

        Args:
            stream: Stream to capture on (None for current)

        Yields:
            Self for chaining
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, graph capture disabled")
            yield self
            return

        self._stream = stream or torch.cuda.current_stream()

        # Ensure stream is idle before capture
        self._stream.synchronize()

        # Create new graph
        self._graph = torch.cuda.CUDAGraph()

        # Record capture start time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(self._stream)

        try:
            # Begin capture
            with torch.cuda.graph(self._graph, stream=self._stream, pool=self._pool):
                yield self

            self._captured = True
            logger.debug(f"CUDA graph '{self.name}' captured successfully")

        except Exception as e:
            logger.error(f"CUDA graph capture failed: {e}")
            self._graph = None
            self._captured = False
            raise

        finally:
            end.record(self._stream)
            end.synchronize()
            self.stats.capture_time_ms = start.elapsed_time(end)

    def replay(self) -> None:
        """
        Replay the captured graph.

        Raises:
            RuntimeError: If no graph has been captured
        """
        if not self._captured or self._graph is None:
            raise RuntimeError(f"Graph '{self.name}' has not been captured")

        start = None
        if logger.isEnabledFor(10):  # DEBUG level
            start = torch.cuda.Event(enable_timing=True)
            start.record(self._stream)

        self._graph.replay()
        self.stats.replay_count += 1

        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(self._stream)
            end.synchronize()
            self.stats.total_replay_time_ms += start.elapsed_time(end)

    def reset(self) -> None:
        """Reset the capture, discarding the graph."""
        self._graph = None
        self._captured = False
        self.stats.replay_count = 0
        self.stats.total_replay_time_ms = 0.0

    def __repr__(self) -> str:
        status = "captured" if self._captured else "not captured"
        return f"CUDAGraphCapture(name='{self.name}', {status})"


class GraphPool:
    """
    Pool of pre-captured CUDA graphs for common operations.

    Maintains a cache of captured graphs keyed by operation
    signature, allowing reuse of graphs for repeated operations.
    """

    def __init__(self, max_graphs: int = 32):
        """
        Initialize the graph pool.

        Args:
            max_graphs: Maximum number of graphs to cache
        """
        self._max_graphs = max_graphs
        self._graphs: Dict[str, CUDAGraphCapture] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

    def get_or_capture(
        self,
        key: str,
        capture_fn: Callable[[], None],
        warmup_iters: int = 3,
    ) -> CUDAGraphCapture:
        """
        Get a cached graph or capture a new one.

        Args:
            key: Unique key for this operation
            capture_fn: Function to capture (called during capture)
            warmup_iters: Number of warmup iterations before capture

        Returns:
            The captured graph
        """
        with self._lock:
            if key in self._graphs:
                # Move to end of access order (LRU)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._graphs[key]

        # Capture new graph
        graph = CUDAGraphCapture(name=key)

        # Warmup
        for _ in range(warmup_iters):
            capture_fn()

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture
        with graph.capture():
            capture_fn()

        with self._lock:
            # Evict LRU if at capacity
            while len(self._graphs) >= self._max_graphs:
                oldest = self._access_order.pop(0)
                del self._graphs[oldest]

            self._graphs[key] = graph
            self._access_order.append(key)

        return graph

    def contains(self, key: str) -> bool:
        """Check if a graph is cached."""
        return key in self._graphs

    def clear(self) -> None:
        """Clear all cached graphs."""
        with self._lock:
            self._graphs.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, CaptureStats]:
        """Get statistics for all cached graphs."""
        return {k: v.stats for k, v in self._graphs.items()}

    @property
    def num_graphs(self) -> int:
        """Number of cached graphs."""
        return len(self._graphs)


class StaticGraphExecutor:
    """
    Executor for models with static computation graphs.

    Automatically captures the model's forward pass as a CUDA graph
    on first execution and replays it on subsequent calls.

    Best suited for:
    - Fixed input shapes
    - No dynamic control flow
    - Repeated inference with same structure
    """

    def __init__(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        warmup_iters: int = 3,
    ):
        """
        Initialize the static graph executor.

        Args:
            model: PyTorch model to execute
            example_input: Example input tensor (defines shape)
            warmup_iters: Warmup iterations before capture
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for graph executor")

        self.model = model
        self._input_shape = example_input.shape
        self._graph = CUDAGraphCapture(name="static_forward")
        self._output: Optional[torch.Tensor] = None
        self._static_input: Optional[torch.Tensor] = None

        self._capture(example_input, warmup_iters)

    def _capture(self, example_input: torch.Tensor, warmup_iters: int) -> None:
        """Capture the model forward pass."""
        device = next(self.model.parameters()).device

        # Create static input buffer
        self._static_input = example_input.clone().to(device)

        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup_iters):
                self._output = self.model(self._static_input)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture
        with torch.no_grad():
            with self._graph.capture():
                self._output = self.model(self._static_input)

        logger.info(f"Static graph captured for input shape {self._input_shape}")

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Execute the model using the captured graph.

        Args:
            input: Input tensor (must match captured shape)

        Returns:
            Model output
        """
        if input.shape != self._input_shape:
            raise RuntimeError(
                f"Input shape {input.shape} doesn't match captured shape {self._input_shape}"
            )

        # Copy input to static buffer
        self._static_input.copy_(input)

        # Replay graph
        self._graph.replay()

        # Return output (already computed in static buffer)
        return self._output

    def reset(self) -> None:
        """Reset and recapture the graph."""
        self._graph.reset()


# Global graph pool singleton
_global_graph_pool: Optional[GraphPool] = None


def get_graph_pool() -> GraphPool:
    """Get or create the global graph pool."""
    global _global_graph_pool
    if _global_graph_pool is None:
        _global_graph_pool = GraphPool()
    return _global_graph_pool


def reset_graph_pool() -> None:
    """Reset the global graph pool."""
    global _global_graph_pool
    if _global_graph_pool is not None:
        _global_graph_pool.clear()
    _global_graph_pool = None
