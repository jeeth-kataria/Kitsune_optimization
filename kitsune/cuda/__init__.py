"""
Kitsune CUDA - GPU acceleration layer

Contains CUDA-specific optimizations:
- Stream pool management for parallel execution
- Event-based synchronization
- CUDA Graph capture and replay
"""

from .stream_pool import (
    CUDAStream,
    StreamPool,
    StreamStats,
    StreamScheduler,
    get_stream_pool,
    reset_stream_pool,
)
from .events import (
    EventManager,
    EventTiming,
    DependencyTracker,
    EventBarrier,
    get_event_manager,
    reset_event_manager,
)
from .graphs import (
    CUDAGraphCapture,
    CaptureStats,
    GraphPool,
    StaticGraphExecutor,
    get_graph_pool,
    reset_graph_pool,
)

__all__ = [
    # Stream Pool
    "CUDAStream",
    "StreamPool",
    "StreamStats",
    "StreamScheduler",
    "get_stream_pool",
    "reset_stream_pool",
    # Events
    "EventManager",
    "EventTiming",
    "DependencyTracker",
    "EventBarrier",
    "get_event_manager",
    "reset_event_manager",
    # CUDA Graphs
    "CUDAGraphCapture",
    "CaptureStats",
    "GraphPool",
    "StaticGraphExecutor",
    "get_graph_pool",
    "reset_graph_pool",
]
