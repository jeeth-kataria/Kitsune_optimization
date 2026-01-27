"""
Kitsune - CUDA-accelerated dynamic task scheduler for PyTorch

A dataflow-driven scheduler that optimizes PyTorch neural network training
through intelligent graph analysis, CUDA stream parallelism, memory pooling,
and kernel fusion.

Example:
    >>> import kitsune
    >>> import torch.nn as nn
    >>>
    >>> model = MyModel().cuda()
    >>> sample_input = torch.randn(64, 784, device="cuda")
    >>>
    >>> # Setup optimizer
    >>> optimizer = kitsune.optimize_model(model, sample_input)
    >>>
    >>> # Training loop with prefetching
    >>> for batch in optimizer.prefetch(dataloader):
    ...     with optimizer.optimize():
    ...         output = model(batch)
    ...         loss = criterion(output, target)
    ...         loss.backward()
    ...
    >>> # View optimization stats
    >>> print(optimizer.summary())
"""

__version__ = "0.1.0"  # Initial release
__author__ = "Kitsune Team"

# Profiler (Week 1)
from .profiler import CUDATimer, MemoryTracker, Profiler, get_logger

# Core (Week 2)
from .core import (
    Task,
    TaskType,
    TaskStatus,
    TaskCost,
    ComputationGraph,
    DataflowScheduler,
    ExecutionPlan,
)

# CUDA (Week 3)
from .cuda import (
    StreamPool,
    CUDAStream,
    EventManager,
    CUDAGraphCapture,
    get_stream_pool,
)

# Memory (Week 4)
from .memory import (
    MemoryPool,
    TensorCache,
    DoubleBuffer,
    CUDAPrefetcher,
    LifetimeAnalyzer,
    get_memory_pool,
    create_prefetched_loader,
)

# API (Week 4 MVP)
from .api import (
    KitsuneOptimizer,
    OptimizationConfig,
    OptimizationStats,
    optimize_model,
)

# Fusion (Week 5)
from .fusion import (
    FusionEngine,
    FusionDetector,
    FusedOperations,
    TRITON_AVAILABLE,
    get_fusion_engine,
)

# AMP (Week 6)
from .amp import (
    AMPConfig,
    PrecisionMode,
    KitsuneGradScaler,
    AMPOptimizer,
    autocast_context,
)

# PyTorch integration
from .pytorch import capture_graph, GraphCapture


# Version info
def get_version() -> str:
    """Return the current version of Kitsune."""
    return __version__


def get_device_info() -> dict:
    """Return information about available CUDA devices."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "kitsune_version": __version__,
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [],
        })

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    return info


def check_compatibility() -> tuple[bool, list[str]]:
    """
    Check if the current environment is compatible with Kitsune.

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    import torch
    from packaging import version

    warnings = []
    is_compatible = True

    # Check PyTorch version
    pytorch_version = torch.__version__.split("+")[0]
    if version.parse(pytorch_version) < version.parse("2.0.0"):
        warnings.append(
            f"PyTorch {pytorch_version} is below minimum 2.0.0. "
            "Some features may not work correctly."
        )
        is_compatible = False

    # Check CUDA availability
    if not torch.cuda.is_available():
        warnings.append(
            "CUDA is not available. Kitsune requires CUDA for acceleration. "
            "Will fall back to baseline PyTorch execution."
        )
        is_compatible = False
    else:
        # Check compute capability
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = props.major + props.minor / 10
            if cc < 6.0:
                warnings.append(
                    f"GPU {i} ({props.name}) has compute capability {props.major}.{props.minor}, "
                    "which is below the minimum 6.0. Performance may be limited."
                )

    return is_compatible, warnings


__all__ = [
    # Version
    "__version__",
    "get_version",
    "get_device_info",
    "check_compatibility",
    # Profiler
    "CUDATimer",
    "MemoryTracker",
    "Profiler",
    "get_logger",
    # Core
    "Task",
    "TaskType",
    "TaskStatus",
    "TaskCost",
    "ComputationGraph",
    "DataflowScheduler",
    "ExecutionPlan",
    # CUDA
    "StreamPool",
    "CUDAStream",
    "EventManager",
    "CUDAGraphCapture",
    "get_stream_pool",
    # Memory
    "MemoryPool",
    "TensorCache",
    "DoubleBuffer",
    "CUDAPrefetcher",
    "LifetimeAnalyzer",
    "get_memory_pool",
    "create_prefetched_loader",
    # API
    "KitsuneOptimizer",
    "OptimizationConfig",
    "OptimizationStats",
    "optimize_model",
    # Fusion
    "FusionEngine",
    "FusionDetector",
    "FusedOperations",
    "TRITON_AVAILABLE",
    "get_fusion_engine",
    # AMP
    "AMPConfig",
    "PrecisionMode",
    "KitsuneGradScaler",
    "AMPOptimizer",
    "autocast_context",
    # PyTorch
    "capture_graph",
    "GraphCapture",
]
