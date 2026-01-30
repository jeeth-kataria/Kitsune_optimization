"""
🦊 Kitsune - PyTorch Model Optimizer (Device-Aware)

A lightweight optimizer that automatically detects your hardware and applies
optimal settings for maximum performance:

Supported Hardware:
- Tesla T4 (SM75): JIT Trace best, 1.15-1.25x speedup
- Ampere GPUs (RTX 30xx, A100): torch.compile + TF32, 1.5-2.5x speedup
- Hopper GPUs (H100): All optimizations + FP8, 2-4x speedup
- CPU: JIT + optional INT8 quantization, 1.2-4x speedup

Simple API:
    >>> import kitsune
    >>> import torch
    >>>
    >>> model = MyModel().cuda().eval()
    >>> sample = torch.randn(1, 3, 224, 224, device='cuda')
    >>>
    >>> # Auto-configures based on your hardware
    >>> optimized = kitsune.optimize(model, sample)
    >>> output = optimized(input_data)  # Optimized inference

With Configuration:
    >>> from kitsune import OptimizationConfig, KitsuneOptimizer
    >>> 
    >>> config = OptimizationConfig(strategy='compile', compile_mode='max-autotune')
    >>> optimizer = KitsuneOptimizer(model, sample, config)
    >>> output = optimizer(input_data)

Hardware Detection:
    >>> from kitsune import detect_hardware, show_performance_guide
    >>> hw = detect_hardware()
    >>> print(hw)  # Shows detected hardware and capabilities
    >>> show_performance_guide()  # Shows expected performance for each device
"""

__version__ = "0.3.0"  # Device-aware optimization release
__author__ = "Kitsune Team"

# Primary API (v2 - tested and working)
from .api.optimizer_v2 import (
    optimize_model,
    optimize,
    get_optimizer,
    KitsuneOptimizer,
    OptimizationConfig,
    benchmark_optimization,
    print_benchmark,
)

# Device-Aware Configuration (NEW)
from .api.device_config import (
    detect_hardware,
    get_optimal_config,
    apply_hardware_optimizations,
    print_hardware_info,
    show_performance_guide,
    HardwareInfo,
    DeviceType,
    T4Config,
    AmpereConfig,
    HopperConfig,
    CPUConfig,
    GenericCUDAConfig,
)

# Simple API (Legacy - kept for compatibility)
try:
    from .api.simple_optimizer import (
        KitsuneConfig,
        OptimizationMode,
    )
except ImportError:
    pass

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

# Hardware-Specific Backends (NEW in v0.3.0)
try:
    from .backends import (
        # Backend selector
        detect_platform,
        get_optimal_backend,
        PlatformType,
        # T4/Turing optimizer
        T4Optimizer,
        T4QuantizationOptimizer,
        T4MixedPrecisionOptimizer,
        T4JITFusionOptimizer,
        # Apple Silicon optimizer
        AppleSiliconOptimizer,
        AppleMPSOptimizer,
        # RTX optimizer
        RTXOptimizer,
        RTXTF32Optimizer,
    )
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    detect_platform = None
    get_optimal_backend = None
    PlatformType = None
    T4Optimizer = None
    AppleSiliconOptimizer = None
    RTXOptimizer = None

# Convenience function for auto-optimization
def auto_optimize(model, sample_input, level="balanced"):
    """
    Automatically detect hardware and apply optimal optimizations.
    
    This is the recommended entry point for most users.
    
    Args:
        model: PyTorch model to optimize
        sample_input: Example input tensor
        level: Optimization level ("conservative", "balanced", "aggressive")
        
    Returns:
        Optimized model on the appropriate device
    
    Example:
        >>> import kitsune
        >>> import torchvision.models as models
        >>> 
        >>> model = models.resnet50()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> optimized = kitsune.auto_optimize(model, x)
    """
    if BACKENDS_AVAILABLE and detect_platform is not None:
        from .backends.backend_selector import auto_optimize as _auto_optimize
        return _auto_optimize(model, sample_input, level)
    else:
        # Fallback to basic optimize
        return optimize(model, sample_input)


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
    # API (Simple - Recommended)
    "optimize_model",
    "optimize",
    "auto_optimize",
    "KitsuneConfig",
    "OptimizationMode",
    # API (Advanced - Original)
    "KitsuneOptimizer",
    "OptimizationConfig",
    "OptimizationStats",
    # Device Detection
    "detect_hardware",
    "get_optimal_config",
    "apply_hardware_optimizations",
    "print_hardware_info",
    "show_performance_guide",
    "HardwareInfo",
    "DeviceType",
    # Hardware-Specific Backends (NEW)
    "detect_platform",
    "get_optimal_backend",
    "PlatformType",
    "T4Optimizer",
    "AppleSiliconOptimizer",
    "RTXOptimizer",
    "BACKENDS_AVAILABLE",
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
