"""
🦊 Kitsune Backend System - Hardware-Specific Optimization

This package provides optimized backends for different hardware platforms:

1. Stable Backend (stable.py):
   - Production-ready optimizations
   - Uses torch.compile, CUDA graphs, TF32, channels-last

2. T4/Turing Backend (t4_optimizer.py):
   - INT8 quantization (61 TOPS on T4)
   - FP16 mixed precision
   - JIT fusion optimized for SM75

3. Apple Silicon Backend (apple_optimizer.py):
   - MPS (Metal Performance Shaders)
   - CoreML conversion for Neural Engine
   - Unified memory optimization

4. RTX Ampere/Ada Backend (rtx_optimizer.py):
   - TF32 acceleration (8x FP32 for matmul)
   - FP8 for RTX 40xx
   - 2:4 structured sparsity
"""

from .stable import StableBackend
from .experimental import ExperimentalBackend

# Hardware-specific backends (import with try/except for compatibility)
try:
    from .t4_optimizer import (
        T4Optimizer,
        T4QuantizationOptimizer,
        T4MixedPrecisionOptimizer,
        T4JITFusionOptimizer,
    )
except ImportError:
    T4Optimizer = None
    T4QuantizationOptimizer = None
    T4MixedPrecisionOptimizer = None
    T4JITFusionOptimizer = None

try:
    from .apple_optimizer import (
        AppleSiliconOptimizer,
        AppleMPSOptimizer,
    )
except ImportError:
    AppleSiliconOptimizer = None
    AppleMPSOptimizer = None

try:
    from .rtx_optimizer import (
        RTXOptimizer,
        RTXTF32Optimizer,
    )
except ImportError:
    RTXOptimizer = None
    RTXTF32Optimizer = None

try:
    from .backend_selector import (
        get_optimal_backend,
        detect_platform,
        PlatformType,
    )
except ImportError:
    get_optimal_backend = None
    detect_platform = None
    PlatformType = None

__all__ = [
    # Legacy
    'StableBackend', 
    'ExperimentalBackend',
    # T4/Turing
    "T4Optimizer",
    "T4QuantizationOptimizer",
    "T4MixedPrecisionOptimizer",
    "T4JITFusionOptimizer",
    # Apple Silicon
    "AppleSiliconOptimizer",
    "AppleMPSOptimizer",
    # RTX
    "RTXOptimizer",
    "RTXTF32Optimizer",
    # Selector
    "get_optimal_backend",
    "detect_platform",
    "PlatformType",
]
