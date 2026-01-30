"""
Kitsune Fusion - Kernel fusion layer

Contains kernel fusion optimizations:
- Fusion pattern detection and matching
- torch.compile backend for automatic fusion
- Triton-based fused kernels (Linux only)
- Pre-fused common operation patterns
"""

from .detector import FusionCandidate, FusionDetector
from .engine import (
    TORCH_COMPILE_AVAILABLE,
    TRITON_AVAILABLE,
    FusedKernel,
    FusedOperations,
    FusionEngine,
    TorchCompileBackend,
    get_fusion_engine,
    reset_fusion_engine,
)
from .patterns import (
    ACTIVATION_OPS,
    BUILTIN_PATTERNS,
    ELEMENTWISE_OPS,
    REDUCTION_OPS,
    FusionPattern,
    FusionType,
    PatternMatcher,
    get_fusion_type_for_op,
    is_fusable,
)

__all__ = [
    # Patterns
    "FusionPattern",
    "FusionType",
    "PatternMatcher",
    "BUILTIN_PATTERNS",
    "ELEMENTWISE_OPS",
    "ACTIVATION_OPS",
    "REDUCTION_OPS",
    "is_fusable",
    "get_fusion_type_for_op",
    # Detector
    "FusionDetector",
    "FusionCandidate",
    # Engine
    "FusionEngine",
    "FusedKernel",
    "FusedOperations",
    "TorchCompileBackend",
    "TRITON_AVAILABLE",
    "TORCH_COMPILE_AVAILABLE",
    "get_fusion_engine",
    "reset_fusion_engine",
]
