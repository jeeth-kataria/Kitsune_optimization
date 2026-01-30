"""
🔍 Automatic Backend Selection

Detects hardware and returns the optimal optimizer.

Priority Order:
1. T4 (Colab) → T4Optimizer with INT8 quantization
2. Apple Silicon → AppleSiliconOptimizer with MPS
3. RTX 40xx → RTXOptimizer with FP8
4. RTX 30xx / Ampere → RTXOptimizer with TF32
5. Other CUDA → Generic CUDA optimizer
6. CPU → CPU-specific optimizations
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Type
from enum import Enum
import logging
import platform

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Detected platform types."""
    NVIDIA_T4 = "nvidia_t4"
    NVIDIA_AMPERE = "nvidia_ampere"
    NVIDIA_HOPPER = "nvidia_hopper"
    NVIDIA_ADA = "nvidia_ada"
    NVIDIA_OTHER = "nvidia_other"
    APPLE_SILICON = "apple_silicon"
    CPU_ONLY = "cpu_only"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Information about the detected platform."""
    platform_type: PlatformType
    device: torch.device
    name: str
    compute_capability: Optional[tuple] = None
    recommended_optimizer: str = "generic"
    optimization_potential: str = "1.2-1.5x"


def detect_platform() -> PlatformInfo:
    """
    Detect the current hardware platform.
    
    Returns:
        PlatformInfo with detected platform details
    """
    # Check for Apple Silicon first (macOS with MPS)
    if platform.system() == 'Darwin':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return PlatformInfo(
                platform_type=PlatformType.APPLE_SILICON,
                device=torch.device('mps'),
                name="Apple Silicon (MPS)",
                recommended_optimizer="AppleSiliconOptimizer",
                optimization_potential="3-5x over CPU"
            )
    
    # Check for CUDA
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        cc = (props.major, props.minor)
        
        # Detect specific GPU type
        if 'T4' in name:
            return PlatformInfo(
                platform_type=PlatformType.NVIDIA_T4,
                device=torch.device('cuda'),
                name=name,
                compute_capability=cc,
                recommended_optimizer="T4Optimizer",
                optimization_potential="2.0-2.5x with INT8"
            )
        
        elif cc >= (9, 0):  # Hopper
            return PlatformInfo(
                platform_type=PlatformType.NVIDIA_HOPPER,
                device=torch.device('cuda'),
                name=name,
                compute_capability=cc,
                recommended_optimizer="RTXOptimizer",
                optimization_potential="3-4x with FP8"
            )
        
        elif cc >= (8, 9):  # Ada Lovelace (RTX 40xx)
            return PlatformInfo(
                platform_type=PlatformType.NVIDIA_ADA,
                device=torch.device('cuda'),
                name=name,
                compute_capability=cc,
                recommended_optimizer="RTXOptimizer",
                optimization_potential="3x+ with FP8"
            )
        
        elif cc >= (8, 0):  # Ampere (RTX 30xx, A100)
            return PlatformInfo(
                platform_type=PlatformType.NVIDIA_AMPERE,
                device=torch.device('cuda'),
                name=name,
                compute_capability=cc,
                recommended_optimizer="RTXOptimizer",
                optimization_potential="2-2.5x with TF32"
            )
        
        else:  # Older CUDA
            return PlatformInfo(
                platform_type=PlatformType.NVIDIA_OTHER,
                device=torch.device('cuda'),
                name=name,
                compute_capability=cc,
                recommended_optimizer="GenericCUDAOptimizer",
                optimization_potential="1.3-1.8x with JIT"
            )
    
    # CPU only
    return PlatformInfo(
        platform_type=PlatformType.CPU_ONLY,
        device=torch.device('cpu'),
        name=f"CPU ({platform.processor() or 'Unknown'})",
        recommended_optimizer="CPUOptimizer",
        optimization_potential="1.1-1.3x with JIT"
    )


def get_optimal_backend(
    platform_info: Optional[PlatformInfo] = None
) -> Union['T4Optimizer', 'AppleSiliconOptimizer', 'RTXOptimizer', 'GenericOptimizer']:
    """
    Get the optimal optimizer for the current platform.
    
    Args:
        platform_info: Optional pre-detected platform info
        
    Returns:
        Appropriate optimizer instance for the platform
    """
    if platform_info is None:
        platform_info = detect_platform()
    
    logger.info(f"🔍 Platform: {platform_info.name}")
    logger.info(f"   Type: {platform_info.platform_type.value}")
    logger.info(f"   Optimizer: {platform_info.recommended_optimizer}")
    logger.info(f"   Potential: {platform_info.optimization_potential}")
    
    try:
        if platform_info.platform_type == PlatformType.NVIDIA_T4:
            from .t4_optimizer import T4Optimizer
            return T4Optimizer()
        
        elif platform_info.platform_type == PlatformType.APPLE_SILICON:
            from .apple_optimizer import AppleSiliconOptimizer
            return AppleSiliconOptimizer()
        
        elif platform_info.platform_type in [
            PlatformType.NVIDIA_ADA,
            PlatformType.NVIDIA_AMPERE,
            PlatformType.NVIDIA_HOPPER
        ]:
            from .rtx_optimizer import RTXOptimizer
            return RTXOptimizer()
        
        else:
            # Fallback to generic optimizer
            return GenericOptimizer(platform_info)
    
    except ImportError as e:
        logger.warning(f"Could not import specialized optimizer: {e}")
        return GenericOptimizer(platform_info)


class GenericOptimizer:
    """
    Generic optimizer for unsupported platforms.
    
    Applies basic optimizations that work everywhere:
    - JIT trace
    - torch.compile (if available)
    """
    
    def __init__(self, platform_info: Optional[PlatformInfo] = None):
        self.platform_info = platform_info or detect_platform()
    
    def optimize(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        use_compile: bool = True
    ) -> nn.Module:
        """
        Apply generic optimizations.
        
        Args:
            model: PyTorch model
            sample_input: Example input
            use_compile: Try torch.compile if available
            
        Returns:
            Optimized model
        """
        logger.info("🔧 Applying generic optimizations...")
        
        model.eval()
        device = self.platform_info.device
        
        # Move to device
        model = model.to(device)
        sample_input = sample_input.to(device)
        
        optimizations = []
        
        # Try torch.compile first (PyTorch 2.x)
        if use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                optimizations.append("torch.compile")
                logger.info("   ✓ torch.compile applied")
            except Exception as e:
                logger.debug(f"torch.compile failed: {e}")
        
        # Fall back to JIT trace
        if not optimizations:
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(model, sample_input)
                    traced = torch.jit.optimize_for_inference(traced)
                    traced = torch.jit.freeze(traced)
                model = traced
                optimizations.append("JIT trace + freeze")
                logger.info("   ✓ JIT trace applied")
            except Exception as e:
                logger.warning(f"JIT trace failed: {e}")
        
        logger.info(f"✅ Applied: {', '.join(optimizations) or 'None'}")
        
        return model


def auto_optimize(
    model: nn.Module,
    sample_input: torch.Tensor,
    level: str = "balanced"
) -> nn.Module:
    """
    Automatically detect platform and apply optimal optimizations.
    
    This is the main entry point for automatic optimization.
    
    Args:
        model: PyTorch model to optimize
        sample_input: Example input tensor
        level: Optimization level ("conservative", "balanced", "aggressive")
        
    Returns:
        Optimized model on the appropriate device
    
    Example:
        >>> model = torchvision.models.resnet18()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> optimized = auto_optimize(model, x, level="balanced")
    """
    platform_info = detect_platform()
    optimizer = get_optimal_backend(platform_info)
    
    logger.info(f"🦊 Auto-optimizing for {platform_info.platform_type.value}...")
    
    # Move input to correct device
    sample_input = sample_input.to(platform_info.device)
    
    # Call appropriate optimize method
    if hasattr(optimizer, 'optimize'):
        # All our optimizers have this signature
        result = optimizer.optimize(model, sample_input)
        
        # Handle different result types
        if hasattr(result, 'model'):
            return result.model
        return result
    
    return model


def print_platform_info():
    """Print detected platform information."""
    info = detect_platform()
    
    print("\n" + "=" * 50)
    print("🦊 Kitsune Platform Detection")
    print("=" * 50)
    print(f"Platform: {info.name}")
    print(f"Type: {info.platform_type.value}")
    print(f"Device: {info.device}")
    if info.compute_capability:
        print(f"Compute: SM{info.compute_capability[0]}{info.compute_capability[1]}")
    print(f"Recommended: {info.recommended_optimizer}")
    print(f"Potential: {info.optimization_potential}")
    print("=" * 50 + "\n")


# Aliases for convenience
detect = detect_platform
get_backend = get_optimal_backend
optimize = auto_optimize
