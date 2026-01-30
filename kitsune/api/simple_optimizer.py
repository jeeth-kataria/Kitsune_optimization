"""
Simplified Kitsune API - Dual-Backend Optimization

This module provides a clean, simple API for model optimization
using the dual-backend architecture.

Usage:
    from kitsune import optimize_model, OptimizationMode
    
    # Stable mode (default - guaranteed speedups)
    optimized = optimize_model(model, sample_input)
    
    # Experimental mode (research/demonstration)
    optimized = optimize_model(model, sample_input, mode="experimental")
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch
import torch.nn as nn
import logging

from ..backends import StableBackend, ExperimentalBackend

logger = logging.getLogger(__name__)


# Type alias for backend modes
OptimizationMode = Literal["stable", "experimental"]


@dataclass
class KitsuneConfig:
    """
    Configuration for Kitsune optimization.
    
    Attributes:
        mode: Backend to use ('stable' or 'experimental')
        use_compile: Enable torch.compile (stable backend only)
        use_cuda_graphs: Enable CUDA graph capture
        num_streams: Number of CUDA streams (experimental backend)
        verbose: Print optimization steps
    """
    mode: OptimizationMode = "stable"
    use_compile: bool = True
    use_cuda_graphs: bool = True
    num_streams: int = 4
    verbose: bool = True


class KitsuneOptimizer:
    """
    Unified optimizer interface for both backends.
    
    This class provides a simple wrapper around the dual-backend
    system, automatically selecting and initializing the appropriate
    backend based on configuration.
    
    Example:
        >>> model = MyModel().cuda()
        >>> sample_input = torch.randn(32, 3, 224, 224).cuda()
        >>> optimizer = KitsuneOptimizer(model, sample_input)
        >>> output = optimizer(input_data)  # Optimized execution
    """
    
    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        config: Optional[KitsuneConfig] = None
    ):
        """
        Initialize Kitsune optimizer with automatic backend selection.
        
        Args:
            model: PyTorch model to optimize
            sample_input: Representative input tensor for optimization
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or KitsuneConfig()
        self.model = model
        
        # Auto-move to CUDA if not already
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            self.model = model.cuda()
            sample_input = sample_input.cuda()
            logger.info("Model moved to CUDA")
        
        # Print mode banner
        if self.config.verbose:
            self._print_banner()
        
        # Select and initialize backend
        if self.config.mode == "experimental":
            self.backend = ExperimentalBackend(self.model, self.config)
        else:
            self.backend = StableBackend(self.model, self.config)
        
        # Apply optimizations
        self.optimized_model = self.backend.optimize(sample_input)
        
    def _print_banner(self):
        """Print optimization mode banner."""
        mode_display = self.config.mode.upper()
        width = 60
        
        print()
        print("=" * width)
        print(f"🦊 KITSUNE OPTIMIZER | Mode: {mode_display}")
        print("=" * width)
        
        if self.config.mode == "stable":
            print("Backend: Production (torch.compile + CUDA graphs)")
            print("Expected: 1.3-2.0x speedup with proven reliability")
        else:
            print("Backend: Experimental (custom kernel scheduling)")
            print("Expected: Variable performance, research demonstration")
        
        print("=" * width)
        print()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute optimized forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.backend.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for __call__."""
        return self(x)


def optimize_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    mode: OptimizationMode = "stable",
    use_compile: bool = True,
    use_cuda_graphs: bool = True,
    verbose: bool = True
) -> KitsuneOptimizer:
    """
    Optimize a PyTorch model for faster inference.
    
    This is the primary entry point for Kitsune optimization.
    It provides a simple interface to the dual-backend system.
    
    Args:
        model: PyTorch model to optimize
        sample_input: Representative input tensor (for graph capture)
        mode: Optimization backend to use
            - "stable": Production backend with guaranteed speedups
            - "experimental": Research backend with custom kernels
        use_compile: Enable torch.compile (stable mode only)
        use_cuda_graphs: Enable CUDA graph capture
        verbose: Print optimization progress
    
    Returns:
        KitsuneOptimizer instance wrapping the optimized model
    
    Example (Stable Mode):
        >>> model = resnet50().cuda()
        >>> sample = torch.randn(32, 3, 224, 224).cuda()
        >>> opt = optimize_model(model, sample)
        >>> output = opt(input_data)  # 1.5-2.0x faster
    
    Example (Experimental Mode):
        >>> opt = optimize_model(model, sample, mode="experimental")
        >>> output = opt(input_data)  # Research/demonstration
    """
    config = KitsuneConfig(
        mode=mode,
        use_compile=use_compile,
        use_cuda_graphs=use_cuda_graphs,
        verbose=verbose
    )
    
    return KitsuneOptimizer(model, sample_input, config)


__all__ = [
    'KitsuneOptimizer',
    'KitsuneConfig',
    'optimize_model',
    'OptimizationMode'
]
