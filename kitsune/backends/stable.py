"""
Stable Backend - Production-Ready Optimizations

Uses industry-standard PyTorch optimization techniques:
- torch.compile for kernel fusion and graph optimization
- CUDA Graphs for launch overhead elimination
- TF32 for Ampere+ GPUs
- Channels Last memory format for CNNs
- Intelligent fallback for dynamic shapes

Expected speedups: 1.3-2.0x on typical workloads
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StableBackend:
    """
    Standard Engineering Backend using PyTorch best practices.
    
    This backend combines multiple proven optimization techniques:
    1. Memory format optimization (channels-last for convs)
    2. TF32 precision for Ampere GPUs
    3. torch.compile for kernel fusion
    4. CUDA graph capture for fixed-shape inference
    
    Advantages:
    - Reliable and well-tested
    - Works across different model architectures
    - Graceful fallback for edge cases
    - Minimal setup required
    """
    
    def __init__(self, model: nn.Module, config):
        self.original_model = model
        self.model = model
        self.config = config
        self.cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.capture_shape: Optional[Tuple] = None
        
    def optimize(self, sample_input: torch.Tensor) -> nn.Module:
        """
        Apply all stable optimizations to the model.
        
        Args:
            sample_input: Representative input tensor for graph capture
            
        Returns:
            Optimized model (or wrapper)
        """
        logger.info("🔧 [Stable Backend] Applying Production Optimizations...")
        
        # 1. Memory Format Optimization (Channels Last for CNNs)
        if sample_input.dim() == 4 and sample_input.size(1) in [1, 3, 4]:  # NCHW format
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                sample_input = sample_input.to(memory_format=torch.channels_last)
                logger.info("   ✓ Applied Channels Last memory format (better cache locality)")
            except Exception as e:
                logger.warning(f"   ⚠ Channels Last failed: {e}, continuing...")
        
        # 2. TF32 Precision (Ampere+ GPUs)
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("   ✓ Enabled TF32 precision (faster matmul on Ampere+)")
        
        # 3. torch.compile (Kernel Fusion & Optimization)
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                logger.info("   ⏳ Compiling model (mode='reduce-overhead')...")
                # reduce-overhead mode optimizes for repeated execution
                # Best when combined with CUDA graphs
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False  # Allow graph breaks for flexibility
                )
                logger.info("   ✓ Model compiled successfully")
            except Exception as e:
                logger.warning(f"   ⚠ torch.compile failed: {e}, continuing without...")
        
        # 4. CUDA Graph Capture (Launch Overhead Elimination)
        if self.config.use_cuda_graphs and torch.cuda.is_available():
            try:
                self._capture_cuda_graph(sample_input)
            except Exception as e:
                logger.warning(f"   ⚠ CUDA graph capture failed: {e}, will use compiled model")
                self.cuda_graph = None
        
        logger.info("🎯 [Stable Backend] Optimization Complete")
        return self.model
    
    def _capture_cuda_graph(self, sample_input: torch.Tensor):
        """
        Capture CUDA graph for fixed-shape inference.
        
        CUDA graphs record the entire sequence of kernel launches
        and replay them with minimal CPU overhead.
        """
        logger.info("   ⏳ Capturing CUDA Graph...")
        
        self.original_model.eval()
        self.capture_shape = tuple(sample_input.shape)
        
        # Create stream for isolation
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        
        # Warmup: torch.compile needs to trace first
        with torch.cuda.stream(stream):
            with torch.no_grad():
                for _ in range(15):  # More warmup for compile to stabilize
                    _ = self.model(sample_input)
        
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        
        # Allocate static tensors
        self.static_input = sample_input.clone()
        
        # Capture graph on original model to avoid compile RNG issues
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.original_model(self.static_input)
        
        torch.cuda.synchronize()
        logger.info(f"   ✓ CUDA Graph captured for shape {self.capture_shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with intelligent execution strategy.
        
        Priority:
        1. CUDA graph replay (fastest, fixed shapes only)
        2. Compiled model (fast, works for all cases)
        3. Original model (fallback)
        """
        # Path 1: CUDA Graph (Fixed shape, eval mode)
        if (self.cuda_graph is not None and 
            not self.original_model.training and
            tuple(x.shape) == self.capture_shape):
            
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            return self.static_output.clone()
        
        # Path 2 & 3: Compiled or original model
        return self.model(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow calling backend directly."""
        return self.forward(x)
