"""
CUDA graph capture for inference acceleration.
Eliminates kernel launch overhead for fixed-shape workloads.
"""
import torch
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for CUDA graph capture."""
    enabled: bool = True
    warmup_iters: int = 20
    pool_size: str = "default"  # Options: default, max, min
    

class CUDAGraphCapture:
    """Manages CUDA graph capture and replay for models."""
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.input_shape: Optional[Tuple] = None
        self._is_captured = False
    
    def can_capture(self) -> bool:
        """Check if CUDA graphs can be used."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot use CUDA graphs")
            return False
        
        if not self.config.enabled:
            logger.info("CUDA graphs disabled by config")
            return False
        
        # CUDA graphs require CUDA 11.0+
        try:
            cuda_version = torch.version.cuda
            if cuda_version and float(cuda_version.split('.')[0]) < 11:
                logger.warning(f"CUDA graphs require CUDA 11.0+, found {cuda_version}")
                return False
        except:
            pass
        
        return True
    
    def capture(
        self, 
        model: torch.nn.Module, 
        sample_input: torch.Tensor
    ) -> bool:
        """
        Capture CUDA graph for a model with fixed input shape.
        
        Args:
            model: Model in eval mode on CUDA
            sample_input: Example input tensor on CUDA
            
        Returns:
            True if capture successful, False otherwise
        """
        if not self.can_capture():
            return False
        
        if not model.training:
            model.eval()
        
        try:
            logger.info("Capturing CUDA graph...")
            
            # Store input shape for validation
            self.input_shape = tuple(sample_input.shape)
            
            # Warmup: compile kernels and stabilize allocations
            logger.debug(f"Warming up for {self.config.warmup_iters} iterations...")
            with torch.no_grad():
                for _ in range(self.config.warmup_iters):
                    _ = model(sample_input)
            
            torch.cuda.synchronize()
            
            # Create static tensors (required for graph capture)
            self.static_input = sample_input.clone()
            
            # Create CUDA graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            # Capture graph
            logger.debug("Capturing computational graph...")
            with torch.cuda.graph(self.cuda_graph):
                self.static_output = model(self.static_input)
            
            torch.cuda.synchronize()
            
            self._is_captured = True
            logger.info(f"✓ CUDA graph captured for input shape {self.input_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"CUDA graph capture failed: {e}")
            logger.debug("Falling back to eager execution")
            self._is_captured = False
            self.cuda_graph = None
            return False
    
    def replay(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Replay captured CUDA graph.
        
        Args:
            input_data: Input tensor (must match captured shape)
            
        Returns:
            Model output
            
        Raises:
            RuntimeError: If shapes don't match or graph not captured
        """
        if not self._is_captured:
            raise RuntimeError("CUDA graph not captured. Call capture() first.")
        
        # Validate input shape
        if tuple(input_data.shape) != self.input_shape:
            raise RuntimeError(
                f"Input shape mismatch. Expected {self.input_shape}, "
                f"got {tuple(input_data.shape)}"
            )
        
        # Copy input to static buffer
        self.static_input.copy_(input_data)
        
        # Replay graph
        self.cuda_graph.replay()
        
        # Return output (copy to avoid graph memory issues)
        return self.static_output.clone()
    
    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Convenience method for replay."""
        return self.replay(input_data)
    
    @property
    def is_captured(self) -> bool:
        """Check if graph is captured and ready."""
        return self._is_captured
