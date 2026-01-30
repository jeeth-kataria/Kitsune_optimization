"""
Main API for Kitsune optimization.
Combines torch.compile + CUDA graphs + memory optimization.
"""
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field

from ..core.compiler import TorchCompiler, CompilerConfig
from ..cuda.graph_capture import CUDAGraphCapture, GraphConfig

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for Kitsune optimization."""
    
    # Compiler settings
    use_compile: bool = True
    compile_mode: str = "max-autotune"  # default, reduce-overhead, max-autotune
    
    # CUDA graph settings
    use_cuda_graphs: bool = True
    graph_warmup_iters: int = 20
    
    # Memory settings
    use_memory_pooling: bool = True
    memory_pool_size_gb: float = 1.0
    
    # Additional optimizations
    use_channels_last: bool = True  # For CNNs
    use_tf32: bool = True  # For Ampere GPUs
    
    # Behavior
    verbose: bool = True
    fallback_on_error: bool = True
    

class KitsuneOptimizer:
    """
    Main optimizer class that combines all Kitsune optimizations.
    
    Example:
        >>> model = MyModel().cuda()
        >>> sample_input = torch.randn(32, 3, 224, 224).cuda()
        >>> optimizer = KitsuneOptimizer(model, sample_input)
        >>> 
        >>> # For inference
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     output = optimizer(input_batch)  # Fast execution
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        config: Optional[OptimizationConfig] = None
    ):
        self.config = config or OptimizationConfig()
        self.original_model = model
        self.optimized_model = model
        self.sample_input = sample_input
        
        # Initialize components
        self.compiler: Optional[TorchCompiler] = None
        self.graph_capture: Optional[CUDAGraphCapture] = None
        
        # Tracking
        self.is_optimized = False
        self.optimization_stats = {}
        
        # Setup logging
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Apply optimizations
        self._optimize()
    
    def _optimize(self):
        """Apply all enabled optimizations."""
        logger.info("🦊 Kitsune: Starting optimization...")
        
        # Move to CUDA if not already
        if not next(self.original_model.parameters()).is_cuda:
            logger.info("Moving model to CUDA...")
            self.optimized_model = self.original_model.cuda()
        
        # 1. Apply torch.compile
        if self.config.use_compile:
            self._apply_compilation()
        
        # 2. Apply additional PyTorch optimizations
        self._apply_pytorch_optimizations()
        
        # 3. Setup CUDA graphs (only if sample input provided)
        if self.config.use_cuda_graphs and self.sample_input is not None:
            self._setup_cuda_graphs()
        
        self.is_optimized = True
        logger.info("✅ Kitsune: Optimization complete!")
        self._print_summary()
    
    def _apply_compilation(self):
        """Apply torch.compile optimization."""
        try:
            compiler_config = CompilerConfig(
                mode=self.config.compile_mode,
                fullgraph=False,  # More compatible
                dynamic=True  # Support dynamic shapes
            )
            
            self.compiler = TorchCompiler(compiler_config)
            
            if self.original_model.training:
                self.optimized_model = self.compiler.optimize_for_training(self.optimized_model)
            else:
                self.optimized_model = self.compiler.optimize_for_inference(self.optimized_model)
            
            self.optimization_stats['compile'] = 'enabled'
            logger.info("✓ torch.compile applied")
            
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            if not self.config.fallback_on_error:
                raise
            self.optimization_stats['compile'] = 'failed'
    
    def _apply_pytorch_optimizations(self):
        """Apply additional PyTorch-level optimizations."""
        try:
            # Enable TF32 for Ampere GPUs (RTX 30xx, A100, etc.)
            if self.config.use_tf32 and torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability()
                if compute_cap[0] >= 8:  # Ampere or newer
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    self.optimization_stats['tf32'] = 'enabled'
                    logger.info(f"✓ TF32 precision enabled (GPU: SM{compute_cap[0]}{compute_cap[1]})")
                else:
                    logger.info(f"⏭️  TF32 not available (GPU: SM{compute_cap[0]}{compute_cap[1]} - requires SM80+)")
            
            # Channels-last memory format for CNNs
            if self.config.use_channels_last:
                if self._is_cnn_model():
                    self.optimized_model = self.optimized_model.to(
                        memory_format=torch.channels_last
                    )
                    if self.sample_input is not None and self.sample_input.dim() == 4:
                        self.sample_input = self.sample_input.to(memory_format=torch.channels_last)
                    self.optimization_stats['channels_last'] = 'enabled'
                    logger.info("✓ Channels-last memory format applied")
            
        except Exception as e:
            logger.warning(f"Additional optimizations failed: {e}")
            if not self.config.fallback_on_error:
                raise
    
    def _setup_cuda_graphs(self):
        """Setup CUDA graph capture for inference."""
        if self.original_model.training:
            logger.info("⏭️  CUDA graphs skipped (model in training mode)")
            return
        
        try:
            graph_config = GraphConfig(
                enabled=True,
                warmup_iters=self.config.graph_warmup_iters
            )
            
            self.graph_capture = CUDAGraphCapture(graph_config)
            
            # Capture graph
            success = self.graph_capture.capture(
                self.optimized_model,
                self.sample_input
            )
            
            if success:
                self.optimization_stats['cuda_graphs'] = 'captured'
                logger.info("✓ CUDA graph captured")
            else:
                self.optimization_stats['cuda_graphs'] = 'failed'
                
        except Exception as e:
            logger.warning(f"CUDA graph capture failed: {e}")
            self.graph_capture = None
            if not self.config.fallback_on_error:
                raise
    
    def _is_cnn_model(self) -> bool:
        """Check if model contains convolutional layers."""
        for module in self.optimized_model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return True
        return False
    
    def _print_summary(self):
        """Print optimization summary."""
        if not self.config.verbose:
            return
            
        logger.info("\n" + "="*60)
        logger.info("🦊 KITSUNE OPTIMIZATION SUMMARY")
        logger.info("="*60)
        for opt_name, status in self.optimization_stats.items():
            status_icon = "✅" if status in ['enabled', 'captured'] else "⚠️"
            logger.info(f"  {status_icon} {opt_name:20s}: {status}")
        logger.info("="*60 + "\n")
    
    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Execute optimized forward pass.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Model output
        """
        # Use CUDA graph if available and shapes match
        if (self.graph_capture is not None and 
            self.graph_capture.is_captured and 
            tuple(input_data.shape) == self.graph_capture.input_shape):
            return self.graph_capture.replay(input_data)
        
        # Otherwise use compiled model
        return self.optimized_model(input_data)
    
    @property
    def model(self) -> torch.nn.Module:
        """Get the optimized model."""
        return self.optimized_model


# Convenience function for quick optimization
def optimize_model(
    model: torch.nn.Module,
    sample_input: Optional[torch.Tensor] = None,
    config: Optional[OptimizationConfig] = None
) -> KitsuneOptimizer:
    """
    Optimize a PyTorch model with Kitsune.
    
    Args:
        model: PyTorch model to optimize
        sample_input: Example input (required for CUDA graphs)
        config: Optimization configuration
    
    Returns:
        KitsuneOptimizer instance
    
    Example:
        >>> model = resnet50().cuda()
        >>> sample_input = torch.randn(32, 3, 224, 224).cuda()
        >>> optimizer = optimize_model(model, sample_input)
        >>> 
        >>> # Use it
        >>> output = optimizer(batch)
    """
    return KitsuneOptimizer(model, sample_input, config)
