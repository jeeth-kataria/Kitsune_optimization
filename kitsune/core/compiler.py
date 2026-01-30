"""
Proper torch.compile integration with backend selection and optimization modes.
"""
import torch
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CompilerConfig:
    """Configuration for torch.compile optimization."""
    mode: str = "max-autotune"  # Options: default, reduce-overhead, max-autotune
    fullgraph: bool = False  # True = faster but stricter, False = more compatible
    dynamic: bool = None  # Allow dynamic shapes
    backend: str = "inductor"  # PyTorch's default backend
    disable: bool = False  # Disable compilation
    

class TorchCompiler:
    """Manages torch.compile optimization for models."""
    
    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()
        self._check_availability()
    
    def _check_availability(self):
        """Check if torch.compile is available."""
        try:
            # torch.compile requires PyTorch 2.0+
            version = torch.__version__.split('.')
            major = int(version[0])
            
            if major < 2:
                logger.warning(f"torch.compile requires PyTorch 2.0+, found {torch.__version__}")
                self.config.disable = True
            else:
                logger.info(f"torch.compile available with PyTorch {torch.__version__}")
        except Exception as e:
            logger.warning(f"Could not verify torch.compile availability: {e}")
            self.config.disable = True
    
    def compile(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Compile model with torch.compile for optimization.
        
        Args:
            model: PyTorch model to compile
            
        Returns:
            Compiled model (or original if compilation fails/disabled)
        """
        if self.config.disable:
            logger.info("torch.compile disabled, returning original model")
            return model
        
        try:
            logger.info(f"Compiling model with mode='{self.config.mode}'")
            
            compiled_model = torch.compile(
                model,
                mode=self.config.mode,
                fullgraph=self.config.fullgraph,
                dynamic=self.config.dynamic,
                backend=self.config.backend
            )
            
            logger.info("✓ Model compilation successful")
            return compiled_model
            
        except Exception as e:
            logger.error(f"torch.compile failed: {e}")
            logger.warning("Falling back to uncompiled model")
            return model
    
    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply inference-specific optimizations.
        
        Args:
            model: Model in eval mode
            
        Returns:
            Optimized model
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
        
        model.eval()
        
        # Apply torch.compile with inference-optimized settings
        old_mode = self.config.mode
        self.config.mode = "max-autotune"  # Best for inference
        
        compiled = self.compile(model)
        
        self.config.mode = old_mode
        return compiled
    
    def optimize_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply training-specific optimizations.
        
        Args:
            model: Model in train mode
            
        Returns:
            Optimized model
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
        
        model.train()
        
        # Apply torch.compile with training-optimized settings
        old_mode = self.config.mode
        self.config.mode = "reduce-overhead"  # Better for training with dynamic shapes
        
        compiled = self.compile(model)
        
        self.config.mode = old_mode
        return compiled
