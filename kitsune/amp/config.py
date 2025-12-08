"""
AMP Configuration - Precision modes and settings.

Manages mixed precision configuration for the Kitsune optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Set, Dict, Any
import torch


class PrecisionMode(Enum):
    """Precision mode for operations."""
    FP32 = auto()       # Full precision (default PyTorch)
    FP16 = auto()       # Half precision (float16)
    BF16 = auto()       # BFloat16 (better dynamic range)
    TF32 = auto()       # TensorFloat32 (NVIDIA Ampere+)
    AUTO = auto()       # Automatic selection based on hardware


@dataclass
class AMPConfig:
    """
    Configuration for Automatic Mixed Precision.

    Attributes:
        enabled: Whether AMP is enabled
        precision_mode: Default precision mode
        grad_scaler_enabled: Whether to use gradient scaling
        init_scale: Initial scale for gradient scaler
        growth_factor: Scale growth factor
        backoff_factor: Scale backoff factor on overflow
        growth_interval: Steps between scale updates
        ops_fp32: Operations that must stay in FP32
        ops_fp16: Operations that should use FP16
    """
    enabled: bool = True
    precision_mode: PrecisionMode = PrecisionMode.AUTO
    grad_scaler_enabled: bool = True
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # Operation precision overrides
    ops_fp32: Set[str] = field(default_factory=lambda: {
        "softmax", "log_softmax", "layer_norm", "batch_norm",
        "group_norm", "loss", "cross_entropy", "mse_loss",
    })
    ops_fp16: Set[str] = field(default_factory=lambda: {
        "linear", "conv1d", "conv2d", "conv3d", "matmul", "bmm",
    })

    # Dynamic loss scaling
    dynamic_scaling: bool = True
    min_scale: float = 1.0
    max_scale: float = 2**24

    def __post_init__(self):
        """Validate and adjust config based on hardware."""
        if self.precision_mode == PrecisionMode.AUTO:
            self.precision_mode = self._detect_best_precision()

    def _detect_best_precision(self) -> PrecisionMode:
        """Detect the best precision mode for current hardware."""
        if not torch.cuda.is_available():
            return PrecisionMode.FP32

        # Check GPU capability
        props = torch.cuda.get_device_properties(0)
        compute_capability = (props.major, props.minor)

        # Ampere and newer (8.0+) support BF16 and TF32
        if compute_capability >= (8, 0):
            # BF16 has better dynamic range than FP16
            if torch.cuda.is_bf16_supported():
                return PrecisionMode.BF16
            return PrecisionMode.FP16

        # Volta and Turing (7.0-7.5) support FP16 well
        if compute_capability >= (7, 0):
            return PrecisionMode.FP16

        # Older GPUs - stick with FP32
        return PrecisionMode.FP32

    def get_dtype(self) -> torch.dtype:
        """Get the torch dtype for current precision mode."""
        if self.precision_mode == PrecisionMode.FP16:
            return torch.float16
        elif self.precision_mode == PrecisionMode.BF16:
            return torch.bfloat16
        else:
            return torch.float32

    def should_use_fp32(self, op_name: str) -> bool:
        """Check if operation should use FP32."""
        return op_name.lower() in self.ops_fp32

    def should_use_fp16(self, op_name: str) -> bool:
        """Check if operation should use reduced precision."""
        return op_name.lower() in self.ops_fp16

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "precision_mode": self.precision_mode.name,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "init_scale": self.init_scale,
            "dtype": str(self.get_dtype()),
        }


# Global AMP configuration
_global_config: Optional[AMPConfig] = None


def get_amp_config() -> AMPConfig:
    """Get or create the global AMP configuration."""
    global _global_config
    if _global_config is None:
        _global_config = AMPConfig()
    return _global_config


def set_amp_config(config: AMPConfig) -> None:
    """Set the global AMP configuration."""
    global _global_config
    _global_config = config


def reset_amp_config() -> None:
    """Reset the global AMP configuration."""
    global _global_config
    _global_config = None
