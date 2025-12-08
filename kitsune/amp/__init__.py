"""
Kitsune AMP - Automatic Mixed Precision integration.

Provides:
- AMP configuration and policy management
- GradScaler integration with memory pools
- Operation-level precision control
- Dynamic loss scaling
"""

from .config import (
    AMPConfig,
    PrecisionMode,
    get_amp_config,
    set_amp_config,
)
from .scaler import (
    KitsuneGradScaler,
    create_grad_scaler,
)
from .autocast import (
    autocast_context,
    mixed_precision_forward,
    get_autocast_dtype,
)
from .optimizer import (
    AMPOptimizer,
    wrap_optimizer_with_amp,
)

__all__ = [
    # Config
    "AMPConfig",
    "PrecisionMode",
    "get_amp_config",
    "set_amp_config",
    # Scaler
    "KitsuneGradScaler",
    "create_grad_scaler",
    # Autocast
    "autocast_context",
    "mixed_precision_forward",
    "get_autocast_dtype",
    # Optimizer
    "AMPOptimizer",
    "wrap_optimizer_with_amp",
]
