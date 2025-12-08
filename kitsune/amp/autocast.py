"""
Autocast Context - Mixed precision context management.

Provides utilities for autocast context management and
operation-level precision control.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Callable, Any, Generator
import torch
import torch.nn as nn

from .config import AMPConfig, PrecisionMode, get_amp_config
from ..profiler import get_logger

logger = get_logger(__name__)


def get_autocast_dtype(config: Optional[AMPConfig] = None) -> torch.dtype:
    """
    Get the appropriate dtype for autocast.

    Args:
        config: AMP configuration

    Returns:
        Torch dtype for autocast
    """
    config = config or get_amp_config()
    return config.get_dtype()


@contextmanager
def autocast_context(
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
    config: Optional[AMPConfig] = None,
    cache_enabled: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager for mixed precision.

    Enhanced wrapper around torch.cuda.amp.autocast with
    Kitsune configuration support.

    Args:
        enabled: Whether autocast is enabled
        dtype: Override dtype
        config: AMP configuration
        cache_enabled: Whether to cache autocast results

    Yields:
        None

    Example:
        with autocast_context():
            output = model(input)
            loss = criterion(output, target)
    """
    config = config or get_amp_config()

    # Determine if we should actually enable
    actual_enabled = enabled and config.enabled

    # Get dtype
    if dtype is None:
        dtype = config.get_dtype()

    # Handle CPU vs CUDA
    if not torch.cuda.is_available():
        # CPU autocast (PyTorch 1.10+)
        try:
            with torch.cpu.amp.autocast(enabled=actual_enabled, dtype=dtype):
                yield
        except (AttributeError, RuntimeError):
            # Fallback for older PyTorch or unsupported dtypes
            yield
    else:
        # CUDA autocast
        with torch.cuda.amp.autocast(
            enabled=actual_enabled,
            dtype=dtype,
            cache_enabled=cache_enabled,
        ):
            yield


def mixed_precision_forward(
    module: nn.Module,
    input_tensor: torch.Tensor,
    config: Optional[AMPConfig] = None,
) -> torch.Tensor:
    """
    Perform forward pass with mixed precision.

    Args:
        module: PyTorch module
        input_tensor: Input tensor
        config: AMP configuration

    Returns:
        Output tensor
    """
    config = config or get_amp_config()

    with autocast_context(config=config):
        return module(input_tensor)


class AutocastModule(nn.Module):
    """
    Wrapper module that automatically uses autocast for forward pass.

    Useful for wrapping models that should always use mixed precision.
    """

    def __init__(
        self,
        module: nn.Module,
        config: Optional[AMPConfig] = None,
    ):
        """
        Initialize autocast wrapper.

        Args:
            module: Module to wrap
            config: AMP configuration
        """
        super().__init__()
        self._module = module
        self._config = config or get_amp_config()

    def forward(self, *args, **kwargs) -> Any:
        """Forward with autocast."""
        with autocast_context(config=self._config):
            return self._module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)


class PrecisionCast:
    """
    Utility for casting tensors between precisions.

    Handles safe casting with overflow checking and
    automatic fallback to FP32 for problematic values.
    """

    def __init__(self, config: Optional[AMPConfig] = None):
        """Initialize precision caster."""
        self._config = config or get_amp_config()
        self._target_dtype = self._config.get_dtype()

    def cast_to_reduced(
        self,
        tensor: torch.Tensor,
        check_overflow: bool = True,
    ) -> torch.Tensor:
        """
        Cast tensor to reduced precision.

        Args:
            tensor: Input tensor
            check_overflow: Whether to check for overflow

        Returns:
            Cast tensor
        """
        if tensor.dtype == self._target_dtype:
            return tensor

        if check_overflow and self._target_dtype == torch.float16:
            # Check for values that would overflow in FP16
            max_val = tensor.abs().max()
            if max_val > 65504:  # FP16 max
                logger.warning(
                    f"Tensor has values > FP16 max ({max_val:.0f}), "
                    "keeping in FP32"
                )
                return tensor

        return tensor.to(self._target_dtype)

    def cast_to_full(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Cast tensor to full precision (FP32).

        Args:
            tensor: Input tensor

        Returns:
            FP32 tensor
        """
        return tensor.float()

    @staticmethod
    def safe_cast(
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Safely cast tensor to target dtype.

        Args:
            tensor: Input tensor
            dtype: Target dtype

        Returns:
            Cast tensor
        """
        if tensor.dtype == dtype:
            return tensor
        return tensor.to(dtype)


def enable_tf32() -> None:
    """
    Enable TensorFloat32 for matrix multiplications.

    TF32 uses 19-bit mantissa (vs 23-bit for FP32) but maintains
    FP32 dynamic range. Available on Ampere GPUs and newer.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cuDNN")


def disable_tf32() -> None:
    """Disable TensorFloat32."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info("TF32 disabled")


def get_precision_info() -> dict:
    """
    Get information about available precision modes.

    Returns:
        Dictionary with precision capabilities
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "fp16_supported": False,
        "bf16_supported": False,
        "tf32_supported": False,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_capability = (props.major, props.minor)

        info["gpu_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"

        # FP16 support (Volta and newer)
        info["fp16_supported"] = compute_capability >= (7, 0)

        # BF16 support (Ampere and newer)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()

        # TF32 support (Ampere and newer)
        info["tf32_supported"] = compute_capability >= (8, 0)

    return info
