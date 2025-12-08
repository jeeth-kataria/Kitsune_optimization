"""
Gradient Scaler - Enhanced gradient scaling for AMP.

Provides gradient scaling with better memory management
and integration with Kitsune's memory pools.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from .config import AMPConfig, get_amp_config, PrecisionMode
from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class ScalerStats:
    """Statistics for gradient scaler."""
    scale: float = 65536.0
    growth_tracker: int = 0
    overflow_count: int = 0
    update_count: int = 0
    total_skipped_steps: int = 0

    def overflow_rate(self) -> float:
        """Get overflow rate."""
        if self.update_count == 0:
            return 0.0
        return self.overflow_count / self.update_count


class KitsuneGradScaler:
    """
    Enhanced gradient scaler with Kitsune integration.

    Features:
    - Dynamic loss scaling with bounds
    - Integration with memory pools for efficient unscaling
    - Statistics tracking for debugging
    - Automatic backoff on persistent overflow

    Usage:
        scaler = KitsuneGradScaler()

        for data, target in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    """

    def __init__(
        self,
        config: Optional[AMPConfig] = None,
        enabled: bool = True,
    ):
        """
        Initialize gradient scaler.

        Args:
            config: AMP configuration
            enabled: Whether scaling is enabled
        """
        self._config = config or get_amp_config()
        self._enabled = enabled and self._config.grad_scaler_enabled

        # Create underlying PyTorch scaler
        if self._enabled:
            self._scaler = GradScaler(
                init_scale=self._config.init_scale,
                growth_factor=self._config.growth_factor,
                backoff_factor=self._config.backoff_factor,
                growth_interval=self._config.growth_interval,
                enabled=True,
            )
        else:
            self._scaler = GradScaler(enabled=False)

        # Statistics
        self._stats = ScalerStats(scale=self._config.init_scale)

        logger.debug(f"KitsuneGradScaler initialized (enabled={self._enabled})")

    @property
    def enabled(self) -> bool:
        """Check if scaler is enabled."""
        return self._enabled

    @property
    def scale(self) -> float:
        """Get current scale value."""
        return self._scaler.get_scale()

    @property
    def stats(self) -> ScalerStats:
        """Get scaler statistics."""
        self._stats.scale = self.scale
        return self._stats

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss for backward pass.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss
        """
        return self._scaler.scale(loss)

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        """Alias for scale_loss."""
        return self.scale_loss(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients in-place.

        Args:
            optimizer: Optimizer with gradients to unscale
        """
        self._scaler.unscale_(optimizer)

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs,
    ) -> Optional[float]:
        """
        Step the optimizer with gradient unscaling.

        Args:
            optimizer: Optimizer to step

        Returns:
            Return value of optimizer.step() or None if skipped
        """
        result = self._scaler.step(optimizer, *args, **kwargs)

        # Track if step was skipped (overflow)
        if result is None:
            self._stats.total_skipped_steps += 1
            self._stats.overflow_count += 1

        return result

    def update(self) -> None:
        """Update the scaler after optimizer step."""
        old_scale = self.scale
        self._scaler.update()
        new_scale = self.scale

        self._stats.update_count += 1
        self._stats.growth_tracker = self._scaler._growth_tracker

        # Log scale changes
        if new_scale != old_scale:
            if new_scale < old_scale:
                logger.debug(f"GradScaler reduced scale: {old_scale:.0f} -> {new_scale:.0f}")
            else:
                logger.debug(f"GradScaler increased scale: {old_scale:.0f} -> {new_scale:.0f}")

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "scaler": self._scaler.state_dict(),
            "stats": {
                "scale": self._stats.scale,
                "overflow_count": self._stats.overflow_count,
                "update_count": self._stats.update_count,
                "total_skipped_steps": self._stats.total_skipped_steps,
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self._scaler.load_state_dict(state_dict["scaler"])
        if "stats" in state_dict:
            stats = state_dict["stats"]
            self._stats.scale = stats.get("scale", self._stats.scale)
            self._stats.overflow_count = stats.get("overflow_count", 0)
            self._stats.update_count = stats.get("update_count", 0)
            self._stats.total_skipped_steps = stats.get("total_skipped_steps", 0)

    def get_scale(self) -> float:
        """Get current scale (PyTorch GradScaler API compatibility)."""
        return self._scaler.get_scale()

    def is_enabled(self) -> bool:
        """Check if scaler is enabled (PyTorch GradScaler API compatibility)."""
        return self._enabled

    def summary(self) -> str:
        """Get scaler summary."""
        lines = [
            "KitsuneGradScaler Summary",
            "=" * 40,
            f"Enabled: {self._enabled}",
            f"Current scale: {self.scale:.0f}",
            f"Updates: {self._stats.update_count}",
            f"Overflows: {self._stats.overflow_count}",
            f"Overflow rate: {self._stats.overflow_rate():.2%}",
            f"Skipped steps: {self._stats.total_skipped_steps}",
        ]
        return "\n".join(lines)


def create_grad_scaler(
    config: Optional[AMPConfig] = None,
    enabled: Optional[bool] = None,
) -> KitsuneGradScaler:
    """
    Create a gradient scaler.

    Args:
        config: AMP configuration
        enabled: Override enabled setting

    Returns:
        Configured gradient scaler
    """
    config = config or get_amp_config()

    if enabled is None:
        enabled = config.enabled and config.grad_scaler_enabled

    return KitsuneGradScaler(config=config, enabled=enabled)
