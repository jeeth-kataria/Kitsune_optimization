"""
AMP Optimizer - Optimizer wrapper with mixed precision support.

Wraps PyTorch optimizers with automatic gradient scaling,
mixed precision casting, and Kitsune integration.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Callable, Iterator
import torch
import torch.nn as nn
from torch.optim import Optimizer

from .config import AMPConfig, get_amp_config
from .scaler import KitsuneGradScaler, create_grad_scaler
from .autocast import autocast_context
from ..profiler import get_logger

logger = get_logger(__name__)


class AMPOptimizer:
    """
    Optimizer wrapper with automatic mixed precision support.

    Combines a PyTorch optimizer with gradient scaling for
    seamless mixed precision training.

    Usage:
        model = MyModel().cuda()
        base_optimizer = torch.optim.Adam(model.parameters())
        optimizer = AMPOptimizer(base_optimizer)

        for data, target in dataloader:
            loss = optimizer.forward_backward(
                model, data, target, criterion
            )
            optimizer.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: Optional[AMPConfig] = None,
        scaler: Optional[KitsuneGradScaler] = None,
    ):
        """
        Initialize AMP optimizer.

        Args:
            optimizer: Base PyTorch optimizer
            config: AMP configuration
            scaler: Gradient scaler (created if not provided)
        """
        self._optimizer = optimizer
        self._config = config or get_amp_config()
        self._scaler = scaler or create_grad_scaler(config=self._config)

        # Training state
        self._step_count = 0
        self._overflow_count = 0

        logger.debug(f"AMPOptimizer wrapping {type(optimizer).__name__}")

    @property
    def optimizer(self) -> Optimizer:
        """Get the underlying optimizer."""
        return self._optimizer

    @property
    def scaler(self) -> KitsuneGradScaler:
        """Get the gradient scaler."""
        return self._scaler

    @property
    def config(self) -> AMPConfig:
        """Get the AMP configuration."""
        return self._config

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero gradients.

        Args:
            set_to_none: Set gradients to None instead of zero
        """
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass with scaled loss.

        Args:
            loss: Loss tensor
        """
        scaled_loss = self._scaler.scale_loss(loss)
        scaled_loss.backward()

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Optimizer step with gradient unscaling.

        Args:
            closure: Optional closure for optimizers that require it

        Returns:
            Loss value if closure provided, else None
        """
        result = self._scaler.step(self._optimizer, closure)

        if result is None:
            self._overflow_count += 1

        self._scaler.update()
        self._step_count += 1

        return result

    def forward_backward(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        target: torch.Tensor,
        criterion: Callable,
    ) -> torch.Tensor:
        """
        Perform forward and backward pass with autocast.

        Convenience method that handles autocast and scaling.

        Args:
            model: Model to train
            input_data: Input data
            target: Target labels
            criterion: Loss function

        Returns:
            Loss value (unscaled)
        """
        self.zero_grad()

        with autocast_context(config=self._config):
            output = model(input_data)
            loss = criterion(output, target)

        self.backward(loss)

        return loss.detach()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "optimizer": self._optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "step_count": self._step_count,
            "overflow_count": self._overflow_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._scaler.load_state_dict(state_dict["scaler"])
        self._step_count = state_dict.get("step_count", 0)
        self._overflow_count = state_dict.get("overflow_count", 0)

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Get optimizer param groups."""
        return self._optimizer.param_groups

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add param group to optimizer."""
        self._optimizer.add_param_group(param_group)

    def summary(self) -> str:
        """Get optimizer summary."""
        lines = [
            "AMPOptimizer Summary",
            "=" * 40,
            f"Base optimizer: {type(self._optimizer).__name__}",
            f"AMP enabled: {self._config.enabled}",
            f"Precision mode: {self._config.precision_mode.name}",
            f"Steps: {self._step_count}",
            f"Overflows: {self._overflow_count}",
            f"Current scale: {self._scaler.scale:.0f}",
        ]
        return "\n".join(lines)


class AMPTrainer:
    """
    Complete training helper with AMP support.

    Provides a high-level API for training with mixed precision,
    including forward, backward, step, and metrics tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        config: Optional[AMPConfig] = None,
    ):
        """
        Initialize AMP trainer.

        Args:
            model: Model to train
            optimizer: Base optimizer
            criterion: Loss function
            config: AMP configuration
        """
        self._model = model
        self._criterion = criterion
        self._config = config or get_amp_config()
        self._amp_optimizer = AMPOptimizer(optimizer, config=self._config)

        # Metrics
        self._train_losses: List[float] = []
        self._epoch_count = 0

    @property
    def model(self) -> nn.Module:
        """Get the model."""
        return self._model

    @property
    def optimizer(self) -> AMPOptimizer:
        """Get the AMP optimizer."""
        return self._amp_optimizer

    def train_step(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """
        Perform a single training step.

        Args:
            input_data: Input data
            target: Target labels

        Returns:
            Loss value
        """
        self._model.train()

        loss = self._amp_optimizer.forward_backward(
            self._model, input_data, target, self._criterion
        )
        self._amp_optimizer.step()

        loss_val = loss.item()
        self._train_losses.append(loss_val)

        return loss_val

    def train_epoch(
        self,
        dataloader,
        verbose: bool = True,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            verbose: Print progress

        Returns:
            Average epoch loss
        """
        self._model.train()
        epoch_losses = []

        for batch_idx, (data, target) in enumerate(dataloader):
            loss = self.train_step(data, target)
            epoch_losses.append(loss)

            if verbose and batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: loss={loss:.4f}")

        self._epoch_count += 1
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        if verbose:
            logger.info(f"Epoch {self._epoch_count}: avg_loss={avg_loss:.4f}")

        return avg_loss

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "model": self._model.state_dict(),
            "optimizer": self._amp_optimizer.state_dict(),
            "epoch_count": self._epoch_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self._model.load_state_dict(state_dict["model"])
        self._amp_optimizer.load_state_dict(state_dict["optimizer"])
        self._epoch_count = state_dict.get("epoch_count", 0)


def wrap_optimizer_with_amp(
    optimizer: Optimizer,
    config: Optional[AMPConfig] = None,
) -> AMPOptimizer:
    """
    Wrap an optimizer with AMP support.

    Args:
        optimizer: PyTorch optimizer
        config: AMP configuration

    Returns:
        Wrapped optimizer
    """
    return AMPOptimizer(optimizer, config=config)
