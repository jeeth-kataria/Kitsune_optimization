"""
Unit tests for AMP (Automatic Mixed Precision).
"""

import pytest
import torch
import torch.nn as nn

from kitsune.amp import (
    AMPConfig,
    PrecisionMode,
    get_amp_config,
    set_amp_config,
    KitsuneGradScaler,
    create_grad_scaler,
    autocast_context,
    mixed_precision_forward,
    get_autocast_dtype,
    AMPOptimizer,
    wrap_optimizer_with_amp,
)
from kitsune.amp.config import reset_amp_config
from kitsune.amp.autocast import (
    AutocastModule,
    PrecisionCast,
    enable_tf32,
    get_precision_info,
)
from kitsune.amp.optimizer import AMPTrainer


class TestAMPConfig:
    """Tests for AMPConfig."""

    def test_config_creation(self):
        """Test creating a config."""
        config = AMPConfig()
        assert config.enabled is True
        assert config.grad_scaler_enabled is True

    def test_precision_mode_auto(self):
        """Test auto precision mode detection."""
        config = AMPConfig(precision_mode=PrecisionMode.AUTO)
        # Should auto-detect
        assert config.precision_mode != PrecisionMode.AUTO

    def test_get_dtype_fp16(self):
        """Test getting FP16 dtype."""
        config = AMPConfig(precision_mode=PrecisionMode.FP16)
        assert config.get_dtype() == torch.float16

    def test_get_dtype_bf16(self):
        """Test getting BF16 dtype."""
        config = AMPConfig(precision_mode=PrecisionMode.BF16)
        assert config.get_dtype() == torch.bfloat16

    def test_get_dtype_fp32(self):
        """Test getting FP32 dtype."""
        config = AMPConfig(precision_mode=PrecisionMode.FP32)
        assert config.get_dtype() == torch.float32

    def test_ops_fp32(self):
        """Test FP32 operation detection."""
        config = AMPConfig()
        assert config.should_use_fp32("softmax")
        assert config.should_use_fp32("layer_norm")
        assert not config.should_use_fp32("linear")

    def test_ops_fp16(self):
        """Test FP16 operation detection."""
        config = AMPConfig()
        assert config.should_use_fp16("linear")
        assert config.should_use_fp16("matmul")
        assert not config.should_use_fp16("softmax")

    def test_to_dict(self):
        """Test config serialization."""
        config = AMPConfig()
        d = config.to_dict()
        assert "enabled" in d
        assert "precision_mode" in d
        assert "dtype" in d

    def test_global_config(self):
        """Test global config management."""
        reset_amp_config()
        config = get_amp_config()
        assert config is not None

        new_config = AMPConfig(enabled=False)
        set_amp_config(new_config)

        assert get_amp_config().enabled is False
        reset_amp_config()


class TestKitsuneGradScaler:
    """Tests for KitsuneGradScaler."""

    def test_scaler_creation(self):
        """Test creating a scaler."""
        scaler = KitsuneGradScaler()
        assert scaler is not None
        assert scaler.enabled is True

    def test_scaler_disabled(self):
        """Test disabled scaler."""
        scaler = KitsuneGradScaler(enabled=False)
        assert scaler.enabled is False

    def test_scale_property(self):
        """Test scale property."""
        scaler = KitsuneGradScaler()
        scale = scaler.scale
        assert scale > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scale_loss(self):
        """Test loss scaling."""
        scaler = KitsuneGradScaler()
        loss = torch.tensor(1.0, device="cuda", requires_grad=True)
        scaled = scaler.scale_loss(loss)
        assert scaled.item() > loss.item()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_step_and_update(self):
        """Test step and update."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = KitsuneGradScaler()

        x = torch.randn(32, 10, device="cuda")
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = y.sum()

        scaler.scale_loss(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def test_state_dict(self):
        """Test state dict."""
        scaler = KitsuneGradScaler()
        state = scaler.state_dict()
        assert "scaler" in state
        assert "stats" in state

    def test_stats(self):
        """Test statistics tracking."""
        scaler = KitsuneGradScaler()
        stats = scaler.stats
        assert hasattr(stats, "scale")
        assert hasattr(stats, "overflow_count")

    def test_create_grad_scaler(self):
        """Test factory function."""
        scaler = create_grad_scaler()
        assert scaler is not None


class TestAutocast:
    """Tests for autocast utilities."""

    def test_get_autocast_dtype(self):
        """Test getting autocast dtype."""
        dtype = get_autocast_dtype()
        assert dtype in [torch.float16, torch.bfloat16, torch.float32]

    def test_autocast_context_cpu(self):
        """Test autocast context on CPU."""
        x = torch.randn(10, 10)
        with autocast_context():
            y = x @ x.T
        assert y is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autocast_context_cuda(self):
        """Test autocast context on CUDA."""
        x = torch.randn(10, 10, device="cuda")
        with autocast_context():
            y = x @ x.T
        # Should be in reduced precision
        assert y.dtype in [torch.float16, torch.bfloat16, torch.float32]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_precision_forward(self):
        """Test mixed precision forward pass."""
        model = nn.Linear(10, 5).cuda()
        x = torch.randn(32, 10, device="cuda")

        output = mixed_precision_forward(model, x)
        assert output.shape == (32, 5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autocast_module(self):
        """Test AutocastModule wrapper."""
        model = nn.Linear(10, 5).cuda()
        wrapped = AutocastModule(model)

        x = torch.randn(32, 10, device="cuda")
        output = wrapped(x)
        assert output.shape == (32, 5)

    def test_precision_cast(self):
        """Test PrecisionCast utility."""
        config = AMPConfig(precision_mode=PrecisionMode.FP16)
        caster = PrecisionCast(config)

        x = torch.randn(10)
        reduced = caster.cast_to_reduced(x)
        assert reduced.dtype == torch.float16

        full = caster.cast_to_full(reduced)
        assert full.dtype == torch.float32

    def test_precision_cast_overflow_check(self):
        """Test overflow checking in precision cast."""
        config = AMPConfig(precision_mode=PrecisionMode.FP16)
        caster = PrecisionCast(config)

        # Value too large for FP16
        x = torch.tensor([100000.0])
        result = caster.cast_to_reduced(x, check_overflow=True)
        # Should stay in FP32
        assert result.dtype == torch.float32

    def test_get_precision_info(self):
        """Test precision info."""
        info = get_precision_info()
        assert "cuda_available" in info
        assert "fp16_supported" in info
        assert "bf16_supported" in info


class TestAMPOptimizer:
    """Tests for AMPOptimizer."""

    def test_optimizer_creation(self):
        """Test creating AMP optimizer."""
        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = AMPOptimizer(base_opt)

        assert amp_opt is not None
        assert amp_opt.optimizer is base_opt

    def test_zero_grad(self):
        """Test zero_grad."""
        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = AMPOptimizer(base_opt)

        x = torch.randn(32, 10)
        y = model(x)
        y.sum().backward()

        # Gradients should exist
        assert model.weight.grad is not None

        amp_opt.zero_grad()
        # Gradients should be None
        assert model.weight.grad is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_forward_backward(self):
        """Test forward_backward convenience method."""
        model = nn.Linear(10, 5).cuda()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = AMPOptimizer(base_opt)

        x = torch.randn(32, 10, device="cuda")
        target = torch.randn(32, 5, device="cuda")
        criterion = nn.MSELoss()

        loss = amp_opt.forward_backward(model, x, target, criterion)
        assert loss is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_step(self):
        """Test optimizer step."""
        model = nn.Linear(10, 5).cuda()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = AMPOptimizer(base_opt)

        old_weight = model.weight.clone()

        x = torch.randn(32, 10, device="cuda")
        target = torch.randn(32, 5, device="cuda")
        criterion = nn.MSELoss()

        amp_opt.forward_backward(model, x, target, criterion)
        amp_opt.step()

        # Weights should have changed
        assert not torch.allclose(model.weight, old_weight)

    def test_state_dict(self):
        """Test state dict."""
        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = AMPOptimizer(base_opt)

        state = amp_opt.state_dict()
        assert "optimizer" in state
        assert "scaler" in state

    def test_wrap_optimizer_with_amp(self):
        """Test factory function."""
        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        amp_opt = wrap_optimizer_with_amp(base_opt)

        assert isinstance(amp_opt, AMPOptimizer)


class TestAMPTrainer:
    """Tests for AMPTrainer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_trainer_creation(self):
        """Test creating trainer."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = AMPTrainer(model, optimizer, criterion)
        assert trainer.model is model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_step(self):
        """Test single training step."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = AMPTrainer(model, optimizer, criterion)

        x = torch.randn(32, 10, device="cuda")
        target = torch.randn(32, 5, device="cuda")

        loss = trainer.train_step(x, target)
        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
