"""
Unit tests for benchmark models.
"""

import pytest
import torch

from tests.benchmarks.models import (
    MLP,
    LeNet,
    ResNet18,
    create_lenet,
    create_mlp,
    create_resnet18,
    get_model_info,
)


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_creation(self):
        """Test MLP can be created."""
        model = MLP()
        assert model is not None

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(input_size=784, hidden_sizes=[256, 128], output_size=10)
        x = torch.randn(32, 784)
        output = model(x)

        assert output.shape == (32, 10)

    def test_mlp_with_image_input(self):
        """Test MLP handles image input (flattening)."""
        model = MLP(input_size=784)
        x = torch.randn(32, 1, 28, 28)  # Image format
        output = model(x)

        assert output.shape == (32, 10)

    def test_mlp_factory(self):
        """Test MLP factory function."""
        model = create_mlp(device="cpu")
        assert model is not None


class TestLeNet:
    """Tests for LeNet model."""

    def test_lenet_creation(self):
        """Test LeNet can be created."""
        model = LeNet()
        assert model is not None

    def test_lenet_forward_mnist(self):
        """Test LeNet forward pass with MNIST-size input."""
        model = LeNet(in_channels=1, num_classes=10, input_size=28)
        x = torch.randn(32, 1, 28, 28)
        output = model(x)

        assert output.shape == (32, 10)

    def test_lenet_forward_cifar(self):
        """Test LeNet forward pass with CIFAR-size input."""
        model = LeNet(in_channels=3, num_classes=10, input_size=32)
        x = torch.randn(32, 3, 32, 32)
        output = model(x)

        assert output.shape == (32, 10)

    def test_lenet_factory(self):
        """Test LeNet factory function."""
        model = create_lenet(device="cpu")
        assert model is not None


class TestResNet18:
    """Tests for ResNet18 model."""

    def test_resnet_creation(self):
        """Test ResNet18 can be created."""
        model = ResNet18()
        assert model is not None

    def test_resnet_forward_small(self):
        """Test ResNet18 forward pass with small input (CIFAR)."""
        model = ResNet18(in_channels=3, num_classes=10, small_input=True)
        x = torch.randn(8, 3, 32, 32)
        output = model(x)

        assert output.shape == (8, 10)

    def test_resnet_forward_large(self):
        """Test ResNet18 forward pass with large input (ImageNet)."""
        model = ResNet18(in_channels=3, num_classes=1000, small_input=False)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)

        assert output.shape == (4, 1000)

    def test_resnet_factory(self):
        """Test ResNet18 factory function."""
        model = create_resnet18(device="cpu")
        assert model is not None


class TestModelInfo:
    """Tests for model info utility."""

    def test_get_model_info_mlp(self):
        """Test getting model info for MLP."""
        model = MLP()
        info = get_model_info(model)

        assert info["name"] == "MLP"
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0
        assert info["total_params_mb"] > 0

    def test_get_model_info_resnet(self):
        """Test getting model info for ResNet."""
        model = ResNet18()
        info = get_model_info(model)

        assert info["name"] == "ResNet18"
        # ResNet18 has ~11M params
        assert info["total_params"] > 10_000_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestModelsOnCUDA:
    """Tests for models on CUDA."""

    def test_mlp_cuda(self):
        """Test MLP on CUDA."""
        model = create_mlp(device="cuda")
        x = torch.randn(32, 784, device="cuda")
        output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (32, 10)

    def test_lenet_cuda(self):
        """Test LeNet on CUDA."""
        model = create_lenet(device="cuda")
        x = torch.randn(32, 1, 28, 28, device="cuda")
        output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (32, 10)

    def test_resnet_cuda(self):
        """Test ResNet18 on CUDA."""
        model = create_resnet18(device="cuda")
        x = torch.randn(8, 3, 32, 32, device="cuda")
        output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (8, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
