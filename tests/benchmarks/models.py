"""
Test models for benchmarking

Provides standard model architectures for performance benchmarking:
- MLP: Simple feedforward network for basic tests
- LeNet: Classic CNN for image classification
- ResNet-18: Modern CNN for realistic benchmarks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for baseline benchmarks.

    Architecture: 784 -> 256 -> 128 -> 10
    Use case: MNIST digit classification
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = None,
        output_size: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class LeNet(nn.Module):
    """
    LeNet-5 style CNN for image classification.

    Architecture:
        Conv(1->6, 5x5) -> ReLU -> MaxPool
        Conv(6->16, 5x5) -> ReLU -> MaxPool
        FC(400->120) -> ReLU
        FC(120->84) -> ReLU
        FC(84->10)

    Use case: MNIST/CIFAR-10 classification
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_size = input_size

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Calculate flattened size
        # After conv1+pool: (input_size/2)
        # After conv2+pool: ((input_size/2 - 4) / 2)
        conv_out_size = (input_size // 2 - 4) // 2
        self.flat_size = 16 * conv_out_size * conv_out_size

        # Classifier
        self.fc1 = nn.Linear(self.flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet-18 for realistic performance benchmarks.

    Architecture: Standard ResNet-18 with [2, 2, 2, 2] blocks
    Use case: CIFAR-10/ImageNet classification
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        small_input: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            small_input: If True, use smaller kernel for CIFAR-10 (32x32)
                        If False, use standard ImageNet config (224x224)
        """
        super().__init__()

        self.in_channels_current = 64

        # Initial convolution - smaller for CIFAR-10
        if small_input:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels_current != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels_current, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels_current, out_channels, stride, downsample)]
        self.in_channels_current = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Factory functions for easy model creation


def create_mlp(
    input_size: int = 784,
    hidden_sizes: list[int] = None,
    output_size: int = 10,
    device: str = "cuda",
) -> MLP:
    """Create an MLP model."""
    model = MLP(input_size, hidden_sizes, output_size)
    return model.to(device) if device else model


def create_lenet(
    in_channels: int = 1,
    num_classes: int = 10,
    input_size: int = 28,
    device: str = "cuda",
) -> LeNet:
    """Create a LeNet model."""
    model = LeNet(in_channels, num_classes, input_size)
    return model.to(device) if device else model


def create_resnet18(
    in_channels: int = 3,
    num_classes: int = 10,
    small_input: bool = True,
    device: str = "cuda",
) -> ResNet18:
    """Create a ResNet-18 model."""
    model = ResNet18(in_channels, num_classes, small_input)
    return model.to(device) if device else model


def get_model_info(model: nn.Module) -> dict:
    """Get information about a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "name": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_mb": total_params * 4 / (1024**2),  # Assuming float32
    }
