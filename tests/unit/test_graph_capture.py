"""
Unit tests for graph capture module.
"""

import pytest
import torch
import torch.nn as nn

from kitsune.pytorch.graph_capture import (
    GraphCapture,
    FXGraphCapture,
    HookGraphCapture,
    GraphCaptureError,
    capture_graph,
)
from kitsune.core.graph import ComputationGraph


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ConvModel(nn.Module):
    """CNN model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DynamicModel(nn.Module):
    """Model with dynamic control flow (won't trace with FX)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.linear(x) * 2
        else:
            return self.linear(x) * 3


class TestFXGraphCapture:
    """Tests for FX-based graph capture."""

    def test_capture_simple_model(self):
        """Test capturing a simple model."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = FXGraphCapture()
        graph = capturer.capture(model, sample_input)

        assert isinstance(graph, ComputationGraph)
        assert graph.num_tasks > 0

    def test_capture_conv_model(self):
        """Test capturing a CNN model."""
        model = ConvModel()
        sample_input = torch.randn(32, 1, 28, 28)

        capturer = FXGraphCapture()
        graph = capturer.capture(model, sample_input)

        assert graph.num_tasks > 0

    def test_traced_module_available(self):
        """Test that traced module is available after capture."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = FXGraphCapture()
        capturer.capture(model, sample_input)

        assert capturer.traced_module is not None

    def test_dynamic_model_fails(self):
        """Test that dynamic models fail FX tracing."""
        model = DynamicModel()
        sample_input = torch.randn(32, 10)

        capturer = FXGraphCapture()

        with pytest.raises(GraphCaptureError):
            capturer.capture(model, sample_input)


class TestHookGraphCapture:
    """Tests for hook-based graph capture."""

    def test_capture_simple_model(self):
        """Test capturing with hooks."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = HookGraphCapture()
        graph = capturer.capture(model, sample_input)

        assert isinstance(graph, ComputationGraph)
        assert graph.num_tasks > 0

    def test_capture_dynamic_model(self):
        """Test that hooks work with dynamic models."""
        model = DynamicModel()
        sample_input = torch.randn(32, 10)

        capturer = HookGraphCapture()
        graph = capturer.capture(model, sample_input)

        assert graph.num_tasks > 0


class TestGraphCapture:
    """Tests for high-level GraphCapture interface."""

    def test_auto_strategy_prefers_fx(self):
        """Test that auto strategy uses FX for static models."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = GraphCapture(strategy="auto")
        graph = capturer.capture(model, sample_input)

        assert capturer.used_strategy == "fx"
        assert graph.num_tasks > 0

    def test_auto_strategy_falls_back_to_hooks(self):
        """Test that auto strategy falls back to hooks for dynamic models."""
        model = DynamicModel()
        sample_input = torch.randn(32, 10)

        capturer = GraphCapture(strategy="auto")
        graph = capturer.capture(model, sample_input)

        assert capturer.used_strategy == "hooks"
        assert graph.num_tasks > 0

    def test_force_fx_strategy(self):
        """Test forcing FX strategy."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = GraphCapture(strategy="fx")
        graph = capturer.capture(model, sample_input)

        assert capturer.used_strategy == "fx"

    def test_force_hooks_strategy(self):
        """Test forcing hooks strategy."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        capturer = GraphCapture(strategy="hooks")
        graph = capturer.capture(model, sample_input)

        assert capturer.used_strategy == "hooks"


class TestCaptureGraphFunction:
    """Tests for capture_graph convenience function."""

    def test_capture_graph_basic(self):
        """Test basic capture_graph usage."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        graph = capture_graph(model, sample_input)

        assert isinstance(graph, ComputationGraph)
        assert graph.num_tasks > 0

    def test_capture_graph_with_strategy(self):
        """Test capture_graph with explicit strategy."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        graph = capture_graph(model, sample_input, strategy="hooks")

        assert graph.num_tasks > 0


class TestGraphStructure:
    """Tests for captured graph structure."""

    def test_graph_has_correct_ops(self):
        """Test that captured graph has expected operation types."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        graph = capture_graph(model, sample_input, strategy="hooks")

        op_types = [task.op_type for task in graph.tasks]

        # Should have linear and relu operations
        assert any("linear" in op.lower() for op in op_types)
        assert any("relu" in op.lower() for op in op_types)

    def test_graph_has_dependencies(self):
        """Test that captured graph has proper dependencies."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        graph = capture_graph(model, sample_input, strategy="hooks")

        # At least some tasks should have dependencies
        tasks_with_deps = [t for t in graph.tasks if t.inputs]
        assert len(tasks_with_deps) > 0

    def test_graph_is_acyclic(self):
        """Test that captured graph is acyclic."""
        model = SimpleModel()
        sample_input = torch.randn(32, 784)

        graph = capture_graph(model, sample_input)

        # Should be able to get topological order without error
        order = graph.topological_order()
        assert len(order) == graph.num_tasks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGraphCaptureOnCUDA:
    """Tests for graph capture on CUDA."""

    def test_capture_cuda_model(self):
        """Test capturing a model on CUDA."""
        model = SimpleModel().cuda()
        sample_input = torch.randn(32, 784, device="cuda")

        graph = capture_graph(model, sample_input)

        assert graph.num_tasks > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
