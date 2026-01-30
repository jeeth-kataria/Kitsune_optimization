"""
Unit tests for kernel fusion.
"""

import pytest
import torch
import torch.nn as nn

from kitsune.core import ComputationGraph
from kitsune.fusion import (
    BUILTIN_PATTERNS,
    FusedOperations,
    FusionCandidate,
    FusionDetector,
    FusionEngine,
    FusionPattern,
    FusionType,
    PatternMatcher,
    is_fusable,
)
from kitsune.pytorch import capture_graph


class TestFusionPattern:
    """Tests for FusionPattern."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = FusionPattern(
            name="test",
            op_sequence=["linear", "relu"],
            fusion_type=FusionType.MATMUL_ACTIVATION,
        )
        assert pattern.name == "test"
        assert pattern.fusion_type == FusionType.MATMUL_ACTIVATION

    def test_pattern_matching(self):
        """Test pattern matching."""
        pattern = FusionPattern(
            name="linear_relu",
            op_sequence=["linear", "relu"],
            fusion_type=FusionType.MATMUL_ACTIVATION,
        )

        assert pattern.matches(["linear", "relu"])
        assert pattern.matches(["linear", "relu", "linear"])
        assert not pattern.matches(["relu", "linear"])
        assert not pattern.matches(["linear"])

    def test_builtin_patterns(self):
        """Test built-in patterns exist."""
        assert len(BUILTIN_PATTERNS) > 0

        # Check specific patterns exist
        pattern_names = [p.name for p in BUILTIN_PATTERNS]
        assert "linear_relu" in pattern_names
        assert "linear_gelu" in pattern_names


class TestPatternMatcher:
    """Tests for PatternMatcher."""

    def test_matcher_creation(self):
        """Test matcher creation."""
        matcher = PatternMatcher()
        assert matcher is not None

    def test_find_matches(self):
        """Test finding matches in op sequence."""
        matcher = PatternMatcher()

        ops = ["linear", "relu", "linear", "gelu"]
        matches = matcher.find_matches(ops)

        # Should find at least linear_relu and linear_gelu
        assert len(matches) >= 2

    def test_find_best_match(self):
        """Test finding best match."""
        matcher = PatternMatcher()

        ops = ["linear", "relu"]
        match = matcher.find_best_match(ops)

        assert match is not None
        pattern, start, end = match
        assert pattern.name == "linear_relu"

    def test_fusable_groups(self):
        """Test getting fusable groups."""
        matcher = PatternMatcher()

        ops = ["linear", "relu", "linear", "gelu"]
        task_ids = [0, 1, 2, 3]

        groups = matcher.get_fusable_groups(ops, task_ids)
        assert len(groups) >= 1


class TestFusionDetector:
    """Tests for FusionDetector."""

    def test_detector_creation(self):
        """Test detector creation."""
        detector = FusionDetector()
        assert detector is not None

    def test_detect_in_graph(self):
        """Test detecting fusion in a graph."""
        detector = FusionDetector()
        graph = ComputationGraph()

        # Create a simple linear chain
        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="linear1", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="relu1", op_type="relu", inputs=[t2.id])
        t4 = graph.add_task(name="linear2", op_type="linear", inputs=[t3.id])
        t5 = graph.add_task(name="relu2", op_type="relu", inputs=[t4.id])

        candidates = detector.detect(graph)

        # Should find fusion opportunities
        assert len(candidates) >= 1

    def test_get_fusion_plan(self):
        """Test getting a fusion plan."""
        detector = FusionDetector()
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="linear", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="relu", op_type="relu", inputs=[t2.id])

        plan = detector.get_fusion_plan(graph)

        assert "candidates" in plan
        assert "estimated_speedup" in plan
        assert "fusion_coverage" in plan


class TestFusionEngine:
    """Tests for FusionEngine."""

    def test_engine_creation(self):
        """Test engine creation."""
        engine = FusionEngine()
        assert engine is not None
        assert engine.backend in ["torch.compile", "triton", "torch.jit"]

    def test_compile_function(self):
        """Test compiling a function."""
        engine = FusionEngine()

        def simple_fn(x):
            return x * 2 + 1

        compiled = engine.compile_function(simple_fn, "simple")

        # Test it works
        x = torch.randn(10)
        result = compiled(x)
        expected = simple_fn(x)

        assert torch.allclose(result, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimize_model(self):
        """Test optimizing a model."""
        engine = FusionEngine()

        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).cuda()

        sample_input = torch.randn(64, 784, device="cuda")

        optimized = engine.optimize_model(model, sample_input)

        # Test it works
        output = optimized(sample_input)
        assert output.shape == (64, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_fusion(self):
        """Test benchmarking fusion."""
        engine = FusionEngine()

        def original(x):
            x = x * 2
            x = x + 1
            return torch.relu(x)

        compiled = engine.compile_function(original, "original")

        x = torch.randn(1000, 1000, device="cuda")

        results = engine.benchmark_fusion(
            original,
            compiled,
            x,
            num_iterations=50,
            warmup=10,
        )

        assert "original_ms" in results
        assert "fused_ms" in results
        assert "speedup" in results


class TestFusedOperations:
    """Tests for pre-fused operations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_linear_relu(self):
        """Test fused linear + ReLU."""
        weight = torch.randn(256, 784, device="cuda")
        bias = torch.randn(256, device="cuda")

        fused = FusedOperations.linear_relu(weight, bias)

        x = torch.randn(64, 784, device="cuda")
        result = fused(x)

        # Compare with unfused
        expected = torch.relu(torch.nn.functional.linear(x, weight, bias))

        assert torch.allclose(result, expected, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_linear_gelu(self):
        """Test fused linear + GELU."""
        weight = torch.randn(256, 784, device="cuda")
        bias = torch.randn(256, device="cuda")

        fused = FusedOperations.linear_gelu(weight, bias)

        x = torch.randn(64, 784, device="cuda")
        result = fused(x)

        # Compare with unfused
        expected = torch.nn.functional.gelu(torch.nn.functional.linear(x, weight, bias))

        assert torch.allclose(result, expected, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_add_relu(self):
        """Test fused add + ReLU."""
        fused = FusedOperations.add_relu()

        x = torch.randn(64, 256, device="cuda")
        y = torch.randn(64, 256, device="cuda")

        result = fused(x, y)
        expected = torch.relu(x + y)

        assert torch.allclose(result, expected)


class TestIsFusable:
    """Tests for is_fusable function."""

    def test_fusable_ops(self):
        """Test that common ops are fusable."""
        assert is_fusable("relu")
        assert is_fusable("gelu")
        assert is_fusable("add")
        assert is_fusable("mul")

    def test_non_fusable_ops(self):
        """Test that some ops are not fusable."""
        assert not is_fusable("unknown_op")
        assert not is_fusable("input")
        assert not is_fusable("output")


class TestFusionWithGraphCapture:
    """Tests for fusion with captured graphs."""

    def test_fusion_detection_mlp(self):
        """Test fusion detection on MLP."""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        sample_input = torch.randn(64, 784)
        graph = capture_graph(model, sample_input)

        detector = FusionDetector()
        candidates = detector.detect(graph)

        # Should find linear+relu patterns
        assert len(candidates) >= 1

    def test_fusion_plan_coverage(self):
        """Test fusion plan coverage."""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

        sample_input = torch.randn(64, 784)
        graph = capture_graph(model, sample_input)

        detector = FusionDetector()
        plan = detector.get_fusion_plan(graph)

        # Should have some fusion coverage
        assert plan["fusion_coverage"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
