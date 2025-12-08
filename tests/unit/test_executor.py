"""
Unit tests for stream-aware executor.
"""

import pytest
import torch
import torch.nn as nn

from kitsune.core.executor import (
    StreamExecutor,
    ModelExecutor,
    ParallelForwardExecutor,
    ExecutionResult,
)
from kitsune.core.scheduler import DataflowScheduler, ExecutionPlan, ScheduleStep
from kitsune.core.graph import ComputationGraph


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ParallelBranchModel(nn.Module):
    """Model with parallel branches."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(784, 256)
        self.branch1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.merge = nn.Linear(256, 10)

    def forward(self, x):
        x = self.stem(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        merged = torch.cat([b1, b2], dim=1)
        return self.merge(merged)


class TestStreamExecutor:
    """Tests for StreamExecutor."""

    def test_executor_creation(self):
        """Test executor creation."""
        executor = StreamExecutor(num_streams=4)
        # Should work even without CUDA
        assert executor is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_executor_enabled(self):
        """Test executor is enabled with CUDA."""
        executor = StreamExecutor(num_streams=4)
        assert executor.enabled

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_plan(self):
        """Test executing a simple plan."""
        from kitsune.core.task import Task

        executor = StreamExecutor(num_streams=4)
        plan = ExecutionPlan(num_streams=4)

        # Create simple tasks
        t1 = Task(id=0, name="task0", op_type="compute")
        t2 = Task(id=1, name="task1", op_type="compute", inputs={0})
        t1.outputs.add(1)

        plan.add_step(t1, stream_id=0)
        plan.add_step(t2, stream_id=1, wait_for=[0])

        # Create kernels
        results = []
        kernels = {
            0: lambda: results.append(0),
            1: lambda: results.append(1),
        }

        result = executor.execute(plan, kernels)

        assert result.tasks_executed == 2
        assert 0 in results
        assert 1 in results


class TestModelExecutor:
    """Tests for ModelExecutor."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_executor_creation(self):
        """Test model executor creation."""
        model = SimpleModel().cuda()
        sample_input = torch.randn(64, 784, device="cuda")

        executor = ModelExecutor(model, sample_input)

        assert executor.plan is not None
        assert executor.graph is not None
        assert executor.graph.num_tasks > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_forward_pass(self):
        """Test forward pass through executor."""
        model = SimpleModel().cuda()
        sample_input = torch.randn(64, 784, device="cuda")

        executor = ModelExecutor(model, sample_input)

        # Execute
        output = executor(sample_input)

        assert output.shape == (64, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark(self):
        """Test benchmarking."""
        model = SimpleModel().cuda()
        sample_input = torch.randn(64, 784, device="cuda")

        executor = ModelExecutor(model, sample_input)

        stats = executor.benchmark(sample_input, num_iterations=10, warmup=3)

        assert "avg_time_ms" in stats
        assert "throughput_per_sec" in stats
        assert stats["num_iterations"] == 10


class TestParallelForwardExecutor:
    """Tests for ParallelForwardExecutor."""

    def test_executor_creation(self):
        """Test executor creation."""
        executor = ParallelForwardExecutor(num_streams=4)
        assert executor is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_parallel_functions(self):
        """Test executing functions in parallel."""
        executor = ParallelForwardExecutor(num_streams=4)

        def fn1():
            return torch.randn(100, device="cuda")

        def fn2():
            return torch.randn(200, device="cuda")

        def fn3():
            return torch.randn(300, device="cuda")

        results = executor.execute_parallel([fn1, fn2, fn3])

        assert len(results) == 3
        assert results[0].shape == (100,)
        assert results[1].shape == (200,)
        assert results[2].shape == (300,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_execute_branches(self):
        """Test executing model branches in parallel."""
        executor = ParallelForwardExecutor(num_streams=4)

        branch1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU()).cuda()
        branch2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU()).cuda()
        branch3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU()).cuda()

        input = torch.randn(64, 256, device="cuda")

        results = executor.execute_branches(input, [branch1, branch2, branch3])

        assert len(results) == 3
        assert results[0].shape == (64, 128)
        assert results[1].shape == (64, 128)
        assert results[2].shape == (64, 64)


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = ExecutionResult()
        assert result.total_time_ms == 0.0
        assert result.tasks_executed == 0

    def test_result_with_values(self):
        """Test result with values."""
        result = ExecutionResult(
            output=torch.zeros(10),
            total_time_ms=5.0,
            tasks_executed=3,
            parallel_efficiency=0.8,
        )

        assert result.total_time_ms == 5.0
        assert result.tasks_executed == 3
        assert result.parallel_efficiency == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
