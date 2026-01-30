"""
Unit tests for the computation graph module.
"""

import pytest

from kitsune.core.graph import ComputationGraph, CycleDetectedError
from kitsune.core.task import Task, TaskStatus, TaskType


class TestComputationGraph:
    """Tests for ComputationGraph."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = ComputationGraph()
        assert graph.is_empty
        assert graph.num_tasks == 0
        assert graph.is_complete

    def test_add_task(self):
        """Test adding tasks to graph."""
        graph = ComputationGraph()

        task = graph.add_task(
            name="linear1",
            op_type="linear",
            input_shapes=[(64, 784)],
            output_shapes=[(64, 256)],
        )

        assert task.id == 0
        assert task.name == "linear1"
        assert graph.num_tasks == 1
        assert task.id in graph

    def test_task_dependencies(self):
        """Test task dependency tracking."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="linear", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="relu", op_type="relu", inputs=[t2.id])

        assert t1.id in t2.inputs
        assert t2.id in t3.inputs
        assert t2.id in t1.outputs
        assert t3.id in t2.outputs

    def test_ready_tasks_initial(self):
        """Test that tasks with no dependencies are immediately ready."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input1", op_type="input")
        t2 = graph.add_task(name="input2", op_type="input")

        ready = graph.get_ready_tasks()
        assert len(ready) == 2
        assert t1 in ready
        assert t2 in ready

    def test_ready_tasks_after_completion(self):
        """Test ready queue updates after task completion."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="linear", op_type="linear", inputs=[t1.id])

        # Initially only t1 is ready
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert t1 in ready

        # After t1 completes, t2 should be ready
        graph.mark_completed(t1.id)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert t2 in ready

    def test_topological_order(self):
        """Test topological ordering."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="linear1", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="linear2", op_type="linear", inputs=[t1.id])
        t4 = graph.add_task(name="add", op_type="add", inputs=[t2.id, t3.id])

        order = graph.topological_order()

        # t1 must come before t2 and t3
        assert order.index(t1) < order.index(t2)
        assert order.index(t1) < order.index(t3)
        # t2 and t3 must come before t4
        assert order.index(t2) < order.index(t4)
        assert order.index(t3) < order.index(t4)

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="a", op_type="op")
        t2 = graph.add_task(name="b", op_type="op", inputs=[t1.id])
        t3 = graph.add_task(name="c", op_type="op", inputs=[t2.id])

        # Manually create a cycle (t1 depends on t3)
        graph.add_dependency(t3.id, t1.id)

        with pytest.raises(CycleDetectedError):
            graph.topological_order()

    def test_parallel_levels(self):
        """Test parallel level grouping."""
        graph = ComputationGraph()

        # Create a diamond pattern
        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="branch1", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="branch2", op_type="linear", inputs=[t1.id])
        t4 = graph.add_task(name="merge", op_type="add", inputs=[t2.id, t3.id])

        levels = graph.get_parallel_levels()

        assert len(levels) == 3
        assert t1 in levels[0]
        assert t2 in levels[1] and t3 in levels[1]  # Same level
        assert t4 in levels[2]

    def test_critical_path(self):
        """Test critical path finding."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="heavy", op_type="linear", inputs=[t1.id])
        t3 = graph.add_task(name="light", op_type="relu", inputs=[t1.id])
        t4 = graph.add_task(name="output", op_type="add", inputs=[t2.id, t3.id])

        # Set costs - t2 is expensive
        t1.cost.estimated_time_us = 1.0
        t2.cost.estimated_time_us = 100.0
        t3.cost.estimated_time_us = 1.0
        t4.cost.estimated_time_us = 1.0

        critical = graph.get_critical_path()

        # Critical path should go through the heavy task
        assert t2 in critical

    def test_remove_task(self):
        """Test task removal."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="a", op_type="op")
        t2 = graph.add_task(name="b", op_type="op", inputs=[t1.id])

        assert graph.num_tasks == 2

        graph.remove_task(t1.id)

        assert graph.num_tasks == 1
        assert t1.id not in graph
        assert t2.id in graph

    def test_graph_validation(self):
        """Test graph validation."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="a", op_type="op")
        t2 = graph.add_task(name="b", op_type="op", inputs=[t1.id])

        is_valid, msg = graph.is_valid()
        assert is_valid

    def test_dot_export(self):
        """Test DOT format export."""
        graph = ComputationGraph()

        t1 = graph.add_task(name="input", op_type="input")
        t2 = graph.add_task(name="output", op_type="output", inputs=[t1.id])

        dot = graph.to_dot()

        assert "digraph" in dot
        assert "input" in dot
        assert "output" in dot
        assert "->" in dot  # Edge


class TestTask:
    """Tests for Task."""

    def test_task_creation(self):
        """Test task creation."""
        task = Task(
            id=0,
            name="linear",
            op_type="linear",
            input_shapes=[(64, 784)],
            output_shapes=[(64, 256)],
        )

        assert task.id == 0
        assert task.name == "linear"
        assert task.status == TaskStatus.PENDING

    def test_task_cost_estimation(self):
        """Test automatic cost estimation."""
        task = Task(
            id=0,
            name="linear",
            op_type="linear",
            input_shapes=[(64, 784), (784, 256)],
            output_shapes=[(64, 256)],
        )

        assert task.cost is not None
        assert task.cost.flops > 0
        assert task.cost.memory_read_bytes > 0

    def test_task_status_transitions(self):
        """Test task status changes."""
        task = Task(id=0, name="test", op_type="op")

        assert task.status == TaskStatus.PENDING

        task.mark_ready()
        assert task.status == TaskStatus.READY
        assert task.is_ready

        task.mark_running()
        assert task.status == TaskStatus.RUNNING

        task.mark_completed("result")
        assert task.status == TaskStatus.COMPLETED
        assert task.is_completed
        assert task.result == "result"

    def test_task_dependencies(self):
        """Test task dependency management."""
        task = Task(id=0, name="test", op_type="op")

        task.add_dependency(1)
        task.add_dependency(2)
        task.add_dependent(3)

        assert task.num_dependencies == 2
        assert task.num_dependents == 1
        assert 1 in task.inputs
        assert 3 in task.outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
