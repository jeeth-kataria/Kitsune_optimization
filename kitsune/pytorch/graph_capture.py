"""
Graph Capture - Extract computation graphs from PyTorch models

Supports multiple capture strategies:
1. torch.fx symbolic tracing (preferred for static graphs)
2. Forward/backward hooks (fallback for dynamic graphs)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from contextlib import contextmanager
import weakref

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Graph, GraphModule, Node

from ..core.graph import ComputationGraph
from ..core.task import Task, TaskType, TaskCost
from ..utils.logging import get_logger

logger = get_logger("graph_capture")


class GraphCaptureError(Exception):
    """Error during graph capture."""
    pass


@dataclass
class CapturedOp:
    """Represents a captured operation during tracing."""
    name: str
    op_type: str
    target: Any
    args: tuple
    kwargs: dict
    input_shapes: List[tuple]
    output_shapes: List[tuple]
    dtype: torch.dtype = torch.float32


class FXGraphCapture:
    """
    Capture computation graph using torch.fx symbolic tracing.

    This is the preferred method for static graphs (no data-dependent
    control flow). Provides clean, analyzable graph representation.
    """

    def __init__(self, concrete_args: Dict[str, Any] = None):
        """
        Args:
            concrete_args: Concrete values for arguments that shouldn't be traced
        """
        self.concrete_args = concrete_args or {}
        self._traced_module: Optional[GraphModule] = None

    def capture(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> ComputationGraph:
        """
        Capture the computation graph from a model.

        Args:
            model: PyTorch model to trace
            sample_input: Representative input for shape inference

        Returns:
            ComputationGraph with tasks and dependencies
        """
        # Try torch.fx tracing
        try:
            self._traced_module = symbolic_trace(model, concrete_args=self.concrete_args)
            return self._fx_to_computation_graph(self._traced_module, sample_input)
        except Exception as e:
            logger.warning(f"FX tracing failed: {e}. Model may have dynamic control flow.")
            raise GraphCaptureError(f"FX tracing failed: {e}")

    def _fx_to_computation_graph(
        self,
        traced: GraphModule,
        sample_input: torch.Tensor,
    ) -> ComputationGraph:
        """Convert FX graph to ComputationGraph."""
        graph = ComputationGraph()

        # Run shape propagation
        shape_info = self._propagate_shapes(traced, sample_input)

        # Map FX nodes to task IDs
        node_to_task: Dict[Node, int] = {}

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                # Input node
                task = graph.add_task(
                    name=node.name,
                    op_type="input",
                    task_type=TaskType.TRANSFER_H2D,
                    input_shapes=[],
                    output_shapes=[shape_info.get(node.name, ())],
                )
                node_to_task[node] = task.id

            elif node.op == "get_attr":
                # Model parameter/buffer
                task = graph.add_task(
                    name=node.name,
                    op_type="parameter",
                    task_type=TaskType.MEMORY,
                    input_shapes=[],
                    output_shapes=[shape_info.get(node.name, ())],
                )
                node_to_task[node] = task.id

            elif node.op == "call_function":
                # Function call (torch.relu, torch.add, etc.)
                op_name = self._get_op_name(node.target)
                input_ids = self._get_input_task_ids(node, node_to_task)
                input_shapes = [shape_info.get(arg.name, ()) for arg in node.args if isinstance(arg, Node)]

                task = graph.add_task(
                    name=node.name,
                    op_type=op_name,
                    task_type=TaskType.COMPUTE,
                    inputs=input_ids,
                    input_shapes=input_shapes,
                    output_shapes=[shape_info.get(node.name, ())],
                )
                node_to_task[node] = task.id

            elif node.op == "call_method":
                # Method call (tensor.view, tensor.reshape, etc.)
                input_ids = self._get_input_task_ids(node, node_to_task)
                input_shapes = [shape_info.get(arg.name, ()) for arg in node.args if isinstance(arg, Node)]

                task = graph.add_task(
                    name=node.name,
                    op_type=node.target,
                    task_type=TaskType.COMPUTE,
                    inputs=input_ids,
                    input_shapes=input_shapes,
                    output_shapes=[shape_info.get(node.name, ())],
                )
                node_to_task[node] = task.id

            elif node.op == "call_module":
                # Module call (self.linear, self.conv, etc.)
                module = traced.get_submodule(node.target)
                op_name = type(module).__name__.lower()
                input_ids = self._get_input_task_ids(node, node_to_task)
                input_shapes = [shape_info.get(arg.name, ()) for arg in node.args if isinstance(arg, Node)]

                task = graph.add_task(
                    name=node.name,
                    op_type=op_name,
                    task_type=TaskType.COMPUTE,
                    inputs=input_ids,
                    input_shapes=input_shapes,
                    output_shapes=[shape_info.get(node.name, ())],
                )
                node_to_task[node] = task.id

            elif node.op == "output":
                # Output node
                input_ids = self._get_input_task_ids(node, node_to_task)
                task = graph.add_task(
                    name="output",
                    op_type="output",
                    task_type=TaskType.TRANSFER_D2H,
                    inputs=input_ids,
                    input_shapes=[shape_info.get(arg.name, ()) for arg in node.args[0] if isinstance(arg, Node)] if isinstance(node.args[0], (list, tuple)) else [],
                    output_shapes=[],
                )
                node_to_task[node] = task.id

        return graph

    def _propagate_shapes(
        self,
        traced: GraphModule,
        sample_input: torch.Tensor,
    ) -> Dict[str, tuple]:
        """Propagate shapes through the traced graph."""
        shape_info = {}

        # Run the model to get intermediate shapes
        class ShapeRecorder(torch.fx.Interpreter):
            def __init__(self, module):
                super().__init__(module)
                self.shapes = {}

            def run_node(self, n):
                result = super().run_node(n)
                if isinstance(result, torch.Tensor):
                    self.shapes[n.name] = tuple(result.shape)
                elif isinstance(result, (list, tuple)):
                    # Handle multiple outputs
                    for i, r in enumerate(result):
                        if isinstance(r, torch.Tensor):
                            self.shapes[f"{n.name}_{i}"] = tuple(r.shape)
                return result

        try:
            recorder = ShapeRecorder(traced)
            recorder.run(sample_input)
            shape_info = recorder.shapes
        except Exception as e:
            logger.warning(f"Shape propagation failed: {e}")

        return shape_info

    def _get_op_name(self, target: Any) -> str:
        """Extract operation name from target."""
        if callable(target):
            if hasattr(target, "__name__"):
                return target.__name__
            elif hasattr(target, "__class__"):
                return target.__class__.__name__
        return str(target)

    def _get_input_task_ids(
        self,
        node: Node,
        node_to_task: Dict[Node, int],
    ) -> List[int]:
        """Get task IDs for node's input dependencies."""
        input_ids = []

        def collect_inputs(arg):
            if isinstance(arg, Node) and arg in node_to_task:
                input_ids.append(node_to_task[arg])
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    collect_inputs(a)

        for arg in node.args:
            collect_inputs(arg)

        return input_ids

    @property
    def traced_module(self) -> Optional[GraphModule]:
        """Get the traced GraphModule (available after capture)."""
        return self._traced_module


class HookGraphCapture:
    """
    Capture computation graph using forward/backward hooks.

    Fallback method that works with dynamic control flow but
    provides less detailed graph information.
    """

    def __init__(self):
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_ops: List[CapturedOp] = []
        self._tensor_to_id: Dict[int, int] = {}  # tensor id -> op index
        self._next_op_id = 0

    def capture(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        capture_backward: bool = False,
    ) -> ComputationGraph:
        """
        Capture computation graph by running the model with hooks.

        Args:
            model: PyTorch model
            sample_input: Input to run through the model
            capture_backward: Also capture backward pass

        Returns:
            ComputationGraph with captured operations
        """
        self._reset()
        self._register_hooks(model)

        try:
            # Forward pass
            with torch.no_grad():
                output = model(sample_input)

            # Build graph from captured ops
            graph = self._build_graph()

            return graph

        finally:
            self._remove_hooks()

    def _reset(self):
        """Reset capture state."""
        self._captured_ops.clear()
        self._tensor_to_id.clear()
        self._next_op_id = 0

    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on all modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(
                    self._make_forward_hook(name, module)
                )
                self._hooks.append(handle)

    def _make_forward_hook(self, module_name: str, module: nn.Module) -> Callable:
        """Create a forward hook for a module."""
        def hook(mod, inputs, output):
            # Get input shapes
            input_shapes = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(tuple(inp.shape))

            # Get output shapes
            output_shapes = []
            if isinstance(output, torch.Tensor):
                output_shapes.append(tuple(output.shape))
            elif isinstance(output, (list, tuple)):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        output_shapes.append(tuple(out.shape))

            # Determine dtype
            dtype = torch.float32
            if isinstance(output, torch.Tensor):
                dtype = output.dtype

            # Record operation
            op = CapturedOp(
                name=module_name,
                op_type=type(module).__name__.lower(),
                target=module,
                args=inputs,
                kwargs={},
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dtype=dtype,
            )
            self._captured_ops.append(op)

        return hook

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _build_graph(self) -> ComputationGraph:
        """Build ComputationGraph from captured operations."""
        graph = ComputationGraph()

        # Simple sequential dependency: each op depends on the previous
        prev_task_id = None

        for i, op in enumerate(self._captured_ops):
            inputs = [prev_task_id] if prev_task_id is not None else []

            task = graph.add_task(
                name=op.name,
                op_type=op.op_type,
                task_type=TaskType.COMPUTE,
                inputs=inputs,
                input_shapes=op.input_shapes,
                output_shapes=op.output_shapes,
                dtype=op.dtype,
            )
            prev_task_id = task.id

        return graph


class GraphCapture:
    """
    High-level graph capture interface.

    Automatically selects the best capture strategy:
    1. Try torch.fx (preferred)
    2. Fall back to hooks if FX fails
    """

    def __init__(
        self,
        strategy: str = "auto",
        concrete_args: Dict[str, Any] = None,
    ):
        """
        Args:
            strategy: "auto", "fx", or "hooks"
            concrete_args: Concrete values for FX tracing
        """
        self.strategy = strategy
        self.concrete_args = concrete_args or {}
        self._fx_capture = FXGraphCapture(concrete_args)
        self._hook_capture = HookGraphCapture()
        self._used_strategy: Optional[str] = None

    def capture(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> ComputationGraph:
        """
        Capture the computation graph from a model.

        Args:
            model: PyTorch model
            sample_input: Representative input for tracing

        Returns:
            ComputationGraph with operations and dependencies
        """
        if self.strategy == "fx":
            self._used_strategy = "fx"
            return self._fx_capture.capture(model, sample_input)

        elif self.strategy == "hooks":
            self._used_strategy = "hooks"
            return self._hook_capture.capture(model, sample_input)

        else:  # auto
            try:
                graph = self._fx_capture.capture(model, sample_input)
                self._used_strategy = "fx"
                logger.info("Graph captured using torch.fx")
                return graph
            except GraphCaptureError:
                logger.info("FX tracing failed, falling back to hooks")
                self._used_strategy = "hooks"
                return self._hook_capture.capture(model, sample_input)

    @property
    def used_strategy(self) -> Optional[str]:
        """Strategy that was used for capture."""
        return self._used_strategy

    @property
    def traced_module(self) -> Optional[GraphModule]:
        """Get FX traced module (if FX was used)."""
        return self._fx_capture.traced_module


def capture_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
    strategy: str = "auto",
) -> ComputationGraph:
    """
    Convenience function to capture a computation graph.

    Args:
        model: PyTorch model
        sample_input: Representative input
        strategy: Capture strategy ("auto", "fx", "hooks")

    Returns:
        ComputationGraph
    """
    capturer = GraphCapture(strategy=strategy)
    return capturer.capture(model, sample_input)
