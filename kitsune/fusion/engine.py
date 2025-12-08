"""
Fusion Engine - JIT compilation and execution of fused kernels.

Uses torch.compile (PyTorch 2.0+) as the primary backend, with
optional Triton support on Linux.
"""

from __future__ import annotations

import sys
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import torch
import torch.nn as nn

from .patterns import FusionPattern, FusionType
from .detector import FusionCandidate, FusionDetector
from ..core.graph import ComputationGraph
from ..profiler import get_logger

logger = get_logger(__name__)

# Check for Triton availability
TRITON_AVAILABLE = False
try:
    if sys.platform != "win32":
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
        logger.info("Triton available for kernel fusion")
except ImportError:
    logger.info("Triton not available, using torch.compile backend")

# Check if torch.compile is available (requires PyTorch 2.0+ on Linux)
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        # Test if it actually works
        _test_fn = lambda x: x + 1
        torch.compile(_test_fn, backend='eager')
        TORCH_COMPILE_AVAILABLE = True
        logger.info("torch.compile available")
except Exception:
    logger.info("torch.compile not available, using torch.jit fallback")


@dataclass
class FusedKernel:
    """
    A compiled fused kernel.

    Attributes:
        name: Kernel identifier
        pattern: Original fusion pattern
        compiled_fn: Compiled function
        input_specs: Input tensor specifications
        backend: Compilation backend used
    """
    name: str
    pattern: FusionPattern
    compiled_fn: Callable
    input_specs: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "torch.compile"
    speedup: float = 1.0


class TorchCompileBackend:
    """
    Fusion backend using torch.compile (PyTorch 2.0+) or torch.jit fallback.

    Uses PyTorch's built-in compilation with inductor backend
    for automatic kernel fusion. Falls back to torch.jit.trace on Windows.
    """

    def __init__(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ):
        """
        Initialize torch.compile backend.

        Args:
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            fullgraph: Whether to require full graph compilation
            dynamic: Whether to allow dynamic shapes
        """
        self._mode = mode
        self._fullgraph = fullgraph
        self._dynamic = dynamic
        self._compiled_cache: Dict[str, Callable] = {}
        self._use_jit = not TORCH_COMPILE_AVAILABLE

    def compile(
        self,
        fn: Callable,
        name: str = "fused",
    ) -> Callable:
        """
        Compile a function using torch.compile or torch.jit.

        Args:
            fn: Function to compile
            name: Cache key

        Returns:
            Compiled function
        """
        if name in self._compiled_cache:
            return self._compiled_cache[name]

        if self._use_jit:
            # Use torch.jit.script as fallback (more limited but works on Windows)
            try:
                compiled = torch.jit.script(fn)
                self._compiled_cache[name] = compiled
                logger.debug(f"Compiled {name} with torch.jit.script")
                return compiled
            except Exception as e:
                logger.debug(f"torch.jit.script failed for {name}: {e}, using original")
                return fn

        try:
            compiled = torch.compile(
                fn,
                mode=self._mode,
                fullgraph=self._fullgraph,
                dynamic=self._dynamic,
            )
            self._compiled_cache[name] = compiled
            logger.debug(f"Compiled {name} with torch.compile (mode={self._mode})")
            return compiled
        except Exception as e:
            logger.warning(f"torch.compile failed for {name}: {e}, using original")
            return fn

    def compile_module(
        self,
        module: nn.Module,
        name: str = "module",
    ) -> nn.Module:
        """
        Compile a module using torch.compile or torch.jit.

        Args:
            module: Module to compile
            name: Cache key

        Returns:
            Compiled module
        """
        if self._use_jit:
            # JIT tracing requires example input, so we skip module compilation
            # and rely on manual fusion patterns instead
            logger.debug(f"Skipping module compilation for {name} (using manual fusion)")
            return module

        try:
            compiled = torch.compile(
                module,
                mode=self._mode,
                fullgraph=self._fullgraph,
                dynamic=self._dynamic,
            )
            logger.debug(f"Compiled module {name} with torch.compile")
            return compiled
        except Exception as e:
            logger.warning(f"torch.compile failed for module {name}: {e}")
            return module


class FusedOperations:
    """
    Collection of pre-fused common operation patterns.

    Provides optimized implementations for frequent patterns.
    """

    @staticmethod
    def linear_relu(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Callable:
        """Create fused linear + ReLU."""
        def fused(x: torch.Tensor) -> torch.Tensor:
            out = torch.nn.functional.linear(x, weight, bias)
            return torch.nn.functional.relu(out)
        return fused

    @staticmethod
    def linear_gelu(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Callable:
        """Create fused linear + GELU."""
        def fused(x: torch.Tensor) -> torch.Tensor:
            out = torch.nn.functional.linear(x, weight, bias)
            return torch.nn.functional.gelu(out)
        return fused

    @staticmethod
    def linear_silu(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Callable:
        """Create fused linear + SiLU."""
        def fused(x: torch.Tensor) -> torch.Tensor:
            out = torch.nn.functional.linear(x, weight, bias)
            return torch.nn.functional.silu(out)
        return fused

    @staticmethod
    def conv_bn_relu(
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> Callable:
        """Create fused conv + batchnorm + ReLU."""
        # Fuse conv and bn weights
        fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)

        def fused(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(fused_conv(x))

        return fused

    @staticmethod
    def add_relu() -> Callable:
        """Create fused add + ReLU."""
        def fused(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(x + y)
        return fused

    @staticmethod
    def mul_add(scale: float, bias: float) -> Callable:
        """Create fused multiply + add."""
        def fused(x: torch.Tensor) -> torch.Tensor:
            return x * scale + bias
        return fused


class FusionEngine:
    """
    Main fusion engine for compiling and executing fused kernels.

    Coordinates pattern detection, kernel compilation, and execution.

    Usage:
        engine = FusionEngine()

        # Compile a model with fusion
        optimized_model = engine.optimize_model(model, sample_input)

        # Or manually fuse operations
        fused_fn = engine.fuse_operations([op1, op2, op3])
    """

    def __init__(
        self,
        backend: str = "auto",
        compile_mode: str = "reduce-overhead",
        enable_triton: bool = True,
    ):
        """
        Initialize fusion engine.

        Args:
            backend: Backend to use ("auto", "torch.compile", "triton")
            compile_mode: torch.compile mode
            enable_triton: Whether to use Triton when available
        """
        self._backend_name = backend
        self._compile_mode = compile_mode
        self._enable_triton = enable_triton and TRITON_AVAILABLE

        # Initialize backends
        self._torch_backend = TorchCompileBackend(mode=compile_mode)
        self._detector = FusionDetector()

        # Kernel cache
        self._kernel_cache: Dict[str, FusedKernel] = {}

        # Determine actual backend
        if backend == "auto":
            if self._enable_triton:
                self._backend_name = "triton"
            elif TORCH_COMPILE_AVAILABLE:
                self._backend_name = "torch.compile"
            else:
                self._backend_name = "torch.jit"

        logger.info(f"FusionEngine initialized with backend: {self._backend_name}")

    @property
    def backend(self) -> str:
        """Get the active backend name."""
        return self._backend_name

    def detect_fusion_opportunities(
        self,
        graph: ComputationGraph,
    ) -> List[FusionCandidate]:
        """
        Detect fusion opportunities in a graph.

        Args:
            graph: Computation graph

        Returns:
            List of fusion candidates
        """
        return self._detector.detect(graph)

    def get_fusion_plan(self, graph: ComputationGraph) -> Dict[str, Any]:
        """
        Create a fusion plan for a graph.

        Args:
            graph: Computation graph

        Returns:
            Fusion plan dictionary
        """
        return self._detector.get_fusion_plan(graph)

    def optimize_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Optimize a model with kernel fusion.

        Args:
            model: PyTorch model
            sample_input: Sample input for tracing
            inplace: Whether to modify model in place

        Returns:
            Optimized model
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)

        # Apply torch.compile for automatic fusion
        optimized = self._torch_backend.compile_module(model, name=type(model).__name__)

        return optimized

    def fuse_sequential(
        self,
        modules: List[nn.Module],
        name: str = "fused_seq",
    ) -> nn.Module:
        """
        Fuse a sequence of modules.

        Args:
            modules: Modules to fuse
            name: Name for the fused module

        Returns:
            Fused module
        """
        sequential = nn.Sequential(*modules)
        return self._torch_backend.compile_module(sequential, name=name)

    def compile_function(
        self,
        fn: Callable,
        name: str = "fn",
    ) -> Callable:
        """
        Compile a function with fusion.

        Args:
            fn: Function to compile
            name: Cache key

        Returns:
            Compiled function
        """
        return self._torch_backend.compile(fn, name)

    def create_fused_linear_activation(
        self,
        linear: nn.Linear,
        activation: str = "relu",
    ) -> Callable:
        """
        Create a fused linear + activation kernel.

        Args:
            linear: Linear layer
            activation: Activation type

        Returns:
            Fused function
        """
        if activation == "relu":
            fn = FusedOperations.linear_relu(linear.weight, linear.bias)
        elif activation == "gelu":
            fn = FusedOperations.linear_gelu(linear.weight, linear.bias)
        elif activation == "silu":
            fn = FusedOperations.linear_silu(linear.weight, linear.bias)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return self._torch_backend.compile(fn, f"linear_{activation}")

    def benchmark_fusion(
        self,
        original_fn: Callable,
        fused_fn: Callable,
        sample_input: torch.Tensor,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark original vs fused function.

        Args:
            original_fn: Original function
            fused_fn: Fused function
            sample_input: Sample input
            num_iterations: Benchmark iterations
            warmup: Warmup iterations

        Returns:
            Benchmark results
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA required for benchmarking"}

        # Warmup
        for _ in range(warmup):
            _ = original_fn(sample_input)
            _ = fused_fn(sample_input)
        torch.cuda.synchronize()

        # Benchmark original
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            _ = original_fn(sample_input)
        end.record()
        end.synchronize()

        original_time = start.elapsed_time(end) / num_iterations

        # Benchmark fused
        start.record()
        for _ in range(num_iterations):
            _ = fused_fn(sample_input)
        end.record()
        end.synchronize()

        fused_time = start.elapsed_time(end) / num_iterations

        return {
            "original_ms": original_time,
            "fused_ms": fused_time,
            "speedup": original_time / fused_time if fused_time > 0 else 1.0,
        }

    def summary(self) -> str:
        """Get engine summary."""
        lines = [
            "FusionEngine Summary",
            "=" * 40,
            f"Backend: {self._backend_name}",
            f"Compile mode: {self._compile_mode}",
            f"Triton available: {TRITON_AVAILABLE}",
            f"Cached kernels: {len(self._kernel_cache)}",
        ]
        return "\n".join(lines)


# Global fusion engine singleton
_global_engine: Optional[FusionEngine] = None


def get_fusion_engine() -> FusionEngine:
    """Get or create the global fusion engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = FusionEngine()
    return _global_engine


def reset_fusion_engine() -> None:
    """Reset the global fusion engine."""
    global _global_engine
    _global_engine = None
