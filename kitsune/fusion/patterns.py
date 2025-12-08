"""
Fusion Patterns - Define patterns for kernel fusion.

Identifies common operation sequences that can be fused into
single kernels for better performance.
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import re

from ..profiler import get_logger

logger = get_logger(__name__)


class FusionType(Enum):
    """Types of kernel fusion."""
    ELEMENTWISE = auto()      # Element-wise ops (add, mul, relu, etc.)
    REDUCTION = auto()        # Reduction ops (sum, mean, etc.)
    MATMUL_BIAS = auto()      # MatMul + Bias addition
    MATMUL_ACTIVATION = auto()  # MatMul + Activation
    CONV_BN = auto()          # Conv + BatchNorm
    CONV_BN_RELU = auto()     # Conv + BatchNorm + ReLU
    ATTENTION = auto()        # Attention pattern (QKV)
    LAYERNORM = auto()        # LayerNorm components
    SOFTMAX = auto()          # Softmax components
    GELU = auto()             # GELU activation
    CUSTOM = auto()           # User-defined pattern


@dataclass
class FusionPattern:
    """
    A pattern that can be fused into a single kernel.

    Attributes:
        name: Pattern identifier
        op_sequence: Sequence of operation types to match
        fusion_type: Type of fusion
        min_ops: Minimum operations to trigger fusion
        priority: Higher priority patterns are matched first
    """
    name: str
    op_sequence: List[str]
    fusion_type: FusionType
    min_ops: int = 2
    priority: int = 0
    constraints: Dict[str, Any] = field(default_factory=dict)

    def matches(self, ops: List[str]) -> bool:
        """
        Check if operation sequence matches this pattern.

        Args:
            ops: List of operation type strings

        Returns:
            True if pattern matches
        """
        if len(ops) < self.min_ops:
            return False

        # Simple sequential matching
        pattern_idx = 0
        for op in ops:
            if pattern_idx >= len(self.op_sequence):
                break
            if self._op_matches(op, self.op_sequence[pattern_idx]):
                pattern_idx += 1

        return pattern_idx == len(self.op_sequence)

    def _op_matches(self, op: str, pattern: str) -> bool:
        """Check if an operation matches a pattern element."""
        op = op.lower()
        pattern = pattern.lower()

        # Exact match
        if op == pattern:
            return True

        # Wildcard matching
        if pattern == "*":
            return True

        # Category matching
        if pattern == "elementwise":
            return op in ELEMENTWISE_OPS
        if pattern == "activation":
            return op in ACTIVATION_OPS
        if pattern == "reduction":
            return op in REDUCTION_OPS

        # Regex matching
        if pattern.startswith("r:"):
            return bool(re.match(pattern[2:], op))

        return False


# Common operation categories
ELEMENTWISE_OPS = {
    "add", "sub", "mul", "div", "neg",
    "relu", "gelu", "silu", "sigmoid", "tanh",
    "exp", "log", "sqrt", "rsqrt", "abs",
    "sin", "cos", "pow",
}

ACTIVATION_OPS = {
    "relu", "gelu", "silu", "sigmoid", "tanh",
    "leaky_relu", "elu", "selu", "softplus",
    "hardswish", "hardsigmoid", "mish",
}

REDUCTION_OPS = {
    "sum", "mean", "max", "min", "prod",
    "var", "std", "norm", "softmax", "log_softmax",
}

MATMUL_OPS = {
    "linear", "matmul", "mm", "bmm", "addmm",
    "conv1d", "conv2d", "conv3d",
}

NORMALIZATION_OPS = {
    "batchnorm", "layernorm", "instancenorm", "groupnorm",
    "batch_norm", "layer_norm", "instance_norm", "group_norm",
}


# Built-in fusion patterns
BUILTIN_PATTERNS = [
    # Linear + Activation patterns
    FusionPattern(
        name="linear_relu",
        op_sequence=["linear", "relu"],
        fusion_type=FusionType.MATMUL_ACTIVATION,
        priority=10,
    ),
    FusionPattern(
        name="linear_gelu",
        op_sequence=["linear", "gelu"],
        fusion_type=FusionType.MATMUL_ACTIVATION,
        priority=10,
    ),
    FusionPattern(
        name="linear_silu",
        op_sequence=["linear", "silu"],
        fusion_type=FusionType.MATMUL_ACTIVATION,
        priority=10,
    ),
    FusionPattern(
        name="linear_sigmoid",
        op_sequence=["linear", "sigmoid"],
        fusion_type=FusionType.MATMUL_ACTIVATION,
        priority=10,
    ),

    # Bias + Activation patterns
    FusionPattern(
        name="add_relu",
        op_sequence=["add", "relu"],
        fusion_type=FusionType.ELEMENTWISE,
        priority=5,
    ),
    FusionPattern(
        name="add_gelu",
        op_sequence=["add", "gelu"],
        fusion_type=FusionType.ELEMENTWISE,
        priority=5,
    ),

    # Conv + BN + Activation
    FusionPattern(
        name="conv_bn_relu",
        op_sequence=["conv2d", "batchnorm", "relu"],
        fusion_type=FusionType.CONV_BN_RELU,
        priority=15,
    ),
    FusionPattern(
        name="conv_bn",
        op_sequence=["conv2d", "batchnorm"],
        fusion_type=FusionType.CONV_BN,
        priority=12,
    ),

    # Multiple elementwise operations
    FusionPattern(
        name="elementwise_chain",
        op_sequence=["elementwise", "elementwise"],
        fusion_type=FusionType.ELEMENTWISE,
        min_ops=2,
        priority=3,
    ),

    # Activation chains
    FusionPattern(
        name="activation_chain",
        op_sequence=["activation", "activation"],
        fusion_type=FusionType.ELEMENTWISE,
        min_ops=2,
        priority=2,
    ),

    # LayerNorm components
    FusionPattern(
        name="layernorm_fused",
        op_sequence=["mean", "sub", "pow", "mean", "add", "rsqrt", "mul"],
        fusion_type=FusionType.LAYERNORM,
        priority=20,
    ),

    # Softmax components
    FusionPattern(
        name="softmax_fused",
        op_sequence=["max", "sub", "exp", "sum", "div"],
        fusion_type=FusionType.SOFTMAX,
        priority=20,
    ),

    # GELU approximation
    FusionPattern(
        name="gelu_approx",
        op_sequence=["mul", "pow", "mul", "add", "mul", "tanh", "add", "mul"],
        fusion_type=FusionType.GELU,
        priority=18,
    ),
]


class PatternMatcher:
    """
    Match fusion patterns in operation sequences.

    Identifies opportunities for kernel fusion in computation graphs.
    """

    def __init__(self, patterns: Optional[List[FusionPattern]] = None):
        """
        Initialize pattern matcher.

        Args:
            patterns: Patterns to match (uses built-in if None)
        """
        self._patterns = patterns or list(BUILTIN_PATTERNS)
        # Sort by priority (higher first)
        self._patterns.sort(key=lambda p: -p.priority)

    def add_pattern(self, pattern: FusionPattern) -> None:
        """Add a custom pattern."""
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda p: -p.priority)

    def find_matches(
        self,
        ops: List[str],
        max_matches: Optional[int] = None,
    ) -> List[Tuple[FusionPattern, int, int]]:
        """
        Find all pattern matches in operation sequence.

        Args:
            ops: List of operation type strings
            max_matches: Maximum matches to return

        Returns:
            List of (pattern, start_idx, end_idx) tuples
        """
        matches = []
        used_indices: Set[int] = set()

        for pattern in self._patterns:
            # Sliding window search
            for i in range(len(ops) - pattern.min_ops + 1):
                if i in used_indices:
                    continue

                window = ops[i:i + len(pattern.op_sequence)]
                if pattern.matches(window):
                    end_idx = i + len(pattern.op_sequence)
                    matches.append((pattern, i, end_idx))

                    # Mark indices as used
                    for j in range(i, end_idx):
                        used_indices.add(j)

                    if max_matches and len(matches) >= max_matches:
                        return matches

        return matches

    def find_best_match(self, ops: List[str]) -> Optional[Tuple[FusionPattern, int, int]]:
        """
        Find the best (highest priority) match.

        Args:
            ops: List of operation type strings

        Returns:
            (pattern, start_idx, end_idx) or None if no match
        """
        matches = self.find_matches(ops, max_matches=1)
        return matches[0] if matches else None

    def get_fusable_groups(
        self,
        ops: List[str],
        task_ids: List[int],
    ) -> List[Tuple[FusionPattern, List[int]]]:
        """
        Get groups of tasks that can be fused.

        Args:
            ops: Operation types
            task_ids: Corresponding task IDs

        Returns:
            List of (pattern, [task_ids]) for fusable groups
        """
        matches = self.find_matches(ops)
        groups = []

        for pattern, start, end in matches:
            group_task_ids = task_ids[start:end]
            groups.append((pattern, group_task_ids))

        return groups


def get_fusion_type_for_op(op: str) -> Optional[FusionType]:
    """
    Get the appropriate fusion type for an operation.

    Args:
        op: Operation type string

    Returns:
        FusionType or None if not fusable
    """
    op = op.lower()

    if op in ELEMENTWISE_OPS or op in ACTIVATION_OPS:
        return FusionType.ELEMENTWISE

    if op in REDUCTION_OPS:
        return FusionType.REDUCTION

    if op in MATMUL_OPS:
        return FusionType.MATMUL_BIAS

    if op in NORMALIZATION_OPS:
        return FusionType.LAYERNORM

    return None


def is_fusable(op: str) -> bool:
    """Check if an operation type is fusable."""
    return get_fusion_type_for_op(op) is not None
