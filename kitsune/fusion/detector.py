"""
Fusion Detector - Detect fusion opportunities in computation graphs.

Analyzes computation graphs to identify sequences of operations
that can be fused into single kernels.
"""

from __future__ import annotations

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.graph import ComputationGraph
from ..core.task import Task
from .patterns import (
    FusionPattern,
    FusionType,
    PatternMatcher,
    BUILTIN_PATTERNS,
    is_fusable,
)
from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class FusionCandidate:
    """
    A candidate group of operations for fusion.

    Attributes:
        pattern: Matched pattern
        tasks: Tasks to be fused
        estimated_speedup: Estimated speedup from fusion
        memory_reduction: Estimated memory reduction in bytes
    """
    pattern: FusionPattern
    tasks: List[Task]
    estimated_speedup: float = 1.0
    memory_reduction: int = 0
    priority: float = 0.0

    @property
    def task_ids(self) -> List[int]:
        """Get task IDs in this fusion group."""
        return [t.id for t in self.tasks]

    @property
    def op_types(self) -> List[str]:
        """Get operation types in this fusion group."""
        return [t.op_type for t in self.tasks]

    def __repr__(self) -> str:
        ops = " -> ".join(self.op_types)
        return f"FusionCandidate({self.pattern.name}: {ops}, speedup={self.estimated_speedup:.2f}x)"


class FusionDetector:
    """
    Detect fusion opportunities in computation graphs.

    Analyzes the dependency structure and operation types to find
    sequences that can be efficiently fused.

    Usage:
        detector = FusionDetector()
        candidates = detector.detect(graph)

        for candidate in candidates:
            print(f"Can fuse: {candidate}")
    """

    def __init__(
        self,
        patterns: Optional[List[FusionPattern]] = None,
        min_speedup: float = 1.05,
        max_fusion_size: int = 8,
    ):
        """
        Initialize fusion detector.

        Args:
            patterns: Fusion patterns to detect (uses built-in if None)
            min_speedup: Minimum estimated speedup to consider
            max_fusion_size: Maximum ops to fuse together
        """
        self._matcher = PatternMatcher(patterns)
        self._min_speedup = min_speedup
        self._max_fusion_size = max_fusion_size

    def detect(self, graph: ComputationGraph) -> List[FusionCandidate]:
        """
        Detect fusion candidates in a computation graph.

        Args:
            graph: Computation graph to analyze

        Returns:
            List of fusion candidates, sorted by priority
        """
        candidates = []

        # Get topological order
        topo_order = graph.topological_order()

        # Find linear chains (single input, single output)
        chains = self._find_linear_chains(graph, topo_order)

        # Match patterns in chains
        for chain in chains:
            chain_candidates = self._detect_in_chain(chain)
            candidates.extend(chain_candidates)

        # Find parallel fusable groups
        parallel_candidates = self._detect_parallel_fusion(graph, topo_order)
        candidates.extend(parallel_candidates)

        # Estimate speedups and filter
        for candidate in candidates:
            candidate.estimated_speedup = self._estimate_speedup(candidate)
            candidate.memory_reduction = self._estimate_memory_reduction(candidate)
            candidate.priority = self._compute_priority(candidate)

        # Filter by minimum speedup
        candidates = [c for c in candidates if c.estimated_speedup >= self._min_speedup]

        # Sort by priority
        candidates.sort(key=lambda c: -c.priority)

        logger.debug(f"Detected {len(candidates)} fusion candidates")
        return candidates

    def _find_linear_chains(
        self,
        graph: ComputationGraph,
        topo_order: List[Task],
    ) -> List[List[Task]]:
        """Find linear chains of tasks (single in, single out)."""
        chains = []
        visited: Set[int] = set()

        for task in topo_order:
            if task.id in visited:
                continue

            # Start a new chain
            chain = [task]
            visited.add(task.id)

            # Extend forward
            current = task
            while True:
                # Check if single output
                if len(current.outputs) != 1:
                    break

                next_id = list(current.outputs)[0]
                next_task = graph.get_task(next_id)

                if next_task is None or next_task.id in visited:
                    break

                # Check if single input
                if len(next_task.inputs) != 1:
                    break

                chain.append(next_task)
                visited.add(next_task.id)
                current = next_task

            if len(chain) >= 2:
                chains.append(chain)

        return chains

    def _detect_in_chain(self, chain: List[Task]) -> List[FusionCandidate]:
        """Detect fusion candidates in a linear chain."""
        candidates = []

        ops = [t.op_type for t in chain]
        task_ids = [t.id for t in chain]

        # Use pattern matcher
        groups = self._matcher.get_fusable_groups(ops, task_ids)

        for pattern, group_ids in groups:
            tasks = [chain[task_ids.index(tid)] for tid in group_ids]
            candidate = FusionCandidate(pattern=pattern, tasks=tasks)
            candidates.append(candidate)

        # Also detect elementwise chains
        elementwise_chain = self._detect_elementwise_chain(chain)
        if elementwise_chain:
            candidates.append(elementwise_chain)

        return candidates

    def _detect_elementwise_chain(self, chain: List[Task]) -> Optional[FusionCandidate]:
        """Detect chains of elementwise operations."""
        elementwise_tasks = []

        for task in chain:
            if is_fusable(task.op_type):
                elementwise_tasks.append(task)
            elif elementwise_tasks:
                break  # End chain on non-fusable op

        if len(elementwise_tasks) >= 2:
            pattern = FusionPattern(
                name="elementwise_chain",
                op_sequence=[t.op_type for t in elementwise_tasks],
                fusion_type=FusionType.ELEMENTWISE,
            )
            return FusionCandidate(pattern=pattern, tasks=elementwise_tasks)

        return None

    def _detect_parallel_fusion(
        self,
        graph: ComputationGraph,
        topo_order: List[Task],
    ) -> List[FusionCandidate]:
        """Detect parallel operations that can be fused."""
        candidates = []

        # Group tasks by their parallel level
        levels = graph.get_parallel_levels()

        for level_tasks in levels:
            if len(level_tasks) < 2:
                continue

            # Group by operation type
            by_op: Dict[str, List[Task]] = {}
            for task in level_tasks:
                op = task.op_type.lower()
                if op not in by_op:
                    by_op[op] = []
                by_op[op].append(task)

            # Same operation type can be batched
            for op, tasks in by_op.items():
                if len(tasks) >= 2 and is_fusable(op):
                    pattern = FusionPattern(
                        name=f"parallel_{op}",
                        op_sequence=[op] * len(tasks),
                        fusion_type=FusionType.ELEMENTWISE,
                    )
                    candidate = FusionCandidate(pattern=pattern, tasks=tasks)
                    candidates.append(candidate)

        return candidates

    def _estimate_speedup(self, candidate: FusionCandidate) -> float:
        """
        Estimate speedup from fusing operations.

        Based on:
        - Reduced kernel launch overhead
        - Better memory locality
        - Reduced intermediate memory traffic
        """
        num_ops = len(candidate.tasks)

        # Base speedup from reduced kernel launches
        # Each kernel launch has ~5-10us overhead on modern GPUs
        kernel_overhead_reduction = min(1.5, 1.0 + 0.1 * (num_ops - 1))

        # Memory bandwidth improvement from fusion
        # Intermediate results stay in registers/L1
        memory_factor = 1.0
        if candidate.pattern.fusion_type == FusionType.ELEMENTWISE:
            # Elementwise fusion has best memory improvement
            memory_factor = 1.0 + 0.15 * (num_ops - 1)
        elif candidate.pattern.fusion_type in (FusionType.MATMUL_ACTIVATION, FusionType.CONV_BN_RELU):
            # Fused compute+activation
            memory_factor = 1.2
        elif candidate.pattern.fusion_type == FusionType.LAYERNORM:
            # LayerNorm fusion is very effective
            memory_factor = 1.4

        # Estimate total speedup (capped at realistic values)
        speedup = kernel_overhead_reduction * memory_factor
        return min(speedup, 3.0)  # Cap at 3x

    def _estimate_memory_reduction(self, candidate: FusionCandidate) -> int:
        """Estimate memory reduction from fusion."""
        # Sum of intermediate tensor sizes (no longer needed)
        reduction = 0

        for i, task in enumerate(candidate.tasks[:-1]):
            # Each intermediate output is saved
            if task.output_shapes:
                for shape in task.output_shapes:
                    numel = 1
                    for dim in shape:
                        numel *= dim
                    # Assume float32
                    reduction += numel * 4

        return reduction

    def _compute_priority(self, candidate: FusionCandidate) -> float:
        """Compute priority score for a candidate."""
        # Combine speedup, memory reduction, and pattern priority
        speedup_score = candidate.estimated_speedup * 10
        memory_score = candidate.memory_reduction / 1e6  # MB
        pattern_score = candidate.pattern.priority

        return speedup_score + memory_score + pattern_score

    def get_fusion_plan(
        self,
        graph: ComputationGraph,
    ) -> Dict[str, Any]:
        """
        Create a complete fusion plan for a graph.

        Args:
            graph: Computation graph

        Returns:
            Fusion plan with candidates and statistics
        """
        candidates = self.detect(graph)

        # Remove overlapping candidates
        selected = self._select_non_overlapping(candidates)

        # Calculate statistics
        total_speedup = 1.0
        total_memory_saved = 0
        tasks_fused = set()

        for candidate in selected:
            total_speedup *= candidate.estimated_speedup
            total_memory_saved += candidate.memory_reduction
            tasks_fused.update(candidate.task_ids)

        return {
            "candidates": selected,
            "num_fusions": len(selected),
            "estimated_speedup": total_speedup,
            "memory_saved_bytes": total_memory_saved,
            "tasks_fused": len(tasks_fused),
            "total_tasks": graph.num_tasks,
            "fusion_coverage": len(tasks_fused) / max(graph.num_tasks, 1),
        }

    def _select_non_overlapping(
        self,
        candidates: List[FusionCandidate],
    ) -> List[FusionCandidate]:
        """Select non-overlapping candidates."""
        selected = []
        used_tasks: Set[int] = set()

        for candidate in candidates:
            # Check if any task is already used
            if any(tid in used_tasks for tid in candidate.task_ids):
                continue

            selected.append(candidate)
            used_tasks.update(candidate.task_ids)

        return selected
