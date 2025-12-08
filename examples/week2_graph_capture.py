"""
Week 2 Demo - Graph Capture and Scheduling

Demonstrates:
- Capturing computation graphs from PyTorch models
- Building dependency DAGs
- Creating execution schedules with parallelism analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from kitsune.pytorch import capture_graph, GraphCapture
from kitsune.core import (
    DataflowScheduler,
    ComputationGraph,
    WavefrontScheduler,
)
from tests.benchmarks.models import create_mlp, create_lenet, create_resnet18


def demo_graph_capture():
    """Demonstrate graph capture capabilities."""
    print("=" * 60)
    print("Graph Capture Demo")
    print("=" * 60)

    # Simple MLP
    print("\n--- MLP Graph Capture ---")
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    sample_input = torch.randn(64, 784)

    capturer = GraphCapture(strategy="auto")
    graph = capturer.capture(model, sample_input)

    print(f"Capture strategy used: {capturer.used_strategy}")
    print(f"Tasks captured: {graph.num_tasks}")
    print(f"Graph valid: {graph.is_valid()[0]}")

    # Print task details
    print("\nTasks:")
    for task in graph.tasks:
        deps = f"deps={list(task.inputs)}" if task.inputs else "no deps"
        print(f"  [{task.id}] {task.name} ({task.op_type}) - {deps}")

    return graph


def demo_parallel_analysis():
    """Demonstrate parallelism analysis."""
    print("\n" + "=" * 60)
    print("Parallelism Analysis Demo")
    print("=" * 60)

    # Create a model with parallel branches
    class ParallelBranchModel(nn.Module):
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

    model = ParallelBranchModel()
    sample_input = torch.randn(64, 784)

    # Capture and analyze
    graph = capture_graph(model, sample_input, strategy="hooks")

    print(f"\nGraph with parallel branches:")
    print(f"  Total tasks: {graph.num_tasks}")

    # Get parallel levels
    levels = graph.get_parallel_levels()
    print(f"  Parallel levels: {len(levels)}")

    for i, level in enumerate(levels):
        task_names = [t.name for t in level]
        print(f"  Level {i}: {len(level)} tasks - {task_names[:3]}{'...' if len(task_names) > 3 else ''}")

    return graph


def demo_scheduler():
    """Demonstrate the dataflow scheduler."""
    print("\n" + "=" * 60)
    print("Dataflow Scheduler Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create scheduler
    scheduler = DataflowScheduler(
        num_streams=4,
        scheduler_type="wavefront",
    )

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    if device == "cuda":
        model = model.cuda()

    sample_input = torch.randn(64, 784, device=device)

    # Capture and schedule
    print("\nCapturing and scheduling...")
    plan = scheduler.capture_and_schedule(model, sample_input)

    print(f"\nExecution Plan:")
    print(f"  Total steps: {len(plan)}")
    print(f"  Streams: {plan.num_streams}")
    print(f"  Estimated time: {plan.estimated_time_us:.2f} µs")

    # Show scheduler summary
    print("\n" + scheduler.summary())

    # Show first few steps
    print("\nFirst 10 steps:")
    for i, step in enumerate(plan.steps[:10]):
        stream = f"[S{step.stream_id}]"
        wait = f" wait:{step.wait_for}" if step.wait_for else ""
        print(f"  {i:3d}. {stream} {step.task.name} ({step.task.op_type}){wait}")

    return plan


def demo_resnet_analysis():
    """Analyze ResNet-18 graph structure."""
    print("\n" + "=" * 60)
    print("ResNet-18 Analysis Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create ResNet-18
    model = create_resnet18(device=device)
    sample_input = torch.randn(8, 3, 32, 32, device=device)

    # Create scheduler and capture
    scheduler = DataflowScheduler(
        num_streams=6,
        scheduler_type="wavefront",
    )

    print("Capturing ResNet-18 graph (this may take a moment)...")

    try:
        plan = scheduler.capture_and_schedule(model, sample_input)

        print(f"\nResNet-18 Graph Analysis:")
        print(f"  Tasks: {scheduler.graph.num_tasks}")
        print(f"  Schedule steps: {len(plan)}")
        print(f"  Estimated time: {plan.estimated_time_us:.2f} µs")

        # Parallelism analysis
        if isinstance(scheduler._scheduler, WavefrontScheduler):
            stats = scheduler._scheduler.get_parallelism_stats(scheduler.graph)
            print(f"\nParallelism Stats:")
            print(f"  Levels: {stats['num_levels']}")
            print(f"  Max parallelism: {stats['max_parallelism']}")
            print(f"  Avg parallelism: {stats['avg_parallelism']:.2f}")

        # Export DOT for visualization
        dot_content = scheduler.graph.to_dot()
        dot_file = project_root / "examples" / "resnet18_graph.dot"
        with open(dot_file, "w") as f:
            f.write(dot_content)
        print(f"\nGraph exported to: {dot_file}")
        print("Visualize with: dot -Tpng resnet18_graph.dot -o resnet18_graph.png")

    except Exception as e:
        print(f"Note: Full ResNet capture may fail with FX. Error: {e}")
        print("Using hooks fallback for analysis...")

        graph = capture_graph(model, sample_input, strategy="hooks")
        print(f"\nResNet-18 (hooks capture):")
        print(f"  Tasks: {graph.num_tasks}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Kitsune Week 2 - Graph Capture & Scheduling Demo")
    print("=" * 60)

    # Basic graph capture
    demo_graph_capture()

    # Parallelism analysis
    demo_parallel_analysis()

    # Scheduler demo
    demo_scheduler()

    # ResNet analysis (if CUDA available)
    if torch.cuda.is_available():
        demo_resnet_analysis()
    else:
        print("\n[Skipping ResNet analysis - CUDA not available]")

    print("\n" + "=" * 60)
    print("Week 2 Demo Complete!")
    print("=" * 60)
    print("\nNext: Week 3 - CUDA Stream Parallelism")


if __name__ == "__main__":
    main()
