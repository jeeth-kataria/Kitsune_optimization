"""
Week 3 Demo - CUDA Stream Parallelism

Demonstrates:
- Stream pool management
- Event-based synchronization
- Parallel execution of independent operations
- Performance comparison vs sequential execution
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import time

from kitsune.cuda import (
    StreamPool,
    StreamScheduler,
    EventManager,
    DependencyTracker,
    CUDAGraphCapture,
)
from kitsune.core import (
    ModelExecutor,
    ParallelForwardExecutor,
    DataflowScheduler,
)
from tests.benchmarks.models import create_mlp, create_resnet18


def demo_stream_pool():
    """Demonstrate stream pool management."""
    print("=" * 60)
    print("Stream Pool Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping stream pool demo")
        return

    # Create pool with 4 streams
    pool = StreamPool(num_streams=4)
    print(f"Created pool with {pool.num_streams} streams")

    # Execute work on different streams
    print("\nExecuting parallel work on streams...")

    results = []
    events = []

    for i in range(8):
        stream = pool.next_stream()

        def work(idx=i):
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            return y.sum().item()

        event = pool.execute_on_stream(
            stream_id=stream.stream_id,
            func=lambda idx=i: results.append((idx, stream.stream_id)),
        )
        events.append(event)

    # Wait for all
    pool.synchronize_all()

    print("Work distribution:")
    for task_id, stream_id in sorted(results):
        print(f"  Task {task_id} -> Stream {stream_id}")

    # Print stats
    print("\nStream statistics:")
    stats = pool.get_stats()
    for stream_id, stat in stats.items():
        print(f"  Stream {stream_id}: {stat.tasks_executed} tasks")


def demo_parallel_branches():
    """Demonstrate parallel branch execution."""
    print("\n" + "=" * 60)
    print("Parallel Branches Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create model with parallel branches
    class MultiHeadModel(nn.Module):
        def __init__(self, num_heads=4):
            super().__init__()
            self.stem = nn.Linear(784, 512)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                )
                for _ in range(num_heads)
            ])
            self.merge = nn.Linear(128 * num_heads, 10)

        def forward(self, x):
            x = self.stem(x)
            head_outputs = [head(x) for head in self.heads]
            merged = torch.cat(head_outputs, dim=1)
            return self.merge(merged)

    model = MultiHeadModel(num_heads=4).cuda()
    input = torch.randn(64, 784, device="cuda")

    # Warmup
    for _ in range(5):
        _ = model(input)
    torch.cuda.synchronize()

    # Benchmark sequential execution
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iterations = 100
    start.record()
    for _ in range(iterations):
        _ = model(input)
    end.record()
    end.synchronize()

    sequential_time = start.elapsed_time(end) / iterations
    print(f"\nSequential execution: {sequential_time:.3f} ms/iteration")

    # Now try parallel head execution
    executor = ParallelForwardExecutor(num_streams=4)

    def parallel_forward(x):
        stem_out = model.stem(x)

        # Execute heads in parallel
        head_outputs = executor.execute_branches(stem_out, list(model.heads))

        merged = torch.cat(head_outputs, dim=1)
        return model.merge(merged)

    # Warmup parallel
    for _ in range(5):
        _ = parallel_forward(input)
    torch.cuda.synchronize()

    # Benchmark parallel
    start.record()
    for _ in range(iterations):
        _ = parallel_forward(input)
    end.record()
    end.synchronize()

    parallel_time = start.elapsed_time(end) / iterations
    print(f"Parallel execution: {parallel_time:.3f} ms/iteration")

    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.0:
        print("Parallel execution is faster!")
    else:
        print("Note: Parallel overhead may exceed gains for small workloads")


def demo_cuda_events():
    """Demonstrate event-based synchronization."""
    print("\n" + "=" * 60)
    print("CUDA Events Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    manager = EventManager(enable_timing=True)
    tracker = DependencyTracker()

    # Simulate a dependency chain
    print("\nExecuting dependency chain with events...")

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    stream3 = torch.cuda.Stream()

    # Task 1 on stream1
    tracker.register_task(1, [])
    with torch.cuda.stream(stream1):
        manager.start_timing("task1")
        x = torch.randn(2000, 2000, device="cuda")
        y = torch.matmul(x, x)
        manager.end_timing("task1")
        tracker.mark_complete(1, stream1)

    # Task 2 on stream2 (depends on task 1)
    tracker.register_task(2, [1])
    with torch.cuda.stream(stream2):
        tracker.wait_for_dependencies(2, stream2)
        manager.start_timing("task2")
        z = torch.matmul(y, x)
        manager.end_timing("task2")
        tracker.mark_complete(2, stream2)

    # Task 3 on stream3 (depends on task 1)
    tracker.register_task(3, [1])
    with torch.cuda.stream(stream3):
        tracker.wait_for_dependencies(3, stream3)
        manager.start_timing("task3")
        w = torch.matmul(y, y)
        manager.end_timing("task3")
        tracker.mark_complete(3, stream3)

    # Synchronize
    stream2.synchronize()
    stream3.synchronize()

    # Print timings
    print("\nTask timings:")
    for name in ["task1", "task2", "task3"]:
        elapsed = manager.get_timing(name, synchronize=False)
        print(f"  {name}: {elapsed:.3f} ms")


def demo_model_executor():
    """Demonstrate model executor with stream parallelism."""
    print("\n" + "=" * 60)
    print("Model Executor Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create model
    model = create_mlp(device="cuda")
    sample_input = torch.randn(64, 784, device="cuda")

    # Create executor
    print("\nCreating ModelExecutor...")
    executor = ModelExecutor(
        model=model,
        sample_input=sample_input,
        num_streams=4,
        scheduler_type="wavefront",
    )

    print(f"Graph: {executor.graph.num_tasks} tasks")
    print(f"Plan: {len(executor.plan)} steps")

    # Benchmark
    print("\nBenchmarking...")
    stats = executor.benchmark(sample_input, num_iterations=100, warmup=20)

    print(f"Average time: {stats['avg_time_ms']:.3f} ms")
    print(f"Throughput: {stats['throughput_per_sec']:.1f} samples/sec")


def demo_speedup_comparison():
    """Compare speedup for different model sizes."""
    print("\n" + "=" * 60)
    print("Speedup Comparison Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    print("\nComparing parallel vs sequential execution...\n")

    # Different workload sizes
    configs = [
        ("Small (100x100)", 100),
        ("Medium (500x500)", 500),
        ("Large (1000x1000)", 1000),
        ("XL (2000x2000)", 2000),
    ]

    pool = StreamPool(num_streams=4)

    for name, size in configs:
        # Sequential: 4 independent matmuls in series
        matrices = [torch.randn(size, size, device="cuda") for _ in range(4)]

        # Warmup
        for m in matrices:
            _ = torch.matmul(m, m)
        torch.cuda.synchronize()

        # Sequential timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            results = [torch.matmul(m, m) for m in matrices]
        end.record()
        end.synchronize()

        seq_time = start.elapsed_time(end) / 10

        # Parallel timing
        def parallel_matmuls():
            events = []
            results = [None] * 4
            for i, m in enumerate(matrices):
                stream = pool.get_stream(i)
                with stream.context():
                    results[i] = torch.matmul(m, m)
                    events.append(stream.record_event())
            for e in events:
                e.synchronize()
            return results

        # Warmup
        parallel_matmuls()

        start.record()
        for _ in range(10):
            parallel_matmuls()
        end.record()
        end.synchronize()

        par_time = start.elapsed_time(end) / 10

        speedup = seq_time / par_time
        print(f"{name:20s}: seq={seq_time:7.2f}ms, par={par_time:7.2f}ms, speedup={speedup:.2f}x")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Kitsune Week 3 - CUDA Stream Parallelism Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")

    demo_stream_pool()
    demo_cuda_events()
    demo_parallel_branches()
    demo_speedup_comparison()

    if torch.cuda.is_available():
        demo_model_executor()

    print("\n" + "=" * 60)
    print("Week 3 Demo Complete!")
    print("=" * 60)
    print("\nNext: Week 4 - Memory Pooling and MVP")


if __name__ == "__main__":
    main()
