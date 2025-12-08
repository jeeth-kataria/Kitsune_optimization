"""
Basic Usage Example

Demonstrates how to use Kitsune's profiling tools and run baseline benchmarks.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

# Import Kitsune components
import kitsune
from kitsune.profiler import Profiler, CUDATimer, MemoryTracker
from tests.benchmarks import run_baseline_benchmark, BenchmarkConfig
from tests.benchmarks.models import create_mlp, create_lenet, get_model_info


def main():
    """Run basic Kitsune demonstration."""

    print("=" * 60)
    print("Kitsune - CUDA-Accelerated Dataflow Scheduler")
    print("=" * 60)

    # Check environment
    print("\n--- Environment Check ---")
    info = kitsune.get_device_info()
    print(f"Kitsune version: {info['kitsune_version']}")
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA available:  {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA version:    {info['cuda_version']}")
        for device in info['devices']:
            print(f"GPU {device['index']}: {device['name']}")
            print(f"  Memory: {device['total_memory_gb']:.1f} GB")
            print(f"  Compute: SM {device['compute_capability']}")

    # Check compatibility
    is_compatible, warnings = kitsune.check_compatibility()
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if not info['cuda_available']:
        print("\nCUDA not available. Skipping GPU benchmarks.")
        return

    # Demo 1: Profiler usage
    print("\n--- Profiler Demo ---")
    device = "cuda"

    model = create_mlp(device=device)
    model_info = get_model_info(model)
    print(f"Model: {model_info['name']}")
    print(f"Parameters: {model_info['total_params']:,} ({model_info['total_params_mb']:.2f} MB)")

    profiler = Profiler(device=0)
    x = torch.randn(64, 784, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Profile a few iterations
    print("\nProfiling 10 training iterations...")
    for i in range(10):
        profiler.start_iteration(i, batch_size=64)

        with profiler.profile("forward"):
            output = model(x)
            loss = criterion(output, y)

        with profiler.profile("backward"):
            optimizer.zero_grad()
            loss.backward()

        with profiler.profile("optimizer"):
            optimizer.step()

        profiler.end_iteration(loss=loss.item())

    print(profiler.summary())

    # Demo 2: Baseline benchmark
    print("\n--- Baseline Benchmark ---")

    config = BenchmarkConfig(
        batch_size=64,
        num_iterations=50,
        warmup_iterations=10,
        input_size=(784,),  # Flattened MNIST
        num_classes=10,
    )

    # MLP benchmark
    print("\nRunning MLP benchmark...")
    mlp = create_mlp(device=device)
    mlp_result = run_baseline_benchmark(mlp, config)
    print(mlp_result.summary())

    # LeNet benchmark
    print("\nRunning LeNet benchmark...")
    lenet_config = BenchmarkConfig(
        batch_size=64,
        num_iterations=50,
        warmup_iterations=10,
        input_size=(1, 28, 28),  # MNIST images
        num_classes=10,
    )
    lenet = create_lenet(device=device)
    lenet_result = run_baseline_benchmark(lenet, lenet_config)
    print(lenet_result.summary())

    print("\n--- Demo Complete ---")
    print("Baseline measurements established. Ready for optimization!")


if __name__ == "__main__":
    main()
