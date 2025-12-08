"""
Week 5 Demo - Kernel Fusion

Demonstrates:
- Fusion pattern detection
- torch.compile automatic fusion
- Pre-fused operation patterns
- Fusion benchmarking

Note: Triton is only available on Linux. On Windows, we use
torch.compile as the backend for kernel fusion.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from kitsune.fusion import (
    FusionEngine,
    FusionDetector,
    FusedOperations,
    PatternMatcher,
    BUILTIN_PATTERNS,
    TRITON_AVAILABLE,
    TORCH_COMPILE_AVAILABLE,
)
from kitsune.pytorch import capture_graph
from tests.benchmarks.models import create_mlp, create_resnet18


def demo_pattern_detection():
    """Demonstrate fusion pattern detection."""
    print("=" * 60)
    print("Pattern Detection Demo")
    print("=" * 60)

    # Show built-in patterns
    print(f"\nBuilt-in fusion patterns: {len(BUILTIN_PATTERNS)}")
    for pattern in BUILTIN_PATTERNS[:8]:
        print(f"  {pattern.name}: {' -> '.join(pattern.op_sequence)} ({pattern.fusion_type.name})")

    # Detect patterns in operation sequence
    print("\n--- Pattern Matching ---")
    matcher = PatternMatcher()

    ops = ["linear", "relu", "linear", "gelu", "add", "relu"]
    print(f"\nOperation sequence: {' -> '.join(ops)}")

    matches = matcher.find_matches(ops)
    print(f"Found {len(matches)} fusion opportunities:")
    for pattern, start, end in matches:
        matched_ops = ops[start:end]
        print(f"  {pattern.name}: positions {start}-{end} ({' -> '.join(matched_ops)})")


def demo_graph_fusion_analysis():
    """Analyze fusion opportunities in computation graphs."""
    print("\n" + "=" * 60)
    print("Graph Fusion Analysis Demo")
    print("=" * 60)

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    sample_input = torch.randn(64, 784)

    # Capture graph
    print("\nCapturing computation graph...")
    graph = capture_graph(model, sample_input)
    print(f"Graph has {graph.num_tasks} tasks")

    # Detect fusion
    detector = FusionDetector()
    plan = detector.get_fusion_plan(graph)

    print(f"\nFusion Analysis:")
    print(f"  Fusion opportunities: {plan['num_fusions']}")
    print(f"  Tasks fusable: {plan['tasks_fused']}/{plan['total_tasks']}")
    print(f"  Coverage: {plan['fusion_coverage']:.1%}")
    print(f"  Estimated speedup: {plan['estimated_speedup']:.2f}x")
    print(f"  Memory saved: {plan['memory_saved_bytes'] / 1024:.1f} KB")

    print("\nFusion candidates:")
    for candidate in plan['candidates'][:5]:
        ops = " -> ".join(candidate.op_types)
        print(f"  {candidate.pattern.name}: {ops} (speedup: {candidate.estimated_speedup:.2f}x)")


def demo_fused_operations():
    """Demonstrate pre-fused operation patterns."""
    print("\n" + "=" * 60)
    print("Fused Operations Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    print("\n--- Fused Linear + Activation ---")

    # Create linear layer
    linear = nn.Linear(784, 256).cuda()

    # Create fused versions
    fused_relu = FusedOperations.linear_relu(linear.weight, linear.bias)
    fused_gelu = FusedOperations.linear_gelu(linear.weight, linear.bias)
    fused_silu = FusedOperations.linear_silu(linear.weight, linear.bias)

    # Test
    x = torch.randn(64, 784, device="cuda")

    # Verify correctness
    result_relu = fused_relu(x)
    expected_relu = torch.relu(linear(x))
    assert torch.allclose(result_relu, expected_relu, rtol=1e-4)
    print("Linear+ReLU fusion: Correct!")

    result_gelu = fused_gelu(x)
    expected_gelu = torch.nn.functional.gelu(linear(x))
    assert torch.allclose(result_gelu, expected_gelu, rtol=1e-4)
    print("Linear+GELU fusion: Correct!")

    print("\n--- Fused Add + ReLU ---")
    fused_add_relu = FusedOperations.add_relu()

    x = torch.randn(64, 256, device="cuda")
    y = torch.randn(64, 256, device="cuda")

    result = fused_add_relu(x, y)
    expected = torch.relu(x + y)
    assert torch.allclose(result, expected)
    print("Add+ReLU fusion: Correct!")

    # Benchmark the fused operations
    print("\n--- Fused Operations Benchmark ---")

    # Benchmark linear + relu (separate vs fused)
    linear_large = nn.Linear(1024, 1024).cuda()
    fused_linear_relu = FusedOperations.linear_relu(linear_large.weight, linear_large.bias)
    x_large = torch.randn(256, 1024, device="cuda")

    # Warmup
    for _ in range(20):
        _ = torch.relu(linear_large(x_large))
        _ = fused_linear_relu(x_large)
    torch.cuda.synchronize()

    # Benchmark separate
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = torch.relu(linear_large(x_large))
    end.record()
    torch.cuda.synchronize()
    separate_time = start.elapsed_time(end) / 100

    # Benchmark fused
    start.record()
    for _ in range(100):
        _ = fused_linear_relu(x_large)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / 100

    print(f"Linear+ReLU (separate): {separate_time:.3f} ms")
    print(f"Linear+ReLU (fused):    {fused_time:.3f} ms")
    print(f"Speedup: {separate_time/fused_time:.2f}x")


def demo_torch_compile():
    """Demonstrate torch.compile for automatic fusion."""
    print("\n" + "=" * 60)
    print("torch.compile Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    engine = FusionEngine()
    print(f"Backend: {engine.backend}")
    print(f"Triton available: {TRITON_AVAILABLE}")

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 10),
    ).cuda()

    sample_input = torch.randn(64, 784, device="cuda")

    # Optimize model
    print("\nOptimizing model with torch.compile...")
    optimized = engine.optimize_model(model, sample_input)

    # Verify correctness
    with torch.no_grad():
        original_out = model(sample_input)
        optimized_out = optimized(sample_input)

    assert torch.allclose(original_out, optimized_out, rtol=1e-4)
    print("Model optimization: Correct!")


def demo_fusion_benchmark():
    """Benchmark fusion speedup."""
    print("\n" + "=" * 60)
    print("Fusion Benchmark Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    engine = FusionEngine()

    print("\n--- Elementwise Chain Fusion ---")

    # Original function with multiple ops
    def original_chain(x):
        x = x * 2
        x = x + 0.5
        x = torch.relu(x)
        x = x * 1.5
        return x

    # Compiled version
    compiled_chain = engine.compile_function(original_chain, "chain")

    x = torch.randn(1000, 1000, device="cuda")

    results = engine.benchmark_fusion(
        original_chain, compiled_chain, x,
        num_iterations=100,
        warmup=20,
    )

    print(f"Original: {results['original_ms']:.3f} ms")
    print(f"Fused: {results['fused_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")

    print("\n--- Linear + Activation Fusion ---")

    linear = nn.Linear(1024, 1024).cuda()

    def original_linear(x):
        return torch.relu(linear(x))

    compiled_linear = engine.compile_function(original_linear, "linear_relu")

    x = torch.randn(256, 1024, device="cuda")

    results = engine.benchmark_fusion(
        original_linear, compiled_linear, x,
        num_iterations=100,
        warmup=20,
    )

    print(f"Original: {results['original_ms']:.3f} ms")
    print(f"Fused: {results['fused_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")


def demo_model_optimization():
    """Demonstrate full model optimization with fusion."""
    print("\n" + "=" * 60)
    print("Model Optimization Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    engine = FusionEngine()

    # Test different models
    models = [
        ("MLP", create_mlp, (64, 784)),
        ("ResNet-18", create_resnet18, (8, 3, 32, 32)),
    ]

    print("\n{:<15} {:>12} {:>12} {:>10}".format(
        "Model", "Original", "Fused", "Speedup"
    ))
    print("-" * 52)

    for name, create_fn, input_shape in models:
        try:
            model = create_fn(device="cuda")
            sample_input = torch.randn(*input_shape, device="cuda")

            # Get original timing
            def original_fn(x):
                with torch.no_grad():
                    return model(x)

            # Optimize model
            optimized = engine.optimize_model(model, sample_input)

            def optimized_fn(x):
                with torch.no_grad():
                    return optimized(x)

            # Benchmark
            results = engine.benchmark_fusion(
                original_fn, optimized_fn, sample_input,
                num_iterations=100,
                warmup=20,
            )

            print("{:<15} {:>10.3f}ms {:>10.3f}ms {:>9.2f}x".format(
                name,
                results['original_ms'],
                results['fused_ms'],
                results['speedup'],
            ))
        except Exception as e:
            print(f"{name:<15} Error: {e}")


def demo_custom_fusion():
    """Demonstrate custom fusion patterns."""
    print("\n" + "=" * 60)
    print("Custom Fusion Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    engine = FusionEngine()

    print("\n--- Custom Fused Function ---")

    # Define a custom computation
    def custom_activation(x, alpha=0.1):
        """Leaky GELU: GELU with leaky negative slope."""
        gelu = torch.nn.functional.gelu(x)
        return torch.where(x > 0, gelu, alpha * x)

    # Compile it
    compiled_custom = engine.compile_function(custom_activation, "leaky_gelu")

    x = torch.randn(1000, 1000, device="cuda")

    results = engine.benchmark_fusion(
        custom_activation, compiled_custom, x,
        num_iterations=100,
        warmup=20,
    )

    print(f"Custom Leaky GELU:")
    print(f"  Original: {results['original_ms']:.3f} ms")
    print(f"  Compiled: {results['fused_ms']:.3f} ms")
    print(f"  Speedup: {results['speedup']:.2f}x")


def main():
    """Run all Week 5 demos."""
    print("=" * 60)
    print("Kitsune Week 5 - Kernel Fusion Demo")
    print("=" * 60)

    print(f"\nTriton available: {TRITON_AVAILABLE}")
    print(f"torch.compile available: {TORCH_COMPILE_AVAILABLE}")
    if not TRITON_AVAILABLE and not TORCH_COMPILE_AVAILABLE:
        print("(Using torch.jit fallback and pre-fused operations)")
    elif not TRITON_AVAILABLE:
        print("(Triton only available on Linux, using torch.compile)")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    demo_pattern_detection()
    demo_graph_fusion_analysis()
    demo_fused_operations()
    demo_torch_compile()
    demo_fusion_benchmark()
    demo_model_optimization()
    demo_custom_fusion()

    print("\n" + "=" * 60)
    print("Week 5 Demo Complete!")
    print("=" * 60)
    print("\nKernel fusion benefits:")
    print("  - Reduced kernel launch overhead")
    print("  - Better memory locality (intermediate results in registers)")
    print("  - Lower memory bandwidth usage")
    print("  - Automatic optimization via torch.compile")
    print("\nNext: Week 6 - AMP Integration and Fine-tuning")


if __name__ == "__main__":
    main()
