"""
🍎 Kitsune Apple Silicon Test

Run locally on your Mac:
    python benchmarks/platform_tests/test_apple.py

Expected: 2-5x speedup with MPS over CPU
"""

import sys
import time
import platform

sys.path.insert(0, '/Users/jeethkataria/Kitsune/KITSUNE_ALGO')

import torch
import torch.nn as nn
import torchvision.models as models

print("=" * 60)
print("🍎 Kitsune Apple Silicon Benchmark")
print("=" * 60)

# Check platform
assert platform.system() == "Darwin", "This test is for macOS only"

# Check MPS
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f"\nPyTorch: {torch.__version__}")
print(f"MPS available: {mps_available}")

# Import kitsune
import kitsune
from kitsune.backends.apple_optimizer import (
    AppleSiliconOptimizer,
    detect_apple_chip,
    MPSOptimizationLevel,
)

# Detect chip
chip = detect_apple_chip()
print(f"\nChip: {chip.chip_type.value.upper()}")
print(f"CPU cores: {chip.cpu_cores}")
print(f"GPU cores: {chip.gpu_cores}")
print(f"Neural Engine: {chip.neural_engine_tops} TOPS")

# Configuration
batch_size = 16  # Smaller batch for Apple Silicon
models_to_test = [
    ("ResNet-18", lambda: models.resnet18(weights=None)),
    ("ResNet-50", lambda: models.resnet50(weights=None)),
    ("MobileNetV3", lambda: models.mobilenet_v3_small(weights=None)),
]

def benchmark(model, x, name, iterations=50, warmup=10):
    """Benchmark with proper MPS sync."""
    model.eval()
    device = x.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'mps':
                torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return sorted(times)[len(times) // 2]

# ============================================
# Test Suite
# ============================================

print("\n" + "=" * 60)
print("📊 Running Apple Silicon Tests")
print("=" * 60)

all_results = {}

for model_name, model_fn in models_to_test:
    print(f"\n{'='*60}")
    print(f"🎯 {model_name}")
    print("=" * 60)
    
    results = {}
    x_cpu = torch.randn(batch_size, 3, 224, 224)
    
    # CPU Baseline
    print("\n  📍 CPU Baseline")
    model_cpu = model_fn().eval()
    cpu_time = benchmark(model_cpu, x_cpu, "CPU")
    results["CPU"] = cpu_time
    print(f"     Result: {cpu_time:.2f} ms")
    
    # CPU + JIT
    print("\n  📍 CPU + JIT")
    with torch.no_grad():
        traced_cpu = torch.jit.trace(model_cpu, x_cpu)
        traced_cpu = torch.jit.optimize_for_inference(traced_cpu)
    jit_cpu_time = benchmark(traced_cpu, x_cpu, "CPU+JIT")
    results["CPU+JIT"] = jit_cpu_time
    print(f"     Result: {jit_cpu_time:.2f} ms ({cpu_time/jit_cpu_time:.2f}x)")
    
    if mps_available:
        # MPS
        print("\n  📍 MPS Backend")
        x_mps = x_cpu.to("mps")
        model_mps = model_fn().to("mps").eval()
        mps_time = benchmark(model_mps, x_mps, "MPS")
        results["MPS"] = mps_time
        print(f"     Result: {mps_time:.2f} ms ({cpu_time/mps_time:.2f}x vs CPU)")
        
        # MPS + Channels Last
        print("\n  📍 MPS + Channels Last")
        model_mps_cl = model_fn().to("mps").to(memory_format=torch.channels_last).eval()
        x_mps_cl = x_mps.to(memory_format=torch.channels_last)
        mps_cl_time = benchmark(model_mps_cl, x_mps_cl, "MPS+CL")
        results["MPS+ChannelsLast"] = mps_cl_time
        print(f"     Result: {mps_cl_time:.2f} ms ({cpu_time/mps_cl_time:.2f}x vs CPU)")
        
        # MPS + torch.compile (if available)
        if hasattr(torch, "compile"):
            print("\n  📍 MPS + torch.compile")
            try:
                model_compile = model_fn().to("mps").eval()
                compiled = torch.compile(model_compile, backend="aot_eager")
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = compiled(x_mps)
                torch.mps.synchronize()
                
                compile_time = benchmark(compiled, x_mps, "compile")
                results["MPS+compile"] = compile_time
                print(f"     Result: {compile_time:.2f} ms ({cpu_time/compile_time:.2f}x vs CPU)")
            except Exception as e:
                print(f"     Skipped: {e}")
        
        # Using AppleSiliconOptimizer
        print("\n  📍 Kitsune AppleSiliconOptimizer")
        optimizer = AppleSiliconOptimizer()
        model_fresh = model_fn()
        result = optimizer.optimize(model_fresh, x_cpu, level=MPSOptimizationLevel.ENHANCED)
        
        x_opt = x_cpu.to(result.device)
        opt_time = benchmark(result.model, x_opt, "Kitsune")
        results["Kitsune"] = opt_time
        print(f"     Result: {opt_time:.2f} ms ({cpu_time/opt_time:.2f}x vs CPU)")
        print(f"     Applied: {', '.join(result.optimizations_applied)}")
    
    # Summary for this model
    print(f"\n  📊 {model_name} Summary:")
    best = min(results, key=results.get)
    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = cpu_time / time_ms
        marker = " 🏆" if name == best else ""
        print(f"     {name:<20} {time_ms:>8.2f} ms  ({speedup:.2f}x){marker}")
    
    all_results[model_name] = results

# ============================================
# Final Summary
# ============================================

print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)

print(f"\n{'Model':<15} {'CPU':<10} {'Best':<10} {'Speedup':<10}")
print("-" * 50)

for model_name, results in all_results.items():
    cpu = results["CPU"]
    best = min(results.values())
    best_name = min(results, key=results.get)
    speedup = cpu / best
    print(f"{model_name:<15} {cpu:>8.2f} {best:>8.2f} {speedup:>8.2f}x ({best_name})")

print("\n" + "=" * 60)
print("✅ Apple Silicon test complete!")
print("=" * 60)
