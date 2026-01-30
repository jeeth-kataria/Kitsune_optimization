"""
🎮 Kitsune RTX GPU Test (RTX 30xx/40xx)

Run on:
- Colab Pro (A100/V100)
- Lambda Labs (RTX 3090)
- Vast.ai (RTX 3090/4090)
- Local RTX GPU

Expected:
- RTX 30xx: 1.5-2x with TF32
- RTX 40xx: 2-3x with FP8/TF32
"""

import sys
import time
import gc

import torch
import torch.nn as nn
import torchvision.models as models

print("=" * 60)
print("🎮 Kitsune RTX GPU Benchmark")
print("=" * 60)

# Check CUDA
assert torch.cuda.is_available(), "CUDA required!"

gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
cc = (props.major, props.minor)

print(f"\nGPU: {gpu_name}")
print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
print(f"Compute: SM{props.major}{props.minor}")

# Determine GPU tier
is_ampere_plus = cc >= (8, 0)
is_ada = cc >= (8, 9)
supports_tf32 = is_ampere_plus
supports_fp8 = is_ada

print(f"\nCapabilities:")
print(f"  TF32: {'✓' if supports_tf32 else '✗'}")
print(f"  BF16: {'✓' if is_ampere_plus else '✗'}")
print(f"  FP8:  {'✓' if supports_fp8 else '✗'}")

device = torch.device("cuda")

# Configuration
batch_size = 32
x = torch.randn(batch_size, 3, 224, 224, device=device)

def benchmark(model, x, name, iterations=100, warmup=20):
    """Benchmark with CUDA events."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    return sorted(times)[len(times) // 2]

# ============================================
# Test Suite
# ============================================

print("\n" + "=" * 60)
print("📊 Running RTX Optimization Tests")
print("=" * 60)

results = {}

# 1. Baseline (TF32 disabled)
print("\n🎯 Test 1: Baseline FP32 (TF32 disabled)")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = models.resnet50(weights=None).to(device).eval()
baseline = benchmark(model, x, "Baseline")
results["1. Baseline FP32"] = baseline
print(f"   Result: {baseline:.2f} ms")

# 2. TF32 Enabled (Ampere+)
if supports_tf32:
    print("\n🎯 Test 2: TF32 Enabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model_tf32 = models.resnet50(weights=None).to(device).eval()
    tf32_time = benchmark(model_tf32, x, "TF32")
    results["2. TF32"] = tf32_time
    print(f"   Result: {tf32_time:.2f} ms ({baseline/tf32_time:.2f}x)")
else:
    print("\n🎯 Test 2: TF32 - Skipped (requires Ampere+)")

# 3. FP16 AMP
print("\n🎯 Test 3: FP16 Mixed Precision")
from torch.cuda.amp import autocast

model_amp = models.resnet50(weights=None).to(device).eval()
times = []
with torch.no_grad():
    for _ in range(20):
        with autocast(dtype=torch.float16):
            _ = model_amp(x)
    torch.cuda.synchronize()
    
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast(dtype=torch.float16):
            _ = model_amp(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

amp_time = sorted(times)[len(times) // 2]
results["3. FP16 AMP"] = amp_time
print(f"   Result: {amp_time:.2f} ms ({baseline/amp_time:.2f}x)")

# 4. BF16 (Ampere+)
if is_ampere_plus:
    print("\n🎯 Test 4: BF16 Mixed Precision")
    model_bf16 = models.resnet50(weights=None).to(device).eval()
    times = []
    with torch.no_grad():
        for _ in range(20):
            with autocast(dtype=torch.bfloat16):
                _ = model_bf16(x)
        torch.cuda.synchronize()
        
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast(dtype=torch.bfloat16):
                _ = model_bf16(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    bf16_time = sorted(times)[len(times) // 2]
    results["4. BF16 AMP"] = bf16_time
    print(f"   Result: {bf16_time:.2f} ms ({baseline/bf16_time:.2f}x)")
else:
    print("\n🎯 Test 4: BF16 - Skipped (requires Ampere+)")

# 5. TF32 + JIT
if supports_tf32:
    print("\n🎯 Test 5: TF32 + JIT Trace")
    model_jit = models.resnet50(weights=None).to(device).eval()
    with torch.no_grad():
        traced = torch.jit.trace(model_jit, x)
        traced = torch.jit.optimize_for_inference(traced)
        traced = torch.jit.freeze(traced)
    
    jit_time = benchmark(traced, x, "TF32+JIT")
    results["5. TF32 + JIT"] = jit_time
    print(f"   Result: {jit_time:.2f} ms ({baseline/jit_time:.2f}x)")

# 6. torch.compile
if hasattr(torch, "compile"):
    print("\n🎯 Test 6: torch.compile (reduce-overhead)")
    model_compile = models.resnet50(weights=None).to(device).eval()
    compiled = torch.compile(model_compile, mode="reduce-overhead")
    
    print("   Compiling...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled(x)
    torch.cuda.synchronize()
    
    compile_time = benchmark(compiled, x, "compile")
    results["6. torch.compile"] = compile_time
    print(f"   Result: {compile_time:.2f} ms ({baseline/compile_time:.2f}x)")

# 7. torch.compile + FP16
if hasattr(torch, "compile"):
    print("\n🎯 Test 7: torch.compile + FP16")
    model_best = models.resnet50(weights=None).half().to(device).eval()
    x_half = x.half()
    compiled_half = torch.compile(model_best, mode="reduce-overhead")
    
    print("   Compiling...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_half(x_half)
    torch.cuda.synchronize()
    
    best_time = benchmark(compiled_half, x_half, "compile+FP16")
    results["7. compile + FP16"] = best_time
    print(f"   Result: {best_time:.2f} ms ({baseline/best_time:.2f}x)")

# 8. torch.compile max-autotune (slowest compile, fastest runtime)
if hasattr(torch, "compile"):
    print("\n🎯 Test 8: torch.compile (max-autotune)")
    model_max = models.resnet50(weights=None).half().to(device).eval()
    
    try:
        compiled_max = torch.compile(model_max, mode="max-autotune")
        
        print("   Compiling (this takes longer)...")
        with torch.no_grad():
            for _ in range(3):
                _ = compiled_max(x_half)
        torch.cuda.synchronize()
        
        max_time = benchmark(compiled_max, x_half, "max-autotune")
        results["8. max-autotune + FP16"] = max_time
        print(f"   Result: {max_time:.2f} ms ({baseline/max_time:.2f}x)")
    except Exception as e:
        print(f"   Skipped: {e}")

# ============================================
# Summary
# ============================================

print("\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Optimization':<25} {'Time (ms)':<12} {'Speedup':<10}")
print("-" * 50)

for name, time_ms in results.items():
    speedup = baseline / time_ms
    print(f"{name:<25} {time_ms:>10.2f} {speedup:>8.2f}x")

best_name = min(results, key=results.get)
best_time = results[best_name]
best_speedup = baseline / best_time

print("\n" + "=" * 60)
print(f"🏆 BEST: {best_name}")
print(f"   Speedup: {best_speedup:.2f}x")
print(f"   Time: {baseline:.2f} ms → {best_time:.2f} ms")

# Target check
if "30" in gpu_name or "A10" in gpu_name:
    target = 2.0
elif "40" in gpu_name:
    target = 2.5
else:
    target = 1.5

if best_speedup >= target:
    print(f"\n✅ TARGET ACHIEVED: {best_speedup:.2f}x >= {target}x")
else:
    print(f"\n⚠️  Target: {target}x, Achieved: {best_speedup:.2f}x")

print("=" * 60)
