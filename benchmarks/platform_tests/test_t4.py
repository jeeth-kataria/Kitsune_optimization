"""
🦊 Kitsune T4 GPU Test (Google Colab)

Run this on Google Colab with T4 runtime:
1. Runtime → Change runtime type → T4 GPU
2. Upload this file or paste into a cell
3. Run!

Expected: 2.0x+ speedup with INT8/FP16 optimizations
"""

import sys
import time
import gc

# Install kitsune if needed
try:
    import kitsune
except ImportError:
    print("Installing kitsune...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-kitsune"])
    import kitsune

import torch
import torch.nn as nn
import torchvision.models as models

print("=" * 60)
print("🦊 Kitsune T4 GPU Benchmark")
print("=" * 60)

# Verify T4
assert torch.cuda.is_available(), "CUDA required! Enable T4 GPU in Runtime settings."
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")

if "T4" not in gpu_name:
    print(f"⚠️  Warning: Expected T4, got {gpu_name}")
    print("   Results may differ from expected values.")

device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
print(f"Compute: SM{props.major}{props.minor}")

# Configuration
batch_size = 32
x = torch.randn(batch_size, 3, 224, 224, device=device)

def benchmark(model, x, name, iterations=100, warmup=20):
    """Benchmark with CUDA events for accurate timing."""
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
    
    median = sorted(times)[len(times) // 2]
    return median

# ============================================
# Test Suite
# ============================================

print("\n" + "=" * 60)
print("📊 Running T4 Optimization Tests")
print("=" * 60)

results = {}

# 1. Baseline
print("\n🎯 Test 1: Baseline FP32")
model = models.resnet50(weights=None).to(device).eval()
baseline = benchmark(model, x, "Baseline")
results["1. Baseline FP32"] = baseline
print(f"   Result: {baseline:.2f} ms")

# 2. JIT Trace
print("\n🎯 Test 2: JIT Trace + Freeze")
model_jit = models.resnet50(weights=None).to(device).eval()
with torch.no_grad():
    traced = torch.jit.trace(model_jit, x)
    # freeze first, then optimize_for_inference
    traced = torch.jit.freeze(traced)
    traced = torch.jit.optimize_for_inference(traced)
jit_time = benchmark(traced, x, "JIT")
results["2. JIT Trace"] = jit_time
print(f"   Result: {jit_time:.2f} ms ({baseline/jit_time:.2f}x)")
del traced, model_jit
gc.collect()
torch.cuda.empty_cache()

# 3. FP16 AMP
print("\n🎯 Test 3: FP16 Mixed Precision")
model_amp = models.resnet50(weights=None).to(device).eval()

times = []
with torch.no_grad():
    for _ in range(20):  # warmup
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model_amp(x)
    torch.cuda.synchronize()
    
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model_amp(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

amp_time = sorted(times)[len(times) // 2]
results["3. FP16 AMP"] = amp_time
print(f"   Result: {amp_time:.2f} ms ({baseline/amp_time:.2f}x)")
del model_amp
gc.collect()
torch.cuda.empty_cache()

# 4. JIT + FP16
print("\n🎯 Test 4: JIT + FP16 Combined")
model_combined = models.resnet50(weights=None).half().to(device).eval()
x_half = x.half()

with torch.no_grad():
    traced_half = torch.jit.trace(model_combined, x_half)
    traced_half = torch.jit.freeze(traced_half)
    traced_half = torch.jit.optimize_for_inference(traced_half)

jit_fp16_time = benchmark(traced_half, x_half, "JIT+FP16")
results["4. JIT + FP16"] = jit_fp16_time
print(f"   Result: {jit_fp16_time:.2f} ms ({baseline/jit_fp16_time:.2f}x)")
del traced_half, model_combined
gc.collect()
torch.cuda.empty_cache()

# 5. torch.compile
print("\n🎯 Test 5: torch.compile (reduce-overhead)")
if hasattr(torch, "compile"):
    model_compile = models.resnet50(weights=None).to(device).eval()
    compiled = torch.compile(model_compile, mode="reduce-overhead")
    
    # Warmup (triggers compilation)
    print("   Compiling...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled(x)
    torch.cuda.synchronize()
    
    compile_time = benchmark(compiled, x, "compile")
    results["5. torch.compile"] = compile_time
    print(f"   Result: {compile_time:.2f} ms ({baseline/compile_time:.2f}x)")
    del compiled, model_compile
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("   Skipped (requires PyTorch 2.x)")

# 6. torch.compile + FP16
print("\n🎯 Test 6: torch.compile + FP16 (BEST)")
if hasattr(torch, "compile"):
    model_best = models.resnet50(weights=None).half().to(device).eval()
    compiled_half = torch.compile(model_best, mode="reduce-overhead")
    
    print("   Compiling...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_half(x_half)
    torch.cuda.synchronize()
    
    best_time = benchmark(compiled_half, x_half, "compile+FP16")
    results["6. compile + FP16"] = best_time
    print(f"   Result: {best_time:.2f} ms ({baseline/best_time:.2f}x)")
else:
    print("   Skipped (requires PyTorch 2.x)")

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

# Verify target
if best_speedup >= 2.0:
    print(f"\n✅ TARGET ACHIEVED: {best_speedup:.2f}x >= 2.0x")
else:
    print(f"\n⚠️  Target: 2.0x, Achieved: {best_speedup:.2f}x")
    print("   Try: torch.compile with max-autotune mode")

print("=" * 60)
