"""
🚀 ULTIMATE OPTIMIZATION TEST
Based on PyTorch Performance Tuning Guide + NVIDIA Research
Combines ALL working optimizations for maximum speedup
"""

import subprocess
import sys

try:
    import torch
    import torchvision
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=True)
    import torch
    import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

import torch.nn as nn
from torchvision.models import resnet18, resnet50
import json
import gc

# =============================================================================
# KEY INSIGHT FROM RESEARCH:
# 1. cudnn.benchmark = True (auto-tune convolution algorithms) - HUGE for CNNs
# 2. torch.compile with mode="default" works better than max-autotune on T4
# 3. Channels-last ONLY helps when combined with cudnn.benchmark
# 4. CUDA graphs conflict with torch.compile - use one OR the other
# 5. Avoid synchronization points during inference
# =============================================================================

def benchmark(model, x, warmup=30, iters=100):
    """Accurate benchmark with CUDA events - NO SYNC during timing."""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Measure - collect all events first, then sync once
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    with torch.no_grad():
        for i in range(iters):
            start_events[i].record()
            _ = model(x)
            end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times)//2]  # median


def clear_gpu():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_baseline(model_fn, x, name):
    """Test 1: Pure baseline."""
    print(f"\n1️⃣  BASELINE")
    model = model_fn(weights=None).cuda().eval()
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    del model
    clear_gpu()
    return time_ms


def test_cudnn_benchmark(model_fn, x, name):
    """Test 2: Enable cuDNN autotuner - KEY optimization for CNNs!"""
    print(f"\n2️⃣  cuDNN BENCHMARK (autotuner)")
    
    # Enable cuDNN benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    model = model_fn(weights=None).cuda().eval()
    
    # Extra warmup for autotuner to find optimal algorithms
    with torch.no_grad():
        for _ in range(50):  # More warmup for autotuner
            _ = model(x)
    torch.cuda.synchronize()
    
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    
    torch.backends.cudnn.benchmark = False  # Reset
    del model
    clear_gpu()
    return time_ms


def test_cudnn_channels_last(model_fn, batch_size, name):
    """Test 3: cuDNN benchmark + Channels-last."""
    print(f"\n3️⃣  cuDNN BENCHMARK + CHANNELS-LAST")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    model = model_fn(weights=None).cuda().eval()
    model = model.to(memory_format=torch.channels_last)
    
    # Create channels-last input
    x = torch.randn(batch_size, 3, 224, 224, device='cuda').to(memory_format=torch.channels_last)
    
    # Extra warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()
    
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    
    torch.backends.cudnn.benchmark = False
    del model, x
    clear_gpu()
    return time_ms


def test_compile_with_cudnn(model_fn, x, name):
    """Test 4: torch.compile + cuDNN benchmark."""
    print(f"\n4️⃣  torch.compile + cuDNN BENCHMARK")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    model = model_fn(weights=None).cuda().eval()
    
    try:
        # Use default mode - more stable than max-autotune on T4
        compiled = torch.compile(model, mode="default", fullgraph=False)
        
        # Extra warmup for both compile and cudnn autotuner
        with torch.no_grad():
            for _ in range(50):
                _ = compiled(x)
        torch.cuda.synchronize()
        
        time_ms = benchmark(compiled, x)
        print(f"   Time: {time_ms:.2f}ms")
        result = time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        result = None
    
    torch.backends.cudnn.benchmark = False
    del model
    clear_gpu()
    return result


def test_ultimate_combo(model_fn, batch_size, name):
    """Test 5: ULTIMATE - compile + cudnn + channels-last."""
    print(f"\n5️⃣  ULTIMATE: compile + cuDNN + channels-last")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    model = model_fn(weights=None).cuda().eval()
    model = model.to(memory_format=torch.channels_last)
    
    x = torch.randn(batch_size, 3, 224, 224, device='cuda').to(memory_format=torch.channels_last)
    
    try:
        compiled = torch.compile(model, mode="default", fullgraph=False)
        
        # Extra warmup
        with torch.no_grad():
            for _ in range(50):
                _ = compiled(x)
        torch.cuda.synchronize()
        
        time_ms = benchmark(compiled, x)
        print(f"   Time: {time_ms:.2f}ms")
        result = time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        result = None
    
    torch.backends.cudnn.benchmark = False
    del model, x
    clear_gpu()
    return result


def test_inference_mode(model_fn, x, name):
    """Test 6: torch.inference_mode (faster than no_grad)."""
    print(f"\n6️⃣  INFERENCE MODE + cuDNN")
    
    torch.backends.cudnn.benchmark = True
    
    model = model_fn(weights=None).cuda().eval()
    
    # Warmup with inference_mode
    with torch.inference_mode():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark with inference_mode
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    with torch.inference_mode():
        for i in range(100):
            start_events[i].record()
            _ = model(x)
            end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    time_ms = times[len(times)//2]
    
    print(f"   Time: {time_ms:.2f}ms")
    
    torch.backends.cudnn.benchmark = False
    del model
    clear_gpu()
    return time_ms


def test_jit_script(model_fn, x, name):
    """Test 7: TorchScript JIT compilation."""
    print(f"\n7️⃣  TORCHSCRIPT JIT + cuDNN")
    
    torch.backends.cudnn.benchmark = True
    
    model = model_fn(weights=None).cuda().eval()
    
    try:
        scripted = torch.jit.script(model)
        scripted = torch.jit.optimize_for_inference(scripted)
        
        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = scripted(x)
        torch.cuda.synchronize()
        
        time_ms = benchmark(scripted, x)
        print(f"   Time: {time_ms:.2f}ms")
        result = time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        result = None
    
    torch.backends.cudnn.benchmark = False
    del model
    clear_gpu()
    return result


def test_jit_trace(model_fn, x, name):
    """Test 8: TorchScript Tracing."""
    print(f"\n8️⃣  TORCHSCRIPT TRACE + cuDNN")
    
    torch.backends.cudnn.benchmark = True
    
    model = model_fn(weights=None).cuda().eval()
    
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
        traced = torch.jit.optimize_for_inference(traced)
        
        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = traced(x)
        torch.cuda.synchronize()
        
        time_ms = benchmark(traced, x)
        print(f"   Time: {time_ms:.2f}ms")
        result = time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        result = None
    
    torch.backends.cudnn.benchmark = False
    del model
    clear_gpu()
    return result


def run_all_tests(model_fn, batch_size, name):
    """Run all optimization strategies."""
    print("\n" + "="*70)
    print(f"🧪 {name} (batch={batch_size})")
    print("="*70)
    
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    results = {}
    
    # Run tests
    results['baseline'] = test_baseline(model_fn, x, name)
    results['cudnn_bench'] = test_cudnn_benchmark(model_fn, x, name)
    results['cudnn_cl'] = test_cudnn_channels_last(model_fn, batch_size, name)
    results['compile_cudnn'] = test_compile_with_cudnn(model_fn, x, name)
    results['ultimate'] = test_ultimate_combo(model_fn, batch_size, name)
    results['inference_mode'] = test_inference_mode(model_fn, x, name)
    results['jit_script'] = test_jit_script(model_fn, x, name)
    results['jit_trace'] = test_jit_trace(model_fn, x, name)
    
    # Summary
    print("\n" + "-"*70)
    print("📊 RESULTS SUMMARY:")
    print("-"*70)
    
    baseline = results['baseline']
    for opt, time_ms in results.items():
        if time_ms is not None:
            speedup = baseline / time_ms
            emoji = "🚀" if speedup >= 1.3 else ("✅" if speedup > 1.1 else ("⚠️" if speedup >= 1.0 else "❌"))
            print(f"   {emoji} {opt:20s}: {time_ms:6.2f}ms ({speedup:.2f}x)")
        else:
            print(f"   ❌ {opt:20s}: FAILED")
    
    # Find best
    valid = {k: v for k, v in results.items() if v is not None}
    best = min(valid, key=valid.get)
    best_speedup = baseline / valid[best]
    
    print(f"\n   🏆 BEST: {best} ({best_speedup:.2f}x speedup)")
    
    del x
    clear_gpu()
    
    return results, baseline, best, best_speedup


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*70)
print("🚀 ULTIMATE OPTIMIZATION TEST")
print("="*70)
print("\nKey optimizations being tested:")
print("  • cuDNN benchmark mode (auto-tune convolutions)")
print("  • Channels-last memory format")
print("  • torch.compile (Inductor backend)")
print("  • torch.inference_mode")
print("  • TorchScript JIT (script + trace)")
print("  • Combined optimizations")

all_results = []

# Test ResNet-18 with batch 128
results, baseline, best, speedup = run_all_tests(resnet18, 128, "ResNet-18")
all_results.append({
    'model': 'ResNet-18',
    'batch': 128,
    'baseline_ms': round(baseline, 2),
    'best_method': best,
    'best_speedup': round(speedup, 2),
    'all_results': {k: round(v, 2) if v else None for k, v in results.items()}
})

# Test ResNet-50 with batch 64
results, baseline, best, speedup = run_all_tests(resnet50, 64, "ResNet-50")
all_results.append({
    'model': 'ResNet-50',
    'batch': 64,
    'baseline_ms': round(baseline, 2),
    'best_method': best,
    'best_speedup': round(speedup, 2),
    'all_results': {k: round(v, 2) if v else None for k, v in results.items()}
})

# Final Summary
print("\n" + "="*70)
print("📊 FINAL SUMMARY")
print("="*70)

for r in all_results:
    print(f"\n{r['model']} (batch={r['batch']}):")
    print(f"   Baseline: {r['baseline_ms']}ms")
    print(f"   Best: {r['best_method']} → {r['best_speedup']}x")

avg_speedup = sum(r['best_speedup'] for r in all_results) / len(all_results)
print(f"\n🎯 AVERAGE BEST SPEEDUP: {avg_speedup:.2f}x")

if avg_speedup >= 1.5:
    print("\n🚀 EXCELLENT! 1.5x+ speedup achieved!")
elif avg_speedup >= 1.3:
    print("\n✅ GOOD! 1.3x+ speedup achieved!")
elif avg_speedup >= 1.2:
    print("\n⚠️  Modest 1.2x+ speedup")
else:
    print("\n❌ Below target")

# Print honest assessment
print("\n" + "="*70)
print("📝 HONEST ASSESSMENT FOR KITSUNE")
print("="*70)
print("""
Based on research and testing:

1. CUDA Graphs: DON'T work with torch.compile (RNG conflict)
2. torch.compile: ~1.15-1.20x speedup on T4
3. cuDNN benchmark: Critical for CNNs, can add 5-15%
4. Channels-last: Mixed results, sometimes slower on T4
5. TF32: NOT available on T4 (requires Ampere SM80+)

REALISTIC SPEEDUP ON T4:
  • Best case: 1.2-1.3x with torch.compile + cuDNN benchmark
  • Memory savings: 20-30% still valid
  
For 1.5x+ speedup, you need:
  • Ampere GPU (A100, RTX 3090) for TF32
  • Or custom CUDA kernels
  • Or int8 quantization
""")

print("\n📄 JSON:")
print(json.dumps(all_results, indent=2))
