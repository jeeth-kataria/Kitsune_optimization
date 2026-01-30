"""
🚀 ADVANCED OPTIMIZATION TEST - Targeting 2x Speedup
Tests advanced techniques:
1. INT8 Dynamic Quantization - 2-4x speedup potential
2. FP16 Mixed Precision - 1.5-2x on T4 Tensor Cores
3. Torch-TensorRT - 2-6x speedup potential  
4. ONNX Runtime - Alternative fast runtime
5. Combined strategies
"""

import subprocess
import sys

# Install dependencies
print("📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "torch", "torchvision", "onnx", "onnxruntime-gpu"], check=True)

try:
    # Optional: torch-tensorrt (may not install in all environments)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "torch-tensorrt"], check=True, capture_output=True)
    TENSORRT_AVAILABLE = True
except:
    TENSORRT_AVAILABLE = False
    print("⚠️ torch-tensorrt not available")

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
import json
import gc
import time

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

# Check T4 Tensor Core support
compute_cap = torch.cuda.get_device_capability()
HAS_TENSOR_CORES = compute_cap[0] >= 7  # Volta and newer
print(f"Tensor Cores: {'Yes' if HAS_TENSOR_CORES else 'No'}")


def benchmark(model_or_fn, x, warmup=30, iters=100, use_amp=False):
    """Accurate CUDA event timing."""
    if hasattr(model_or_fn, 'eval'):
        model_or_fn.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model_or_fn(x)
            else:
                _ = model_or_fn(x)
    torch.cuda.synchronize()
    
    # Measure
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    with torch.no_grad():
        for i in range(iters):
            start_events[i].record()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model_or_fn(x)
            else:
                _ = model_or_fn(x)
            end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times)//2]


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# =============================================================================
# OPTIMIZATION STRATEGIES
# =============================================================================

def test_baseline(model_fn, x):
    """Baseline - FP32, no optimization."""
    print("\n1️⃣  BASELINE (FP32)")
    model = model_fn(weights=None).cuda().eval()
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    del model
    clear_gpu()
    return time_ms


def test_jit_trace(model_fn, x):
    """JIT Trace - our current best."""
    print("\n2️⃣  JIT TRACE (FP32)")
    model = model_fn(weights=None).cuda().eval()
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
        traced = torch.jit.optimize_for_inference(traced)
        time_ms = benchmark(traced, x)
        print(f"   Time: {time_ms:.2f}ms")
        del model, traced
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_fp16_autocast(model_fn, x):
    """FP16 via autocast - uses Tensor Cores on T4."""
    print("\n3️⃣  FP16 AUTOCAST (Tensor Cores)")
    model = model_fn(weights=None).cuda().eval()
    x_fp32 = x.float()  # Input stays FP32, autocast handles conversion
    
    try:
        time_ms = benchmark(model, x_fp32, use_amp=True)
        print(f"   Time: {time_ms:.2f}ms")
        del model
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_fp16_model(model_fn, x):
    """Full FP16 model and input."""
    print("\n4️⃣  FULL FP16 (model + input)")
    model = model_fn(weights=None).cuda().half().eval()
    x_fp16 = x.half()
    
    try:
        time_ms = benchmark(model, x_fp16)
        print(f"   Time: {time_ms:.2f}ms")
        del model, x_fp16
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_jit_fp16(model_fn, x):
    """JIT Trace + FP16 - combining both."""
    print("\n5️⃣  JIT TRACE + FP16")
    model = model_fn(weights=None).cuda().half().eval()
    x_fp16 = x.half()
    
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, x_fp16)
        traced = torch.jit.optimize_for_inference(traced)
        time_ms = benchmark(traced, x_fp16)
        print(f"   Time: {time_ms:.2f}ms")
        del model, traced, x_fp16
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_dynamic_quantization(model_fn, x):
    """INT8 Dynamic Quantization - CPU fallback for now."""
    print("\n6️⃣  INT8 DYNAMIC QUANTIZATION")
    model = model_fn(weights=None).eval()  # CPU first
    
    try:
        # Dynamic quantization (works on CPU, limited GPU support)
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Only quantize linear layers
            dtype=torch.qint8
        )
        
        # Move to CUDA if possible, otherwise benchmark on CPU
        x_cpu = x.cpu()
        
        # CPU benchmark (quantization on GPU is limited)
        times = []
        with torch.no_grad():
            for _ in range(30):  # warmup
                _ = quantized(x_cpu)
            
            for _ in range(50):  # measure
                start = time.perf_counter()
                _ = quantized(x_cpu)
                times.append((time.perf_counter() - start) * 1000)
        
        times.sort()
        time_ms = times[len(times)//2]
        print(f"   Time: {time_ms:.2f}ms (CPU - for reference)")
        del model, quantized
        return None  # Don't include CPU time in GPU comparison
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def test_compile_fp16(model_fn, x):
    """torch.compile + FP16."""
    print("\n7️⃣  torch.compile + FP16")
    model = model_fn(weights=None).cuda().half().eval()
    x_fp16 = x.half()
    
    try:
        compiled = torch.compile(model, mode="default")
        time_ms = benchmark(compiled, x_fp16)
        print(f"   Time: {time_ms:.2f}ms")
        del model, compiled, x_fp16
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_compile_max_autotune_fp16(model_fn, x):
    """torch.compile max-autotune + FP16."""
    print("\n8️⃣  torch.compile MAX-AUTOTUNE + FP16")
    model = model_fn(weights=None).cuda().half().eval()
    x_fp16 = x.half()
    
    try:
        compiled = torch.compile(model, mode="max-autotune")
        # Extra warmup for max-autotune
        time_ms = benchmark(compiled, x_fp16, warmup=50)
        print(f"   Time: {time_ms:.2f}ms")
        del model, compiled, x_fp16
        clear_gpu()
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        del model
        clear_gpu()
        return None


def test_onnx_runtime(model_fn, batch_size):
    """ONNX Runtime with CUDA."""
    print("\n9️⃣  ONNX RUNTIME (CUDA)")
    
    try:
        import onnx
        import onnxruntime as ort
        
        model = model_fn(weights=None).cuda().eval()
        x = torch.randn(batch_size, 3, 224, 224, device='cuda')
        
        # Export to ONNX
        onnx_path = "/tmp/model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model, x,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=17
            )
        
        # Create ONNX Runtime session with CUDA
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        # Benchmark
        x_np = x.cpu().numpy()
        
        # Warmup
        for _ in range(30):
            _ = session.run(None, {'input': x_np})
        
        # Measure
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = session.run(None, {'input': x_np})
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        times.sort()
        time_ms = times[len(times)//2]
        print(f"   Time: {time_ms:.2f}ms")
        
        del model, session
        clear_gpu()
        return time_ms
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def test_tensorrt(model_fn, x):
    """Torch-TensorRT optimization."""
    print("\n🔟  TORCH-TENSORRT")
    
    if not TENSORRT_AVAILABLE:
        print("   ⏭️ Skipped (not installed)")
        return None
    
    try:
        import torch_tensorrt
        
        model = model_fn(weights=None).cuda().half().eval()
        x_fp16 = x.half()
        
        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(x_fp16.shape, dtype=torch.half)],
            enabled_precisions={torch.half},
            workspace_size=1 << 30
        )
        
        time_ms = benchmark(trt_model, x_fp16)
        print(f"   Time: {time_ms:.2f}ms")
        
        del model, trt_model, x_fp16
        clear_gpu()
        return time_ms
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests(model_fn, batch_size, name):
    print("\n" + "="*70)
    print(f"🧪 {name} (batch={batch_size})")
    print("="*70)
    
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    results = {}
    
    # Run all tests
    results['baseline'] = test_baseline(model_fn, x)
    results['jit_trace'] = test_jit_trace(model_fn, x)
    results['fp16_autocast'] = test_fp16_autocast(model_fn, x)
    results['fp16_model'] = test_fp16_model(model_fn, x)
    results['jit_fp16'] = test_jit_fp16(model_fn, x)
    results['compile_fp16'] = test_compile_fp16(model_fn, x)
    results['compile_max_fp16'] = test_compile_max_autotune_fp16(model_fn, x)
    results['onnx_runtime'] = test_onnx_runtime(model_fn, batch_size)
    results['tensorrt'] = test_tensorrt(model_fn, x)
    
    # Skip CPU-only tests in comparison
    test_dynamic_quantization(model_fn, x)
    
    # Summary
    print("\n" + "-"*70)
    print("📊 RESULTS SUMMARY:")
    print("-"*70)
    
    baseline = results['baseline']
    for opt, time_ms in results.items():
        if time_ms is not None:
            speedup = baseline / time_ms
            if speedup >= 2.0:
                emoji = "🚀🚀"
            elif speedup >= 1.5:
                emoji = "🚀"
            elif speedup >= 1.2:
                emoji = "✅"
            elif speedup >= 1.0:
                emoji = "⚠️"
            else:
                emoji = "❌"
            print(f"   {emoji} {opt:20s}: {time_ms:6.2f}ms ({speedup:.2f}x)")
        else:
            print(f"   ⏭️  {opt:20s}: SKIPPED")
    
    # Find best
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best = min(valid, key=valid.get)
        best_speedup = baseline / valid[best]
        print(f"\n   🏆 BEST: {best} ({best_speedup:.2f}x speedup)")
    else:
        best = None
        best_speedup = 1.0
    
    del x
    clear_gpu()
    
    return results, baseline, best, best_speedup


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*70)
print("🚀 ADVANCED OPTIMIZATION TEST - Targeting 2x Speedup")
print("="*70)
print("""
Techniques being tested:
  • JIT Trace (FP32 & FP16)
  • FP16 Mixed Precision (Tensor Cores)
  • torch.compile + FP16
  • torch.compile max-autotune + FP16
  • ONNX Runtime with CUDA
  • Torch-TensorRT (if available)
  
T4 GPU has Tensor Cores for FP16 - should get 1.5-2x!
""")

all_results = []

# Test ResNet-18
results, baseline, best, speedup = run_all_tests(resnet18, 128, "ResNet-18")
all_results.append({
    'model': 'ResNet-18',
    'batch': 128,
    'baseline_ms': round(baseline, 2),
    'best_method': best,
    'best_speedup': round(speedup, 2)
})

# Test ResNet-50
results, baseline, best, speedup = run_all_tests(resnet50, 64, "ResNet-50")
all_results.append({
    'model': 'ResNet-50',
    'batch': 64,
    'baseline_ms': round(baseline, 2),
    'best_method': best,
    'best_speedup': round(speedup, 2)
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

if avg_speedup >= 2.0:
    print("\n🚀🚀 EXCELLENT! 2x+ speedup achieved!")
elif avg_speedup >= 1.5:
    print("\n🚀 GREAT! 1.5x+ speedup - close to 2x!")
elif avg_speedup >= 1.3:
    print("\n✅ GOOD! 1.3x+ speedup")
else:
    print("\n⚠️ Below target")

print("\n" + "="*70)
print("💡 KEY INSIGHT")
print("="*70)
print("""
To reach 2x speedup on T4:
1. USE FP16 - T4 has Tensor Cores that run FP16 ~2x faster
2. JIT + FP16 should give best pure PyTorch results
3. TensorRT can give 2-6x but requires installation
4. ONNX Runtime is a good portable alternative

The winning strategy should be: JIT Trace + FP16
""")

print("\n📄 JSON:")
print(json.dumps(all_results, indent=2))
