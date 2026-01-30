"""
🔬 Kitsune Optimization Validation Benchmark
Tests the FIXED implementation to verify 1.5-2.0x speedups
Run this in Colab with T4 GPU or better
"""

import subprocess
import sys

# Install dependencies
print("📦 Installing dependencies...")
try:
    import torch
    import torchvision
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=True)
    import torch
    import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if not torch.cuda.is_available():
    print("❌ GPU required! Enable GPU in Colab: Runtime → Change runtime type")
    sys.exit(1)

# Install Kitsune from local (or pip install torch-kitsune for published version)
print("\n📦 Installing Kitsune...")
subprocess.run([sys.executable, "-m", "pip", "install", "torch-kitsune"], check=True)

import torch.nn as nn
from torchvision.models import resnet18, resnet50
import time
import json

# Import Kitsune with new fixed API
from kitsune import optimize_model, OptimizationConfig

def measure_time(model, input_data, num_iterations=100, warmup=20):
    """Measure inference time with CUDA events (most accurate method)."""
    if hasattr(model, 'eval'):
        model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data) if not callable(model) or hasattr(model, 'forward') else model(input_data)
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(input_data) if not callable(model) or hasattr(model, 'forward') else model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    # Calculate statistics
    times.sort()
    median_time = times[len(times) // 2]
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5
    
    return median_time, std_time


def test_model(model_fn, batch_size, model_name):
    """Test a specific model configuration."""
    print(f"\n{'='*70}")
    print(f"📊 Testing: {model_name} (batch={batch_size})")
    print(f"{'='*70}")
    
    # Create model and input
    model = model_fn(weights=None).cuda()
    model.eval()
    input_data = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Measure baseline
    print("⏱️  Measuring baseline (no optimization)...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    baseline_time, baseline_std = measure_time(model, input_data)
    baseline_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Baseline: {baseline_time:.2f}ms ± {baseline_std:.2f}ms ({baseline_memory:.0f}MB)")
    
    # Apply Kitsune optimization
    print("\n🦊 Applying Kitsune optimization...")
    torch.cuda.reset_peak_memory_stats()
    
    config = OptimizationConfig(
        use_compile=True,
        compile_mode="max-autotune",
        use_cuda_graphs=True,
        use_channels_last=True,
        use_tf32=True,
        verbose=True
    )
    
    optimizer = optimize_model(model, input_data, config)
    
    # Measure optimized
    print("\n⏱️  Measuring optimized (Kitsune)...", flush=True)
    optimized_time, optimized_std = measure_time(optimizer, input_data)
    optimized_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Optimized: {optimized_time:.2f}ms ± {optimized_std:.2f}ms ({optimized_memory:.0f}MB)")
    
    # Calculate improvements
    speedup = baseline_time / optimized_time
    memory_reduction = (baseline_memory - optimized_memory) / baseline_memory * 100
    
    # Results
    print(f"\n{'='*70}")
    print(f"📈 RESULTS:")
    print(f"   Baseline:       {baseline_time:6.2f}ms  ({baseline_memory:6.0f}MB)")
    print(f"   Optimized:      {optimized_time:6.2f}ms  ({optimized_memory:6.0f}MB)")
    print(f"   ⚡ Speedup:      {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")
    print(f"   💾 Memory:       {memory_reduction:+.1f}%")
    
    # Determine if it meets target
    target_speedup = 1.5  # Realistic target from the fix guide
    if speedup >= target_speedup:
        print(f"   ✅ PASS - Meets {target_speedup}x target!")
    elif speedup >= 1.2:
        print(f"   ⚠️  GOOD - Close to target ({target_speedup}x)")
    else:
        print(f"   ❌ FAIL - Below target ({target_speedup}x)")
    
    # Cleanup
    del model, optimizer, input_data
    torch.cuda.empty_cache()
    
    return {
        'model': model_name,
        'batch_size': batch_size,
        'baseline_time_ms': round(baseline_time, 2),
        'optimized_time_ms': round(optimized_time, 2),
        'speedup': round(speedup, 2),
        'memory_reduction_pct': round(memory_reduction, 1),
        'passes_target': speedup >= target_speedup
    }


def main():
    print("="*70)
    print("🔬 KITSUNE OPTIMIZATION VALIDATION")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Test configurations - focus on realistic workloads
    tests = [
        (resnet18, 256, "ResNet-18"),
        (resnet18, 512, "ResNet-18 (large batch)"),
        (resnet50, 128, "ResNet-50"),
        (resnet50, 256, "ResNet-50 (large batch)"),
    ]
    
    results = []
    
    for model_class, batch_size, name in tests:
        try:
            result = test_model(model_class, batch_size, name)
            results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  Skipping {name} (batch={batch_size}) - Out of memory")
                torch.cuda.empty_cache()
            else:
                print(f"\n❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    if not results:
        print("\n❌ No tests completed successfully")
        return
    
    passed = sum(1 for r in results if r['passes_target'])
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total} (target: ≥1.5x speedup)")
    print(f"\nDetailed results:\n")
    
    for result in results:
        status = "✅" if result['passes_target'] else ("⚠️" if result['speedup'] >= 1.2 else "❌")
        print(f"{status} {result['model']:30s} (batch={result['batch_size']:3d})")
        print(f"   Speedup: {result['speedup']:.2f}x")
        print(f"   Memory:  {result['memory_reduction_pct']:+.1f}%")
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    
    # Final verdict
    print("\n" + "="*70)
    print(f"🎯 AVERAGE SPEEDUP: {avg_speedup:.2f}x")
    print("="*70)
    
    if avg_speedup >= 1.5:
        print("\n✅ SUCCESS! Kitsune delivers 1.5x+ speedup as promised!")
        print("   The optimization fixes are working correctly.")
    elif avg_speedup >= 1.2:
        print("\n⚠️  GOOD! Kitsune shows meaningful improvements (1.2x+)")
        print("   Benefits scale with larger models and batch sizes.")
    else:
        print("\n❌ NEEDS WORK! Speedups below 1.2x")
        print("   Try larger models or investigate optimization pipeline.")
    
    print("\n💡 Note: Speedups scale with:")
    print("   • Model complexity (ResNet-50 > ResNet-18)")
    print("   • Batch size (512 > 256)")
    print("   • GPU generation (Ampere with TF32 > Turing)")
    
    # JSON output for easy parsing
    print("\n" + "="*70)
    print("📄 JSON RESULTS (copy for documentation):")
    print("="*70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
