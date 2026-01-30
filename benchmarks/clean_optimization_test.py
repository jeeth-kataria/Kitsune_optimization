"""
🔬 CLEAN OPTIMIZATION TEST
Tests each optimization SEPARATELY to find what works
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

def benchmark(model, x, warmup=20, iters=100):
    """Accurate benchmark with CUDA events."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    times.sort()
    return times[len(times)//2]  # median


def test_baseline(model, x):
    """Test 1: Pure baseline - no optimizations."""
    print("\n1️⃣  BASELINE (no optimization)")
    model = model.cuda().eval()
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    return time_ms


def test_channels_last(model, x):
    """Test 2: Channels-last memory format only."""
    print("\n2️⃣  CHANNELS-LAST only")
    model = model.cuda().eval()
    model = model.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    time_ms = benchmark(model, x)
    print(f"   Time: {time_ms:.2f}ms")
    return time_ms


def test_compile_default(model, x):
    """Test 3: torch.compile with default mode."""
    print("\n3️⃣  torch.compile (mode='default')")
    model = model.cuda().eval()
    try:
        compiled = torch.compile(model, mode="default")
        time_ms = benchmark(compiled, x)
        print(f"   Time: {time_ms:.2f}ms")
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def test_compile_reduce_overhead(model, x):
    """Test 4: torch.compile with reduce-overhead mode."""
    print("\n4️⃣  torch.compile (mode='reduce-overhead')")
    model = model.cuda().eval()
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        time_ms = benchmark(compiled, x)
        print(f"   Time: {time_ms:.2f}ms")
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def test_cuda_graphs_only(model, x):
    """Test 5: CUDA graphs only (NO torch.compile)."""
    print("\n5️⃣  CUDA GRAPHS only (no compile)")
    model = model.cuda().eval()
    
    try:
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(x)
        torch.cuda.synchronize()
        
        # Create static tensors
        static_input = x.clone()
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)
        
        torch.cuda.synchronize()
        
        # Benchmark graph replay
        def graph_forward(inp):
            static_input.copy_(inp)
            graph.replay()
            return static_output.clone()
        
        # Warmup replay
        with torch.no_grad():
            for _ in range(20):
                _ = graph_forward(x)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = graph_forward(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        times.sort()
        time_ms = times[len(times)//2]
        print(f"   Time: {time_ms:.2f}ms")
        return time_ms
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def test_channels_last_compile(model, x):
    """Test 6: Channels-last + torch.compile."""
    print("\n6️⃣  CHANNELS-LAST + torch.compile")
    model = model.cuda().eval()
    model = model.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        time_ms = benchmark(compiled, x)
        print(f"   Time: {time_ms:.2f}ms")
        return time_ms
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None


def run_all_tests(model_fn, batch_size, name):
    """Run all optimization strategies on a model."""
    print("\n" + "="*70)
    print(f"🧪 {name} (batch={batch_size})")
    print("="*70)
    
    results = {}
    
    # Create fresh model and input for each test
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Test 1: Baseline
    model = model_fn(weights=None)
    results['baseline'] = test_baseline(model, x)
    del model
    torch.cuda.empty_cache()
    
    # Test 2: Channels-last
    model = model_fn(weights=None)
    x_cl = torch.randn(batch_size, 3, 224, 224, device='cuda')
    results['channels_last'] = test_channels_last(model, x_cl)
    del model, x_cl
    torch.cuda.empty_cache()
    
    # Test 3: torch.compile default
    model = model_fn(weights=None)
    results['compile_default'] = test_compile_default(model, x)
    del model
    torch.cuda.empty_cache()
    
    # Test 4: torch.compile reduce-overhead
    model = model_fn(weights=None)
    results['compile_reduce'] = test_compile_reduce_overhead(model, x)
    del model
    torch.cuda.empty_cache()
    
    # Test 5: CUDA graphs only
    model = model_fn(weights=None)
    results['cuda_graphs'] = test_cuda_graphs_only(model, x)
    del model
    torch.cuda.empty_cache()
    
    # Test 6: Channels-last + compile
    model = model_fn(weights=None)
    x_cl = torch.randn(batch_size, 3, 224, 224, device='cuda')
    results['cl_compile'] = test_channels_last_compile(model, x_cl)
    del model, x_cl
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "-"*70)
    print("📊 RESULTS SUMMARY:")
    print("-"*70)
    
    baseline = results['baseline']
    for opt, time_ms in results.items():
        if time_ms is not None:
            speedup = baseline / time_ms
            status = "✅" if speedup > 1.1 else ("⚠️" if speedup >= 1.0 else "❌")
            print(f"   {status} {opt:20s}: {time_ms:6.2f}ms ({speedup:.2f}x)")
        else:
            print(f"   ❌ {opt:20s}: FAILED")
    
    # Find best
    valid = {k: v for k, v in results.items() if v is not None}
    best = min(valid, key=valid.get)
    best_speedup = baseline / valid[best]
    
    print(f"\n   🏆 BEST: {best} ({best_speedup:.2f}x speedup)")
    
    return results, baseline, best, best_speedup


# Main
print("\n" + "="*70)
print("🔬 INDIVIDUAL OPTIMIZATION COMPARISON")
print("="*70)

all_results = []

# Test ResNet-18
results, baseline, best, speedup = run_all_tests(resnet18, 128, "ResNet-18")
all_results.append({
    'model': 'ResNet-18',
    'batch': 128,
    'baseline_ms': round(baseline, 2),
    'best_method': best,
    'best_speedup': round(speedup, 2),
    'all_results': {k: round(v, 2) if v else None for k, v in results.items()}
})

# Test ResNet-50
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

if avg_speedup >= 1.3:
    print("\n✅ GOOD! Found optimization that works!")
elif avg_speedup >= 1.1:
    print("\n⚠️  Modest improvement possible")
else:
    print("\n❌ No significant speedup available")

print("\n📄 JSON:")
print(json.dumps(all_results, indent=2))
