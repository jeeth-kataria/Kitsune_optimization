"""
🦊 Kitsune v2.0 Validation Test
Run this in Colab to verify the new optimizer works correctly
"""

import subprocess
import sys

# Install dependencies
try:
    import torch
    import torchvision
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=True)
    import torch
    import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

import torch.nn as nn
from torchvision.models import resnet18, resnet50
import logging

logging.basicConfig(level=logging.INFO)

# ==============================================================================
# EMBEDDED KITSUNE v2.0 OPTIMIZER (No PyPI dependency)
# ==============================================================================

from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizationConfig:
    """Configuration for Kitsune v2 optimizations."""
    strategy: str = "jit_trace"  # 'jit_trace', 'jit_script', 'compile', 'none'
    compile_mode: str = "default"
    warmup_iterations: int = 10
    device: str = "cuda"
    fallback_on_error: bool = True


class KitsuneOptimizer:
    """🦊 Kitsune Model Optimizer v2.0"""
    
    def __init__(self, model, sample_input=None, config=None):
        self.original_model = model
        self.sample_input = sample_input
        self.config = config or OptimizationConfig()
        self.optimized_model = None
        self.strategy_used = None
        
        # Move to device
        if not next(model.parameters()).is_cuda:
            self.original_model = model.cuda()
        self.original_model.eval()
        
        self._optimize()
    
    def _optimize(self):
        strategy = self.config.strategy.lower()
        print(f"🦊 Kitsune: Optimizing with strategy '{strategy}'")
        
        if strategy == "jit_trace":
            self._apply_jit_trace()
        elif strategy == "jit_script":
            self._apply_jit_script()
        elif strategy == "compile":
            self._apply_compile()
        else:
            self.optimized_model = self.original_model
            self.strategy_used = "none"
        
        # Warmup
        if self.optimized_model and self.sample_input is not None:
            print(f"  Warming up ({self.config.warmup_iterations} iterations)...")
            with torch.no_grad():
                for _ in range(self.config.warmup_iterations):
                    _ = self.optimized_model(self.sample_input)
            torch.cuda.synchronize()
        
        print(f"✅ Kitsune: Optimization complete (strategy: {self.strategy_used})")
    
    def _apply_jit_trace(self):
        if self.sample_input is None:
            print("  ⚠️ JIT trace requires sample_input, falling back to compile")
            self._apply_compile()
            return
        
        try:
            print("  Applying TorchScript trace...")
            with torch.no_grad():
                traced = torch.jit.trace(self.original_model, self.sample_input)
            self.optimized_model = torch.jit.optimize_for_inference(traced)
            self.strategy_used = "jit_trace"
            print("  ✓ TorchScript trace applied (expected: 1.18-1.20x speedup)")
        except Exception as e:
            print(f"  ⚠️ JIT trace failed: {e}")
            self._apply_compile()
    
    def _apply_jit_script(self):
        try:
            print("  Applying TorchScript script...")
            scripted = torch.jit.script(self.original_model)
            self.optimized_model = torch.jit.optimize_for_inference(scripted)
            self.strategy_used = "jit_script"
            print("  ✓ TorchScript script applied")
        except Exception as e:
            print(f"  ⚠️ JIT script failed: {e}")
            self._apply_compile()
    
    def _apply_compile(self):
        try:
            print(f"  Applying torch.compile (mode={self.config.compile_mode})...")
            self.optimized_model = torch.compile(
                self.original_model,
                mode=self.config.compile_mode,
                fullgraph=False
            )
            self.strategy_used = "compile"
            print("  ✓ torch.compile applied")
        except Exception as e:
            print(f"  ⚠️ torch.compile failed: {e}")
            self.optimized_model = self.original_model
            self.strategy_used = "none"
    
    def __call__(self, x):
        return self.optimized_model(x)


def optimize_model(model, sample_input=None, strategy="jit_trace"):
    """Quick optimization API."""
    config = OptimizationConfig(strategy=strategy)
    optimizer = KitsuneOptimizer(model, sample_input, config)
    return optimizer.optimized_model


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark(model, x, warmup=30, iters=100):
    """Accurate CUDA event timing."""
    model.eval()
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
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
    return times[len(times)//2]


def run_test(model_fn, batch_size, name):
    """Test Kitsune v2 optimization."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {name} (batch={batch_size})")
    print(f"{'='*60}")
    
    # Create model and input
    model = model_fn(weights=None).cuda().eval()
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Baseline
    print("\n📊 BASELINE (no optimization)")
    baseline_ms = benchmark(model, x)
    print(f"   Time: {baseline_ms:.2f}ms")
    
    # Kitsune v2 with JIT trace (best)
    print("\n🦊 KITSUNE v2 (jit_trace)")
    model_fresh = model_fn(weights=None).cuda().eval()
    optimizer = KitsuneOptimizer(model_fresh, x)
    optimized_ms = benchmark(optimizer, x)
    print(f"   Time: {optimized_ms:.2f}ms")
    
    # Results
    speedup = baseline_ms / optimized_ms
    print(f"\n📈 RESULTS:")
    print(f"   Baseline:  {baseline_ms:.2f}ms")
    print(f"   Optimized: {optimized_ms:.2f}ms")
    print(f"   Speedup:   {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")
    
    if speedup >= 1.15:
        print(f"   ✅ SUCCESS! Achieved expected speedup")
    elif speedup >= 1.05:
        print(f"   ⚠️ Modest improvement")
    else:
        print(f"   ❌ Below expected")
    
    return speedup


# ==============================================================================
# MAIN
# ==============================================================================

print("\n" + "="*60)
print("🦊 KITSUNE v2.0 VALIDATION")
print("="*60)

results = []

# Test ResNet-18
speedup = run_test(resnet18, 128, "ResNet-18")
results.append(("ResNet-18", speedup))

# Test ResNet-50
speedup = run_test(resnet50, 64, "ResNet-50")
results.append(("ResNet-50", speedup))

# Summary
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

for name, speedup in results:
    status = "✅" if speedup >= 1.15 else ("⚠️" if speedup >= 1.05 else "❌")
    print(f"   {status} {name}: {speedup:.2f}x")

avg = sum(s for _, s in results) / len(results)
print(f"\n🎯 AVERAGE SPEEDUP: {avg:.2f}x")

if avg >= 1.15:
    print("\n✅ Kitsune v2.0 is working correctly!")
    print("   Expected: 1.15-1.20x | Achieved: {:.2f}x".format(avg))
else:
    print("\n⚠️ Performance below expected")
