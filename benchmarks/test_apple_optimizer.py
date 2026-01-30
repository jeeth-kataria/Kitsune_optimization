"""
🍎 Quick test for Apple Silicon optimizer on macOS
"""

import sys
import time

sys.path.insert(0, '/Users/jeethkataria/Kitsune/KITSUNE_ALGO')

import torch
import torch.nn as nn

print("=" * 60)
print("🍎 Apple Silicon Optimizer Test")
print("=" * 60)

# Check environment
print(f"\nPyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

# Import our optimizers
from kitsune.backends.apple_optimizer import (
    AppleSiliconOptimizer,
    detect_apple_chip,
    is_apple_silicon
)
from kitsune.backends.backend_selector import detect_platform, print_platform_info

# Print platform info
print_platform_info()

# Test chip detection
chip_info = detect_apple_chip()
print(f"\n🔍 Detected Chip: {chip_info.chip_type.value}")
print(f"   CPU cores: {chip_info.cpu_cores}")
print(f"   GPU cores: {chip_info.gpu_cores}")
print(f"   Neural Engine: {chip_info.neural_engine_tops} TOPS")

# Create a simple model
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

print("\n📦 Creating test model...")
model = SimpleConvNet()
x = torch.randn(8, 3, 64, 64)

# Initialize optimizer
optimizer = AppleSiliconOptimizer()

print(f"\n✅ MPS available: {optimizer.mps_available}")

# Optimize
if optimizer.mps_available:
    print("\n🔧 Applying MPS optimization...")
    result = optimizer.optimize(model, x)
    
    print(f"\n📊 Optimization Result:")
    print(f"   Applied: {', '.join(result.optimizations_applied)}")
    print(f"   Expected speedup: {result.speedup_estimate:.1f}x")
    print(f"   Device: {result.device}")
    
    # Benchmark
    print("\n⏱️ Running benchmark...")
    bench_result = optimizer.benchmark(model, x, iterations=50)
    
    print(f"\n📈 Benchmark Results:")
    print(f"   CPU: {bench_result['cpu_ms']:.2f} ms")
    print(f"   MPS: {bench_result['mps_ms']:.2f} ms")
    print(f"   Actual speedup: {bench_result['speedup']:.2f}x")
else:
    print("\n⚠️ MPS not available, testing CPU-only path...")
    
    # Test JIT on CPU
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, x)
        
    # Benchmark
    times_orig = []
    times_jit = []
    
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times_orig.append((time.perf_counter() - start) * 1000)
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = traced(x)
        times_jit.append((time.perf_counter() - start) * 1000)
    
    orig_ms = sorted(times_orig)[len(times_orig) // 2]
    jit_ms = sorted(times_jit)[len(times_jit) // 2]
    
    print(f"\n📈 CPU Benchmark:")
    print(f"   Original: {orig_ms:.2f} ms")
    print(f"   JIT: {jit_ms:.2f} ms")
    print(f"   Speedup: {orig_ms / jit_ms:.2f}x")

print("\n" + "=" * 60)
print("✅ Apple Silicon optimizer test complete!")
print("=" * 60)
