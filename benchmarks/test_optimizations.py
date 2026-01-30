#!/usr/bin/env python3
"""
Test new optimization implementation - CUDA graphs + torch.compile
Should show actual speedups now!
"""

import subprocess
import sys

print("📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-kitsune"], check=True)

import torch
import torch.nn as nn
import json
import time
from kitsune import optimize_model

print("\n🔍 System Check")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if not torch.cuda.is_available():
    print("❌ No GPU!")
    sys.exit(1)

# Test model
class TestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096), nn.ReLU(), nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)

def benchmark(model, input_data, num_iter=100):
    """Benchmark model inference."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Benchmark
    timings = []
    with torch.no_grad():
        for _ in range(num_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    
    return sum(timings) / len(timings)

print("\n" + "="*70)
print("🚀 Testing New Optimizations")
print("="*70)

device = torch.device('cuda')
num_runs = 3
all_speedups = []

for run in range(num_runs):
    print(f"\nRun {run+1}/{num_runs}:")
    
    # Create fresh models
    model_baseline = TestMLP().to(device)
    model_optimized = TestMLP().to(device)
    model_optimized.load_state_dict(model_baseline.state_dict())
    
    input_data = torch.randn(256, 1024, device=device)
    
    # Baseline
    print("  Baseline...", end=" ")
    baseline_time = benchmark(model_baseline, input_data)
    print(f"{baseline_time:.2f}ms")
    
    # Optimized with new wrapper
    print("  Kitsune (CUDA graphs + compile)...", end=" ")
    optimizer = optimize_model(model_optimized, input_data)
    optimized_time = benchmark(optimizer.model, input_data)
    print(f"{optimized_time:.2f}ms")
    
    speedup = baseline_time / optimized_time
    print(f"  ⚡ Speedup: {speedup:.2f}x")
    all_speedups.append(speedup)
    
    del model_baseline, model_optimized, input_data
    torch.cuda.empty_cache()

avg_speedup = sum(all_speedups) / len(all_speedups)

print("\n" + "="*70)
print(f"📊 RESULTS: Average Speedup = {avg_speedup:.2f}x")
print("="*70)

if avg_speedup > 1.15:
    print("✅ SUCCESS! Real speedups achieved!")
elif avg_speedup > 1.05:
    print("⚠️  Modest improvement, but working")
else:
    print("❌ No significant speedup yet")

print(f"\nIndividual runs: {', '.join([f'{s:.2f}x' for s in all_speedups])}")
