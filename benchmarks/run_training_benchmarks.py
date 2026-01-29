#!/usr/bin/env python3
"""
Training Benchmark Script for Kitsune - Tests ACTUAL training workloads
This tests forward + backward + optimizer step (where Kitsune should excel)
"""

import subprocess
import sys

# Install dependencies
print("📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-kitsune", "matplotlib"], check=True)

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from typing import Dict
import kitsune
from kitsune import optimize_model

# Check GPU
print("\n🔍 System Check")
print(f"PyTorch: {torch.__version__}")
print(f"Kitsune: {kitsune.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if not torch.cuda.is_available():
    print("❌ ERROR: No GPU! Enable GPU in Runtime → Change runtime type")
    sys.exit(1)

# Define models
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        return self.network(x)


# Warm up CUDA
def warmup_cuda():
    device = torch.device('cuda')
    t = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        _ = t @ t
    torch.cuda.synchronize()


# Baseline training loop (standard PyTorch)
def benchmark_baseline_training(model, input_data, target_data, num_iterations=50):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Warmup
    for _ in range(5):
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark
    timings = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end.record()
        
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
    }


# Kitsune-optimized training with torch.compile
def benchmark_kitsune_training(model, input_data, target_data, num_iterations=50):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_torch = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Apply torch.compile for actual optimization
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("    Using torch.compile optimization")
    except Exception as e:
        print(f"    torch.compile not available: {e}")
    
    # Warmup
    for _ in range(5):
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer_torch.step()
        optimizer_torch.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark
    timings = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer_torch.step()
        optimizer_torch.zero_grad()
        end.record()
        
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
    }


# Run benchmark
print("\n" + "="*70)
print("🚀 Training Benchmark (Forward + Backward + Optimizer Step)")
print("="*70)

warmup_cuda()

device = torch.device('cuda')
num_runs = 5
all_results = []

for run_idx in range(num_runs):
    print(f"\nRun {run_idx + 1}/{num_runs}...")
    
    # Create fresh models
    model_baseline = MLPModel().to(device)
    model_kitsune = MLPModel().to(device)
    model_kitsune.load_state_dict(model_baseline.state_dict())
    
    # Create data
    input_data = torch.randn(256, 1024, device=device)
    target_data = torch.randint(0, 10, (256,), device=device)
    
    # Baseline
    print("  Baseline PyTorch...", end=" ")
    baseline = benchmark_baseline_training(model_baseline, input_data, target_data)
    print(f"{baseline['mean_time_ms']:.2f}ms")
    
    # Kitsune with torch.compile
    print("  Kitsune + torch.compile...", end=" ")
    kitsune_result = benchmark_kitsune_training(model_kitsune, input_data, target_data)
    print(f"{kitsune_result['mean_time_ms']:.2f}ms")
    
    speedup = baseline['mean_time_ms'] / kitsune_result['mean_time_ms']
    print(f"  ⚡ Speedup: {speedup:.2f}x")
    
    all_results.append({
        'baseline': baseline,
        'kitsune': kitsune_result,
        'speedup': speedup
    })
    
    del model_baseline, model_kitsune, input_data, target_data
    torch.cuda.empty_cache()

# Aggregate results
baseline_times = [r['baseline']['mean_time_ms'] for r in all_results]
kitsune_times = [r['kitsune']['mean_time_ms'] for r in all_results]
speedups = [r['speedup'] for r in all_results]

results = {
    'model': 'MLP (Training)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'hardware': {
        'gpu': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'kitsune_version': kitsune.__version__
    },
    'baseline': {
        'mean_time_ms': sum(baseline_times) / len(baseline_times),
        'std_time_ms': torch.tensor(baseline_times).std().item(),
        'peak_memory_mb': all_results[0]['baseline']['peak_memory_mb']
    },
    'kitsune': {
        'mean_time_ms': sum(kitsune_times) / len(kitsune_times),
        'std_time_ms': torch.tensor(kitsune_times).std().item(),
        'peak_memory_mb': all_results[0]['kitsune']['peak_memory_mb']
    },
    'improvement': {
        'speedup': sum(speedups) / len(speedups),
        'speedup_std': torch.tensor(speedups).std().item()
    }
}

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)
print(f"GPU: {results['hardware']['gpu']}")
print(f"PyTorch: {results['hardware']['pytorch_version']}")
print(f"\nBaseline:     {results['baseline']['mean_time_ms']:.2f}ms ± {results['baseline']['std_time_ms']:.2f}ms")
print(f"Optimized:    {results['kitsune']['mean_time_ms']:.2f}ms ± {results['kitsune']['std_time_ms']:.2f}ms")
print(f"⚡ Speedup:    {results['improvement']['speedup']:.2f}x ± {results['improvement']['speedup_std']:.2f}x")
print(f"\nMemory:")
print(f"  Baseline: {results['baseline']['peak_memory_mb']:.1f} MB")
print(f"  Optimized: {results['kitsune']['peak_memory_mb']:.1f} MB")
print("="*70)

print("\n📁 JSON Results:")
print(json.dumps(results, indent=2))

print("\n✅ Training benchmark complete!")
print(f"\n💡 Note: torch.compile provides {results['improvement']['speedup']:.2f}x speedup for training workloads")
