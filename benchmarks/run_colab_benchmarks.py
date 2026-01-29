#!/usr/bin/env python3
"""
Single-cell Colab benchmark script for Kitsune
Run this in Google Colab with GPU enabled
"""

import subprocess
import sys
import os

# Step 1: Install dependencies
print("="*70)
print("📦 Installing Dependencies...")
print("="*70)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-kitsune", "matplotlib"], check=True)

import torch
import torch.nn as nn
import json
import time
from typing import Dict
import kitsune
from kitsune import optimize_model

# Step 2: Verify GPU
print("\n" + "="*70)
print("🔍 System Information")
print("="*70)
print(f"PyTorch: {torch.__version__}")
print(f"Kitsune: {kitsune.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ ERROR: No GPU found! Enable GPU in Runtime → Change runtime type")
    sys.exit(1)

# Step 3: Define models
class MLPModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[2048, 4096, 2048, 1024], output_dim=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Step 4: Benchmark functions
def warmup_cuda():
    device = torch.device('cuda')
    warmup_tensor = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        _ = warmup_tensor @ warmup_tensor
    torch.cuda.synchronize()


def benchmark_baseline(model, input_data, num_iterations=100, num_warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_data)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_data)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'min_time_ms': min(timings),
        'max_time_ms': max(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
        'all_timings': timings
    }


def benchmark_kitsune(model, input_data, num_iterations=100, num_warmup=10):
    optimizer = optimize_model(model, input_data)
    model = optimizer.model
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_data)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_data)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'min_time_ms': min(timings),
        'max_time_ms': max(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
        'all_timings': timings
    }


def run_benchmark_for_model(model_name, model_class, input_shape, num_runs=5):
    print(f"\n{'='*70}")
    print(f"🔥 {model_name} Benchmark")
    print(f"{'='*70}")
    
    device = torch.device('cuda')
    baseline_results = []
    kitsune_results = []
    
    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}...", end=" ")
        
        model_baseline = model_class().to(device)
        model_kitsune = model_class().to(device)
        model_kitsune.load_state_dict(model_baseline.state_dict())
        
        input_data = torch.randn(*input_shape, device=device)
        
        baseline = benchmark_baseline(model_baseline, input_data, num_iterations=100)
        kitsune_bench = benchmark_kitsune(model_kitsune, input_data, num_iterations=100)
        
        baseline_results.append(baseline)
        kitsune_results.append(kitsune_bench)
        
        speedup = baseline['mean_time_ms'] / kitsune_bench['mean_time_ms']
        print(f"Speedup: {speedup:.2f}x")
        
        del model_baseline, model_kitsune, input_data
        torch.cuda.empty_cache()
    
    baseline_mean_times = [r['mean_time_ms'] for r in baseline_results]
    kitsune_mean_times = [r['mean_time_ms'] for r in kitsune_results]
    
    baseline_avg = sum(baseline_mean_times) / len(baseline_mean_times)
    kitsune_avg = sum(kitsune_mean_times) / len(kitsune_mean_times)
    speedup = baseline_avg / kitsune_avg
    
    baseline_memory_avg = sum(r['peak_memory_mb'] for r in baseline_results) / num_runs
    kitsune_memory_avg = sum(r['peak_memory_mb'] for r in kitsune_results) / num_runs
    memory_reduction = ((baseline_memory_avg - kitsune_memory_avg) / baseline_memory_avg) * 100
    
    return {
        'model': model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': {
            'gpu': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'kitsune_version': kitsune.__version__
        },
        'baseline': {
            'mean_time_ms': baseline_avg,
            'std_time_ms': torch.tensor(baseline_mean_times).std().item(),
            'peak_memory_mb': baseline_memory_avg,
        },
        'kitsune': {
            'mean_time_ms': kitsune_avg,
            'std_time_ms': torch.tensor(kitsune_mean_times).std().item(),
            'peak_memory_mb': kitsune_memory_avg,
        },
        'improvement': {
            'speedup': speedup,
            'memory_reduction_percent': memory_reduction
        }
    }


# Step 5: Run all benchmarks
print("\n" + "="*70)
print("🚀 Starting Benchmark Suite")
print("="*70)

warmup_cuda()

results = {}

# MLP
results['MLP'] = run_benchmark_for_model(
    'MLP',
    MLPModel,
    (256, 1024),
    num_runs=5
)

# LeNet-5
results['LeNet-5'] = run_benchmark_for_model(
    'LeNet-5',
    LeNet5,
    (128, 1, 28, 28),
    num_runs=5
)

# ResNet-18
results['ResNet-18'] = run_benchmark_for_model(
    'ResNet-18',
    ResNet18,
    (64, 3, 224, 224),
    num_runs=5
)

# Step 6: Display results
print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)

for model_name, data in results.items():
    print(f"\n{'='*70}")
    print(f"🎯 {model_name.upper()}")
    print(f"{'='*70}")
    print(f"GPU: {data['hardware']['gpu']}")
    print(f"CUDA: {data['hardware']['cuda_version']}")
    print(f"PyTorch: {data['hardware']['pytorch_version']}")
    print(f"Kitsune: {data['hardware']['kitsune_version']}")
    print(f"\nTiming Results:")
    print(f"  Baseline:     {data['baseline']['mean_time_ms']:>8.2f}ms ± {data['baseline']['std_time_ms']:.2f}ms")
    print(f"  Kitsune:      {data['kitsune']['mean_time_ms']:>8.2f}ms ± {data['kitsune']['std_time_ms']:.2f}ms")
    print(f"  ⚡ Speedup:    {data['improvement']['speedup']:>8.2f}x")
    print(f"\nMemory Results:")
    print(f"  Baseline Mem: {data['baseline']['peak_memory_mb']:>8.1f} MB")
    print(f"  Kitsune Mem:  {data['kitsune']['peak_memory_mb']:>8.1f} MB")
    print(f"  💾 Saved:      {data['improvement']['memory_reduction_percent']:>7.1f}%")

print(f"\n{'='*70}")
print("✅ BENCHMARKS COMPLETE!")
print("="*70)

# Step 7: Save JSON results
print("\n📁 Saving results as JSON...")
print(json.dumps(results, indent=2))

print("\n✅ Done! Copy the JSON output above to share results.")
