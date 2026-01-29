#!/usr/bin/env python3
"""
ResNet-18 Benchmark Script for Kitsune Optimizer
Tests deep residual network for image classification
"""

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from typing import Dict

try:
    import kitsune
    from kitsune import optimize
except ImportError:
    print("ERROR: Kitsune not installed. Run: pip install torch-kitsune")
    exit(1)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18"""
    
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
    """ResNet-18 architecture"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classification head
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


def create_dummy_data(batch_size=64, device='cuda'):
    """Create dummy ImageNet-like input data"""
    return torch.randn(batch_size, 3, 224, 224, device=device)


def warmup_cuda():
    """Warmup CUDA to ensure stable timings"""
    device = torch.device('cuda')
    warmup_tensor = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        _ = warmup_tensor @ warmup_tensor
    torch.cuda.synchronize()


def benchmark_baseline(model: nn.Module, input_data: torch.Tensor,
                       num_iterations: int = 100, num_warmup: int = 10) -> Dict:
    """Benchmark baseline PyTorch without Kitsune"""
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Benchmark
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
    
    # Memory measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_data)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'min_time_ms': min(timings),
        'max_time_ms': max(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
        'all_timings': timings
    }


def benchmark_kitsune(model: nn.Module, input_data: torch.Tensor,
                      num_iterations: int = 100, num_warmup: int = 10) -> Dict:
    """Benchmark with Kitsune optimization"""
    
    # Apply Kitsune optimization
    model = optimize(
        model,
        enable_amp=True,
        enable_fusion=True,
        enable_stream_parallel=True,
        enable_memory_pool=True,
        enable_prefetch=True
    )
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Benchmark
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
    
    # Memory measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_data)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    return {
        'mean_time_ms': sum(timings) / len(timings),
        'min_time_ms': min(timings),
        'max_time_ms': max(timings),
        'std_time_ms': torch.tensor(timings).std().item(),
        'peak_memory_mb': peak_memory,
        'all_timings': timings
    }


def run_benchmark(num_runs: int = 5, num_iterations: int = 100) -> Dict:
    """Run complete benchmark suite"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Benchmarks require GPU.")
    
    device = torch.device('cuda')
    print(f"Running benchmarks on: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Kitsune Version: {kitsune.__version__}")
    print()
    
    warmup_cuda()
    
    baseline_results = []
    kitsune_results = []
    
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}...")
        
        # Create fresh model and data for each run
        model_baseline = ResNet18().to(device)
        model_kitsune = ResNet18().to(device)
        
        # Ensure same weights
        model_kitsune.load_state_dict(model_baseline.state_dict())
        
        input_data = create_dummy_data(device=device)
        
        # Baseline
        print("  Baseline PyTorch...")
        baseline = benchmark_baseline(model_baseline, input_data, num_iterations)
        baseline_results.append(baseline)
        print(f"    Mean: {baseline['mean_time_ms']:.2f}ms")
        
        # Kitsune
        print("  Kitsune Optimized...")
        kitsune_bench = benchmark_kitsune(model_kitsune, input_data, num_iterations)
        kitsune_results.append(kitsune_bench)
        print(f"    Mean: {kitsune_bench['mean_time_ms']:.2f}ms")
        print(f"    Speedup: {baseline['mean_time_ms'] / kitsune_bench['mean_time_ms']:.2f}x")
        print()
        
        # Cleanup
        del model_baseline, model_kitsune, input_data
        torch.cuda.empty_cache()
    
    # Aggregate results
    baseline_mean_times = [r['mean_time_ms'] for r in baseline_results]
    kitsune_mean_times = [r['mean_time_ms'] for r in kitsune_results]
    
    baseline_avg = sum(baseline_mean_times) / len(baseline_mean_times)
    kitsune_avg = sum(kitsune_mean_times) / len(kitsune_mean_times)
    speedup = baseline_avg / kitsune_avg
    
    baseline_memory_avg = sum(r['peak_memory_mb'] for r in baseline_results) / num_runs
    kitsune_memory_avg = sum(r['peak_memory_mb'] for r in kitsune_results) / num_runs
    memory_reduction = ((baseline_memory_avg - kitsune_memory_avg) / baseline_memory_avg) * 100
    
    results = {
        'model': 'ResNet-18',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': {
            'gpu': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'kitsune_version': kitsune.__version__
        },
        'configuration': {
            'num_runs': num_runs,
            'num_iterations': num_iterations,
            'batch_size': 64,
            'input_shape': [3, 224, 224],
            'num_classes': 1000
        },
        'baseline': {
            'mean_time_ms': baseline_avg,
            'std_time_ms': torch.tensor(baseline_mean_times).std().item(),
            'peak_memory_mb': baseline_memory_avg,
            'all_runs': baseline_results
        },
        'kitsune': {
            'mean_time_ms': kitsune_avg,
            'std_time_ms': torch.tensor(kitsune_mean_times).std().item(),
            'peak_memory_mb': kitsune_memory_avg,
            'all_runs': kitsune_results
        },
        'improvement': {
            'speedup': speedup,
            'memory_reduction_percent': memory_reduction
        }
    }
    
    return results


def save_results(results: Dict, output_dir: Path):
    """Save benchmark results to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'resnet18_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file


def print_summary(results: Dict):
    """Print benchmark summary"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY - ResNet-18")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"GPU: {results['hardware']['gpu']}")
    print(f"Runs: {results['configuration']['num_runs']}")
    print()
    print(f"Baseline:        {results['baseline']['mean_time_ms']:.2f}ms ± {results['baseline']['std_time_ms']:.2f}ms")
    print(f"Kitsune:         {results['kitsune']['mean_time_ms']:.2f}ms ± {results['kitsune']['std_time_ms']:.2f}ms")
    print(f"Speedup:         {results['improvement']['speedup']:.2f}x")
    print()
    print(f"Baseline Memory: {results['baseline']['peak_memory_mb']:.1f} MB")
    print(f"Kitsune Memory:  {results['kitsune']['peak_memory_mb']:.1f} MB")
    print(f"Memory Reduction: {results['improvement']['memory_reduction_percent']:.1f}%")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark ResNet-18 with Kitsune')
    parser.add_argument('--runs', type=int, default=5, help='Number of benchmark runs')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per run')
    parser.add_argument('--output', type=str, default='../results', help='Output directory')
    
    args = parser.parse_args()
    
    print("ResNet-18 Benchmark Starting...")
    print()
    
    results = run_benchmark(num_runs=args.runs, num_iterations=args.iterations)
    
    output_dir = Path(__file__).parent / args.output
    save_results(results, output_dir)
    
    print_summary(results)
