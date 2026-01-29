#!/usr/bin/env python3
"""
MLP Benchmark Script for Kitsune Optimizer
Tests multi-layer perceptron with multiple hidden layers
"""

import torch
import torch.nn as nn
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Import Kitsune
try:
    import kitsune
    from kitsune import optimize
except ImportError:
    print("ERROR: Kitsune not installed. Run: pip install torch-kitsune")
    exit(1)


class MLPModel(nn.Module):
    """Multi-Layer Perceptron for benchmark testing"""
    
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


def create_dummy_data(batch_size=256, input_dim=1024, device='cuda'):
    """Create dummy input data for benchmarking"""
    return torch.randn(batch_size, input_dim, device=device)


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
    device = input_data.device
    
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
    device = input_data.device
    
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
        model_baseline = MLPModel().to(device)
        model_kitsune = MLPModel().to(device)
        
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
        'model': 'MLP',
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
            'batch_size': 256,
            'input_dim': 1024,
            'hidden_dims': [2048, 4096, 2048, 1024],
            'output_dim': 10
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
    output_file = output_dir / 'mlp_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file


def print_summary(results: Dict):
    """Print benchmark summary"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY - MLP")
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
    
    parser = argparse.ArgumentParser(description='Benchmark MLP with Kitsune')
    parser.add_argument('--runs', type=int, default=5, help='Number of benchmark runs')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per run')
    parser.add_argument('--output', type=str, default='../results', help='Output directory')
    
    args = parser.parse_args()
    
    print("MLP Benchmark Starting...")
    print()
    
    results = run_benchmark(num_runs=args.runs, num_iterations=args.iterations)
    
    output_dir = Path(__file__).parent / args.output
    save_results(results, output_dir)
    
    print_summary(results)
