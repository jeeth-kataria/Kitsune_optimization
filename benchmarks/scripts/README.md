# Benchmark Scripts

This directory contains executable benchmark scripts for testing Kitsune's performance against baseline PyTorch.

## Quick Start

Run all benchmarks:
```bash
cd benchmarks/scripts
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

## Individual Benchmarks

### MLP Benchmark
```bash
python3 benchmark_mlp.py --runs 5 --iterations 100
```

Tests a multi-layer perceptron with hidden layers [2048, 4096, 2048, 1024].

### LeNet-5 Benchmark
```bash
python3 benchmark_lenet5.py --runs 5 --iterations 100
```

Tests classic CNN architecture on MNIST-sized inputs (28×28).

### ResNet-18 Benchmark
```bash
python3 benchmark_resnet18.py --runs 5 --iterations 100
```

Tests deep residual network on ImageNet-sized inputs (224×224).

## Command-Line Options

All benchmark scripts support:
- `--runs N` - Number of complete benchmark runs (default: 5)
- `--iterations N` - Iterations per run for timing (default: 100)
- `--output DIR` - Output directory for results (default: ../results)

## Visualization

Generate charts from results:
```bash
python3 visualize_results.py
```

Creates:
- `speedup_comparison.png` - Baseline vs Kitsune timing
- `memory_comparison.png` - Memory usage comparison
- `optimization_breakdown.png` - Cumulative optimization impact
- `results_table.png` - Summary table

## Requirements

- CUDA-capable GPU
- PyTorch 2.0+
- Kitsune (torch-kitsune)
- matplotlib (for visualization)
- numpy

Install:
```bash
pip install torch-kitsune matplotlib numpy
```

## Output Format

Results are saved as JSON in `../results/`:
```json
{
  "model": "ResNet-18",
  "timestamp": "2026-01-29 12:00:00",
  "hardware": { ... },
  "configuration": { ... },
  "baseline": {
    "mean_time_ms": 156.3,
    "std_time_ms": 2.1,
    "peak_memory_mb": 2048.0
  },
  "kitsune": {
    "mean_time_ms": 72.1,
    "std_time_ms": 1.5,
    "peak_memory_mb": 1536.0
  },
  "improvement": {
    "speedup": 2.17,
    "memory_reduction_percent": 25.0
  }
}
```

## CI/CD Integration

Benchmarks run automatically via GitHub Actions on:
- Push to main branch
- Pull requests
- Weekly schedule
- Manual workflow dispatch

See `.github/workflows/benchmarks.yml` for configuration.

## Reproducibility

Each benchmark:
1. Runs CUDA warmup (10 iterations)
2. Creates identical models with same weights
3. Uses CUDA events for precise timing
4. Measures peak memory with `torch.cuda.max_memory_allocated()`
5. Performs 5 separate runs with 100 iterations each
6. Reports mean ± std across runs

Environment details are captured in results JSON for verification.
