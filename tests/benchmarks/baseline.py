"""
Baseline Benchmark Suite

Provides baseline PyTorch performance measurements for comparison
with Kitsune optimizations.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitsune.profiler import CUDATimer, MemoryTracker, Profiler


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    batch_size: int = 64
    num_iterations: int = 100
    warmup_iterations: int = 10
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    # Model-specific configs
    input_size: tuple = (1, 28, 28)  # Default for MNIST
    num_classes: int = 10

    # Data generation
    num_samples: int = 1000


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    model_name: str
    config: BenchmarkConfig

    # Timing (milliseconds)
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    # Phase breakdown
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float

    # Throughput
    throughput_samples_per_sec: float

    # Memory (MB)
    peak_memory_mb: float
    model_memory_mb: float

    # Raw data
    iteration_times: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.model_name}, "
            f"time={self.mean_time_ms:.2f}±{self.std_time_ms:.2f}ms, "
            f"throughput={self.throughput_samples_per_sec:.1f} samples/s, "
            f"peak_mem={self.peak_memory_mb:.1f}MB)"
        )

    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            "=" * 60,
            f"Benchmark Results: {self.model_name}",
            "=" * 60,
            f"\nConfiguration:",
            f"  Batch size:  {self.config.batch_size}",
            f"  Iterations:  {self.config.num_iterations}",
            f"  Device:      {self.config.device}",
            f"  Input size:  {self.config.input_size}",
            f"\nTiming (per iteration):",
            f"  Mean:        {self.mean_time_ms:.2f} ms",
            f"  Std:         {self.std_time_ms:.2f} ms",
            f"  Min/Max:     {self.min_time_ms:.2f} / {self.max_time_ms:.2f} ms",
            f"\nPhase Breakdown:",
            f"  Forward:     {self.forward_time_ms:.2f} ms ({self.forward_time_ms/self.mean_time_ms*100:.1f}%)",
            f"  Backward:    {self.backward_time_ms:.2f} ms ({self.backward_time_ms/self.mean_time_ms*100:.1f}%)",
            f"  Optimizer:   {self.optimizer_time_ms:.2f} ms ({self.optimizer_time_ms/self.mean_time_ms*100:.1f}%)",
            f"\nThroughput:",
            f"  {self.throughput_samples_per_sec:.1f} samples/second",
            f"\nMemory:",
            f"  Model:       {self.model_memory_mb:.1f} MB",
            f"  Peak:        {self.peak_memory_mb:.1f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)


def generate_synthetic_data(
    config: BenchmarkConfig,
    device: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for benchmarking."""
    device = device or config.device

    # Generate random input
    batch_shape = (config.num_samples,) + config.input_size
    X = torch.randn(batch_shape, dtype=config.dtype)

    # Generate random labels
    y = torch.randint(0, config.num_classes, (config.num_samples,))

    return X.to(device), y.to(device)


def run_baseline_benchmark(
    model: nn.Module,
    config: BenchmarkConfig,
    optimizer_fn: Callable = None,
    loss_fn: nn.Module = None,
    data: tuple[torch.Tensor, torch.Tensor] = None,
) -> BenchmarkResult:
    """
    Run baseline PyTorch benchmark.

    Args:
        model: PyTorch model to benchmark
        config: Benchmark configuration
        optimizer_fn: Function to create optimizer (default: Adam)
        loss_fn: Loss function (default: CrossEntropyLoss)
        data: Optional (X, y) data tuple. If None, generates synthetic data.

    Returns:
        BenchmarkResult with timing and memory statistics
    """
    device = config.device
    model = model.to(device)
    model.train()

    # Setup optimizer and loss
    if optimizer_fn is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer_fn(model.parameters())

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Generate or use provided data
    if data is None:
        X, y = generate_synthetic_data(config, device)
    else:
        X, y = data
        X, y = X.to(device), y.to(device)

    # Create data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Get model memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        model_memory = torch.cuda.memory_allocated(device) / (1024**2)
    else:
        model_memory = 0.0

    # Profiler setup
    timer = CUDATimer(enabled=True)
    memory = MemoryTracker(enabled=True)

    # Storage for iteration times
    iteration_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []

    # Training loop
    iteration = 0
    total_iterations = config.warmup_iterations + config.num_iterations

    while iteration < total_iterations:
        for batch_x, batch_y in dataloader:
            if iteration >= total_iterations:
                break

            is_warmup = iteration < config.warmup_iterations

            # Clear memory stats for this iteration
            if not is_warmup and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            timer.start("forward")
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            fwd_result = timer.stop("forward")

            # Backward pass
            timer.start("backward")
            loss.backward()
            bwd_result = timer.stop("backward")

            # Optimizer step
            timer.start("optimizer")
            optimizer.step()
            opt_result = timer.stop("optimizer")

            # Record times (skip warmup)
            if not is_warmup:
                total_time = (
                    fwd_result.cuda_time_ms + bwd_result.cuda_time_ms + opt_result.cuda_time_ms
                )
                iteration_times.append(total_time)
                forward_times.append(fwd_result.cuda_time_ms)
                backward_times.append(bwd_result.cuda_time_ms)
                optimizer_times.append(opt_result.cuda_time_ms)

            iteration += 1

    # Calculate statistics
    import statistics

    mean_time = statistics.mean(iteration_times)
    std_time = statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0.0

    # Get peak memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory = 0.0

    # Cleanup
    del X, y, dataset, dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        model_name=model.__class__.__name__,
        config=config,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        min_time_ms=min(iteration_times),
        max_time_ms=max(iteration_times),
        forward_time_ms=statistics.mean(forward_times),
        backward_time_ms=statistics.mean(backward_times),
        optimizer_time_ms=statistics.mean(optimizer_times),
        throughput_samples_per_sec=config.batch_size / (mean_time / 1000),
        peak_memory_mb=peak_memory,
        model_memory_mb=model_memory,
        iteration_times=iteration_times,
    )


def run_inference_benchmark(
    model: nn.Module,
    config: BenchmarkConfig,
    data: torch.Tensor = None,
) -> BenchmarkResult:
    """
    Run inference-only benchmark (no backward pass).

    Args:
        model: PyTorch model to benchmark
        config: Benchmark configuration
        data: Optional input data. If None, generates synthetic data.

    Returns:
        BenchmarkResult with timing statistics
    """
    device = config.device
    model = model.to(device)
    model.eval()

    # Generate or use provided data
    if data is None:
        X, _ = generate_synthetic_data(config, device)
    else:
        X = data.to(device)

    # Get model memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        model_memory = torch.cuda.memory_allocated(device) / (1024**2)
    else:
        model_memory = 0.0

    timer = CUDATimer(enabled=True)
    iteration_times = []

    total_iterations = config.warmup_iterations + config.num_iterations

    with torch.no_grad():
        for i in range(total_iterations):
            # Get batch
            start_idx = (i * config.batch_size) % len(X)
            end_idx = start_idx + config.batch_size
            if end_idx > len(X):
                batch_x = X[: config.batch_size]
            else:
                batch_x = X[start_idx:end_idx]

            is_warmup = i < config.warmup_iterations

            if not is_warmup and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            timer.start("inference")
            _ = model(batch_x)
            result = timer.stop("inference")

            if not is_warmup:
                iteration_times.append(result.cuda_time_ms)

    import statistics

    mean_time = statistics.mean(iteration_times)
    std_time = statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0.0

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory = 0.0

    return BenchmarkResult(
        model_name=f"{model.__class__.__name__} (inference)",
        config=config,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        min_time_ms=min(iteration_times),
        max_time_ms=max(iteration_times),
        forward_time_ms=mean_time,
        backward_time_ms=0.0,
        optimizer_time_ms=0.0,
        throughput_samples_per_sec=config.batch_size / (mean_time / 1000),
        peak_memory_mb=peak_memory,
        model_memory_mb=model_memory,
        iteration_times=iteration_times,
    )


def compare_results(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult,
) -> str:
    """Generate comparison between baseline and optimized results."""
    speedup = baseline.mean_time_ms / optimized.mean_time_ms
    improvement_pct = (speedup - 1) * 100

    memory_diff = optimized.peak_memory_mb - baseline.peak_memory_mb
    memory_pct = (memory_diff / baseline.peak_memory_mb) * 100 if baseline.peak_memory_mb > 0 else 0

    lines = [
        "=" * 70,
        "Performance Comparison",
        "=" * 70,
        f"\n{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>12}",
        "-" * 70,
        f"{'Time (ms)':<25} {baseline.mean_time_ms:>15.2f} {optimized.mean_time_ms:>15.2f} {speedup:>11.2f}x",
        f"{'Throughput (samples/s)':<25} {baseline.throughput_samples_per_sec:>15.1f} {optimized.throughput_samples_per_sec:>15.1f} {speedup:>11.2f}x",
        f"{'Peak Memory (MB)':<25} {baseline.peak_memory_mb:>15.1f} {optimized.peak_memory_mb:>15.1f} {memory_pct:>+10.1f}%",
        "-" * 70,
        f"\nSpeedup: {speedup:.2f}x ({improvement_pct:+.1f}% faster)",
        "=" * 70,
    ]

    return "\n".join(lines)
