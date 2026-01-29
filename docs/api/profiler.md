# Profiler API

The profiler module provides detailed performance analysis and metrics collection.

---

## Overview

The profiler module provides comprehensive performance analysis:

- **CUDATimer**: High-precision GPU timing
- **MemoryTracker**: GPU memory usage tracking
- **MetricsCollector**: General metrics collection
- **Reporter**: Report generation

## Key Features

- **CUDA event timing**: Precise GPU operation timing
- **Memory profiling**: Track allocations and usage
- **Metrics collection**: Gather custom metrics
- **Report generation**: Generate detailed reports

---

## Usage Examples

### Basic Profiling

```python
from kitsune.profiler import CUDATimer

timer = CUDATimer()

# Time a code block
with timer.measure('forward_pass'):
    output = model(input)

# Get elapsed time
elapsed = timer.get_elapsed('forward_pass')
print(f"Forward pass: {elapsed:.2f} ms")
```

### Memory Profiling

```python
from kitsune.profiler import MemoryTracker

tracker = MemoryTracker()

# Track memory usage
with tracker:
    for batch in dataloader:
        output = model(batch)
        loss.backward()

# Print memory report
tracker.print_summary()

# Get detailed stats
stats = tracker.get_stats()
print(f"Peak memory: {stats['peak_allocated'] / 1e9:.2f} GB")
```

### Comprehensive Metrics

```python
from kitsune.profiler import MetricsCollector

collector = MetricsCollector()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Record metrics
        collector.record('batch_size', data.size(0))
        
        with collector.measure_time('iteration'):
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        collector.record('loss', loss.item())

# Generate report
report = collector.generate_report()
print(report)
```

### Custom Reporter

```python
from kitsune.profiler import Reporter, MetricsCollector

collector = MetricsCollector()
# ... collect metrics ...

# Create reporter
reporter = Reporter(collector)

# Generate different report formats
reporter.save_json('metrics.json')
reporter.save_html('report.html')
reporter.save_csv('metrics.csv')

# Print summary
reporter.print_summary()
```

---

## Profiling Features

### 1. CUDA Event Timing

High-precision GPU timing using CUDA events:

```python
from kitsune.profiler import CUDATimer

timer = CUDATimer()

timer.start('kernel_launch')
# GPU operation
output = model(input)
timer.stop('kernel_launch')

elapsed = timer.get_elapsed('kernel_launch')
```

### 2. Memory Tracking

Track GPU memory allocations and deallocations:

```python
from kitsune.profiler import MemoryTracker

tracker = MemoryTracker(track_allocations=True)

# Track all allocations
with tracker:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)

# View allocation timeline
timeline = tracker.get_allocation_timeline()
```

### 3. Stream Profiling

Profile multi-stream execution:

```python
from kitsune.profiler import StreamProfiler

profiler = StreamProfiler(num_streams=4)

# Profile stream usage
with profiler:
    # Multi-stream execution
    for stream_id, task in enumerate(tasks):
        with torch.cuda.stream(streams[stream_id]):
            execute_task(task)

# View stream efficiency
profiler.print_stream_efficiency()
```

---

## Configuration Options

### CUDATimer Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `synchronize` | bool | `True` | Synchronize before timing |
| `warmup` | int | `3` | Warmup iterations |
| `repeat` | int | `10` | Timing repetitions |

### MemoryTracker Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_allocations` | bool | `True` | Track individual allocations |
| `track_timeline` | bool | `True` | Record allocation timeline |
| `interval` | int | `100` | Sampling interval (ms) |

---

## Performance Analysis

### Identifying Bottlenecks

```python
from kitsune.profiler import MetricsCollector

collector = MetricsCollector()

# Profile different components
with collector.measure_time('data_loading'):
    data = next(dataloader)

with collector.measure_time('forward'):
    output = model(data)

with collector.measure_time('backward'):
    loss.backward()

# Analyze bottlenecks
analysis = collector.analyze_bottlenecks()
print(f"Bottleneck: {analysis['bottleneck']}")
```

### Comparing Configurations

```python
from kitsune.profiler import compare_runs

# Run 1: Baseline
metrics1 = profile_training(config1)

# Run 2: With Kitsune
metrics2 = profile_training(config2)

# Compare
comparison = compare_runs(metrics1, metrics2)
print(f"Speedup: {comparison['speedup']:.2f}x")
```

---

## Report Formats

### JSON Report

```python
reporter.save_json('profile.json')
```

Output structure:
```json
{
  "timing": {
    "forward_pass": {"mean": 24.3, "std": 1.2, "min": 22.1, "max": 27.5},
    "backward_pass": {"mean": 31.5, "std": 2.1, "min": 28.3, "max": 35.2}
  },
  "memory": {
    "peak_allocated": 8589934592,
    "peak_reserved": 10737418240
  },
  "throughput": {
    "samples_per_sec": 1245.6,
    "iterations_per_sec": 41.2
  }
}
```

### HTML Report

```python
reporter.save_html('report.html')
```

Generates interactive HTML report with:
- Charts and visualizations
- Detailed metrics tables
- Timeline views
- Comparison graphs

### CSV Export

```python
reporter.save_csv('metrics.csv')
```

Exports metrics in CSV format for analysis in Excel, pandas, etc.

---

## Best Practices

### 1. Profile Representative Workloads

```python
# Use actual training data, not synthetic
collector = MetricsCollector()

# Profile multiple epochs
for epoch in range(5):
    # ... training code with profiling ...
    pass

# Get stable averages
report = collector.generate_report()
```

### 2. Measure End-to-End Performance

```python
with timer.measure('full_iteration'):
    with timer.measure('data_loading'):
        data = next(dataloader)
    
    with timer.measure('forward'):
        output = model(data)
    
    with timer.measure('backward'):
        loss.backward()
    
    with timer.measure('optimizer'):
        optimizer.step()
```

### 3. Compare Against Baseline

```python
# Baseline run
baseline_metrics = run_with_profiling(use_kitsune=False)

# Kitsune run
kitsune_metrics = run_with_profiling(use_kitsune=True)

# Compare
speedup = baseline_metrics['time'] / kitsune_metrics['time']
print(f"Speedup: {speedup:.2f}x")
```

---

## See Also

- [Optimizer API](optimizer.md) - Main optimizer interface
- [Profiling Guide](../user-guide/profiling.md) - Detailed profiling guide
- [Benchmarks](../benchmarks/methodology.md) - Benchmarking methodology
