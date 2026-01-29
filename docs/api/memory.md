# Memory API

The memory module provides advanced memory management including pooling, prefetching, and lifetime analysis.

---

## Overview

The memory module provides efficient memory management:

- **MemoryPool**: GPU memory pooling
- **DoubleBuffer**: Double buffering for data transfer
- **Prefetcher**: Data prefetching
- **LifetimeAnalyzer**: Memory lifetime analysis

## Key Features

- **Memory pooling**: Reduce allocation overhead
- **Prefetching**: Hide data transfer latency
- **Lifetime analysis**: Optimize memory reuse
- **Double buffering**: Overlap compute and data transfer

---

## Usage Examples

### Memory Pool

```python
from kitsune.memory import MemoryPool

# Create memory pool
pool = MemoryPool(
    pool_size=512 * 1024 * 1024,  # 512 MB
    device='cuda'
)

# Allocate from pool
tensor = pool.allocate(shape=(1024, 1024), dtype=torch.float32)

# Free memory
pool.free(tensor)

# View statistics
stats = pool.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Double Buffering

```python
from kitsune.memory import DoubleBuffer

# Create double buffer for data loading
buffer = DoubleBuffer(
    buffer_size=(64, 3, 224, 224),
    dtype=torch.float32,
    device='cuda'
)

for data in dataloader:
    # Prefetch next batch while processing current
    current = buffer.get_current()
    buffer.prefetch_next(data)
    
    # Process current batch
    output = model(current)
    
    # Swap buffers
    buffer.swap()
```

### Memory Prefetching

```python
from kitsune.memory import Prefetcher

prefetcher = Prefetcher(
    lookahead=2,  # Prefetch 2 batches ahead
    device='cuda'
)

# Wrap dataloader
prefetch_loader = prefetcher.wrap(dataloader)

for data, target in prefetch_loader:
    # Data is already on GPU when loop starts
    output = model(data)
```

### Lifetime Analysis

```python
from kitsune.memory import LifetimeAnalyzer

analyzer = LifetimeAnalyzer()

# Analyze tensor lifetimes
for step, tensors in enumerate(training_tensors):
    analyzer.record_usage(step, tensors)

# Get optimization suggestions
suggestions = analyzer.get_optimization_suggestions()
print(f"Can reuse {suggestions['reusable_bytes'] / 1e9:.2f} GB")
```

---

## Configuration Options

### MemoryPool Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size` | int | `512MB` | Total pool size in bytes |
| `block_size` | int | `4MB` | Allocation block size |
| `device` | str | `'cuda'` | Target device |
| `defragment` | bool | `True` | Enable auto-defragmentation |

### Prefetcher Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookahead` | int | `2` | Batches to prefetch |
| `num_workers` | int | `2` | Prefetch worker threads |
| `pin_memory` | bool | `True` | Use pinned memory |

---

## Best Practices

### 1. Use Memory Pooling for Frequent Allocations

```python
from kitsune.memory import MemoryPool

pool = MemoryPool(pool_size=1024 * 1024 * 1024)  # 1 GB

# Frequent allocations benefit from pooling
for iteration in range(1000):
    temp = pool.allocate((256, 256))
    # ... use temp ...
    pool.free(temp)
```

### 2. Monitor Memory Usage

```python
from kitsune.profiler import MemoryTracker

tracker = MemoryTracker()

with tracker:
    # Training code
    for batch in dataloader:
        output = model(batch)

# Print memory report
tracker.print_summary()
```

### 3. Optimize Lifetime with Analysis

```python
from kitsune.memory import LifetimeAnalyzer

analyzer = LifetimeAnalyzer()

# Record lifetimes during training
for step in range(num_steps):
    tensors = get_active_tensors()
    analyzer.record_usage(step, tensors)

# Get reuse opportunities
reuse_plan = analyzer.compute_reuse_plan()
```

---

## Memory Optimization Tips

### Reduce Peak Memory

```python
# Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)
```

### Enable Memory Efficient Settings

```python
from kitsune import KitsuneOptimizer

optimizer = KitsuneOptimizer(
    base_optimizer,
    memory_efficient=True,  # Enable memory optimizations
    prefetch_enabled=True,   # Prefetch data
    pool_size=512 * 1024 * 1024  # 512 MB pool
)
```

---

## See Also

- [Optimizer API](optimizer.md) - Main optimizer interface
- [Memory Management Guide](../user-guide/memory-management.md) - Detailed guide
- [Profiler API](profiler.md) - Memory profiling
