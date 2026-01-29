# Quick Start Guide

Get started with Kitsune in just 5 minutes! This guide walks you through the basics of integrating Kitsune into your PyTorch training workflow.

---

## Prerequisites

Before starting, ensure you have:

- ✅ Kitsune installed ([Installation Guide](installation.md))
- ✅ PyTorch ≥ 2.0.0
- ✅ CUDA-enabled GPU (recommended)
- ✅ Basic familiarity with PyTorch

---

## Step 1: Import Kitsune

Start by importing Kitsune alongside your usual PyTorch imports:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kitsune import KitsuneOptimizer
```

That's it! Kitsune is designed to work seamlessly with existing PyTorch code.

---

## Step 2: Create Your Model

Define your model as you normally would. Kitsune works with any PyTorch model:

```python
# Example: Simple feedforward network
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
).cuda()

# Or use any pre-trained model
# from torchvision.models import resnet50
# model = resnet50(pretrained=True).cuda()
```

!!! tip "Model Compatibility"
    Kitsune works with all PyTorch models: custom architectures, torchvision models, HuggingFace transformers, and more!

---

## Step 3: Wrap Your Optimizer

Create your optimizer as usual, then wrap it with `KitsuneOptimizer`:

```python
# Create base optimizer
base_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# Wrap with Kitsune
optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_amp=True,        # Automatic mixed precision
    enable_fusion=True,     # Kernel fusion
    num_streams=4,          # Stream parallelism
    profile=True            # Performance profiling
)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_amp` | `False` | Enable automatic mixed precision training |
| `enable_fusion` | `False` | Enable kernel fusion for elementwise ops |
| `num_streams` | `4` | Number of CUDA streams for parallelism |
| `profile` | `False` | Enable detailed performance profiling |
| `memory_pool_size` | `512MB` | Size of memory pool for allocation |
| `capture_graph` | `True` | Use CUDA graphs when possible |

---

## Step 4: Train As Usual

Your training loop remains unchanged! Kitsune optimizes execution automatically:

```python
# Standard training loop
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to GPU
        data, target = data.cuda(), target.cuda()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (Kitsune optimizes here!)
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

!!! success "Zero Code Changes"
    Notice how the training loop is identical to standard PyTorch - no special APIs or modifications needed!

---

## Step 5: View Performance Results

After training, view Kitsune's performance analysis:

```python
# Print performance summary
optimizer.print_summary()
```

Example output:

```
╔════════════════════════════════════════════════════════════╗
║             Kitsune Performance Summary                    ║
╚════════════════════════════════════════════════════════════╝

Performance Metrics:
  Total iterations:        1000
  Avg iteration time:      24.3 ms  (baseline: 45.2 ms)
  Speedup:                 1.86x
  Throughput:              41.2 samples/sec
  
Stream Parallelism:
  Streams utilized:        4
  Avg stream efficiency:   87.3%
  Concurrent ops:          12.4 (avg)
  
Kernel Fusion:
  Fused kernels:           156
  Fusion efficiency:       91.2%
  Memory saved:            2.3 GB
  
Memory Management:
  Peak memory:             8.2 GB
  Pool hits:               94.7%
  Allocation overhead:     -32.1%

Mixed Precision:
  FP16 coverage:           78.4%
  Memory savings:          41.2%
  Accuracy delta:          +0.02%
```

---

## Complete Example

Here's a full end-to-end example:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from kitsune import KitsuneOptimizer

# 1. Create synthetic dataset
X = torch.randn(10000, 1024)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Define model
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# 3. Create optimizer
base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_amp=True,
    enable_fusion=True,
    num_streams=4
)

# 4. Training loop
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}')

# 5. View results
optimizer.print_summary()
```

---

## Advanced Configuration

### Custom Stream Assignment

Control which operations run on which streams:

```python
from kitsune import KitsuneOptimizer, StreamConfig

config = StreamConfig(
    num_streams=6,
    priority_streams=[0, 1],  # High-priority ops
    compute_streams=[2, 3, 4], # Compute-heavy ops
    memory_streams=[5]         # Memory-bound ops
)

optimizer = KitsuneOptimizer(
    base_optimizer,
    stream_config=config
)
```

### Selective Fusion

Enable fusion only for specific operation types:

```python
from kitsune.fusion import FusionConfig

fusion_config = FusionConfig(
    enable_elementwise=True,   # Fuse elementwise ops
    enable_reduction=True,      # Fuse reductions
    enable_matmul=False,        # Don't fuse matmuls
    max_fusion_size=8          # Max ops per fused kernel
)

optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_fusion=True,
    fusion_config=fusion_config
)
```

### Profiling Options

Get detailed profiling data:

```python
from kitsune.profiler import ProfilerConfig

profiler_config = ProfilerConfig(
    trace_events=True,         # CUDA event tracing
    track_memory=True,          # Memory usage tracking
    log_interval=10,           # Log every 10 steps
    save_report='profile.json' # Save to file
)

optimizer = KitsuneOptimizer(
    base_optimizer,
    profile=True,
    profiler_config=profiler_config
)

# After training, generate detailed report
optimizer.save_profile_report('detailed_report.html')
```

---

## Common Patterns

### Mixed Precision Training

```python
optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_amp=True,
    amp_dtype=torch.float16,  # or torch.bfloat16
    amp_growth_interval=2000
)

# Training loop remains the same!
# Kitsune handles scaling and gradient management
```

### Gradient Accumulation

```python
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(num_epochs):
    # ... training loop ...
    scheduler.step()
```

---

## Next Steps

Now that you've got the basics, explore more advanced features:

<div class="grid cards" markdown>

-   :material-lightning-bolt: **Stream Parallelism**
    
    Learn how Kitsune achieves parallelism
    
    [:octicons-arrow-right-24: Stream Guide](../user-guide/stream-parallelism.md)

-   :material-merge: **Kernel Fusion**
    
    Understand fusion patterns and optimization
    
    [:octicons-arrow-right-24: Fusion Guide](../user-guide/kernel-fusion.md)

-   :material-memory: **Memory Management**
    
    Optimize memory usage and allocation
    
    [:octicons-arrow-right-24: Memory Guide](../user-guide/memory-management.md)

-   :material-chart-line: **Profiling**
    
    Deep dive into performance analysis
    
    [:octicons-arrow-right-24: Profiling Guide](../user-guide/profiling.md)

</div>

---

## Tips for Best Performance

!!! tip "Performance Tips"
    1. **Enable all features**: Start with `enable_amp=True` and `enable_fusion=True`
    2. **Tune stream count**: Try 4-8 streams depending on model complexity
    3. **Profile first**: Use `profile=True` to identify bottlenecks
    4. **Warm up**: Run a few iterations before benchmarking
    5. **CUDA graphs**: Let Kitsune capture graphs automatically for repeated patterns

!!! warning "Common Pitfalls"
    - Don't mix `.cpu()` and `.cuda()` operations frequently
    - Avoid small batch sizes (< 16) which limit parallelism
    - Ensure CUDA is properly initialized before creating optimizer
    - Use consistent dtypes to maximize fusion opportunities

---

## Troubleshooting

**Q: My speedup is less than expected**

- Check GPU utilization with `nvidia-smi`
- Enable profiling to identify bottlenecks
- Try increasing `num_streams`
- Ensure batch size is large enough

**Q: Getting CUDA errors**

- Verify CUDA version compatibility
- Check memory usage (reduce batch size if needed)
- Try disabling CUDA graphs: `capture_graph=False`

**Q: Accuracy degradation with AMP**

- Use `torch.bfloat16` instead of `float16`
- Adjust gradient scaling parameters
- Monitor loss scaling in profiler output

---

## Getting Help

- 📖 [User Guide](../user-guide/overview.md) - Comprehensive documentation
- 🔧 [API Reference](../api/optimizer.md) - Detailed API docs
- 💬 [GitHub Discussions](https://github.com/yourusername/kitsune/discussions) - Ask questions
- 🐛 [Issue Tracker](https://github.com/yourusername/kitsune/issues) - Report bugs

Happy optimizing! 🦊
