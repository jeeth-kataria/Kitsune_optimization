# AMP (Automatic Mixed Precision) API

The AMP module provides automatic mixed precision training capabilities.

---

## Overview

Kitsune integrates with PyTorch's automatic mixed precision (AMP) to provide efficient FP16/BF16 training. The AMP module includes:

- **GradScaler**: Enhanced gradient scaling with memory management
- **autocast**: Context manager for automatic precision casting
- **AMPConfig**: Configuration for AMP settings

---

## AMPConfig

::: kitsune.amp.config.AMPConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

---

## GradScaler

Enhanced gradient scaler that extends PyTorch's `torch.cuda.amp.GradScaler` with:

- Better integration with Kitsune's memory pools
- Detailed overflow tracking
- Performance statistics

**Note**: See PyTorch's [GradScaler documentation](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler) for base functionality.

---

## autocast

Context manager for automatic mixed precision casting.

**Note**: This is a wrapper around PyTorch's `torch.cuda.amp.autocast`. See [PyTorch AMP documentation](https://pytorch.org/docs/stable/amp.html) for details.

---

## Usage Examples

### Basic AMP Usage

```python
from kitsune.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Custom AMP Configuration

```python
from kitsune.amp import AMPConfig, GradScaler

config = AMPConfig(
    dtype=torch.float16,
    growth_interval=2000,
    backoff_factor=0.5,
    growth_factor=2.0,
    enabled=True
)

scaler = GradScaler(
    init_scale=2.**16,
    growth_interval=config.growth_interval,
    backoff_factor=config.backoff_factor
)
```

### Selective AMP

```python
from kitsune.amp import autocast

# Enable AMP only for specific operations
with autocast(enabled=True):
    # These operations use FP16
    hidden = model.encoder(input)

# Back to FP32
output = model.decoder(hidden)
```

---

## Configuration Options

### AMPConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | torch.dtype | `torch.float16` | Target precision (float16/bfloat16) |
| `enabled` | bool | `True` | Enable/disable AMP |
| `growth_interval` | int | `2000` | Steps between scale factor increases |
| `backoff_factor` | float | `0.5` | Scale reduction on overflow |
| `growth_factor` | float | `2.0` | Scale increase factor |
| `init_scale` | float | `65536.0` | Initial gradient scale |

---

## Best Practices

### 1. Use BFloat16 for Better Stability

```python
config = AMPConfig(dtype=torch.bfloat16)
```

BFloat16 has wider dynamic range than Float16, reducing overflow risk.

### 2. Monitor Gradient Scaling

```python
scaler = GradScaler()

for epoch in range(num_epochs):
    # ... training loop ...
    
    if epoch % 10 == 0:
        scale = scaler.get_scale()
        print(f"Current gradient scale: {scale}")
```

### 3. Handle Gradient Clipping

```python
from kitsune.amp import GradScaler

scaler = GradScaler()

# Unscale before clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Then step
scaler.step(optimizer)
scaler.update()
```

---

## See Also

- [Optimizer API](optimizer.md) - Main optimizer interface
- [Mixed Precision Guide](../user-guide/amp.md) - Detailed AMP guide
