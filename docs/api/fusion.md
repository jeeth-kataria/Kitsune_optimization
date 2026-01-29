# Fusion API

The fusion module handles automatic kernel fusion for optimizing elementwise operations.

---

## Overview

The fusion module provides automatic kernel fusion capabilities:

- **FusionEngine**: Main fusion engine
- **FusionDetector**: Pattern detection
- **FusionPattern**: Fusion pattern definitions

## Key Features

- **Automatic detection**: Identify fusible operation sequences
- **Pattern matching**: Match against known fusion patterns
- **Code generation**: Generate fused kernels
- **Performance tracking**: Monitor fusion effectiveness

---

## Usage Examples

### Basic Fusion

```python
from kitsune.fusion import FusionEngine

engine = FusionEngine(enable_fusion=True)

# Detect and fuse operations
fused_ops = engine.fuse(operations)
```

### Custom Fusion Patterns

```python
from kitsune.fusion import FusionPattern, FusionEngine

# Define custom pattern
pattern = FusionPattern(
    name='custom_activation',
    ops=['linear', 'relu', 'dropout'],
    fusion_type='elementwise'
)

# Register pattern
engine = FusionEngine()
engine.register_pattern(pattern)
```

### Fusion Configuration

```python
from kitsune.fusion import FusionConfig, FusionEngine

config = FusionConfig(
    enable_elementwise=True,
    enable_reduction=True,
    enable_matmul=False,
    max_fusion_size=8,
    min_fusion_benefit=1.2  # Minimum 1.2x speedup
)

engine = FusionEngine(config=config)
```

---

## Supported Fusion Patterns

### Elementwise Fusion

Fuses consecutive elementwise operations:

- Activation functions (ReLU, GELU, Sigmoid, etc.)
- Arithmetic operations (+, -, *, /)
- Normalization operations

Example:
```python
# Original: 3 separate kernels
x = input + bias
x = F.relu(x)
x = x * scale

# Fused: Single kernel
x = fused_elementwise(input, bias, scale)
```

### Reduction Fusion

Fuses reduction operations:

- Mean, sum, max, min
- Softmax, log_softmax
- Layer normalization

### Matmul Fusion

Fuses matrix operations with activations:

- Linear + activation
- Matmul + bias + activation

---

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_elementwise` | bool | `True` | Enable elementwise fusion |
| `enable_reduction` | bool | `True` | Enable reduction fusion |
| `enable_matmul` | bool | `False` | Enable matmul fusion |
| `max_fusion_size` | int | `8` | Max ops per fused kernel |
| `min_fusion_benefit` | float | `1.1` | Minimum speedup threshold |

---

## Performance Tips

### 1. Use Compatible Operations

Fusion works best with:
- Same dtype operations
- Same device tensors
- Compatible shapes

### 2. Profile Fusion Effectiveness

```python
from kitsune.fusion import FusionEngine

engine = FusionEngine(profile=True)
# ... run operations ...

stats = engine.get_fusion_stats()
print(f"Fused kernels: {stats['num_fused']}")
print(f"Fusion efficiency: {stats['efficiency']:.2%}")
```

### 3. Custom Fusion for Domain-Specific Patterns

```python
from kitsune.fusion import FusionPattern

# Custom pattern for transformer attention
attention_pattern = FusionPattern(
    name='fused_attention',
    ops=['matmul', 'softmax', 'matmul'],
    fusion_type='attention'
)
```

---

## See Also

- [Optimizer API](optimizer.md) - Main optimizer interface
- [Kernel Fusion Guide](../user-guide/kernel-fusion.md) - Detailed fusion guide
