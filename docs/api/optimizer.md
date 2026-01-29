# Optimizer API

The `KitsuneOptimizer` is the main entry point for using Kitsune. It wraps any PyTorch optimizer and adds automatic optimization features.

---

## KitsuneOptimizer

The main optimizer class that wraps PyTorch optimizers and applies Kitsune's optimizations.

### Constructor

```python
KitsuneOptimizer(
    optimizer: torch.optim.Optimizer,
    enable_amp: bool = False,
    enable_fusion: bool = False,
    num_streams: int = 4,
    profile: bool = False,
    amp_config: Optional[AMPConfig] = None,
    fusion_config: Optional[FusionConfig] = None,
    stream_config: Optional[StreamConfig] = None,
    memory_pool_size: int = 512 * 1024 * 1024,
    capture_graph: bool = True
)
```

### Parameters

- `optimizer` (torch.optim.Optimizer): Base PyTorch optimizer to wrap
- `enable_amp` (bool): Enable automatic mixed precision. Default: False
- `enable_fusion` (bool): Enable kernel fusion. Default: False
- `num_streams` (int): Number of CUDA streams for parallelism. Default: 4
- `profile` (bool): Enable performance profiling. Default: False
- `amp_config` (AMPConfig, optional): AMP configuration
- `fusion_config` (FusionConfig, optional): Fusion configuration
- `stream_config` (StreamConfig, optional): Stream configuration
- `memory_pool_size` (int): Memory pool size in bytes. Default: 512MB
- `capture_graph` (bool): Enable CUDA graph capture. Default: True

---

## Usage Examples

### Basic Usage

```python
import torch
from kitsune import KitsuneOptimizer

# Create base optimizer
base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Wrap with Kitsune
optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_amp=True,
    enable_fusion=True,
    num_streams=4
)

# Use in training loop
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Advanced Configuration

```python
from kitsune import KitsuneOptimizer
from kitsune.amp import AMPConfig
from kitsune.fusion import FusionConfig

# Configure AMP
amp_config = AMPConfig(
    dtype=torch.float16,
    growth_interval=2000,
    backoff_factor=0.5
)

# Configure fusion
fusion_config = FusionConfig(
    enable_elementwise=True,
    enable_reduction=True,
    max_fusion_size=8
)

# Create optimizer with custom configs
optimizer = KitsuneOptimizer(
    base_optimizer,
    enable_amp=True,
    amp_config=amp_config,
    enable_fusion=True,
    fusion_config=fusion_config,
    num_streams=6,
    profile=True
)
```

---

## Configuration Classes

### AMPConfig

Configuration class for automatic mixed precision settings.

**Attributes:**
- `dtype` (torch.dtype): Target precision (torch.float16 or torch.bfloat16)
- `enabled` (bool): Enable/disable AMP
- `growth_interval` (int): Steps between scale factor increases
- `backoff_factor` (float): Scale reduction on overflow
- `growth_factor` (float): Scale increase factor
- `init_scale` (float): Initial gradient scale

### FusionConfig

Configuration class for kernel fusion settings.

**Attributes:**
- `enable_elementwise` (bool): Enable elementwise fusion
- `enable_reduction` (bool): Enable reduction fusion
- `enable_matmul` (bool): Enable matmul fusion
- `max_fusion_size` (int): Maximum operations per fused kernel
- `min_fusion_benefit` (float): Minimum speedup threshold

### StreamConfig

Configuration class for stream parallelism settings.

**Attributes:**
- `num_streams` (int): Number of CUDA streams
- `priority_streams` (List[int]): High-priority stream indices
- `compute_streams` (List[int]): Compute-heavy operation streams
- `memory_streams` (List[int]): Memory-bound operation streams

---

## Methods Reference

### step()

Performs a single optimization step.

```python
optimizer.step()
```

**Parameters**: None

**Returns**: None

**Description**: Executes the optimization step with Kitsune's enhancements including stream parallelism, kernel fusion, and automatic mixed precision.

### zero_grad()

Clears gradients of all optimized tensors.

```python
optimizer.zero_grad(set_to_none=True)
```

**Parameters**:

- `set_to_none` (bool, optional): If True, set gradients to None instead of zero. Default: True

**Returns**: None

### state_dict()

Returns the state of the optimizer as a dictionary.

```python
state = optimizer.state_dict()
```

**Returns**: dict - Contains optimizer state and Kitsune-specific state

### load_state_dict()

Loads the optimizer state.

```python
optimizer.load_state_dict(state_dict)
```

**Parameters**:

- `state_dict` (dict): Optimizer state dictionary

### print_summary()

Prints a detailed performance summary.

```python
optimizer.print_summary()
```

**Returns**: None

**Description**: Displays performance metrics including speedup, stream efficiency, fusion statistics, and memory usage.

### save_profile()

Saves profiling data to a file.

```python
optimizer.save_profile(filepath='profile.json')
```

**Parameters**:

- `filepath` (str): Path to save the profile data

**Returns**: None

---

## Properties

### `optimizer.base_optimizer`

Access the underlying PyTorch optimizer.

```python
base_opt = optimizer.base_optimizer
learning_rate = base_opt.param_groups[0]['lr']
```

### `optimizer.profiler`

Access the profiler instance (if profiling is enabled).

```python
if optimizer.profiler:
    metrics = optimizer.profiler.get_metrics()
```

### `optimizer.scheduler`

Access the execution scheduler.

```python
scheduler = optimizer.scheduler
num_streams = scheduler.num_streams
```

---

## See Also

- [Scheduler API](scheduler.md) - Execution scheduling
- [AMP API](amp.md) - Mixed precision training
- [Fusion API](fusion.md) - Kernel fusion
- [Quick Start Guide](../getting-started/quickstart.md) - Usage examples
