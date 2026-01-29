# Installation

This guide covers everything you need to install and configure Kitsune for your PyTorch projects.

---

## Requirements

Before installing Kitsune, ensure your system meets these requirements:

| Component | Minimum Version | Recommended | Notes |
|-----------|----------------|-------------|-------|
| **Python** | 3.8 | 3.10+ | Full type hint support in 3.10+ |
| **PyTorch** | 2.0.0 | 2.1.0+ | CUDA graph support improved in 2.1+ |
| **CUDA** | 11.8 | 12.1+ | Required for GPU acceleration |
| **GPU Memory** | 4GB | 8GB+ | Depends on model size |
| **Triton** | 2.0.0 | 2.1.0+ | Optional, for kernel fusion |

!!! info "CUDA Compatibility"
    Kitsune requires CUDA for GPU acceleration. CPU-only mode is supported but offers limited performance benefits.

---

## Installation Methods

### PyPI Installation (Recommended)

The simplest way to install Kitsune is via pip:

```bash
pip install kitsune-torch
```

This installs the core package with all required dependencies.

### From Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/yourusername/kitsune.git
cd kitsune

# Install in development mode
pip install -e .
```

!!! tip "Development Mode"
    Installing with `-e` creates an editable installation, so changes to the source code are immediately reflected without reinstalling.

### Docker Installation

For a containerized environment with all dependencies pre-configured:

```bash
# Pull the official image
docker pull kitsune/kitsune-torch:latest

# Run with GPU support
docker run --gpus all -it kitsune/kitsune-torch:latest
```

---

## Optional Dependencies

Kitsune supports optional features that require additional packages:

### Kernel Fusion (Triton)

For automatic kernel fusion capabilities:

```bash
pip install kitsune-torch[triton]
```

Or install Triton separately:

```bash
pip install triton>=2.0.0
```

### Development Tools

For contributing or running tests:

```bash
pip install kitsune-torch[dev]
```

This includes:

- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pre-commit` - Git hooks

### All Features

To install everything:

```bash
pip install kitsune-torch[all]
```

---

## Verification

After installation, verify that Kitsune is working correctly:

### Basic Import Test

```python
import kitsune
print(f"Kitsune version: {kitsune.__version__}")
```

### CUDA Availability

```python
import torch
from kitsune import KitsuneOptimizer

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check Kitsune CUDA features
from kitsune.cuda import CUDAGraphManager
print("CUDA graph support: OK")
```

### Full System Check

Run the built-in diagnostic:

```bash
python -m kitsune.diagnostics
```

Expected output:

```
✓ PyTorch version: 2.1.0
✓ CUDA available: True
✓ CUDA version: 12.1
✓ GPU: NVIDIA A100-SXM4-40GB
✓ Triton available: True
✓ Triton version: 2.1.0
✓ Kitsune installation: OK

All checks passed! Kitsune is ready to use.
```

---

## Troubleshooting

### CUDA Not Found

**Issue**: `RuntimeError: CUDA not available`

**Solutions**:

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. Check CUDA environment variables:
   ```bash
   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH
   ```

### Triton Compilation Errors

**Issue**: `Triton compilation failed`

**Solutions**:

1. Update Triton:
   ```bash
   pip install --upgrade triton
   ```

2. Disable fusion temporarily:
   ```python
   optimizer = KitsuneOptimizer(
       base_optimizer,
       enable_fusion=False  # Disable until Triton is fixed
   )
   ```

3. Check Triton compatibility:
   ```python
   import triton
   print(f"Triton version: {triton.__version__}")
   ```

### Memory Errors

**Issue**: `CUDA out of memory`

**Solutions**:

1. Reduce stream count:
   ```python
   optimizer = KitsuneOptimizer(
       base_optimizer,
       num_streams=2  # Reduce from default 4
   )
   ```

2. Enable gradient checkpointing:
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

3. Monitor memory usage:
   ```python
   from kitsune.profiler import MemoryTracker
   
   tracker = MemoryTracker()
   # ... training code ...
   tracker.print_summary()
   ```

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'kitsune'`

**Solutions**:

1. Verify installation:
   ```bash
   pip list | grep kitsune
   ```

2. Check Python path:
   ```python
   import sys
   print('\n'.join(sys.path))
   ```

3. Reinstall in current environment:
   ```bash
   pip install --force-reinstall kitsune-torch
   ```

---

## Platform-Specific Notes

### Linux

Kitsune works out-of-the-box on most Linux distributions with CUDA installed.

### Windows

!!! warning "Windows Support"
    Windows support is experimental. Some features (like CUDA graphs) may have limitations.

Additional steps:
1. Install Visual Studio Build Tools
2. Set up CUDA environment variables
3. Use Anaconda for easier dependency management

### macOS

!!! note "macOS Limitations"
    macOS doesn't support CUDA. MPS (Metal Performance Shaders) support is planned but not yet available.

For development on macOS without GPU:
```bash
pip install kitsune-torch --no-deps
pip install torch torchvision  # CPU-only versions
```

---

## Next Steps

Now that Kitsune is installed, check out:

- [Quick Start Guide](quickstart.md) - Get up and running in 5 minutes
- [User Guide](../user-guide/overview.md) - Deep dive into features
- [API Reference](../api/optimizer.md) - Complete API documentation

---

## Getting Help

If you encounter issues not covered here:

- 📖 Check the [FAQ](../user-guide/faq.md)
- 💬 Ask on [GitHub Discussions](https://github.com/yourusername/kitsune/discussions)
- 🐛 Report bugs on [GitHub Issues](https://github.com/yourusername/kitsune/issues)
- 📧 Email: support@kitsune-ml.org
