# 🦊 Kitsune v0.1.0 - Initial Release

**First production release of Kitsune - CUDA-accelerated dataflow optimizer for PyTorch!**

[![PyPI](https://img.shields.io/pypi/v/torch-kitsune)](https://pypi.org/project/torch-kitsune/)
[![Python](https://img.shields.io/pypi/pyversions/torch-kitsune)](https://pypi.org/project/torch-kitsune/)
[![License](https://img.shields.io/github/license/jeeth-kataria/Kitsune_optimization)](https://github.com/jeeth-kataria/Kitsune_optimization/blob/main/LICENSE)

---

## 🚀 Installation

```bash
pip install torch-kitsune
```

Then in your code:
```python
import kitsune

optimizer = kitsune.KitsuneOptimizer(
    torch.optim.Adam,
    model.parameters(),
    lr=1e-3
)
```

---

## ✨ Key Features

### 🏆 **2-2.2x Speedup** on Consumer GPUs
Proven performance gains across MLP, CNN, and ResNet architectures on NVIDIA RTX 3050 (4GB VRAM)

### 🔌 **Drop-in Integration**
Single-line optimizer wrapper - no code changes needed to your existing PyTorch training loops

### 🧠 **Intelligent Multi-Stream Scheduling**
Dependency-aware execution across 4-8 CUDA streams for maximum parallelism

### 💾 **Zero-Copy Memory Pooling**
Smart tensor reuse with size-class binning reduces GPU allocations by 80%

### ⚡ **Automatic Kernel Fusion**
Triton-based fusion of common patterns (LayerNorm, Dropout, etc.) reduces kernel launches by 30-50%

### 🎯 **Mixed Precision (AMP)**
Automatic FP16/BF16 conversion with dynamic loss scaling for 1.5-2x throughput boost

### 📈 **CUDA Graph Caching**
Capture and replay execution graphs for 15-25% overhead reduction

---

## 📊 Benchmark Results

Measured on **NVIDIA RTX 3050 (4GB VRAM)**:

| Model | Baseline (ms/iter) | Kitsune (ms/iter) | **Speedup** |
|-------|-------------------|-------------------|-------------|
| **MLP** | 45 | 22 | **2.0x** ⚡ |
| **LeNet-5** | 38 | 18 | **2.1x** ⚡ |
| **ResNet-18** | 125 | 58 | **2.2x** ⚡ |

---

## 🎯 What's Included

### Core Modules
- **Stream Scheduler** - Dataflow-aware CUDA stream management
- **Memory Pool** - Zero-allocation tensor recycling system
- **Kernel Fusion** - Pattern-based operation fusion engine
- **CUDA Graphs** - Automatic graph capture and replay
- **AMP Integration** - Seamless mixed precision support

### Developer Tools
- Comprehensive profiling and metrics
- Built-in benchmark suite
- Extensive documentation with examples
- 95%+ test coverage

### Documentation
- [Quick Start Guide](https://jeeth-kataria.github.io/Kitsune_optimization/getting-started/quickstart/)
- [API Reference](https://jeeth-kataria.github.io/Kitsune_optimization/api/optimizer/)
- [Benchmarking Guide](https://jeeth-kataria.github.io/Kitsune_optimization/benchmarks/methodology/)
- [Example Notebooks](https://github.com/jeeth-kataria/Kitsune_optimization/tree/main/examples)

---

## 🔧 Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **CUDA Toolkit**: 11.0+
- **GPU**: NVIDIA GPU with Compute Capability 6.0+
- **Triton**: 2.1+ (optional, Linux only - for kernel fusion)

**Recommended**: NVIDIA RTX 3050/3060 or better (4GB+ VRAM)

---

## 📦 Package Details

- **PyPI Package**: [`torch-kitsune`](https://pypi.org/project/torch-kitsune/)
- **Import Name**: `kitsune`
- **Version**: 0.1.0
- **License**: MIT

---

## 🙏 Acknowledgments

This project was developed to make GPU-accelerated deep learning more accessible on resource-constrained hardware. Special thanks to the PyTorch and Triton communities for their excellent tools and documentation.

---

## 📚 Learn More

- 📖 [Full Documentation](https://jeeth-kataria.github.io/Kitsune_optimization/)
- 🐛 [Report Issues](https://github.com/jeeth-kataria/Kitsune_optimization/issues)
- 💬 [Discussions](https://github.com/jeeth-kataria/Kitsune_optimization/discussions)
- ⭐ [Star on GitHub](https://github.com/jeeth-kataria/Kitsune_optimization)

---

**Install now and start accelerating your PyTorch training!** 🚀

```bash
pip install torch-kitsune
```
