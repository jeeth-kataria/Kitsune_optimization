# 📊 Kitsune Benchmarks

This directory contains comprehensive benchmark results, methodology, and reproduction scripts for Kitsune's performance claims.

## 🎯 Summary Results

Measured on **NVIDIA RTX 3050 (4GB VRAM)** with PyTorch 2.0+ and CUDA 11.8:

| Model | Architecture | Baseline (ms/iter) | Kitsune (ms/iter) | **Speedup** | Memory Savings |
|-------|--------------|-------------------|-------------------|-------------|----------------|
| **MLP** | 3-layer FC (MNIST) | 45.2 | 22.6 | **2.00x** ⚡ | 35% |
| **LeNet-5** | CNN (MNIST) | 38.4 | 18.3 | **2.10x** ⚡ | 42% |
| **ResNet-18** | Deep CNN (CIFAR-10) | 125.7 | 57.8 | **2.17x** ⚡ | 38% |

**Average Speedup: 2.09x across all architectures**

---

## 📁 Directory Structure

```
benchmarks/
├── README.md                    # This file
├── methodology.md               # Detailed benchmark methodology
├── hardware_specs.md            # Hardware and software environment
├── results/
│   ├── mlp_results.json         # Raw MLP benchmark data
│   ├── lenet5_results.json      # Raw LeNet-5 benchmark data
│   └── resnet18_results.json    # Raw ResNet-18 benchmark data
├── charts/
│   ├── speedup_comparison.png   # Speedup bar chart
│   ├── memory_savings.png       # Memory usage comparison
│   └── optimization_breakdown.png # Impact of each optimization
└── scripts/
    ├── run_all_benchmarks.sh    # Run complete benchmark suite
    ├── benchmark_mlp.py         # MLP benchmark script
    ├── benchmark_lenet5.py      # LeNet-5 benchmark script
    └── benchmark_resnet18.py    # ResNet-18 benchmark script
```

---

## 🚀 Quick Start - Reproduce Results

### Prerequisites
```bash
# Install Kitsune
pip install torch-kitsune

# Install benchmark dependencies
pip install matplotlib tensorboard pytest-benchmark
```

### Run Benchmarks
```bash
# Run all benchmarks (takes ~10-15 minutes)
cd benchmarks
bash scripts/run_all_benchmarks.sh

# Or run individual benchmarks
python scripts/benchmark_mlp.py
python scripts/benchmark_lenet5.py
python scripts/benchmark_resnet18.py
```

### View Results
Results are saved to `results/*.json` and automatically generate charts in `charts/`.

---

## 📈 Detailed Results

### MLP (3-Layer Fully Connected)
- **Dataset**: MNIST (28x28 grayscale images)
- **Architecture**: 784 → 512 → 256 → 10
- **Batch Size**: 64
- **Optimizer**: Adam (lr=1e-3)

**Performance:**
- Baseline: 45.2 ms/iteration
- Kitsune: 22.6 ms/iteration
- **Speedup: 2.00x**
- Memory Saved: 35%

### LeNet-5 (Convolutional Neural Network)
- **Dataset**: MNIST
- **Architecture**: Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10)
- **Batch Size**: 64
- **Optimizer**: Adam (lr=1e-3)

**Performance:**
- Baseline: 38.4 ms/iteration
- Kitsune: 18.3 ms/iteration
- **Speedup: 2.10x**
- Memory Saved: 42%

### ResNet-18 (Deep Residual Network)
- **Dataset**: CIFAR-10 (32x32 RGB images)
- **Architecture**: 18-layer ResNet with skip connections
- **Batch Size**: 128
- **Optimizer**: SGD (lr=0.1, momentum=0.9)

**Performance:**
- Baseline: 125.7 ms/iteration
- Kitsune: 57.8 ms/iteration
- **Speedup: 2.17x**
- Memory Saved: 38%

---

## 🔬 Optimization Breakdown

Impact of individual optimizations on **ResNet-18**:

| Optimization | Time (ms/iter) | Speedup vs Baseline | Cumulative Speedup |
|--------------|----------------|---------------------|-------------------|
| Baseline PyTorch | 125.7 | 1.00x | 1.00x |
| + Stream Parallelism | 92.3 | 1.36x | 1.36x |
| + Memory Pooling | 78.1 | 1.61x | 1.61x |
| + Kernel Fusion | 65.4 | 1.92x | 1.92x |
| + CUDA Graphs | 57.8 | **2.17x** | **2.17x** |

**Each optimization contributes:**
- Stream Parallelism: 36% improvement
- Memory Pooling: 25% improvement (on top of streams)
- Kernel Fusion: 19% improvement (on top of pooling)
- CUDA Graphs: 13% improvement (final polish)

---

## 🖥️ Hardware & Environment

See [hardware_specs.md](hardware_specs.md) for complete details.

**GPU:**
- NVIDIA GeForce RTX 3050 (Laptop)
- 4GB GDDR6 VRAM
- CUDA Compute Capability 8.6
- Driver Version: 535.154.05

**Software:**
- PyTorch 2.1.0
- CUDA 11.8
- cuDNN 8.7.0
- Python 3.10.12
- Ubuntu 22.04 LTS

---

## 📐 Methodology

See [methodology.md](methodology.md) for detailed methodology.

**Key Points:**
- Each benchmark run 100 iterations with 10 warmup iterations
- Results averaged over 5 separate runs
- GPU temperature monitored (stayed below 75°C)
- Power limit set to default (no overclocking)
- No other GPU processes running during tests
- Timing measured using CUDA events for accuracy

---

## 📊 Charts & Visualizations

### Speedup Comparison
![Speedup Comparison](charts/speedup_comparison.png)

Bar chart showing baseline vs Kitsune performance across all models.

### Memory Savings
![Memory Savings](charts/memory_savings.png)

Comparison of peak GPU memory usage between baseline and Kitsune.

### Optimization Breakdown
![Optimization Breakdown](charts/optimization_breakdown.png)

Waterfall chart showing the impact of each optimization layer.

---

## 🔄 Reproducing on Your Hardware

Want to test on your own GPU? Follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/jeeth-kataria/Kitsune_optimization.git
   cd Kitsune_optimization/benchmarks
   ```

2. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run benchmarks**
   ```bash
   bash scripts/run_all_benchmarks.sh
   ```

4. **Compare results**
   - Your results will be saved to `results/your_gpu_*.json`
   - Charts will be generated in `charts/`
   - Share your results by opening a GitHub Discussion!

---

## 📝 Reporting Your Results

Found different results on your hardware? We'd love to hear about it!

**Please open a [GitHub Discussion](https://github.com/jeeth-kataria/Kitsune_optimization/discussions) with:**
- Your GPU model and VRAM
- Operating System
- PyTorch/CUDA versions
- Benchmark results (JSON file)
- Any unexpected behavior

---

## 🎯 Benchmark Goals

These benchmarks aim to demonstrate:

✅ **Consistent speedup** across different architectures (MLP, CNN, ResNet)  
✅ **Reproducible results** with detailed methodology  
✅ **Real-world performance** on consumer hardware (RTX 3050)  
✅ **Memory efficiency** alongside speed improvements  
✅ **Transparent testing** with all scripts available  

---

## ⚠️ Known Limitations

- Results measured on specific hardware (RTX 3050 Laptop)
- Performance may vary on different GPU architectures
- Triton kernel fusion only tested on Linux
- Batch sizes optimized for 4GB VRAM
- Training only (inference benchmarks coming soon)

---

## 📖 Related Documentation

- [Main README](../README.md) - Project overview
- [Methodology](methodology.md) - Detailed testing methodology
- [Hardware Specs](hardware_specs.md) - Complete environment details
- [API Documentation](https://jeeth-kataria.github.io/Kitsune_optimization) - Full API reference

---

## 📞 Questions?

- 💬 [GitHub Discussions](https://github.com/jeeth-kataria/Kitsune_optimization/discussions)
- 🐛 [Issue Tracker](https://github.com/jeeth-kataria/Kitsune_optimization/issues)
- 📧 Contact maintainer via GitHub

---

**Last Updated:** January 29, 2026  
**Benchmark Version:** v0.1.0  
**Kitsune Version:** 0.1.0
