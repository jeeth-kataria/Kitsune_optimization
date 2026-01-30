# Changelog

All notable changes to Kitsune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-01-30

### Added

#### Hardware-Specific Backends 🚀
- **T4 Optimizer** (`kitsune/backends/t4_optimizer.py`): Optimized for Tesla T4 GPUs
  - INT8 quantization support (61 TOPS)
  - FP16 mixed precision (65 TFLOPS)
  - JIT trace → freeze → optimize_for_inference pipeline
  - Achieved **4.06x speedup** on Google Colab T4

- **Apple Silicon Optimizer** (`kitsune/backends/apple_optimizer.py`): Native M1/M2/M3 support
  - MPS backend with channels-last memory format
  - Chip detection (M1/M2/M3/M4)
  - CoreML integration for Neural Engine
  - Achieved **45x speedup** on M1 Pro

- **RTX Optimizer** (`kitsune/backends/rtx_optimizer.py`): For RTX 30xx/40xx GPUs
  - TF32 tensor core acceleration
  - FP8 support for RTX 40 series
  - Sparsity optimizations
  - CUDA graphs for repeated patterns

- **Backend Selector** (`kitsune/backends/backend_selector.py`): Auto hardware detection
  - `detect_platform()`: Identifies T4, RTX, Apple Silicon
  - `get_optimal_backend()`: Returns best optimizer
  - `auto_optimize()`: One-line optimization API

#### Platform Test Suite
- `benchmarks/platform_tests/test_t4.py`: Comprehensive T4 benchmark (Colab-ready)
- `benchmarks/platform_tests/test_apple.py`: Apple Silicon benchmark
- `benchmarks/platform_tests/test_rtx.py`: RTX GPU benchmark

### Performance Results

| Platform | Model | Speedup |
|----------|-------|---------|
| **T4 (Colab)** | ResNet-50 | **4.06x** |
| **Apple M1 Pro** | MobileNetV3 | **45.7x** |
| **Apple M1 Pro** | ResNet-50 | **34.7x** |
| **Apple M1 Pro** | ResNet-18 | **21.9x** |

### Fixed
- JIT operation order: trace → freeze → optimize_for_inference (was causing errors)
- Deprecated `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda', ...)`

---

## [Unreleased]

### Added
- CONTRIBUTING.md with development guidelines
- CHANGELOG.md for tracking project evolution
- CODE_OF_CONDUCT.md for community standards
- GitHub issue and PR templates

### Fixed
- Fixed typo in README.md installation path

---

## [0.1.0] - 2026-01-27

### Added

#### Core Features
- **CUDA Stream Parallelism**: Execute independent operations concurrently across 4-8 streams
- **Zero-Copy Memory Pooling**: Intelligent tensor reuse with size-class binning (512B to 64MB bins)
- **Kernel Fusion Engine**: torch.compile and Triton-based fusion of common patterns
- **Automatic Mixed Precision (AMP)**: Integrated FP16/BF16 support with dynamic loss scaling
- **Dataflow Scheduler**: Dependency-aware scheduling with priority-based execution
- **CUDA Graph Caching**: Capture and replay execution graphs for repeated patterns

#### API
- `KitsuneOptimizer`: Drop-in replacement for PyTorch optimizers
- Configurable stream count, memory pool size, and fusion options
- Built-in profiling with `profile=True` option
- Statistics API via `optimizer.get_stats()`

#### Memory Management
- `MemoryPool`: Pre-allocated CUDA memory with size-class binning
- `DoubleBuffer`: Overlapped compute and memory transfer
- `Prefetcher`: Async memory prefetching for data loading
- Lifetime tracking for automatic tensor reuse

#### Profiling & Metrics
- `CUDATimer`: Accurate GPU timing with stream synchronization
- `MemoryTracker`: Track allocations, peak usage, and fragmentation
- `ProfileReporter`: Generate HTML/JSON profiling reports

#### Testing & Examples
- 12 comprehensive unit test files
- Benchmark suite for MLP, CNN, and ResNet architectures
- 6 example scripts demonstrating all features

### Performance

Measured on NVIDIA RTX 3050 (4GB VRAM):

| Model | Baseline | Kitsune | Speedup |
|-------|----------|---------|---------|
| MLP (3-layer, MNIST) | 45 ms/iter | 22 ms/iter | **2.0x** |
| LeNet-5 (MNIST) | 38 ms/iter | 18 ms/iter | **2.1x** |
| ResNet-18 (CIFAR-10) | 125 ms/iter | 58 ms/iter | **2.2x** |

### Technical Details
- **Stream Management**: Priority-based stream allocation with event synchronization
- **Memory Efficiency**: 80% reduction in CUDA allocations through pooling
- **Fusion Patterns**: LayerNorm+Dropout, BatchNorm+ReLU, Attention patterns
- **Compatibility**: Graceful CPU fallback when CUDA unavailable

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.3.0 | 2026-01-30 | Hardware backends: 4x on T4, 45x on Apple Silicon |
| 0.1.0 | 2026-01-27 | Initial release with full optimization stack |

---

[Unreleased]: https://github.com/jeeth-kataria/Kitsune_optimization/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jeeth-kataria/Kitsune_optimization/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/jeeth-kataria/Kitsune_optimization/releases/tag/v0.1.0
