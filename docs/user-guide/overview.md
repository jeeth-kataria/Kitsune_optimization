# User Guide Overview

Welcome to the Kitsune User Guide! This comprehensive guide covers all aspects of using Kitsune to optimize your PyTorch training workflows.

---

## What You'll Learn

This guide is organized into focused chapters covering different aspects of Kitsune:

### 🚀 [Stream Parallelism](stream-parallelism.md)
Learn how Kitsune achieves concurrent execution across multiple CUDA streams to maximize GPU utilization.

**Topics covered:**
- Understanding stream parallelism
- How Kitsune schedules operations
- Tuning stream count for your model
- Best practices for stream-based optimization

### 🔗 [Kernel Fusion](kernel-fusion.md)
Discover how automatic kernel fusion reduces memory bandwidth and kernel launch overhead.

**Topics covered:**
- Fusion pattern detection
- Supported fusion types
- Custom fusion configuration
- Performance impact analysis

### 💾 [Memory Management](memory-management.md)
Master advanced memory techniques including pooling, prefetching, and lifetime analysis.

**Topics covered:**
- Memory pool configuration
- Double buffering for data loading
- Lifetime analysis and reuse
- Reducing peak memory usage

### ⚡ [Mixed Precision Training](amp.md)
Accelerate training with automatic mixed precision while maintaining accuracy.

**Topics covered:**
- FP16 vs BFloat16
- Gradient scaling
- Selective precision
- Debugging precision issues

### 📊 [Profiling & Benchmarking](profiling.md)
Use Kitsune's profiling tools to understand and optimize performance.

**Topics covered:**
- Performance profiling
- Memory tracking
- Identifying bottlenecks
- Comparing configurations

---

## Getting Help

- 💬 [GitHub Discussions](https://github.com/yourusername/kitsune/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/kitsune/issues)
- 📧 Email: support@kitsune-ml.org

---

## Contributing to Documentation

Found an error or want to improve the docs? Contributions are welcome!

1. Fork the repository
2. Edit the documentation in `docs/`
3. Submit a pull request

See [Contributing Guidelines](../contributing.md) for details.
