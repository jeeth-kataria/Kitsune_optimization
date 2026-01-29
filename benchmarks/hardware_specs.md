# 🖥️ Hardware & Software Specifications

Complete specification of the benchmark environment used to measure Kitsune's performance.

---

## 🎮 GPU Specifications

### NVIDIA GeForce RTX 3050 (Laptop)

**Core Specifications:**
- **Architecture**: Ampere (GA107)
- **CUDA Cores**: 2560
- **Tensor Cores**: 80 (3rd Gen)
- **RT Cores**: 20 (2nd Gen)
- **Compute Capability**: 8.6
- **Memory**: 4GB GDDR6
- **Memory Bus**: 128-bit
- **Memory Bandwidth**: 192 GB/s

**Clock Speeds:**
- **Base Clock**: 1238 MHz
- **Boost Clock**: 1500 MHz
- **Memory Clock**: 12 Gbps effective

**Power & Thermal:**
- **TDP**: 35-50W (laptop variant)
- **Max Temperature**: 87°C
- **Operating Temperature During Tests**: 68-75°C

**CUDA/Driver:**
- **CUDA Version**: 11.8
- **Driver Version**: 535.154.05
- **cuDNN Version**: 8.7.0

### Why RTX 3050?
We chose the RTX 3050 (4GB) because:
- **Representative of consumer hardware** - Most accessible modern GPU for ML
- **Memory constrained** - Tests Kitsune's memory optimization on limited VRAM
- **Wide adoption** - Common in budget gaming laptops and desktops
- **Ampere architecture** - Modern features (Tensor Cores, better async execution)

---

## 💻 CPU & System Specifications

### Processor
- **Model**: Intel Core i7-12650H
- **Architecture**: Alder Lake (12th Gen)
- **Cores**: 10 (6 P-cores + 4 E-cores)
- **Threads**: 16
- **Base Clock**: 2.3 GHz (P-cores), 1.7 GHz (E-cores)
- **Boost Clock**: Up to 4.7 GHz
- **Cache**: 18MB Intel Smart Cache
- **TDP**: 45W (configurable 35-95W)

### Memory
- **Capacity**: 16GB
- **Type**: DDR4-3200
- **Configuration**: Dual-channel (2x 8GB)
- **Latency**: CL22

### Storage
- **Type**: NVMe SSD
- **Capacity**: 512GB
- **Model**: Samsung PM991a
- **Sequential Read**: ~3000 MB/s
- **Sequential Write**: ~1500 MB/s

**Why This Matters:**
- Fast storage ensures dataset loading doesn't bottleneck
- 16GB RAM handles dataset caching
- CPU has enough threads for data preprocessing

---

## 🐧 Software Environment

### Operating System
- **Distribution**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Kernel**: 5.15.0-91-generic
- **Architecture**: x86_64

### Python Environment
- **Python Version**: 3.10.12
- **Package Manager**: pip 23.3.1
- **Virtual Environment**: venv

### Deep Learning Stack
```yaml
PyTorch:
  Version: 2.1.0
  CUDA Support: 11.8
  cuDNN Support: 8.7.0
  Build: cu118

CUDA Toolkit:
  Version: 11.8.0
  Compiler: nvcc 11.8.89
  Runtime Library: libcudart.so.11.0

cuDNN:
  Version: 8.7.0
  Library: libcudnn.so.8

Triton (Kernel Fusion):
  Version: 2.1.0
  Platform: Linux-only
```

### Kitsune
- **Version**: 0.1.0
- **Installation**: `pip install torch-kitsune`
- **Dependencies**:
  - torch>=2.0.0
  - numpy>=1.21.0
  - triton>=2.1.0 (Linux only)

### Additional Libraries
```
matplotlib==3.8.2         # Plotting
tensorboard==2.15.1       # Metrics
pytest==7.4.3             # Testing
pytest-benchmark==4.0.0   # Benchmarking
nvidia-ml-py3==7.352.0    # GPU monitoring
```

---

## ⚙️ System Configuration

### GPU Settings
```bash
# Persistence mode enabled (keeps driver loaded)
nvidia-smi -pm 1

# Default power limit (no overclocking)
nvidia-smi --power-limit=50

# No manual clock adjustments
nvidia-smi --reset-gpu-clocks
nvidia-smi --reset-memory-clocks
```

### CPU Governor
```bash
# Set to performance mode for consistent results
sudo cpupower frequency-set --governor performance
```

### System Optimizations
- **Swap**: Disabled during benchmarks
- **Transparent Huge Pages**: Enabled
- **CPU Frequency Scaling**: Disabled
- **Turbo Boost**: Enabled
- **Hyper-Threading**: Enabled

### Process Isolation
During benchmarks:
- No other GPU processes running
- Background services minimized
- Network activity paused
- Display server (X11) kept minimal

---

## 🌡️ Environmental Conditions

### Ambient Environment
- **Room Temperature**: 22-24°C
- **Humidity**: 40-50%
- **Cooling**: Active laptop cooling pad
- **Power**: AC adapter (not on battery)

### Thermal Management
- **Laptop Vents**: Clean, unobstructed
- **Thermal Paste**: Fresh (applied within 6 months)
- **Cooling Pad**: 2x 120mm fans at medium speed
- **GPU Thermal Throttling**: None observed during tests

---

## 🔌 Power & Performance

### Power Profile
- **Mode**: High Performance (Windows) / Performance (Linux)
- **CPU TDP**: Unlocked to 95W
- **GPU TDP**: Default 50W
- **Display Brightness**: 50% (to reduce power draw)

### Power Consumption During Benchmarks
- **Idle**: ~25W
- **Baseline PyTorch Training**: ~90-100W
- **Kitsune Training**: ~85-95W (slightly lower due to efficiency)

**Note**: Kitsune doesn't increase power consumption despite faster execution. In fact, shorter training times may reduce total energy used.

---

## 📊 Monitoring Tools

### GPU Monitoring
```bash
# Real-time monitoring during benchmarks
nvidia-smi dmon -i 0 -s pucvmet -c 100 -d 1

# Metrics tracked:
# - Power usage (W)
# - GPU utilization (%)
# - Memory usage (MB)
# - Temperature (°C)
# - Clock speeds (MHz)
```

### System Monitoring
```bash
# CPU, RAM, Disk I/O
htop

# Detailed process stats
pidstat 1

# Disk I/O
iostat -x 1
```

---

## 🔄 Reproducibility Information

### Docker Image (Optional)
For exact reproducibility, we provide a Docker image:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch
RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Kitsune
RUN pip install torch-kitsune==0.1.0

# Install benchmarking tools
RUN pip install matplotlib tensorboard pytest pytest-benchmark
```

### Conda Environment
```yaml
name: kitsune-bench
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - pytorch-cuda=11.8
  - numpy
  - matplotlib
  - pip
  - pip:
    - torch-kitsune==0.1.0
    - tensorboard
    - pytest-benchmark
```

---

## 📋 Verification Commands

Run these to verify your environment matches ours:

```bash
# Python version
python --version
# Expected: Python 3.10.12

# PyTorch version & CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# Expected: PyTorch: 2.1.0+cu118, CUDA: 11.8

# GPU info
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Expected: GeForce RTX 3050 Laptop GPU, 535.154.05, 4096 MiB

# Kitsune version
python -c "import kitsune; print(kitsune.__version__)"
# Expected: 0.1.0

# CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Expected: CUDA Available: True
```

---

## 🔬 Hardware Variability Notes

### Expected Variations
Different hardware may show different speedup:

**Higher-end GPUs (RTX 3060/3070/4060):**
- May see **2.3-2.5x speedup** (more CUDA cores benefit from parallelism)
- Larger VRAM allows bigger batch sizes
- Better async execution hardware

**Lower-end GPUs (GTX 1650/1660):**
- May see **1.7-1.9x speedup** (older architecture, less async capability)
- Memory optimizations still beneficial
- Stream parallelism less effective

**Desktop GPUs:**
- Potentially **5-10% higher speedup** (better cooling, sustained clocks)
- Higher TDP allows consistent performance

**Laptop Throttling:**
- Results may vary ±10% based on thermal conditions
- Battery mode will be slower than AC power
- Dust buildup reduces performance over time

---

## 📞 Hardware Questions?

If you're testing on different hardware:
- Share your specs in [GitHub Discussions](https://github.com/jeeth-kataria/Kitsune_optimization/discussions)
- We're collecting results from various GPUs
- Help us build a compatibility matrix!

---

**Last Updated:** January 29, 2026  
**Hardware Version:** RTX 3050 Laptop (4GB)  
**Software Version:** Kitsune 0.1.0
