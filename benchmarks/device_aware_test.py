"""
🦊 Kitsune Device-Aware Benchmark Test

Tests the new device-aware configuration system:
1. Detects hardware automatically
2. Applies optimal configuration for detected device
3. Benchmarks all optimization strategies
4. Shows expected vs actual speedup

Supported configurations:
- T4 (SM75): JIT Trace, avoid channels-last
- Ampere (SM80+): torch.compile + TF32 + channels-last  
- Hopper (SM90+): All + FP8 potential
- CPU: JIT + INT8 quantization
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
import json
import gc

# Import Kitsune
try:
    import kitsune
    from kitsune import (
        detect_hardware,
        get_optimal_config,
        apply_hardware_optimizations,
        show_performance_guide,
        optimize_model,
        benchmark_optimization,
        print_benchmark,
        OptimizationConfig,
        KitsuneOptimizer,
        DeviceType,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure kitsune package is installed")
    sys.exit(1)


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_hardware_detection():
    """Test hardware detection."""
    print_header("🔍 Hardware Detection")
    
    hw = detect_hardware()
    print(hw)
    
    print(f"\n📋 Optimal Configuration:")
    config = get_optimal_config(hw)
    
    for key, value in config.items():
        if key != 'hardware_info':
            print(f"  {key}: {value}")
    
    return hw


def test_auto_optimization(hw):
    """Test auto-optimization with device awareness."""
    print_header("🚀 Auto-Optimization Test")
    
    if hw.device_type == DeviceType.CPU:
        device = 'cpu'
        batch_size = 32
    else:
        device = 'cuda'
        batch_size = 64
    
    print(f"Device: {device}, Batch size: {batch_size}")
    
    # Create model
    model = resnet18(weights=None)
    if device == 'cuda':
        model = model.cuda()
    model.eval()
    
    sample_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Test auto-optimization
    print("\n🦊 Running auto-optimization...")
    optimizer = KitsuneOptimizer(model, sample_input)
    
    print(f"\nOptimization Info:")
    info = optimizer.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Quick inference test
    print("\n✅ Running inference test...")
    with torch.no_grad():
        output = optimizer(sample_input)
    print(f"  Output shape: {output.shape}")
    
    clear_gpu()
    return True


def test_benchmark_all_strategies(hw):
    """Benchmark all optimization strategies."""
    print_header("📊 Benchmark All Strategies")
    
    if hw.device_type == DeviceType.CPU:
        device = 'cpu'
        batch_size = 16
    else:
        device = 'cuda'
        batch_size = 64
    
    model = resnet18(weights=None)
    if device == 'cuda':
        model = model.cuda()
    model.eval()
    
    sample_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    print(f"Model: ResNet-18, Batch: {batch_size}, Device: {device}")
    
    results = benchmark_optimization(
        model, 
        sample_input,
        strategies=["none", "jit_trace", "jit_script", "compile"],
        iterations=50,
        warmup=10
    )
    
    print_benchmark(results)
    
    # Validate against expected speedups
    print("\n📈 Performance Analysis:")
    
    if hw.device_type in (DeviceType.T4, DeviceType.TURING):
        expected_min = 1.15
        expected_max = 1.25
        expected_name = "T4/Turing"
    elif hw.device_type in (DeviceType.AMPERE, DeviceType.ADA_LOVELACE):
        expected_min = 1.3
        expected_max = 2.5
        expected_name = "Ampere"
    elif hw.device_type == DeviceType.HOPPER:
        expected_min = 1.5
        expected_max = 4.0
        expected_name = "Hopper"
    elif hw.device_type == DeviceType.CPU:
        expected_min = 1.05
        expected_max = 1.5
        expected_name = "CPU"
    else:
        expected_min = 1.1
        expected_max = 1.5
        expected_name = "Generic CUDA"
    
    best_speedup = 1.0
    best_strategy = "none"
    
    for strategy, data in results['strategies'].items():
        if 'speedup' in data and data['speedup'] > best_speedup:
            best_speedup = data['speedup']
            best_strategy = strategy
    
    print(f"  Expected ({expected_name}): {expected_min:.2f}x - {expected_max:.2f}x")
    print(f"  Actual best: {best_speedup:.2f}x ({best_strategy})")
    
    if best_speedup >= expected_min:
        print(f"  ✅ Performance within expected range!")
    else:
        print(f"  ⚠️  Below expected range (may improve with more warmup)")
    
    clear_gpu()
    return results


def test_specific_configs(hw):
    """Test specific hardware configurations."""
    print_header("🔧 Device-Specific Configuration Tests")
    
    if hw.device_type == DeviceType.CPU:
        device = 'cpu'
        batch_size = 16
    else:
        device = 'cuda'
        batch_size = 32
    
    model = resnet18(weights=None)
    if device == 'cuda':
        model = model.cuda()
    model.eval()
    
    sample_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    configs_to_test = []
    
    if hw.device_type in (DeviceType.T4, DeviceType.TURING):
        configs_to_test = [
            ("T4 Default (JIT)", OptimizationConfig(strategy='jit_trace', auto_configure=False)),
            ("T4 + Compile", OptimizationConfig(strategy='compile', compile_mode='default', auto_configure=False)),
        ]
    elif hw.device_type in (DeviceType.AMPERE, DeviceType.ADA_LOVELACE):
        configs_to_test = [
            ("Ampere Default", OptimizationConfig(strategy='compile', compile_mode='max-autotune', use_tf32=True, auto_configure=False)),
            ("Ampere + Channels-last", OptimizationConfig(strategy='compile', use_channels_last=True, use_tf32=True, auto_configure=False)),
        ]
    elif hw.device_type == DeviceType.CPU:
        configs_to_test = [
            ("CPU JIT", OptimizationConfig(strategy='jit_trace', device='cpu', auto_configure=False)),
            ("CPU Compile", OptimizationConfig(strategy='compile', device='cpu', auto_configure=False)),
        ]
    else:
        configs_to_test = [
            ("Generic JIT", OptimizationConfig(strategy='jit_trace', auto_configure=False)),
            ("Generic Compile", OptimizationConfig(strategy='compile', auto_configure=False)),
        ]
    
    results = {}
    
    for name, config in configs_to_test:
        print(f"\n  Testing: {name}")
        try:
            optimizer = KitsuneOptimizer(model, sample_input, config)
            
            # Time it
            times = []
            with torch.no_grad():
                for _ in range(30):
                    if device == 'cuda':
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        _ = optimizer(sample_input)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                    else:
                        import time
                        start = time.perf_counter()
                        _ = optimizer(sample_input)
                        times.append((time.perf_counter() - start) * 1000)
            
            median = sorted(times)[len(times)//2]
            results[name] = {'median_ms': median, 'strategy': optimizer.strategy_used}
            print(f"    ✓ {median:.2f}ms (strategy: {optimizer.strategy_used})")
            
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"    ✗ Error: {e}")
    
    clear_gpu()
    return results


def test_memory_efficiency(hw):
    """Test memory usage with optimizations."""
    print_header("💾 Memory Efficiency Test")
    
    if hw.device_type == DeviceType.CPU:
        print("  Skipping GPU memory test on CPU")
        return
    
    batch_size = 64
    model = resnet50(weights=None).cuda().eval()
    sample_input = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    torch.cuda.synchronize()
    
    baseline_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Baseline peak memory: {baseline_mem:.2f} GB")
    
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    
    # With optimization
    optimizer = KitsuneOptimizer(model, sample_input)
    
    with torch.no_grad():
        for _ in range(10):
            _ = optimizer(sample_input)
    torch.cuda.synchronize()
    
    optimized_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Optimized peak memory: {optimized_mem:.2f} GB")
    
    savings = (baseline_mem - optimized_mem) / baseline_mem * 100
    print(f"  Memory savings: {savings:.1f}%")
    
    clear_gpu()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  🦊 KITSUNE DEVICE-AWARE OPTIMIZATION TEST")
    print("="*70)
    print(f"  Version: {kitsune.__version__}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
    
    # Test 1: Hardware Detection
    hw = test_hardware_detection()
    
    # Apply hardware optimizations globally
    apply_hardware_optimizations(hw)
    
    # Test 2: Auto-optimization
    test_auto_optimization(hw)
    
    # Test 3: Benchmark all strategies
    test_benchmark_all_strategies(hw)
    
    # Test 4: Device-specific configs
    test_specific_configs(hw)
    
    # Test 5: Memory efficiency
    if hw.device_type != DeviceType.CPU:
        test_memory_efficiency(hw)
    
    # Show performance guide
    print_header("📖 Performance Guide")
    show_performance_guide()
    
    print_header("✅ All Tests Complete")
    print(f"  Hardware: {hw.device_name}")
    print(f"  Configuration: {hw.device_type.value}")
    
    if hw.device_type in (DeviceType.T4, DeviceType.TURING):
        print("\n  💡 T4/Turing Tips:")
        print("     - Use JIT Trace (default)")
        print("     - Avoid channels-last (hurts performance)")
        print("     - Consider INT8 quantization for more speedup")
    elif hw.device_type in (DeviceType.AMPERE, DeviceType.ADA_LOVELACE):
        print("\n  💡 Ampere Tips:")
        print("     - TF32 is enabled automatically")
        print("     - torch.compile max-autotune works well")
        print("     - Channels-last helps CNNs")
    elif hw.device_type == DeviceType.CPU:
        print("\n  💡 CPU Tips:")
        print("     - Enable INT8 quantization for 2-4x speedup")
        print("     - JIT Trace is most compatible")
        print("     - Consider batch size optimization")


if __name__ == "__main__":
    main()
