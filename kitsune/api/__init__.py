"""
Kitsune API - User-facing API layer (Device-Aware)

Contains device-aware optimization:
- KitsuneOptimizer: Auto-configures based on hardware (T4, Ampere, CPU, etc.)
- OptimizationConfig: Configuration for optimizations
- optimize_model: Quick setup helper
- Device detection and hardware-specific configs
"""

from .optimizer_v2 import (
    KitsuneOptimizer,
    OptimizationConfig,
    optimize_model,
    optimize,
    get_optimizer,
    benchmark_optimization,
    print_benchmark,
)

from .device_config import (
    detect_hardware,
    get_optimal_config,
    apply_hardware_optimizations,
    print_hardware_info,
    show_performance_guide,
    HardwareInfo,
    DeviceType,
    T4Config,
    AmpereConfig,
    HopperConfig,
    CPUConfig,
    GenericCUDAConfig,
)

# Legacy compatibility
try:
    from .optimizer import OptimizationStats
except ImportError:
    OptimizationStats = None

__all__ = [
    # Core API
    "KitsuneOptimizer",
    "OptimizationConfig",
    "optimize_model",
    "optimize",
    "get_optimizer",
    "benchmark_optimization",
    "print_benchmark",
    # Device config
    "detect_hardware",
    "get_optimal_config",
    "apply_hardware_optimizations",
    "print_hardware_info",
    "show_performance_guide",
    "HardwareInfo",
    "DeviceType",
    "T4Config",
    "AmpereConfig",
    "HopperConfig",
    "CPUConfig",
    "GenericCUDAConfig",
    # Legacy
    "OptimizationStats",
]
