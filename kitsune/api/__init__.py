"""
Kitsune API - User-facing API layer (Device-Aware)

Contains device-aware optimization:
- KitsuneOptimizer: Auto-configures based on hardware (T4, Ampere, CPU, etc.)
- OptimizationConfig: Configuration for optimizations
- optimize_model: Quick setup helper
- Device detection and hardware-specific configs
"""

from .device_config import (
    AmpereConfig,
    CPUConfig,
    DeviceType,
    GenericCUDAConfig,
    HardwareInfo,
    HopperConfig,
    T4Config,
    apply_hardware_optimizations,
    detect_hardware,
    get_optimal_config,
    print_hardware_info,
    show_performance_guide,
)
from .optimizer_v2 import (
    KitsuneOptimizer,
    OptimizationConfig,
    benchmark_optimization,
    get_optimizer,
    optimize,
    optimize_model,
    print_benchmark,
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
