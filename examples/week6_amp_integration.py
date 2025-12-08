"""
Week 6 Demo - AMP Integration

Demonstrates:
- Automatic Mixed Precision configuration
- Gradient scaling with KitsuneGradScaler
- AMPOptimizer for seamless AMP training
- Precision mode detection and benchmarking
- Full training loop with AMP

Note: Best performance with Ampere GPUs (RTX 30xx) or newer
that support BF16 and TF32.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kitsune.amp import (
    AMPConfig,
    PrecisionMode,
    KitsuneGradScaler,
    autocast_context,
    AMPOptimizer,
    get_autocast_dtype,
)
from kitsune.amp.autocast import get_precision_info, enable_tf32
from kitsune.amp.optimizer import AMPTrainer
from tests.benchmarks.models import create_mlp, create_resnet18


def demo_precision_info():
    """Show available precision modes."""
    print("=" * 60)
    print("Precision Capabilities")
    print("=" * 60)

    info = get_precision_info()

    print(f"\nCUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"GPU: {info.get('gpu_name', 'Unknown')}")
        print(f"Compute capability: {info.get('compute_capability', 'Unknown')}")
        print(f"\nSupported precision modes:")
        print(f"  FP16: {info['fp16_supported']}")
        print(f"  BF16: {info['bf16_supported']}")
        print(f"  TF32: {info['tf32_supported']}")


def demo_amp_config():
    """Demonstrate AMP configuration."""
    print("\n" + "=" * 60)
    print("AMP Configuration Demo")
    print("=" * 60)

    # Auto-detect best precision
    config = AMPConfig(precision_mode=PrecisionMode.AUTO)
    print(f"\nAuto-detected precision: {config.precision_mode.name}")
    print(f"Using dtype: {config.get_dtype()}")

    # Show configuration
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Show operation precision rules
    print("\nOperations using FP32 (for numerical stability):")
    for op in sorted(list(config.ops_fp32)[:5]):
        print(f"  - {op}")

    print("\nOperations using reduced precision:")
    for op in sorted(list(config.ops_fp16)[:5]):
        print(f"  - {op}")


def demo_grad_scaler():
    """Demonstrate gradient scaler."""
    print("\n" + "=" * 60)
    print("Gradient Scaler Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create scaler
    scaler = KitsuneGradScaler()
    print(f"\nInitial scale: {scaler.scale:.0f}")

    # Create simple model and optimizer
    model = nn.Linear(100, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Simulate training steps
    print("\nSimulating training steps...")
    for step in range(10):
        x = torch.randn(32, 100, device="cuda")

        with torch.cuda.amp.autocast():
            y = model(x)
            loss = y.sum()

        scaler.scale_loss(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 3 == 0:
            print(f"  Step {step}: scale={scaler.scale:.0f}")

    print(f"\nFinal scale: {scaler.scale:.0f}")
    print(scaler.summary())


def demo_autocast():
    """Demonstrate autocast context."""
    print("\n" + "=" * 60)
    print("Autocast Context Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create tensors
    x = torch.randn(1000, 1000, device="cuda")
    weight = torch.randn(1000, 1000, device="cuda")

    print(f"\nInput dtype: {x.dtype}")

    # Without autocast
    result_fp32 = x @ weight
    print(f"Result without autocast: {result_fp32.dtype}")

    # With autocast
    with autocast_context():
        result_amp = x @ weight
    print(f"Result with autocast: {result_amp.dtype}")

    # Verify correctness
    max_diff = (result_fp32 - result_amp.float()).abs().max()
    print(f"Max difference: {max_diff:.6f}")


def demo_amp_optimizer():
    """Demonstrate AMPOptimizer."""
    print("\n" + "=" * 60)
    print("AMPOptimizer Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).cuda()

    # Create base optimizer and wrap with AMP
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    amp_optimizer = AMPOptimizer(base_optimizer)

    # Training loop
    criterion = nn.CrossEntropyLoss()

    print("\nTraining with AMPOptimizer...")
    for step in range(10):
        # Synthetic data
        x = torch.randn(64, 784, device="cuda")
        target = torch.randint(0, 10, (64,), device="cuda")

        # Forward/backward with AMP
        loss = amp_optimizer.forward_backward(model, x, target, criterion)
        amp_optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    print(amp_optimizer.summary())


def demo_amp_benchmark():
    """Benchmark FP32 vs AMP."""
    print("\n" + "=" * 60)
    print("AMP Benchmark Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).cuda()

    x = torch.randn(256, 1024, device="cuda")

    # Warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    # Benchmark FP32
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    fp32_time = start.elapsed_time(end) / 100

    # Benchmark AMP
    start.record()
    for _ in range(100):
        with autocast_context():
            _ = model(x)
    end.record()
    torch.cuda.synchronize()
    amp_time = start.elapsed_time(end) / 100

    print(f"\nForward pass benchmark (batch=256, hidden=2048):")
    print(f"  FP32: {fp32_time:.3f} ms")
    print(f"  AMP:  {amp_time:.3f} ms")
    print(f"  Speedup: {fp32_time/amp_time:.2f}x")


def demo_full_training():
    """Demonstrate full training loop with AMP."""
    print("\n" + "=" * 60)
    print("Full Training Demo")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Create synthetic dataset
    num_samples = 1000
    x_data = torch.randn(num_samples, 784)
    y_data = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model
    model = create_mlp(device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create AMP trainer
    trainer = AMPTrainer(model, optimizer, criterion)

    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    print("\nTraining complete!")


def demo_training_benchmark():
    """Benchmark full training loop FP32 vs AMP."""
    print("\n" + "=" * 60)
    print("Training Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    import time

    # Create synthetic dataset
    num_samples = 512
    x_data = torch.randn(num_samples, 784)
    y_data = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x_data, y_data)

    def train_fp32():
        """Train with FP32."""
        model = create_mlp(device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=64)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

    def train_amp():
        """Train with AMP."""
        model = create_mlp(device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=64)

        trainer = AMPTrainer(model, optimizer, criterion)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            trainer.train_step(batch_x, batch_y)

        torch.cuda.synchronize()

    # Warmup
    train_fp32()
    train_amp()

    # Benchmark FP32
    start = time.perf_counter()
    for _ in range(10):
        train_fp32()
    fp32_time = (time.perf_counter() - start) / 10 * 1000

    # Benchmark AMP
    start = time.perf_counter()
    for _ in range(10):
        train_amp()
    amp_time = (time.perf_counter() - start) / 10 * 1000

    print(f"\nTraining epoch benchmark (512 samples):")
    print(f"  FP32: {fp32_time:.1f} ms")
    print(f"  AMP:  {amp_time:.1f} ms")
    print(f"  Speedup: {fp32_time/amp_time:.2f}x")


def main():
    """Run all Week 6 demos."""
    print("=" * 60)
    print("Kitsune Week 6 - AMP Integration Demo")
    print("=" * 60)

    demo_precision_info()
    demo_amp_config()
    demo_grad_scaler()
    demo_autocast()
    demo_amp_optimizer()
    demo_amp_benchmark()
    demo_full_training()
    demo_training_benchmark()

    print("\n" + "=" * 60)
    print("Week 6 Demo Complete!")
    print("=" * 60)
    print("\nAMP benefits:")
    print("  - Up to 2x faster training with minimal code changes")
    print("  - Reduced GPU memory usage (half precision)")
    print("  - Automatic gradient scaling prevents underflow")
    print("  - Works with existing PyTorch models")
    print("\nNext: Week 7-8 - Polish and Competition Submission")


if __name__ == "__main__":
    main()
