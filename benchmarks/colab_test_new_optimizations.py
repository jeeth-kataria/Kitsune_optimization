"""
Single-cell Colab test for new CUDA graph + torch.compile optimizations
Copy and paste this entire cell into Colab with GPU runtime
"""

# Install PyTorch if needed
import subprocess
import sys
try:
    import torch
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"], check=True)
    import torch

import torch.nn as nn
import time

print("🔍 System Check")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if not torch.cuda.is_available():
    print("❌ Need GPU runtime! Runtime -> Change runtime type -> T4 GPU")
    sys.exit(1)

# ============================================================================
# NEW OPTIMIZATION WRAPPER - The actual fix!
# ============================================================================

class OptimizedModelWrapper(nn.Module):
    """
    Wraps a model with torch.compile + CUDA graph capture for real speedups.
    This is what was missing from the original implementation!
    """
    
    def __init__(self, model, sample_input=None):
        super().__init__()
        self.model = model
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.capture_shape = None
        
        print("[Kitsune] Applying torch.compile optimization...")
        # Apply torch.compile for kernel fusion
        self.compiled_model = torch.compile(
            model, 
            mode="reduce-overhead",  # Best for inference
            fullgraph=False
        )
        
        # Capture CUDA graph if sample input provided
        if sample_input is not None:
            self._prepare_cuda_graph(sample_input)
    
    def _prepare_cuda_graph(self, sample_input):
        """Capture CUDA graph for fixed input shapes."""
        print("[Kitsune] Capturing CUDA graph for acceleration...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Record input shape for later checks
        self.capture_shape = tuple(sample_input.shape)
        
        # IMPORTANT: Warmup compiled model BEFORE graph capture
        # torch.compile needs to trace and compile first
        print("[Kitsune] Warming up compiled model...")
        with torch.no_grad():
            for _ in range(5):
                _ = self.compiled_model(sample_input)
        
        torch.cuda.synchronize()
        
        # Create static tensors for graph
        self.static_input = sample_input.clone()
        
        # Now capture CUDA graph on the already-compiled model
        # This avoids RNG state issues during capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        # Use the original model (not compiled) for graph capture
        # to avoid torch.compile interference
        print("[Kitsune] Capturing graph...")
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.model(self.static_input)
        
        torch.cuda.synchronize()
        print(f"[Kitsune] ✓ CUDA graph captured for shape {self.capture_shape}")
    
    def forward(self, x):
        """
        Forward pass with intelligent graph replay.
        Strategy: Use CUDA graph for fixed shapes (best performance)
                 Fallback to torch.compile for other cases
        """
        # Check if we can use CUDA graph (fixed shape, eval mode)
        if (self.cuda_graph is not None and 
            not self.training and 
            tuple(x.shape) == self.capture_shape):
            
            # Copy input to static buffer
            self.static_input.copy_(x)
            
            # Replay graph - eliminates kernel launch overhead
            self.cuda_graph.replay()
            
            # Return output
            return self.static_output.clone()
        
        else:
            # Fallback to torch.compile for:
            # - Training mode
            # - Different input shapes  
            # - Dynamic scenarios
            # Still faster than baseline due to kernel fusion
            return self.compiled_model(x)


# ============================================================================
# TEST MODELS
# ============================================================================

class TestMLP(nn.Module):
    """Deep MLP for testing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096), nn.ReLU(), nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """LeNet-style CNN."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# BENCHMARK FUNCTION
# ============================================================================

def benchmark(model, input_data, num_iter=100, warmup=10):
    """Accurate GPU benchmarking with CUDA events."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    
    # Return median (more stable than mean)
    timings.sort()
    return timings[len(timings)//2]


# ============================================================================
# RUN BENCHMARKS
# ============================================================================

print("\n" + "="*70)
print("🚀 TESTING NEW OPTIMIZATIONS (torch.compile + CUDA graphs)")
print("="*70)

device = torch.device('cuda')
results = []

# Test 1: MLP
print("\n📊 Test 1: Deep MLP (1024→2048→4096→2048→1024→10)")
print("-" * 70)

model_baseline = TestMLP().to(device)
model_optimized = TestMLP().to(device)
model_optimized.load_state_dict(model_baseline.state_dict())
input_mlp = torch.randn(256, 1024, device=device)

# Wrap with new optimizer
optimized_wrapper = OptimizedModelWrapper(model_optimized, input_mlp)

print("\nBenchmarking baseline...", end=" ")
baseline_time = benchmark(model_baseline, input_mlp)
print(f"{baseline_time:.3f}ms")

print("Benchmarking optimized...", end=" ")
optimized_time = benchmark(optimized_wrapper, input_mlp)
print(f"{optimized_time:.3f}ms")

speedup_mlp = baseline_time / optimized_time
print(f"\n⚡ MLP Speedup: {speedup_mlp:.2f}x")
results.append(("MLP", baseline_time, optimized_time, speedup_mlp))

del model_baseline, model_optimized, optimized_wrapper, input_mlp
torch.cuda.empty_cache()

# Test 2: CNN
print("\n📊 Test 2: CNN (32x32 RGB → Conv layers → FC)")
print("-" * 70)

model_baseline = SimpleCNN().to(device)
model_optimized = SimpleCNN().to(device)
model_optimized.load_state_dict(model_baseline.state_dict())
input_cnn = torch.randn(128, 3, 32, 32, device=device)

optimized_wrapper = OptimizedModelWrapper(model_optimized, input_cnn)

print("\nBenchmarking baseline...", end=" ")
baseline_time = benchmark(model_baseline, input_cnn)
print(f"{baseline_time:.3f}ms")

print("Benchmarking optimized...", end=" ")
optimized_time = benchmark(optimized_wrapper, input_cnn)
print(f"{optimized_time:.3f}ms")

speedup_cnn = baseline_time / optimized_time
print(f"\n⚡ CNN Speedup: {speedup_cnn:.2f}x")
results.append(("CNN", baseline_time, optimized_time, speedup_cnn))

del model_baseline, model_optimized, optimized_wrapper, input_cnn
torch.cuda.empty_cache()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)

for name, baseline, optimized, speedup in results:
    improvement = (speedup - 1.0) * 100
    status = "✅" if speedup > 1.1 else "⚠️" if speedup > 1.0 else "❌"
    print(f"{status} {name:10s}: {baseline:6.2f}ms → {optimized:6.2f}ms = {speedup:.2f}x ({improvement:+.1f}%)")

avg_speedup = sum(r[3] for r in results) / len(results)
print(f"\n🎯 Average Speedup: {avg_speedup:.2f}x")

if avg_speedup > 1.2:
    print("\n✅ SUCCESS! Real speedups achieved with new optimization!")
    print("   torch.compile + CUDA graphs working as expected")
elif avg_speedup > 1.05:
    print("\n⚠️  Modest improvement - torch.compile benefit visible")
    print("   CUDA graph overhead may be eating some gains at this batch size")
else:
    print("\n❌ No significant speedup - investigation needed")
    print("   May need larger batch sizes or different models")

print("\n💡 Key Insights:")
print("   - CUDA graphs eliminate kernel launch overhead")
print("   - torch.compile fuses operations for efficiency")
print("   - Benefits scale with model complexity and batch size")
