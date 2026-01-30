"""
🦊 KITSUNE HYBRID BACKEND TEST
Single-cell Colab test for stable production backend
Tests torch.compile + CUDA graphs + TF32 + memory formats
Copy entire cell into Colab with GPU runtime (T4 or better)
"""

import subprocess
import sys
try:
    import torch
    import torchvision
except:
    print("📦 Installing PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=True)
    import torch
    import torchvision

import torch.nn as nn
import time

print("="*70)
print("🔍 SYSTEM CHECK")
print("="*70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Compute Capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")

if not torch.cuda.is_available():
    print("\n❌ CUDA not available! Please enable GPU runtime:")
    print("   Runtime → Change runtime type → T4 GPU")
    sys.exit(1)

# ============================================================================
# STABLE BACKEND IMPLEMENTATION
# ============================================================================

class StableBackend:
    """
    Production-grade optimization backend using industry-standard techniques:
    - torch.compile for kernel fusion
    - CUDA graphs for launch overhead reduction
    - TF32 for Ampere+ GPUs
    - Channels-last memory format for CNNs
    """
    
    def __init__(self, model, sample_input, verbose=True):
        self.original_model = model
        self.model = model
        self.sample_input = sample_input
        self.verbose = verbose
        
        # Graph capture state
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.capture_shape = None
        
        self.optimizations_applied = []
        
    def optimize(self):
        """Apply all stable optimizations."""
        if self.verbose:
            print("\n" + "="*70)
            print("🔧 STABLE BACKEND: Applying Production Optimizations")
            print("="*70)
        
        # 1. Memory Format Optimization (Channels Last for CNNs)
        if self.sample_input.dim() == 4:  # 4D = CNN input (N, C, H, W)
            self._apply_channels_last()
        
        # 2. TF32 Precision (Ampere+ GPUs)
        self._enable_tf32()
        
        # 3. torch.compile (Kernel Fusion)
        self._apply_torch_compile()
        
        # 4. CUDA Graph Capture
        self._capture_cuda_graph()
        
        if self.verbose:
            print("\n✅ Optimization Complete!")
            print(f"   Applied: {', '.join(self.optimizations_applied)}")
        
        return self
    
    def _apply_channels_last(self):
        """Convert to channels-last memory format for better cache locality."""
        try:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.sample_input = self.sample_input.to(memory_format=torch.channels_last)
            self.optimizations_applied.append("Channels-Last")
            if self.verbose:
                print("   ✓ Applied channels-last memory format")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠ Channels-last failed: {e}")
    
    def _enable_tf32(self):
        """Enable TensorFloat-32 for Ampere+ GPUs (3x faster matmul)."""
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] >= 8:  # Ampere (A100, A10, RTX 30xx) or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.optimizations_applied.append("TF32")
            if self.verbose:
                print(f"   ✓ Enabled TF32 (GPU: SM{compute_cap[0]}{compute_cap[1]})")
        else:
            if self.verbose:
                print(f"   ⚠ TF32 not available (GPU: SM{compute_cap[0]}{compute_cap[1]})")
    
    def _apply_torch_compile(self):
        """Apply torch.compile for kernel fusion."""
        try:
            if self.verbose:
                print("   ⏳ Compiling model (mode='reduce-overhead')...")
            
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",  # Best for inference with CUDA graphs
                fullgraph=False
            )
            self.optimizations_applied.append("torch.compile")
            
            if self.verbose:
                print("   ✓ Model compiled successfully")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠ torch.compile failed: {e}")
    
    def _capture_cuda_graph(self):
        """Capture CUDA graph for eliminating kernel launch overhead."""
        try:
            if self.verbose:
                print("   ⏳ Capturing CUDA graph...")
            
            self.model.eval()
            self.capture_shape = tuple(self.sample_input.shape)
            
            # Critical: Warmup compiled model FIRST
            # torch.compile needs to trace before graph capture
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(self.sample_input)
            
            torch.cuda.synchronize()
            
            # Allocate static buffers
            self.static_input = self.sample_input.clone()
            
            # Capture graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.cuda_graph):
                self.static_output = self.original_model(self.static_input)
            
            torch.cuda.synchronize()
            self.optimizations_applied.append("CUDA-Graph")
            
            if self.verbose:
                print(f"   ✓ CUDA graph captured for shape {self.capture_shape}")
        
        except Exception as e:
            if self.verbose:
                print(f"   ⚠ CUDA graph capture failed: {e}")
            self.cuda_graph = None
    
    def eval(self):
        """Set model to eval mode (for compatibility with benchmark function)."""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set model to train mode (for compatibility)."""
        self.model.train(mode)
        return self
    
    def __call__(self, x):
        """Forward pass with intelligent execution strategy."""
        # Use CUDA graph for fixed shapes (fastest path)
        if (self.cuda_graph is not None and 
            not self.model.training and 
            tuple(x.shape) == self.capture_shape):
            
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            return self.static_output.clone()
        
        # Fallback to compiled model
        return self.model(x)


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def benchmark_model(model, input_data, num_iter=100, warmup=20, desc="Model"):
    """Accurate GPU benchmarking with CUDA events."""
    model.eval()
    
    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Benchmark phase
    timings = []
    with torch.no_grad():
        for _ in range(num_iter):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = model(input_data)
            end_event.record()
            
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    
    # Return median (more robust than mean)
    timings.sort()
    median_time = timings[len(timings) // 2]
    
    return median_time


def get_memory_stats():
    """Get current GPU memory usage."""
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
        'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
    }


# ============================================================================
# TEST SUITE
# ============================================================================

def run_test(model_fn, input_shape, model_name, batch_size=None):
    """Run benchmark for a specific model."""
    print("\n" + "="*70)
    print(f"📊 TEST: {model_name}")
    print("="*70)
    
    # Create models
    device = torch.device('cuda')
    
    model_baseline = model_fn().to(device).eval()
    model_optimized = model_fn().to(device).eval()
    
    # Create input
    input_tensor = torch.randn(*input_shape, device=device)
    
    # Apply optimizations
    backend = StableBackend(model_optimized, input_tensor, verbose=True)
    backend.optimize()
    
    # Benchmark baseline
    print(f"\n⏱️  Benchmarking baseline...")
    baseline_time = benchmark_model(model_baseline, input_tensor, desc="Baseline")
    print(f"   Baseline: {baseline_time:.3f}ms")
    
    # Benchmark optimized
    print(f"\n⏱️  Benchmarking optimized...")
    optimized_time = benchmark_model(backend, input_tensor, desc="Optimized")
    print(f"   Optimized: {optimized_time:.3f}ms")
    
    # Results
    speedup = baseline_time / optimized_time
    improvement = (speedup - 1.0) * 100
    
    print(f"\n{'='*70}")
    print(f"📈 RESULTS:")
    print(f"   Baseline:  {baseline_time:6.2f}ms")
    print(f"   Optimized: {optimized_time:6.2f}ms")
    print(f"   Speedup:   {speedup:.2f}x ({improvement:+.1f}%)")
    
    if speedup > 1.2:
        print(f"   Status:    ✅ SIGNIFICANT IMPROVEMENT")
    elif speedup > 1.05:
        print(f"   Status:    ⚠️  MODEST IMPROVEMENT")
    else:
        print(f"   Status:    ❌ MINIMAL IMPROVEMENT")
    
    # Cleanup
    del model_baseline, model_optimized, input_tensor, backend
    torch.cuda.empty_cache()
    
    return {
        'model': model_name,
        'baseline': baseline_time,
        'optimized': optimized_time,
        'speedup': speedup,
        'improvement': improvement
    }


# ============================================================================
# RUN TESTS
# ============================================================================

print("\n" + "="*70)
print("🚀 KITSUNE STABLE BACKEND BENCHMARK SUITE")
print("="*70)

results = []

# Test 1: ResNet-50 (Large production model)
print("\n\n🧪 BATCH 1: PRODUCTION MODELS")
try:
    from torchvision.models import resnet50
    
    # Small batch (latency-sensitive)
    result = run_test(
        model_fn=lambda: resnet50(weights=None),
        input_shape=(32, 3, 224, 224),
        model_name="ResNet-50 (batch=32)"
    )
    results.append(result)
    
    # Larger batch (throughput-oriented)
    result = run_test(
        model_fn=lambda: resnet50(weights=None),
        input_shape=(128, 3, 224, 224),
        model_name="ResNet-50 (batch=128)"
    )
    results.append(result)
    
except Exception as e:
    print(f"⚠️  ResNet-50 test failed: {e}")


# Test 2: MobileNetV2 (Efficient model)
print("\n\n🧪 BATCH 2: EFFICIENT MODELS")
try:
    from torchvision.models import mobilenet_v2
    
    result = run_test(
        model_fn=lambda: mobilenet_v2(weights=None),
        input_shape=(64, 3, 224, 224),
        model_name="MobileNetV2 (batch=64)"
    )
    results.append(result)
    
except Exception as e:
    print(f"⚠️  MobileNetV2 test failed: {e}")


# Test 3: Deep MLP (Custom model)
print("\n\n🧪 BATCH 3: CUSTOM MODELS")

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 4096), nn.ReLU(), nn.BatchNorm1d(4096), nn.Dropout(0.2),
            nn.Linear(4096, 8192), nn.ReLU(), nn.BatchNorm1d(8192), nn.Dropout(0.2),
            nn.Linear(8192, 4096), nn.ReLU(), nn.BatchNorm1d(4096), nn.Dropout(0.2),
            nn.Linear(4096, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 1000)
        )
    def forward(self, x):
        return self.net(x)

result = run_test(
    model_fn=DeepMLP,
    input_shape=(512, 2048),
    model_name="Deep MLP (batch=512)"
)
results.append(result)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "="*70)
print("📊 FINAL SUMMARY - STABLE BACKEND PERFORMANCE")
print("="*70)

for r in results:
    status = "✅" if r['speedup'] > 1.2 else "⚠️ " if r['speedup'] > 1.05 else "❌"
    print(f"{status} {r['model']:30s}: {r['speedup']:.2f}x ({r['improvement']:+5.1f}%)")

avg_speedup = sum(r['speedup'] for r in results) / len(results)
avg_improvement = (avg_speedup - 1.0) * 100

print(f"\n{'='*70}")
print(f"🎯 AVERAGE SPEEDUP: {avg_speedup:.2f}x ({avg_improvement:+.1f}%)")
print(f"{'='*70}")

if avg_speedup > 1.3:
    print("\n✅ EXCELLENT! Stable backend delivers significant speedups!")
    print("   Production-ready for deployment.")
elif avg_speedup > 1.15:
    print("\n✅ GOOD! Stable backend shows meaningful improvements!")
    print("   Benefits scale with model size and batch size.")
elif avg_speedup > 1.05:
    print("\n⚠️  MODEST! Some improvement but not dramatic.")
    print("   Consider larger models or batch sizes for better gains.")
else:
    print("\n❌ MINIMAL! Optimizations not showing significant benefit.")
    print("   May need different models or hardware configuration.")

print("\n💡 KEY INSIGHTS:")
print("   • Speedups scale with model complexity (larger models = bigger gains)")
print("   • Larger batch sizes amortize optimization overhead")
print("   • TF32 provides free 3x matmul speedup on Ampere+ GPUs")
print("   • CUDA graphs eliminate kernel launch overhead (~10-20% gain)")
print("   • torch.compile fuses operations for 1.2-1.5x typical speedup")
print("\n" + "="*70)
