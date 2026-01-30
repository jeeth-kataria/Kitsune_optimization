"""
🔬 STANDALONE Kitsune Fix Validation
All optimization code embedded - no dependencies on PyPI package
Run this in Colab with T4 GPU
"""

import subprocess
import sys

# Install only PyTorch
print("📦 Installing PyTorch...")
try:
    import torch
    import torchvision
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=True)
    import torch
    import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if not torch.cuda.is_available():
    print("❌ GPU required!")
    sys.exit(1)

import torch.nn as nn
from torchvision.models import resnet18, resnet50
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EMBEDDED OPTIMIZATION CODE (New Fixed Implementation)
# ============================================================================

class TorchCompiler:
    """Manages torch.compile optimization."""
    
    def __init__(self, mode="max-autotune"):
        self.mode = mode
        self.available = self._check_availability()
    
    def _check_availability(self):
        try:
            version = int(torch.__version__.split('.')[0])
            return version >= 2
        except:
            return False
    
    def compile(self, model):
        if not self.available:
            logger.warning("torch.compile not available, skipping")
            return model
        
        try:
            logger.info(f"Compiling model (mode='{self.mode}')...")
            compiled = torch.compile(model, mode=self.mode, fullgraph=False, dynamic=True)
            logger.info("✓ Model compiled")
            return compiled
        except Exception as e:
            logger.warning(f"Compilation failed: {e}")
            return model


class CUDAGraphCapture:
    """CUDA graph capture for fixed-shape inference."""
    
    def __init__(self, warmup_iters=20):
        self.warmup_iters = warmup_iters
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.input_shape = None
        self._captured = False
    
    def capture(self, model, sample_input):
        """Capture CUDA graph."""
        if not torch.cuda.is_available():
            return False
        
        try:
            logger.info("Capturing CUDA graph...")
            model.eval()
            self.input_shape = tuple(sample_input.shape)
            
            # Warmup
            with torch.no_grad():
                for _ in range(self.warmup_iters):
                    _ = model(sample_input)
            
            torch.cuda.synchronize()
            
            # Create static tensors
            self.static_input = sample_input.clone()
            
            # Capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                self.static_output = model(self.static_input)
            
            torch.cuda.synchronize()
            self._captured = True
            logger.info(f"✓ CUDA graph captured for shape {self.input_shape}")
            return True
            
        except Exception as e:
            logger.error(f"CUDA graph capture failed: {e}")
            self._captured = False
            return False
    
    def replay(self, input_data):
        """Replay captured graph."""
        if not self._captured:
            raise RuntimeError("Graph not captured")
        
        if tuple(input_data.shape) != self.input_shape:
            raise RuntimeError(f"Shape mismatch: expected {self.input_shape}, got {tuple(input_data.shape)}")
        
        self.static_input.copy_(input_data)
        self.cuda_graph.replay()
        return self.static_output.clone()
    
    @property
    def is_captured(self):
        return self._captured


class KitsuneOptimizer:
    """Fixed Kitsune optimizer with working optimizations."""
    
    def __init__(self, model, sample_input=None):
        self.model = model
        self.sample_input = sample_input
        self.graph_capture = None
        
        # Move to CUDA
        if not next(model.parameters()).is_cuda:
            self.model = model.cuda()
        
        self.model.eval()
        
        # Apply optimizations
        self._optimize()
    
    def _optimize(self):
        """Apply all optimizations."""
        logger.info("🦊 Kitsune: Starting optimization...")
        
        # 1. Enable TF32 (Ampere+ GPUs)
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability()
            if compute_cap[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info(f"✓ TF32 enabled (SM{compute_cap[0]}{compute_cap[1]})")
            else:
                logger.info(f"⏭️  TF32 not available (SM{compute_cap[0]}{compute_cap[1]})")
        
        # 2. Channels-last for CNNs
        if self._is_cnn():
            self.model = self.model.to(memory_format=torch.channels_last)
            if self.sample_input is not None and self.sample_input.dim() == 4:
                self.sample_input = self.sample_input.to(memory_format=torch.channels_last)
            logger.info("✓ Channels-last format applied")
        
        # 3. torch.compile
        compiler = TorchCompiler(mode="max-autotune")
        self.model = compiler.compile(self.model)
        
        # 4. CUDA graphs
        if self.sample_input is not None:
            self.graph_capture = CUDAGraphCapture(warmup_iters=20)
            self.graph_capture.capture(self.model, self.sample_input)
        
        logger.info("✅ Optimization complete!")
    
    def _is_cnn(self):
        """Check if model has conv layers."""
        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                return True
        return False
    
    def __call__(self, x):
        """Execute optimized forward pass."""
        # Try CUDA graph first
        if (self.graph_capture and 
            self.graph_capture.is_captured and 
            tuple(x.shape) == self.graph_capture.input_shape):
            return self.graph_capture.replay(x)
        
        # Fallback to compiled model
        return self.model(x)


# ============================================================================
# BENCHMARK CODE
# ============================================================================

def measure_time(model, input_data, num_iter=100, warmup=20):
    """Accurate CUDA event timing."""
    if hasattr(model, 'eval'):
        model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    times.sort()
    return times[len(times)//2], sum(times)/len(times)


def test_model(model_fn, batch_size, name):
    """Test a model configuration."""
    print(f"\n{'='*70}")
    print(f"📊 {name} (batch={batch_size})")
    print(f"{'='*70}")
    
    # Create model and input
    model = model_fn(weights=None).cuda().eval()
    input_data = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Baseline
    print("⏱️  Baseline...", end=" ", flush=True)
    torch.cuda.reset_peak_memory_stats()
    baseline_time, _ = measure_time(model, input_data)
    baseline_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{baseline_time:.2f}ms ({baseline_mem:.0f}MB)")
    
    # Optimized
    print("🦊 Kitsune...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    optimizer = KitsuneOptimizer(model, input_data)
    
    print("⏱️  Optimized...", end=" ", flush=True)
    opt_time, _ = measure_time(optimizer, input_data)
    opt_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{opt_time:.2f}ms ({opt_mem:.0f}MB)")
    
    # Results
    speedup = baseline_time / opt_time
    mem_save = (baseline_mem - opt_mem) / baseline_mem * 100
    
    print(f"\n📈 Results:")
    print(f"   ⚡ Speedup: {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")
    print(f"   💾 Memory:  {mem_save:+.1f}%")
    
    status = "✅ PASS" if speedup >= 1.5 else ("⚠️ GOOD" if speedup >= 1.2 else "❌ FAIL")
    print(f"   {status} (target: 1.5x)")
    
    # Cleanup
    del model, optimizer, input_data
    torch.cuda.empty_cache()
    
    return {
        'model': name,
        'batch': batch_size,
        'baseline_ms': round(baseline_time, 2),
        'optimized_ms': round(opt_time, 2),
        'speedup': round(speedup, 2),
        'memory_save_pct': round(mem_save, 1),
        'passes': speedup >= 1.5
    }


def main():
    print("="*70)
    print("🔬 KITSUNE FIX VALIDATION")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")
    
    tests = [
        (resnet18, 256, "ResNet-18"),
        (resnet18, 512, "ResNet-18 (large)"),
        (resnet50, 128, "ResNet-50"),
        (resnet50, 256, "ResNet-50 (large)"),
    ]
    
    results = []
    
    for model_fn, batch, name in tests:
        try:
            result = test_model(model_fn, batch, name)
            results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  Skipped - OOM")
                torch.cuda.empty_cache()
            else:
                raise
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if not results:
        print("❌ No tests completed")
        return
    
    passed = sum(1 for r in results if r['passes'])
    print(f"\nPassed: {passed}/{len(results)}\n")
    
    for r in results:
        status = "✅" if r['passes'] else ("⚠️" if r['speedup'] >= 1.2 else "❌")
        print(f"{status} {r['model']:25s} {r['speedup']:.2f}x  ({r['memory_save_pct']:+.1f}% mem)")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    
    print(f"\n{'='*70}")
    print(f"🎯 AVERAGE: {avg_speedup:.2f}x")
    print("="*70)
    
    if avg_speedup >= 1.5:
        print("\n✅ SUCCESS! 1.5x+ speedup achieved!")
    elif avg_speedup >= 1.2:
        print("\n⚠️  GOOD! 1.2x+ speedup (close to target)")
    else:
        print("\n❌ Below target (<1.2x)")
    
    print("\n📄 JSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
