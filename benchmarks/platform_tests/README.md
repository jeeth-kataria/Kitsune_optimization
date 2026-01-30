# 🦊 Kitsune Platform-Specific Testing

This directory contains test scripts for each supported platform.

## Testing Instructions

### 1. T4 GPU (Google Colab - FREE)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `test_t4.py` or use the notebook `../Kitsune_T4_Optimization.ipynb`
3. Runtime → Change runtime type → T4 GPU
4. Run all cells

**Expected Results:**
- Baseline: ~90-100ms (ResNet-50, batch=32)
- JIT + FP16: ~45-55ms (2.0x speedup)
- torch.compile + FP16: ~40-50ms (2.0-2.2x speedup)

### 2. Apple Silicon (Local - Your Mac)

```bash
cd KITSUNE_ALGO
python benchmarks/platform_tests/test_apple.py
```

**Expected Results:**
- MPS vs CPU: 2-4x speedup
- With CoreML (if installed): 3-5x speedup

### 3. RTX 30xx/40xx (Colab Pro or Cloud)

**Option A: Colab Pro ($10/mo)**
1. Colab Pro → Runtime → A100 or V100
2. Upload `test_rtx.py`

**Option B: Lambda Labs (~$0.80/hr for RTX 3090)**
1. Create account at [lambdalabs.com](https://lambdalabs.com)
2. Launch RTX 3090 instance
3. Clone repo and run test

**Option C: Vast.ai (~$0.30/hr for RTX 3090)**
1. Create account at [vast.ai](https://vast.ai)
2. Search for RTX 3090/4090 instances
3. Run Docker with PyTorch

**Expected Results:**
- RTX 3090 with TF32: 1.5-2x speedup
- RTX 4090 with FP8: 2-3x speedup

### 4. CI/CD Testing (GitHub Actions)

The GitHub workflow will automatically test:
- Python imports (all platforms)
- CPU optimization paths (Linux, macOS, Windows)
- MPS on macOS runners

## Test Matrix

| Test | T4 | Apple | RTX | A100 | CPU |
|------|----|----|-----|------|-----|
| Import check | ✅ | ✅ | ✅ | ✅ | ✅ |
| JIT trace | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT8 quant | ✅ | ❌ | ✅ | ✅ | ✅ |
| FP16 AMP | ✅ | ✅ | ✅ | ✅ | ❌ |
| TF32 | ❌ | ❌ | ✅ | ✅ | ❌ |
| FP8 | ❌ | ❌ | RTX40 | ✅ | ❌ |
| MPS | ❌ | ✅ | ❌ | ❌ | ❌ |
| CoreML | ❌ | ✅ | ❌ | ❌ | ❌ |
| torch.compile | ✅ | ✅ | ✅ | ✅ | ✅ |
