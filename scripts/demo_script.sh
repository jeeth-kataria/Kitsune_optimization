#!/bin/bash

# Demo Script for Kitsune Terminal Recording
# This script demonstrates Kitsune installation and benchmarking
# Record with: asciinema rec demo.cast
# Convert to GIF: agg demo.cast demo.gif

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Clear screen
clear

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║          🦊 Kitsune Performance Demo                       ║${NC}"
echo -e "${BLUE}║     CUDA-Accelerated Dataflow Scheduler for PyTorch       ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
sleep 2

# System info
echo -e "${GREEN}[1/4] System Information${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
import torch
import platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA: Not available')
"
echo ""
sleep 3

# Installation
echo -e "${GREEN}[2/4] Installing Kitsune${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}$ pip install -e .${NC}"
sleep 1
pip install -e . > /dev/null 2>&1
echo "✓ Kitsune installed successfully"
echo ""
sleep 2

# Verify installation
echo -e "${GREEN}[3/4] Verifying Installation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
import kitsune
print(f'✓ Kitsune version: {kitsune.__version__}')
info = kitsune.get_system_info()
print(f'✓ CUDA available: {info[\"cuda_available\"]}')
if info['cuda_available']:
    print(f'✓ CUDA streams: {info[\"num_streams\"]}')
"
echo ""
sleep 2

# Run benchmark
echo -e "${GREEN}[4/4] Running Performance Benchmark${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}$ python examples/basic_usage.py${NC}"
echo ""
sleep 1

# Simple benchmark demo
python -c "
import torch
import torch.nn as nn
import kitsune
import time

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

print('Training with baseline PyTorch...')
model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
data = torch.randn(64, 784).cuda()

# Baseline timing
start = time.time()
for _ in range(50):
    optimizer.zero_grad()
    output = model(data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
baseline_time = (time.time() - start) * 1000 / 50

print(f'Baseline: {baseline_time:.2f} ms/iter')
print()
print('Training with Kitsune...')

# Kitsune timing
model = SimpleModel().cuda()
optimizer = kitsune.KitsuneOptimizer(
    torch.optim.Adam,
    model.parameters(),
    num_streams=4
)

start = time.time()
for _ in range(50):
    optimizer.zero_grad()
    output = model(data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
kitsune_time = (time.time() - start) * 1000 / 50

speedup = baseline_time / kitsune_time

print(f'Kitsune: {kitsune_time:.2f} ms/iter')
print()
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print(f'🚀 SPEEDUP: {speedup:.2f}x faster!')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
"

echo ""
echo -e "${GREEN}✅ Demo complete!${NC}"
echo ""
echo "Learn more: https://github.com/jeeth-kataria/Kitsune_optimization"
echo ""
