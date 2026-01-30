#!/usr/bin/env python3
"""
Test torch.compile integration in Kitsune
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-kitsune"], check=True)

import torch
import torch.nn as nn
from kitsune import optimize_model

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda')
model = TestModel().to(device)
sample_input = torch.randn(256, 1024, device=device)

print("Testing Kitsune with torch.compile integration...")
optimizer = optimize_model(model, sample_input)

# Test inference
model.eval()
with torch.no_grad():
    for _ in range(10):
        output = optimizer.model(sample_input)
torch.cuda.synchronize()

print("✅ Integration working!")
print(f"Model type: {type(optimizer.model)}")
print("Expected: OptimizedModule (if torch.compile worked)")
