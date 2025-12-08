"""
Kitsune PyTorch - PyTorch integration layer

Contains PyTorch-specific utilities:
- Computation graph capture (FX/hooks)
- Forward/backward hooks
- Module wrapping utilities
"""

from .graph_capture import (
    GraphCapture,
    FXGraphCapture,
    HookGraphCapture,
    GraphCaptureError,
    capture_graph,
)

__all__ = [
    "GraphCapture",
    "FXGraphCapture",
    "HookGraphCapture",
    "GraphCaptureError",
    "capture_graph",
]
