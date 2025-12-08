"""
Double Buffering - Overlap computation with data transfer.

Implements ping-pong buffering to hide H2D/D2H transfer latency
by overlapping data movement with GPU computation.
"""

from __future__ import annotations

import threading
from typing import Optional, List, Callable, Any, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import torch

from ..profiler import get_logger
from ..cuda import StreamPool, get_stream_pool

logger = get_logger(__name__)


class BufferState(Enum):
    """State of a buffer in the double buffer system."""
    EMPTY = auto()      # Buffer is empty/available
    LOADING = auto()    # Data is being loaded
    READY = auto()      # Data is ready for consumption
    COMPUTING = auto()  # Being used for computation


@dataclass
class Buffer:
    """A single buffer with state tracking."""
    data: torch.Tensor
    state: BufferState = BufferState.EMPTY
    event: Optional[torch.cuda.Event] = None

    def is_ready(self) -> bool:
        """Check if buffer is ready for use."""
        if self.state != BufferState.READY:
            return False
        if self.event is not None:
            return self.event.query()
        return True


class DoubleBuffer:
    """
    Double buffer for overlapping data transfer with computation.

    Uses two buffers in a ping-pong fashion:
    - While one buffer is being used for computation
    - The other buffer is being loaded with next data

    Usage:
        double_buf = DoubleBuffer(shape=(64, 784), dtype=torch.float32)

        for batch in dataloader:
            # Get next ready buffer (blocks if necessary)
            buf = double_buf.get_compute_buffer()

            # Start loading next batch into other buffer
            double_buf.start_load(next_batch)

            # Use current buffer for computation
            output = model(buf)

            # Signal computation done
            double_buf.finish_compute()
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        pinned: bool = True,
    ):
        """
        Initialize double buffer.

        Args:
            shape: Shape of each buffer
            dtype: Data type
            device: Target device (GPU)
            pinned: Whether to use pinned memory for host buffers
        """
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._shape = shape
        self._dtype = dtype
        self._pinned = pinned

        # Create two device buffers
        self._buffers = [
            Buffer(data=torch.empty(shape, dtype=dtype, device=self._device)),
            Buffer(data=torch.empty(shape, dtype=dtype, device=self._device)),
        ]

        # Create host staging buffer (pinned for faster transfers)
        if pinned and torch.cuda.is_available():
            self._host_buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            self._host_buffer = torch.empty(shape, dtype=dtype)

        # Current buffer indices
        self._compute_idx = 0
        self._load_idx = 1

        # Transfer stream (separate from compute)
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream()

        # Synchronization
        self._lock = threading.Lock()

        logger.debug(f"DoubleBuffer created: shape={shape}, dtype={dtype}")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get buffer shape."""
        return self._shape

    @property
    def compute_buffer(self) -> torch.Tensor:
        """Get the current compute buffer."""
        return self._buffers[self._compute_idx].data

    @property
    def load_buffer(self) -> torch.Tensor:
        """Get the current load buffer."""
        return self._buffers[self._load_idx].data

    def start_load(self, data: torch.Tensor, non_blocking: bool = True) -> None:
        """
        Start loading data into the load buffer.

        Args:
            data: Data to load (can be on CPU or GPU)
            non_blocking: Whether to use non-blocking transfer
        """
        buf = self._buffers[self._load_idx]

        with self._lock:
            if buf.state == BufferState.COMPUTING:
                raise RuntimeError("Cannot load into buffer while it's being used for compute")

            buf.state = BufferState.LOADING

        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                buf.data.copy_(data, non_blocking=non_blocking)
                buf.event = torch.cuda.Event()
                buf.event.record()
        else:
            buf.data.copy_(data)
            buf.event = None

        with self._lock:
            buf.state = BufferState.READY

    def get_compute_buffer(self, wait: bool = True) -> torch.Tensor:
        """
        Get the compute buffer, optionally waiting for it to be ready.

        Args:
            wait: Whether to wait for buffer to be ready

        Returns:
            The compute buffer tensor

        Raises:
            RuntimeError: If buffer not ready and wait=False
        """
        buf = self._buffers[self._compute_idx]

        if buf.state == BufferState.LOADING:
            if wait:
                self._wait_for_load(self._compute_idx)
            else:
                raise RuntimeError("Compute buffer not ready")

        with self._lock:
            buf.state = BufferState.COMPUTING

        return buf.data

    def finish_compute(self) -> None:
        """Signal that computation on current buffer is done."""
        buf = self._buffers[self._compute_idx]

        with self._lock:
            buf.state = BufferState.EMPTY

    def swap(self) -> None:
        """Swap compute and load buffers."""
        with self._lock:
            self._compute_idx, self._load_idx = self._load_idx, self._compute_idx

    def _wait_for_load(self, idx: int) -> None:
        """Wait for a specific buffer's load to complete."""
        buf = self._buffers[idx]
        if buf.event is not None:
            buf.event.synchronize()

    def synchronize(self) -> None:
        """Synchronize all pending transfers."""
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()


class TripleBuffer(DoubleBuffer):
    """
    Triple buffer for even better overlap.

    Uses three buffers:
    - One for current computation
    - One being loaded
    - One ready for next computation

    This allows loading to continue even while waiting for computation.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        pinned: bool = True,
    ):
        """Initialize triple buffer."""
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._shape = shape
        self._dtype = dtype
        self._pinned = pinned

        # Create three device buffers
        self._buffers = [
            Buffer(data=torch.empty(shape, dtype=dtype, device=self._device)),
            Buffer(data=torch.empty(shape, dtype=dtype, device=self._device)),
            Buffer(data=torch.empty(shape, dtype=dtype, device=self._device)),
        ]

        # Host staging buffer
        if pinned and torch.cuda.is_available():
            self._host_buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            self._host_buffer = torch.empty(shape, dtype=dtype)

        # Buffer indices
        self._compute_idx = 0
        self._ready_idx = 1
        self._load_idx = 2

        # Transfer stream
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream()

        self._lock = threading.Lock()

    def advance(self) -> None:
        """Advance buffer indices (rotate)."""
        with self._lock:
            # Current compute -> can be loaded
            # Ready -> becomes compute
            # Loading -> becomes ready
            self._buffers[self._compute_idx].state = BufferState.EMPTY
            self._compute_idx = self._ready_idx
            self._ready_idx = self._load_idx
            self._load_idx = (self._load_idx + 1) % 3


class PrefetchBuffer:
    """
    Prefetch buffer for data loading with automatic batching.

    Wraps a data iterator and prefetches next batches in background.
    """

    def __init__(
        self,
        iterator: Iterator,
        device: Optional[torch.device] = None,
        prefetch_count: int = 2,
    ):
        """
        Initialize prefetch buffer.

        Args:
            iterator: Data iterator to wrap
            device: Target device
            prefetch_count: Number of batches to prefetch
        """
        self._iterator = iterator
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prefetch_count = prefetch_count

        # Prefetch queue
        self._queue: List[Any] = []
        self._exhausted = False

        # Transfer stream
        self._stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

        # Fill initial prefetch
        self._fill_queue()

    def _fill_queue(self) -> None:
        """Fill the prefetch queue."""
        while len(self._queue) < self._prefetch_count and not self._exhausted:
            try:
                item = next(self._iterator)
                # Transfer to device in background
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        item = self._to_device(item)
                else:
                    item = self._to_device(item)
                self._queue.append(item)
            except StopIteration:
                self._exhausted = True
                break

    def _to_device(self, item: Any) -> Any:
        """Transfer item to device."""
        if isinstance(item, torch.Tensor):
            return item.to(self._device, non_blocking=True)
        elif isinstance(item, (list, tuple)):
            return type(item)(self._to_device(x) for x in item)
        elif isinstance(item, dict):
            return {k: self._to_device(v) for k, v in item.items()}
        return item

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            if self._exhausted:
                raise StopIteration
            self._fill_queue()
            if not self._queue:
                raise StopIteration

        # Synchronize to ensure data is ready
        if self._stream is not None:
            self._stream.synchronize()

        item = self._queue.pop(0)

        # Start prefetching next
        self._fill_queue()

        return item


class H2DOverlap:
    """
    Host-to-Device transfer with computation overlap.

    Manages asynchronous H2D transfers to overlap with GPU computation.
    """

    def __init__(
        self,
        num_buffers: int = 2,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize H2D overlap manager.

        Args:
            num_buffers: Number of staging buffers
            device: Target device
        """
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_buffers = num_buffers

        # Staging buffers (allocated on first use)
        self._buffers: Dict[Tuple, List[torch.Tensor]] = {}

        # Transfer streams
        self._streams: List[torch.cuda.Stream] = []
        if torch.cuda.is_available():
            self._streams = [torch.cuda.Stream() for _ in range(num_buffers)]

        self._current_buffer = 0
        self._events: List[Optional[torch.cuda.Event]] = [None] * num_buffers

    def transfer(
        self,
        data: torch.Tensor,
        wait_for_compute: bool = True,
    ) -> torch.Tensor:
        """
        Transfer data to device with overlap.

        Args:
            data: Data on CPU
            wait_for_compute: Whether to wait for previous compute

        Returns:
            Tensor on device
        """
        if not torch.cuda.is_available():
            return data.to(self._device)

        # Get or create staging buffer
        key = (data.shape, data.dtype)
        if key not in self._buffers:
            self._buffers[key] = [
                torch.empty(data.shape, dtype=data.dtype, device=self._device)
                for _ in range(self._num_buffers)
            ]

        buf_idx = self._current_buffer
        self._current_buffer = (self._current_buffer + 1) % self._num_buffers

        # Wait for previous transfer on this buffer
        if self._events[buf_idx] is not None:
            self._events[buf_idx].synchronize()

        # Transfer on dedicated stream
        stream = self._streams[buf_idx]
        device_buf = self._buffers[key][buf_idx]

        with torch.cuda.stream(stream):
            device_buf.copy_(data, non_blocking=True)
            self._events[buf_idx] = torch.cuda.Event()
            self._events[buf_idx].record()

        # Make default stream wait for transfer
        if wait_for_compute:
            torch.cuda.current_stream().wait_event(self._events[buf_idx])

        return device_buf

    def synchronize(self) -> None:
        """Synchronize all pending transfers."""
        for stream in self._streams:
            stream.synchronize()
