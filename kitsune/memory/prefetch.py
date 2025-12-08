"""
Async Data Prefetching - Overlap data loading with computation.

Implements prefetching strategies for DataLoaders to hide I/O latency.
"""

from __future__ import annotations

import threading
import queue
from typing import Optional, Iterator, Any, Callable, List, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from ..profiler import get_logger

logger = get_logger(__name__)


@dataclass
class PrefetchedBatch:
    """A prefetched batch with metadata."""
    data: Any
    index: int
    event: Optional[torch.cuda.Event] = None

    def wait(self) -> None:
        """Wait for this batch to be ready."""
        if self.event is not None:
            self.event.synchronize()


class AsyncPrefetcher:
    """
    Asynchronous data prefetcher for PyTorch DataLoader.

    Prefetches batches in a background thread and transfers them
    to GPU asynchronously to overlap with computation.

    Usage:
        prefetcher = AsyncPrefetcher(dataloader, device="cuda")

        for batch in prefetcher:
            # batch is already on GPU
            output = model(batch)
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        num_prefetch: int = 2,
    ):
        """
        Initialize async prefetcher.

        Args:
            dataloader: PyTorch DataLoader to wrap
            device: Target device for prefetched data
            num_prefetch: Number of batches to prefetch
        """
        self._dataloader = dataloader
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_prefetch = num_prefetch

        # Prefetch queue
        self._queue: queue.Queue = queue.Queue(maxsize=num_prefetch)

        # Transfer stream
        self._stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

        # Control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._iterator: Optional[Iterator] = None
        self._batch_index = 0

    def _prefetch_worker(self) -> None:
        """Background worker that prefetches batches."""
        try:
            for batch in self._iterator:
                if self._stop_event.is_set():
                    break

                # Transfer to device on dedicated stream
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        device_batch = self._to_device(batch)
                        event = torch.cuda.Event()
                        event.record()
                else:
                    device_batch = self._to_device(batch)
                    event = None

                prefetched = PrefetchedBatch(
                    data=device_batch,
                    index=self._batch_index,
                    event=event,
                )
                self._batch_index += 1

                # This will block if queue is full
                self._queue.put(prefetched)

        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
            self._queue.put(None)  # Signal error

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
        """Start prefetching and return iterator."""
        self._stop_event.clear()
        self._iterator = iter(self._dataloader)
        self._batch_index = 0

        # Start prefetch thread
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

        return self

    def __next__(self):
        """Get next prefetched batch."""
        try:
            prefetched = self._queue.get(timeout=60)  # 60s timeout
        except queue.Empty:
            raise StopIteration

        if prefetched is None:
            raise StopIteration

        # Wait for transfer to complete
        prefetched.wait()

        return prefetched.data

    def __len__(self):
        """Return length of underlying dataloader."""
        return len(self._dataloader)

    def stop(self) -> None:
        """Stop prefetching."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class CUDAPrefetcher:
    """
    CUDA-optimized prefetcher using pinned memory and streams.

    More efficient than AsyncPrefetcher for GPU training as it
    uses CUDA streams for overlapped H2D transfers.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize CUDA prefetcher.

        Args:
            dataloader: PyTorch DataLoader to wrap
            device: Target CUDA device
        """
        self._dataloader = dataloader
        self._device = device or torch.device("cuda")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDAPrefetcher requires CUDA")

        # Transfer stream (high priority for overlap)
        self._stream = torch.cuda.Stream()

        # Current batch (on GPU)
        self._next_batch: Optional[Any] = None
        self._iterator: Optional[Iterator] = None

    def _preload(self) -> None:
        """Preload next batch."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self._stream):
            self._next_batch = self._to_device(batch)

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
        """Start iteration with preloading."""
        self._iterator = iter(self._dataloader)
        self._preload()
        return self

    def __next__(self):
        """Get next batch."""
        # Wait for preload to complete
        torch.cuda.current_stream().wait_stream(self._stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        # Start preloading next
        self._preload()

        return batch

    def __len__(self):
        """Return length of underlying dataloader."""
        return len(self._dataloader)


class PinnedDataLoader:
    """
    DataLoader wrapper that ensures pinned memory for efficient H2D transfer.

    Automatically handles pin_memory collation for faster GPU transfers.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize pinned dataloader.

        Args:
            dataloader: Base DataLoader
            device: Target device
        """
        self._dataloader = dataloader
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if pin_memory is already enabled
        if hasattr(dataloader, 'pin_memory') and not dataloader.pin_memory:
            logger.warning(
                "DataLoader does not have pin_memory=True. "
                "Consider enabling it for better H2D transfer performance."
            )

    def __iter__(self):
        for batch in self._dataloader:
            yield self._to_device(batch)

    def _to_device(self, item: Any) -> Any:
        """Transfer item to device."""
        if isinstance(item, torch.Tensor):
            return item.to(self._device, non_blocking=True)
        elif isinstance(item, (list, tuple)):
            return type(item)(self._to_device(x) for x in item)
        elif isinstance(item, dict):
            return {k: self._to_device(v) for k, v in item.items()}
        return item

    def __len__(self):
        return len(self._dataloader)


def create_prefetched_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    prefetch_factor: int = 2,
    use_cuda_prefetch: bool = True,
) -> Iterator:
    """
    Create a prefetched data loader.

    Args:
        dataloader: Base DataLoader
        device: Target device
        prefetch_factor: Number of batches to prefetch
        use_cuda_prefetch: Whether to use CUDA-optimized prefetcher

    Returns:
        Prefetched iterator
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and use_cuda_prefetch:
        return CUDAPrefetcher(dataloader, device)
    else:
        return AsyncPrefetcher(dataloader, device, prefetch_factor)
