"""
Unit tests for data prefetching.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kitsune.memory.double_buffer import DoubleBuffer, H2DOverlap, TripleBuffer
from kitsune.memory.prefetch import (
    AsyncPrefetcher,
    CUDAPrefetcher,
    PinnedDataLoader,
    create_prefetched_loader,
)


class TestDoubleBuffer:
    """Tests for DoubleBuffer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_buffer_creation(self):
        """Test double buffer creation."""
        buf = DoubleBuffer(shape=(64, 256), dtype=torch.float32)
        assert buf.shape == (64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_start_load(self):
        """Test starting a load operation."""
        buf = DoubleBuffer(shape=(64, 256))
        data = torch.randn(64, 256)

        buf.start_load(data)
        buf.synchronize()

        # Data should be in load buffer
        assert buf.load_buffer.shape == (64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swap(self):
        """Test swapping buffers."""
        buf = DoubleBuffer(shape=(64, 256))

        # Load into load buffer
        data = torch.randn(64, 256)
        buf.start_load(data)
        buf.synchronize()

        # Swap
        buf.swap()

        # Now compute buffer should have the data
        compute = buf.get_compute_buffer()
        assert compute.shape == (64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_double_buffer_workflow(self):
        """Test complete double buffer workflow."""
        buf = DoubleBuffer(shape=(64, 256))

        # Simulate loading batches
        for i in range(3):
            data = torch.randn(64, 256)
            buf.start_load(data)
            buf.synchronize()
            buf.swap()

            compute = buf.get_compute_buffer()
            # Simulate computation
            result = compute.sum()
            buf.finish_compute()


class TestTripleBuffer:
    """Tests for TripleBuffer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triple_buffer_creation(self):
        """Test triple buffer creation."""
        buf = TripleBuffer(shape=(64, 256))
        assert buf.shape == (64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_advance(self):
        """Test advancing buffer indices."""
        buf = TripleBuffer(shape=(64, 256))

        # Load data
        data = torch.randn(64, 256)
        buf.start_load(data)
        buf.synchronize()

        # Advance
        buf.advance()

        # Should be able to continue
        compute = buf.get_compute_buffer()
        assert compute is not None


class TestH2DOverlap:
    """Tests for H2D overlap manager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_overlap_creation(self):
        """Test overlap manager creation."""
        overlap = H2DOverlap(num_buffers=2)
        assert overlap is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_transfer(self):
        """Test H2D transfer."""
        overlap = H2DOverlap(num_buffers=2)

        cpu_data = torch.randn(64, 256)
        gpu_data = overlap.transfer(cpu_data)

        assert gpu_data.device.type == "cuda"
        assert gpu_data.shape == cpu_data.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multiple_transfers(self):
        """Test multiple transfers."""
        overlap = H2DOverlap(num_buffers=2)

        for _ in range(5):
            cpu_data = torch.randn(64, 256)
            gpu_data = overlap.transfer(cpu_data)
            assert gpu_data.device.type == "cuda"

        overlap.synchronize()


class TestAsyncPrefetcher:
    """Tests for AsyncPrefetcher."""

    def test_prefetcher_creation(self):
        """Test prefetcher creation."""
        # Create simple dataset
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        prefetcher = AsyncPrefetcher(dataloader, device=torch.device("cpu"))
        assert prefetcher is not None

    def test_iteration(self):
        """Test iterating through prefetcher."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        prefetcher = AsyncPrefetcher(dataloader, device=torch.device("cpu"))

        count = 0
        for batch in prefetcher:
            count += 1
            assert len(batch) == 2  # data and labels

        assert count == 10  # 100 samples / 10 batch size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_prefetch(self):
        """Test prefetching to GPU."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        prefetcher = AsyncPrefetcher(dataloader, device=torch.device("cuda"))

        for batch_data, batch_labels in prefetcher:
            assert batch_data.device.type == "cuda"
            assert batch_labels.device.type == "cuda"
            break  # Just check first batch


class TestCUDAPrefetcher:
    """Tests for CUDAPrefetcher."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prefetcher_creation(self):
        """Test CUDA prefetcher creation."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        prefetcher = CUDAPrefetcher(dataloader)
        assert prefetcher is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_iteration(self):
        """Test iterating through CUDA prefetcher."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        prefetcher = CUDAPrefetcher(dataloader)

        count = 0
        for batch_data, batch_labels in prefetcher:
            assert batch_data.device.type == "cuda"
            count += 1

        assert count == 10


class TestPinnedDataLoader:
    """Tests for PinnedDataLoader."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pinned_loader(self):
        """Test pinned data loader."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10, pin_memory=True)

        pinned = PinnedDataLoader(dataloader, device=torch.device("cuda"))

        for batch_data, batch_labels in pinned:
            assert batch_data.device.type == "cuda"
            break


class TestCreatePrefetchedLoader:
    """Tests for create_prefetched_loader factory."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_cuda_prefetcher(self):
        """Test creating CUDA prefetcher."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        loader = create_prefetched_loader(
            dataloader,
            device=torch.device("cuda"),
            use_cuda_prefetch=True,
        )

        assert isinstance(loader, CUDAPrefetcher)

    def test_create_async_prefetcher(self):
        """Test creating async prefetcher."""
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        loader = create_prefetched_loader(
            dataloader,
            device=torch.device("cpu"),
            use_cuda_prefetch=False,
        )

        assert isinstance(loader, AsyncPrefetcher)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
