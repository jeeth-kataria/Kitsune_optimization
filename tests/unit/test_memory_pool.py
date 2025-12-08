"""
Unit tests for memory pool.
"""

import pytest
import torch

from kitsune.memory.pool import (
    MemoryPool,
    TensorCache,
    SizeClass,
    get_memory_pool,
    reset_memory_pool,
)


class TestSizeClass:
    """Tests for size class binning."""

    def test_minimum_size(self):
        """Test minimum size class."""
        assert SizeClass.get_size_class(1) == 512
        assert SizeClass.get_size_class(100) == 512
        assert SizeClass.get_size_class(512) == 512

    def test_power_of_two_rounding(self):
        """Test rounding up to power of two."""
        assert SizeClass.get_size_class(513) == 1024
        assert SizeClass.get_size_class(1000) == 1024
        assert SizeClass.get_size_class(1025) == 2048

    def test_large_sizes(self):
        """Test large size handling."""
        # Under max binned size
        assert SizeClass.get_size_class(1 << 20) == 1 << 20  # 1MB

        # Over max binned size (exact size)
        huge = (1 << 30) + 1000
        assert SizeClass.get_size_class(huge) == huge

    def test_all_classes(self):
        """Test getting all size classes."""
        classes = SizeClass.get_all_classes()
        assert classes[0] == 512
        assert all(classes[i] == classes[i-1] * 2 for i in range(1, len(classes)))


class TestMemoryPool:
    """Tests for MemoryPool."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pool_creation(self):
        """Test pool creation."""
        pool = MemoryPool(device=torch.device("cuda"))
        assert pool.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_allocate_tensor(self):
        """Test tensor allocation."""
        pool = MemoryPool(device=torch.device("cuda"))

        tensor = pool.allocate(shape=(64, 256), dtype=torch.float32)

        assert tensor.shape == (64, 256)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_deallocate_tensor(self):
        """Test tensor deallocation."""
        pool = MemoryPool(device=torch.device("cuda"))

        tensor = pool.allocate(shape=(64, 256))
        result = pool.deallocate(tensor)

        assert result is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cache_hit(self):
        """Test cache hit on reallocation."""
        pool = MemoryPool(device=torch.device("cuda"))

        # Allocate and deallocate
        t1 = pool.allocate(shape=(64, 256))
        pool.deallocate(t1)

        # Should hit cache
        t2 = pool.allocate(shape=(64, 256))

        stats = pool.get_stats()
        assert stats["cache_hits"] >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_different_shapes(self):
        """Test allocation of different shapes."""
        pool = MemoryPool(device=torch.device("cuda"))

        t1 = pool.allocate(shape=(32, 128))
        t2 = pool.allocate(shape=(64, 256))
        t3 = pool.allocate(shape=(128, 512))

        assert t1.shape == (32, 128)
        assert t2.shape == (64, 256)
        assert t3.shape == (128, 512)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_allocate_like(self):
        """Test allocate_like method."""
        pool = MemoryPool(device=torch.device("cuda"))

        template = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        tensor = pool.allocate_like(template)

        assert tensor.shape == template.shape
        assert tensor.dtype == template.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_temp_allocation_context(self):
        """Test temporary allocation context manager."""
        pool = MemoryPool(device=torch.device("cuda"))

        with pool.allocate_temp(shape=(64, 256)) as tensor:
            assert tensor.shape == (64, 256)
            # Tensor is valid here

        # After context, tensor should be deallocated
        stats = pool.get_stats()
        assert stats["total_deallocations"] >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stats(self):
        """Test statistics tracking."""
        pool = MemoryPool(device=torch.device("cuda"))

        pool.allocate(shape=(64, 256))
        pool.allocate(shape=(64, 256))

        stats = pool.get_stats()
        assert stats["total_allocations"] == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clear(self):
        """Test clearing the pool."""
        pool = MemoryPool(device=torch.device("cuda"))

        t = pool.allocate(shape=(64, 256))
        pool.deallocate(t)

        pool.clear()
        assert pool.cached_bytes == 0


class TestTensorCache:
    """Tests for TensorCache."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cache_creation(self):
        """Test cache creation."""
        cache = TensorCache(device=torch.device("cuda"))
        assert cache is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_tensor(self):
        """Test getting tensor from cache."""
        cache = TensorCache(device=torch.device("cuda"))

        tensor = cache.get(shape=(64, 256))
        assert tensor.shape == (64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_put_and_get(self):
        """Test putting and getting tensor."""
        cache = TensorCache(device=torch.device("cuda"))

        # Get tensor
        t1 = cache.get(shape=(64, 256))

        # Return it
        cache.put(t1)

        # Get again - should be same tensor (from cache)
        t2 = cache.get(shape=(64, 256))
        assert t2.data_ptr() == t1.data_ptr()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_with_zero(self):
        """Test getting zeroed tensor."""
        cache = TensorCache(device=torch.device("cuda"))

        tensor = cache.get(shape=(64, 256), zero=True)
        assert torch.all(tensor == 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prealloc(self):
        """Test pre-allocation."""
        cache = TensorCache(device=torch.device("cuda"))

        cache.prealloc(shape=(64, 256), count=4)

        # Should be able to get 4 without allocation
        tensors = [cache.get(shape=(64, 256)) for _ in range(4)]
        assert len(tensors) == 4


class TestGlobalMemoryPool:
    """Tests for global memory pool."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_global_pool(self):
        """Test getting global pool."""
        reset_memory_pool()
        pool = get_memory_pool()
        assert pool is not None

        # Same pool returned
        pool2 = get_memory_pool()
        assert pool2 is pool

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_global_pool(self):
        """Test resetting global pool."""
        pool1 = get_memory_pool()
        reset_memory_pool()
        pool2 = get_memory_pool()

        assert pool1 is not pool2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
