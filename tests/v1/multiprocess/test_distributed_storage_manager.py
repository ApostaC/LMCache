# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for distributed StorageManager.

These tests verify the behavior of StorageManager as described in the
interface docstrings. The tests focus on black-box testing without
accessing private members.

Test Structure:
- Each public method has its own test class
- Tests verify behavior documented in docstrings
- Tests are designed to be extendible for future function additions
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ReserveResult,
)
from lmcache.v1.multiprocess.distributed.config import (
    L1MemoryManagerConfig,
    L1ObjectManagerConfig,
    StorageManagerConfig,
)

try:
    # First Party
    from lmcache.v1.multiprocess.distributed.storage_manager import StorageManager
except ImportError:
    pytest.skip(
        "StorageManager not available (missing dependencies)",
        allow_module_level=True,
    )

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def should_use_lazy_alloc() -> bool:
    """Determine if lazy allocation should be used based on CUDA availability."""
    return torch.cuda.is_available()


# =============================================================================
# Helper Functions
# =============================================================================


def create_object_key(
    chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0
) -> ObjectKey:
    """Create an ObjectKey for testing."""
    return ObjectKey(chunk_hash=chunk_hash, model_name=model_name, kv_rank=kv_rank)


def create_object_keys(
    count: int, model_name: str = "test_model", kv_rank: int = 0
) -> list[ObjectKey]:
    """Create a list of unique ObjectKeys for testing."""
    return [create_object_key(i, model_name, kv_rank) for i in range(count)]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_memory_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def basic_object_config():
    """Create a basic L1ObjectManagerConfig for testing."""
    return L1ObjectManagerConfig(lock_ttl_seconds=60)


@pytest.fixture
def storage_config(basic_memory_config, basic_object_config):
    """Create a StorageManagerConfig for testing."""
    return StorageManagerConfig(
        l1_memory_manager_config=basic_memory_config,
        l1_object_manager_config=basic_object_config,
    )


@pytest.fixture
def small_memory_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion.

    Note: Minimum size is 64MB due to LazyMemoryAllocator's PIN_CHUNK_SIZE.
    """
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB (same as final)
        align_bytes=0x1000,
    )


@pytest.fixture
def small_storage_config(small_memory_config, basic_object_config):
    """Create a small StorageManagerConfig to test memory exhaustion."""
    return StorageManagerConfig(
        l1_memory_manager_config=small_memory_config,
        l1_object_manager_config=basic_object_config,
    )


@pytest.fixture
def storage_manager(storage_config):
    """Create a fresh StorageManager instance for each test."""
    manager = StorageManager(storage_config)
    yield manager
    # Cleanup if needed (StorageManager may need a close method)


@pytest.fixture
def small_storage_manager(small_storage_config):
    """Create a small StorageManager instance for memory exhaustion tests."""
    manager = StorageManager(small_storage_config)
    yield manager


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def multi_tensor_layout():
    """Create a MemoryLayoutDesc with multiple tensor shapes."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512]), torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16, torch.bfloat16],
    )


@pytest.fixture
def large_layout():
    """Create a large MemoryLayoutDesc that will exhaust small memory.

    Each allocation is 8MB (2M elements * 4 bytes).
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2048, 1024])],  # 2M elements * 4 bytes = 8MB
        dtypes=[torch.float32],
    )


@pytest.fixture
def test_keys():
    """Create a list of 20 unique test ObjectKeys."""
    return create_object_keys(20)


# =============================================================================
# Tests for StorageManager.reserve()
# =============================================================================


class TestReserveBasicBehavior:
    """
    Basic behavior tests for StorageManager.reserve() method.

    Per the docstring:
    - Reserve storage for the given object keys non-blockingly.
    - Returns dict[ObjectKey, ReserveResult]
    - Each ReserveResult contains: memory_object, success, is_new
    """

    def test_reserve_returns_dict_with_all_keys(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test that reserve returns a dict containing all requested keys."""
        keys = test_keys[:3]

        result = storage_manager.reserve(keys, basic_layout)

        assert isinstance(result, dict)
        assert len(result) == len(keys)
        for key in keys:
            assert key in result

    def test_reserve_result_structure(self, storage_manager, test_keys, basic_layout):
        """Test that each result is a ReserveResult with correct fields."""
        keys = test_keys[:3]

        result = storage_manager.reserve(keys, basic_layout)

        for key, reserve_result in result.items():
            assert isinstance(reserve_result, ReserveResult)
            assert hasattr(reserve_result, "memory_object")
            assert hasattr(reserve_result, "success")
            assert hasattr(reserve_result, "is_new")

    def test_reserve_new_keys_returns_success_and_is_new(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test that reserving new keys returns success=True and is_new=True."""
        keys = test_keys[:3]

        result = storage_manager.reserve(keys, basic_layout)

        for key in keys:
            reserve_result = result[key]
            assert reserve_result.success is True
            assert reserve_result.is_new is True
            assert reserve_result.memory_object is not None

    def test_reserve_returns_valid_memory_objects(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test that successful reservations return valid memory objects."""
        keys = test_keys[:3]

        result = storage_manager.reserve(keys, basic_layout)

        for key in keys:
            mem_obj = result[key].memory_object
            if result[key].success:
                assert mem_obj is not None
                # Verify the memory object is valid
                assert mem_obj.is_valid()

    def test_reserve_empty_keys(self, storage_manager, basic_layout):
        """Test that reserving empty key list returns empty dict."""
        result = storage_manager.reserve([], basic_layout)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_reserve_single_key(self, storage_manager, test_keys, basic_layout):
        """Test reserving a single key."""
        keys = [test_keys[0]]

        result = storage_manager.reserve(keys, basic_layout)

        assert len(result) == 1
        assert keys[0] in result
        assert result[keys[0]].success is True
        assert result[keys[0]].is_new is True

    def test_reserve_many_keys(self, storage_manager, test_keys, basic_layout):
        """Test reserving many keys at once."""
        keys = test_keys[:15]

        result = storage_manager.reserve(keys, basic_layout)

        assert len(result) == len(keys)
        for key in keys:
            assert key in result
            assert result[key].success is True


class TestReserveExistingKeys:
    """
    Tests for reserve() behavior with existing keys.

    Per the docstring:
    - is_new=False for previously existing objects
    """

    def test_reserve_existing_key_returns_is_new_false(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test that reserving an existing key returns is_new=False."""
        keys = test_keys[:3]

        # First reservation - keys should be new
        result1 = storage_manager.reserve(keys, basic_layout)
        for key in keys:
            assert result1[key].is_new is True

        # Commit the keys to make them "existing"
        storage_manager.commit(keys)

        # Second reservation - keys should not be new
        result2 = storage_manager.reserve(keys, basic_layout)
        for key in keys:
            # Keys already exist, so is_new should be False
            if result2[key].success:
                assert result2[key].is_new is False


class TestReserveMemoryAllocation:
    """
    Tests for reserve() memory allocation behavior.
    """

    def test_reserve_with_multi_tensor_layout(
        self, storage_manager, test_keys, multi_tensor_layout
    ):
        """Test reserving with a multi-tensor memory layout."""
        keys = test_keys[:3]

        result = storage_manager.reserve(keys, multi_tensor_layout)

        assert len(result) == len(keys)
        for key in keys:
            assert result[key].success is True
            assert result[key].memory_object is not None

    def test_reserve_memory_exhaustion(
        self, small_storage_manager, test_keys, large_layout
    ):
        """Test reserve behavior when memory is exhausted.

        When memory is exhausted, reserve should return success=False
        for keys that cannot be allocated, not raise an exception.
        """
        # Try to allocate many large objects to exhaust memory
        keys = test_keys[:15]

        result = small_storage_manager.reserve(keys, large_layout)

        # Should return a result for all keys
        assert len(result) == len(keys)

        # Some keys may have failed due to memory exhaustion
        # Those should have success=False and memory_object=None
        for key, reserve_result in result.items():
            if not reserve_result.success:
                assert reserve_result.memory_object is None


class TestReserveDistinctMemoryObjects:
    """
    Tests to verify that each key gets a distinct memory object.
    """

    def test_reserve_returns_distinct_memory_objects(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test that each key receives a distinct memory object."""
        keys = test_keys[:5]

        result = storage_manager.reserve(keys, basic_layout)

        memory_objects = [
            result[key].memory_object
            for key in keys
            if result[key].success and result[key].memory_object is not None
        ]

        # All memory objects should be distinct (different object instances)
        assert len(memory_objects) == len(set(id(obj) for obj in memory_objects))


class TestReserveWithDifferentModels:
    """
    Tests for reserve() with keys from different models.
    """

    def test_reserve_keys_from_different_models(self, storage_manager, basic_layout):
        """Test reserving keys with different model names."""
        keys = [
            create_object_key(0, model_name="model_a"),
            create_object_key(0, model_name="model_b"),
            create_object_key(0, model_name="model_c"),
        ]

        result = storage_manager.reserve(keys, basic_layout)

        assert len(result) == 3
        for key in keys:
            assert key in result
            assert result[key].success is True

    def test_reserve_keys_with_different_kv_ranks(self, storage_manager, basic_layout):
        """Test reserving keys with different kv_ranks."""
        keys = [
            create_object_key(0, kv_rank=0),
            create_object_key(0, kv_rank=1),
            create_object_key(0, kv_rank=2),
        ]

        result = storage_manager.reserve(keys, basic_layout)

        assert len(result) == 3
        for key in keys:
            assert key in result
            assert result[key].success is True


class TestReserveNonBlocking:
    """
    Tests for reserve() non-blocking behavior.
    """

    def test_reserve_is_non_blocking(self, storage_manager, test_keys, basic_layout):
        """Test that reserve returns immediately without blocking.

        This test verifies that reserve() does not block indefinitely
        even when called multiple times.
        """
        keys = test_keys[:5]

        # Multiple reserve calls should all return without blocking
        for _ in range(3):
            result = storage_manager.reserve(keys, basic_layout)
            assert isinstance(result, dict)


class TestReserveThreadSafety:
    """
    Tests for reserve() thread safety.

    These tests verify that reserve() can be called safely from
    multiple threads concurrently.
    """

    def test_concurrent_reserve_different_keys(self, storage_manager, basic_layout):
        """Test concurrent reserve operations with different keys."""
        num_threads = 10
        keys_per_thread = 5

        def reserve_keys(thread_id):
            keys = create_object_keys(
                keys_per_thread,
                model_name=f"model_{thread_id}",
            )
            result = storage_manager.reserve(keys, basic_layout)
            return result

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_keys, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All operations should succeed
        assert len(results) == num_threads
        for result in results:
            assert len(result) == keys_per_thread

    def test_concurrent_reserve_same_keys(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test concurrent reserve operations with same keys.

        When multiple threads try to reserve the same keys,
        each key should be successfully reserved by exactly one thread
        (as new), and others should either get existing keys or fail.
        """
        num_threads = 5
        keys = test_keys[:3]

        results = []
        results_lock = threading.Lock()

        def reserve_keys():
            result = storage_manager.reserve(keys, basic_layout)
            with results_lock:
                results.append(result)
            return result

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_keys) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        # All threads should have received results
        assert len(results) == num_threads

        # Check that for each key, at most one thread got is_new=True
        for key in keys:
            new_count = sum(1 for r in results if r[key].success and r[key].is_new)
            # At most one thread should have created each key as new
            assert new_count <= 1

    def test_stress_concurrent_reserve(self, storage_manager, basic_layout):
        """Stress test with many concurrent reserve operations."""
        num_threads = 20
        keys_per_thread = 10

        def reserve_keys(thread_id):
            keys = create_object_keys(
                keys_per_thread,
                model_name=f"stress_model_{thread_id}",
            )
            result = storage_manager.reserve(keys, basic_layout)
            # Verify basic result structure
            assert len(result) == keys_per_thread
            return True

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_keys, i) for i in range(num_threads)]
            for f in as_completed(futures):
                assert f.result() is True


# =============================================================================
# Tests for Reserve and Commit Ownership Semantics
# =============================================================================


class TestReserveCommitOwnership:
    """
    Tests for reserve() and commit() ownership semantics.

    These tests verify the interaction between reserve and commit,
    particularly around ownership transfer and key availability.
    """

    def test_reserve_after_partial_commit(
        self, storage_manager, test_keys, basic_layout
    ):
        """Test reserve behavior after partially committing reserved keys.

        Scenario:
        1. Reserve 10 keys (0-9)
        2. Commit keys 0-4
        3. Try to reserve keys 0-2: should succeed with is_new=False
        4. Try to reserve keys 3-7:
           - Keys 3-4 should succeed with is_new=False (committed)
           - Keys 5-7 should NOT succeed (still held by first reservation)
        """
        # Step 1: Reserve 10 keys (indices 0-9)
        all_keys = test_keys[:10]
        result1 = storage_manager.reserve(all_keys, basic_layout)

        # All 10 keys should be successfully reserved as new
        assert len(result1) == 10
        for key in all_keys:
            assert result1[key].success is True
            assert result1[key].is_new is True

        # Step 2: Commit keys 0-4
        keys_to_commit = test_keys[:5]  # keys 0, 1, 2, 3, 4
        storage_manager.commit(keys_to_commit)

        # Step 3: Try to reserve keys 0-2
        # These were committed, so they should be available and is_new=False
        keys_0_to_2 = test_keys[:3]  # keys 0, 1, 2
        result2 = storage_manager.reserve(keys_0_to_2, basic_layout)

        assert len(result2) == 3
        for key in keys_0_to_2:
            assert result2[key].success is True, (
                f"Key {key} should be successfully reserved after commit"
            )
            assert result2[key].is_new is False, (
                f"Key {key} should have is_new=False (was previously committed)"
            )

        # Step 4: Try to reserve keys 3-7
        keys_3_to_7 = test_keys[3:8]  # keys 3, 4, 5, 6, 7
        result3 = storage_manager.reserve(keys_3_to_7, basic_layout)

        assert len(result3) == 5

        # Keys 3-4 were committed, should succeed with is_new=False
        for key in test_keys[3:5]:  # keys 3, 4
            assert result3[key].success is True, (
                f"Key {key} should be successfully reserved (was committed)"
            )
            assert result3[key].is_new is False, (
                f"Key {key} should have is_new=False (was previously committed)"
            )

        # Keys 5-7 were reserved but NOT committed, should fail
        # (still held by the first reservation)
        for key in test_keys[5:8]:  # keys 5, 6, 7
            assert result3[key].success is False, (
                f"Key {key} should NOT be successfully reserved "
                "(still owned by first reservation)"
            )
