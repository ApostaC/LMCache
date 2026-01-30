# SPDX-License-Identifier: Apache-2.0
"""
Main interface for distributed MP storage manager
"""

# Standard
from contextlib import contextmanager
from typing import Iterator
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    ReserveResult,
)
from lmcache.v1.multiprocess.distributed.config import StorageManagerConfig
from lmcache.v1.multiprocess.distributed.error import (
    ErrorType,
    is_successful,
    strerror,
)
from lmcache.v1.multiprocess.distributed.memory_manager import L1MemoryManager
from lmcache.v1.multiprocess.distributed.object_manager import L1ObjectManager

logger = init_logger(__name__)


# Helper function
def maybe_report_unprocessed_error(error: ErrorType, parent_func: str) -> None:
    """
    Report the error if it's not SUCCESS.
    Currently, it just raises an exception.
    """
    if not is_successful(error):
        error_msg = strerror(error)
        logger.error(
            "[StorageManager.%s] Unprocessed Error: %s", parent_func, error_msg
        )


def sm_synchronized(func):
    """
    Decorator to synchronize StorageManager methods.
    Currently a placeholder for future synchronization logic.
    """

    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


class StorageManager:
    """
    Main interface for the distributed MP storage manager with all-non-blocking
    interfaces.
    In the future, there should be ONLY 1 thread calling into this class.
    """

    def __init__(self, config: StorageManagerConfig):
        # NOTE: Experimental -- use a global lock to simulate "single-threaded"
        # behavior. We need to make sure all the functions are non-blocking
        self._lock = threading.Lock()
        self._memory_manager = L1MemoryManager(config.l1_memory_manager_config)
        self._object_manager = L1ObjectManager(config.l1_object_manager_config)

        # Eviction policy

        # Remote backends

        # Monitoring

    @sm_synchronized
    def reserve(
        self, keys: list[ObjectKey], memory_desc: MemoryLayoutDesc
    ) -> dict[ObjectKey, ReserveResult]:
        """
        Reserve storage for the given object keys non-blockingly.

        By calling this, the caller expresses the willingness to get the ownership
        of the objects corresponding to the given keys to write to.
        This function is not guaranteed to return all the keys requested because
        some keys may not be "writable" at the moment due to other writers holding
        the ownership.

        Args:
            keys (Iterable[ObjectKey]): The object keys to reserve storage for.
            memory_desc (MemoryLayoutDesc): The memory layout description for the
            objects.

        Returns:
            dict[ObjectKey, ReserveResult]: A dictionary mapping each requested
            requested key to the ReserveResult. Each ReserveResult contains:
            - memory_object: The reserved MemoryObj if successful, else None.
            - success: Whether the reservation was successful.
            - is_new: Whether the reserved object is newly created or previously
                      existing.
        """
        object_states = self._object_manager.query_states(keys)
        new_keys: list[ObjectKey] = []
        existing_keys: list[ObjectKey] = []

        failed_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, ReserveResult] = {}

        for key, state in zip(keys, object_states, strict=False):
            if not state.exists():
                new_keys.append(key)
            else:
                existing_keys.append(key)

        # For new keys, alloc
        if new_keys:
            err, mem_objs = self._memory_manager.allocate(memory_desc, len(new_keys))

            if not is_successful(err):
                # Allocation failed for all new keys
                for key in new_keys:
                    failed_keys.append(key)
            else:
                # pre-reserve and post-reserve for new keys
                # NOTE: since it's single-threaded, we can assume the reserve
                # should be successful for all keys
                res = self._object_manager.prereserve_forced(new_keys)
                maybe_report_unprocessed_error(
                    res.error, "reserve() -> prereserved_forced()"
                )

                res = self._object_manager.postreserve_must(new_keys, mem_objs)
                maybe_report_unprocessed_error(
                    res.error, "reserve() -> postreserved_must()"
                )
                ret.update(
                    {
                        key: ReserveResult(
                            memory_object=mem_obj, success=True, is_new=True
                        )
                        for key, mem_obj in zip(new_keys, mem_objs, strict=False)
                    }
                )

        # For existing keys, try to mark_reserve with force
        if existing_keys:
            mr_result, mem_objs = self._object_manager.mark_reserved(
                existing_keys, force=True
            )
            failed_keys.extend(mr_result.failed_keys)
            for key, mem_obj in zip(mr_result.success_keys, mem_objs, strict=False):
                ret[key] = ReserveResult(
                    memory_object=mem_obj, success=True, is_new=False
                )

        # For failed keys, return failure
        for key in failed_keys:
            ret[key] = ReserveResult(memory_object=None, success=False, is_new=False)
        return ret

    @sm_synchronized
    def commit(self, keys: list[ObjectKey]) -> None:
        """
        Commit the changes made to the objects corresponding to the given keys
        non-blockingly.

        By calling this, the caller releases the ownership of the objects and
        makes the changes visible to other readers.

        Args:
            keys (Iterable[ObjectKey]): The object keys to commit changes for.
        """
        res = self._object_manager.commit(keys, force=True)

        # NOTE: since it's single-threaded, we can assume that all the
        # commits should be successful because no body can change the reserved
        # objects before commit.
        maybe_report_unprocessed_error(res.error, "commit() -> commit()")

    @contextmanager
    def retrieve(self, keys: list[ObjectKey]) -> Iterator[list[MemoryObj]]:
        """
        Retrieve the objects from L1 cache corresponding to the given keys
        non-blockingly for read.

        Lock and unlock semantics:
          This function assumes that `submit_prefetch()` was called on the objects
        and the objects are locked properly.
          The caller need to call mark_retrieve_finished() after finished processing
        the retrieved objects to "unlock" the objects.

        Error handling:
          This function follows "all or nothing" semantics when error happens.
          If error happens internally during retrieve (i.e., object is not in L1
        cache), an empty list will be returned. But this function will also do
        `unlock` on all the other successfully retrieved objects. Therefore, the
        caller NEED TO CALL `submit_prefetch()` AGAIN on the failed objects to
        ensure proper locking before calling `retrieve()` again.
          If error happens when the caller is processing the retrieved objects,
        this function will handle the unlock internally when exiting the context.

        Example Usage:
        ```python
        with storage_manager.retrieve(keys) as objects:
            for obj in objects:
                # process obj
                ...

        # After processing is finished
        storage_manager.mark_retrieve_finished(keys)
        ```

        Args:
            keys (Iterable[ObjectKey]): The object keys to retrieve.

        Yields:
            list[MemoryObj]: A list of MemoryObj or None for each requested
            key. Guaranteed to have the same length as the input keys. If any
            object is not found in L1 cache, an empty list is yielded.
        """
        # First, verify if all the keys are there in L1 cache
        good_keys = []
        memory_objs = []
        all_good = True
        with self._lock:
            object_states = self._object_manager.query_states(keys)
            for key, state in zip(key, object_states, strict=False):
                if state.exists() and state.is_in_l1_cache():
                    good_keys.append(key)
                    memory_objs.append(state.memory_obj)
                else:
                    all_good = False
                    logger.error(
                        "[StorageManager.retrieve()] Key %s not found in L1 cache.", key
                    )

        try:
            yield memory_objs if all_good else []

        except:
            # If error happens during processing, unlock all the good keys
            if good_keys:
                self._object_manager.unlock(good_keys)
            raise
        finally:
            pass

    @sm_synchronized
    def mark_retrieve_finished(self, keys: list[ObjectKey]) -> None:
        """
        Mark the retrieve operation as finished for the given object keys
        non-blockingly.

        If the object is temporary, we will delete it from L1 cache and free
        it.

        This function unlocks the objects that were previously locked by
        `submit_prefetch()`.

        Args:
            keys (Iterable[ObjectKey]): The object keys to mark as finished.
        """
        # Unlock first
        res = self._object_manager.unlock(keys)
        maybe_report_unprocessed_error(
            res.error, "mark_retrieve_finished() -> unlock()"
        )

        # For temporary objects, delete and free
        temporary_key_obj: dict[ObjectKey, MemoryObj] = {}
        for key, state in zip(
            key, self._object_manager.query_states(keys), strict=False
        ):
            if state.is_temporary():
                temporary_key_obj[key] = state.memory_obj

        if temporary_keys:
            res = self._object_manager.delete_committed(
                temporary_key_obj.keys(), force=True
            )
            maybe_report_unprocessed_error(
                res.error, "mark_retrieve_finished() -> delete_committed()"
            )
            objs = [temporary_key_obj[key] for key in res.success_keys]
            err = self._memory_manager.free(objs)
            maybe_report_unprocessed_error(err, "mark_retrieve_finished() -> free()")

    def submit_prefetch(self, keys: list[ObjectKey]) -> PrefetchHandle:
        """
        Submit a prefetch task for the given object keys non-blockingly.

        Once the prefetch is finished, the objects will be available in
        L1 cache and be locked.

        Args:
            keys (Iterable[ObjectKey]): The object keys to prefetch.

        Returns:
            PrefetchHandle: A handle to track the prefetch operation.
        """
        pass

    def query_prefetch_status(self, handle: PrefetchHandle) -> int:
        """
        TODO: write the description of this function
        """
        pass

    def lookup_local(self, keys: list[ObjectKey]) -> int:
        """
        Lookup the local L1 for the given object keys non-blockingly.

        This function does prefix lookup. It does NOT lock the objects.

        Args:
            keys (Iterable[ObjectKey]): The object keys to lookup.

        Returns:
            The number of keys found in the local L1 cache.
        """
        pass

    def delete(self, keys: list[ObjectKey]) -> None:
        """
        Delete the objects corresponding to the given keys non-blockingly.

        Args:
            keys (Iterable[ObjectKey]): The object keys to delete.
        """
        # NOTE: we can only delete committed objects in the underlying object manager
        pass
