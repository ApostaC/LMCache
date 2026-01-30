# SPDX-License-Identifier: Apache-2.0
"""
Eviction module to determine the what to evict from L1 cache
"""

# Standard
from abc import ABC, abstractmethod
import warnings

# First Party
from lmcache.v1.multiprocess.distributed.api import ObjectKey
from lmcache.v1.multiprocess.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class EvictionPolicy(ABC):
    """
    Base class for eviction policies
    """

    @abstractmethod
    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to register
        """
        pass

    @abstractmethod
    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that new keys have been created.

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        pass

    @abstractmethod
    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been accessed.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        pass

    def on_keys_locked(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been locked and should
        not be evicted before unlock or expire.

        Args:
            keys (list[ObjectKey]): The keys that have been locked

        .. deprecated::
            This method is deprecated until we have a reliable way to track
            lock expiration as callbacks.
        """
        warnings.warn(
            "on_keys_locked is deprecated and has no effect. "
            "Lock tracking is not reliable.",
            DeprecationWarning,
            stacklevel=2,
        )

    def on_keys_unlocked(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been unlocked and can be evicted.

        Args:
            keys (list[ObjectKey]): The keys that have been unlocked

        .. deprecated::
            This method is deprecated and has no effect because lock tracking is
            not reliable.
        """
        warnings.warn(
            "on_keys_unlocked is deprecated and has no effect. "
            "Lock tracking is not reliable.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def on_keys_deleted(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been deleted.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        pass

    @abstractmethod
    def get_eviction_actions(self, expected_ratio: float) -> list[EvictionAction]:
        """
        Get the eviction actions to evict objects from L1 cache.

        Args:
            expected_ratio (float): A hint indicating approximately what fraction
                of tracked keys should be evicted. Value should be in range [0.0, 1.0].
                For example, 0.1 means roughly 10% of keys should be evicted.
                This is a hint and the policy may return more or fewer keys.

        Returns:
            list[EvictionAction]: The eviction actions to perform. Each
                action contains the keys and one eviction destination.

        Notes:
            The eviction action may not be successfully executed, or it
            may be executed asynchronously. Therefore, the eviction policy
            should not assume that the objects are evicted immediately, but
            it should use `on_keys_deleted` to know when the objects are actually
            deleted.
        """
        pass
