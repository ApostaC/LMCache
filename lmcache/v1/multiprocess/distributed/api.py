# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj


@dataclass(frozen=True)
class ObjectKey:
    """
    The unique identifier for an object in the distributed storage manager
    """

    chunk_hash: int
    """ Content hash of this particular chunk """

    model_name: str
    """ Name of the model this chunk belongs to """

    kv_rank: int
    """ The rank that uniquely identifies the slice of the KV cache """


@dataclass(frozen=True)
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post_init__(self):
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )


@dataclass(frozen=True)
class ReserveResult:
    """
    Result of a reserve operation for a given key
    """

    memory_object: MemoryObj | None
    """ The reserved memory object """

    success: bool
    """ Whether the reservation was successful """

    is_new: bool
    """ Whether the reserved object is newly created """


@dataclass(frozen=True)
class PrefetchHandle:
    """
    TODO: think about what to put here
    """

    pass
