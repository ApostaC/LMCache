from dataclasses import dataclass
import torch
from typing import Tuple 
import abc

@dataclass
class BlenderOutput:
    """The output of the cacheblend module

    :ivar torch.Tensor q: The short Q tensor with selected tokens
    :ivar torch.Tensor k: The long K tensor with the updated values
    :ivar torch.Tensor v: The long V tensor with the updated values
    :ivar torch.Tensor positions: The positions of the selected Q tokens in 
        the input sequence
    """
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor


@dataclass
class BlenderRetrieverResult:
    """The result of the cacheblend retriever

    :ivar torch.Tensor k: The K tensor of a single layer
    :ivar torch.Tensor v: The V tensor of a single layer
    :ivar torch.Tensor valid_mask: The valid mask
    """
    k: torch.Tensor
    v: torch.Tensor
    valid_mask: torch.Tensor

class BlendRetrieverTask(metaclass=abc.ABCMeta):
    """The KV retrieval task created by the BlendRetriever"""

    @abc.abstractmethod
    def result(self, layer_id: int) -> BlenderRetrieverResult:
        """Blocking function to get a single layer of K and V tensor.
        The returned the K and V tensor should match the length of the input tokens
        passed to the `BlenderRetriever.new_request` function.
        If the KV of a token is not available, the `vaild_mask` will be 0, and the
        correponding values in the KV tensor will be undefined.

        :param int layer_id: the layer id 
        :return: The BlenderRetrieverResult object
        :rtype: BlenderRetrieverResult
        """
        pass

class BlendRetriever(metaclass=abc.ABCMeta):
    """The interface for the cacheblend retriever to retrieve the KV caches

    It takes in input tokens and ROI as input, and launch some tasks (maybe 
    async), and return a BlendRetrieverTask to retrieve the KV caches.
    """

    @abc.abstractmethod
    def new_request(
            self, 
            input_tokens: torch.Tensor, 
            query_start_loc: torch.Tensor,
        ) -> BlendRetrieverTask:
        """Create a new BlendRetrieverTask to retrieve the KV caches.
        It may launch async tasks in the background during the retrieval.

        :param torch.Tensor input_tokens: The input tokens, could include
            multiple requests in a batch
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be
            the number of requests in the batch.

        :return: The retriever task to retrieve the KV caches
        :rtype: BlendRetrieverTask
        """
        pass

class BlendExecutor(metaclass=abc.ABCMeta):
    """The interface for the cacheblend executor to blend the retrieved KV 
    with fresh KVs
    """

    @abc.abstractmethod
    def blend(
        self,
        layer_id: int,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        fresh_q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> BlenderOutput:
        """This function blends the retrieved KV with fresh KVs, and
        returns the short Q + long KV (blended) + positions of the tokens in Q

        :param int layer_id: The layer id
        :param torch.Tensor retrieved_k: The retrieved K tensor
        :param torch.Tensor retrieved_v: The retrieved V tensor
        :param torch.Tensor fresh_q: The fresh Q tensor from QKV split
        :param torch.Tensor fresh_k: The fresh K tensor from QKV split
        :param torch.Tensor fresh_v: The fresh V tensor from QKV split
        :param torch.Tensor positions: The positions in the input of the
            tokens in the fresh_q
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be
            the number of requests in the batch.

        :return: The blended Q, K, V, and positions
        """
        pass

@dataclass
class BlendMetadata:
    """The class for cacheblend metadata

    For a single request, there is only one [start, end] tuple in the ROI list
    Tokens:          |-------------------------------------------------|
    Prefix caches:   |*********|
    ROI:                       ^(start)                                ^(end)
    selected tokens: |               *   *      *    *  *    ***       |
    positions:                       ^   ^      ^    ^  ^    ^^^        

    When there are multiple requests in a batch, ROI list will have multiple
    start-end tuples

    :ivar int processed_layer_count: The number of processed layers
    :ivar torch.Tensor positions: The positions of the selected tokens 
        in the input sequence
    :ivar BlendRetrieverHandle retriever_handler: The retriever handler 
        to retrieve the KV caches
    """
    processed_layer_count: int
    positions: torch.Tensor
    retrieval_tasks: BlendRetrieverTask
