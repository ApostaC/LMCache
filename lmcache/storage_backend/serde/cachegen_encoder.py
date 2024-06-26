import io
import pickle
import torchac
import torchac_cuda
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger

logger = init_logger(__name__)

def torch_quant(bins: int, qA: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float32)
    
    return xq, max1

def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    # TODO: this function can be optimized, we don't really need this
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

def _renorm_cast_cdf_(cdf, precision):
    """ The cdf normalization function in torchac
    """
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf

def _split_kv(tensor: torch.Tensor) -> torch.Tensor:
    """
    Split a blob KV tensor to K and V tensors with the merged heads

    Input:
        tensor: the KV tensor with shape [num_layers, 2, num_tokens, num_heads, head_size]

    Returns:
        K and V tensors with shape [num_layers, num_tokens, num_channels]
    """
    num_layers, _, num_tokens, num_heads, head_size = tensor.shape
    return torch.unbind(tensor.reshape(num_layers, 2, num_tokens, num_heads * head_size), dim=1)

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """
    Convert floatingpoint CDF to integers. See README for more info.
  
    The idea is the following:
    When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
      cdf in [0, 1)
    (note that 1 should not be included.)
    We now want to convert this to int16 but make sure we do not get
    the same value twice, as this would break the arithmetic coder
    (you need a strictly monotonically increasing function).
    So, if needs_normalization==True, we multiply the input CDF
    with 2**16 - (Lp - 1). This means that now,
      cdf in [0, 2**16 - (Lp - 1)].
    Then, in a final step, we add an arange(Lp), which is just a line with
    slope one. This ensure that for sure, we will get unique, strictly
    monotonically increasing CDFs, which are in [0, 2**16)
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
      2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
      new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
      r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
      cdf.add_(r)
    return cdf

class CacheGenEncoderImpl:
    def __init__(self, **kwargs) -> None:
        """ 
        Fields: 
        - fp_kv: should be a tensor of shape (num_layers, num_tokens, num_channels)
        - fp_v: should be a tensor of shape (num_layers, num_tokens, num_channels)
        """
        self.fp_k = kwargs["fp_k"]
        self.fp_v = kwargs["fp_v"]
        
        self.quantized_key = {}
        self.max_tensors_key = {}  
        self.quantized_value = {}
        self.max_tensors_value = {} 
        self.config = kwargs["config"]
        
    def quantize(self):
        """ Quantize the key and value tensors 
        (self.fp_k and self.fp_v) 
        """
        for layer in range(len(self.fp_k)):
            if layer < self.config["key_first_layers"]:
                bins = self.config["key_first_bins"]
            elif layer < self.config["key_second_layers"]:
                bins = self.config["key_second_bins"]
            else:
                bins = self.config["key_third_bins"]

            tmp = torch_quant(bins, self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + bins // 2 - 1
            self.max_tensors_key[layer] = tmp[1]

        for layer in range(len(self.fp_v)):
            if layer < self.config["value_first_layers"]:
                bins = self.config["value_first_bins"]
            else:
                bins = self.config["value_second_bins"]
            tmp = torch_quant(bins, self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ bins // 2 - 1
            self.max_tensors_value[layer] = tmp[1]
            
    def compute_cdf(self, is_key):
        """
        Compute the CDF based on the quantized tensors
        Field: 
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        """
        # TODO: Add start_index here
        channels = self.fp_k[0].shape[-1]
        tokens = self.fp_k[0].shape[0]
        
        def process_batch(X, max_val):
            """
            input shape should be [channels, tokens]
            """
            nchannels, ntokens = X.shape
            one_hot = torch.nn.functional.one_hot(X.long(), num_classes=max_val + 1).to(torch.float32)  # Use float32 to avoid integer overflow
            counts = one_hot.sum(dim=1) / ntokens
            ret = torch.cumsum(counts, dim=1).roll(1)
            ret[:, 0] = 0
            return ret

        def process_layers(X, max_val):
            """
            x is a iterator of dict values
            each element's shape is [tokens, channels]
            """
            results = []
            for x in X:
                ''' do permute here '''
                batch_counts = process_batch(x.cuda().permute(1, 0), max_val)
                results.append(batch_counts)

            final_counts = torch.cat(results, dim=0)
            
            return final_counts
        
        if is_key:
            X = self.quantized_key.values()
        else:
            X = self.quantized_value.values()
        value_range = 32
        cdfs = process_layers(X, value_range) # 4096 is batch size, ==> 18GB GPU memory
        final_cdf = cdfs.reshape((len(self.fp_k), channels, value_range+1)).cpu()
                
        return final_cdf

def encode_function(kv, config, chunk_size) -> CacheGenEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache
    """
    logger.debug(f"Jiayi: encode chunk size: {chunk_size}")
    num_heads, head_size = kv.shape[-2:]
    output_dict = {}
    fp_k, fp_v = _split_kv(kv)
    l = fp_k.shape[0]
    encoder = CacheGenEncoderImpl(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize()
    cdf_k = encoder.compute_cdf(is_key=True)
    encode_input_key = torch.stack(list(encoder.quantized_key.values()))
    
    cdf_v = encoder.compute_cdf(is_key=False)
    encode_input_value = torch.stack(list(encoder.quantized_value.values()))
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0).cpu()
    current_index = 0
    start_indices = []
    bytestreams = []
    cdf_int = _convert_to_int_and_normalize(cdf, True)
    for l in range(cdf.shape[0]):
        for i in range(chunk_size):
            bits = torchac.encode_int16_normalized_cdf(
                    cdf_int[l:l+1],
                    encode_input[l:l+1, i].to(torch.int16))
            bytestreams.append(bits)
            length = len(bits)
            start_indices += [current_index]
            current_index += length
    #print(len(b"".join(bytestreams)))
    #print(type(bytestreams[0]))
    #print(len(start_indices))
    #print(start_indices[:100])
    #print(bytestreams[0])
    #print(cdf.shape)
    output = CacheGenEncoderOutput(
        bytestream = b"".join(bytestreams),
        start_indices = torch.tensor(start_indices).int(),
        cdf = _renorm_cast_cdf_(cdf.float(), 16),
        max_tensors_key = concat_max(encoder.max_tensors_key),
        max_tensors_value = concat_max(encoder.max_tensors_value),
        num_heads = num_heads,
        head_size = head_size,
    )
    return output

#TODO(Jiayi): The current implmentation does not have any performance gain
def encode_function_gpu(kv, config, chunk_size) -> CacheGenEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache (with cuda)
    """
    logger.debug(f"Jiayi: encode_cuda chunk size: {chunk_size}")
    num_heads, head_size = kv.shape[-2:]
    output_dict = {}
    fp_k, fp_v = _split_kv(kv)
    l = fp_k.shape[0]
    encoder = CacheGenEncoderImpl(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize()
    cdf_k = encoder.compute_cdf(is_key=True)
    encode_input_key = torch.stack(list(encoder.quantized_key.values()))
    
    cdf_v = encoder.compute_cdf(is_key=False)
    encode_input_value = torch.stack(list(encoder.quantized_value.values()))
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0).cpu()
    current_index = 0
    start_indices = []
    bytestreams = []
    #cdf_int = _convert_to_int_and_normalize(cdf, True)
    #cdf_int = cdf
    
    
    cdf_temp = cdf.unsqueeze(1).repeat(1,chunk_size, 1, 1)
    encode_input = encode_input.to(torch.int16).squeeze(0)
    
    
    cdf_int = _convert_to_int_and_normalize(cdf_temp, True)
    
    Lp = cdf_int.shape[-1]
    cdf_int = cdf_int.reshape(-1, Lp)
    encode_input = encode_input.reshape(-1)
    
    print(f"cdf_shape: {cdf_int.shape}")
    print(f"cdf_type: {cdf_int.dtype}")
    print(f"cdf_device: {cdf_int.device}")
    print(f"encode_shape: {encode_input.shape}")
    print(f"encode_type: {encode_input.dtype}")
    print(f"encode_device: {encode_input.device}")
    
    all_bits_cuda = torchac_cuda.encode_fast(cdf_int,
                                           encode_input,
                                           max_out_size=10000,
                                           blockNum=64,
                                           threadNum=chunk_size)
    #import mytorchac_cuda
    #all_bits_cuda = mytorchac_cuda.encode_cuda(cdf_int,
    #                                        encode_input,
    #                                        10000,
    #                                        64,
    #                                        chunk_size)
    index = 0
    for bits in all_bits_cuda:
        
        bytestreams.append(bits)
        start_indices.append(index)
        index += len(bits)
    
    print(len(b"".join(bytestreams)))
    print(type(bytestreams[0]))
    print(len(start_indices))
    print(start_indices[:100])
    print(bytestreams[0])
    print(cdf.shape)
    output = CacheGenEncoderOutput(
        bytestream = b"".join(bytestreams),
        start_indices = torch.tensor(start_indices).int(),
        cdf = _renorm_cast_cdf_(cdf.float(), 16),
        max_tensors_key = concat_max(encoder.max_tensors_key),
        max_tensors_value = concat_max(encoder.max_tensors_value),
        num_heads = num_heads,
        head_size = head_size,
    )
    return output

class CacheGenSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        
    def to_bytes(
            self,
            tensor: torch.Tensor
        ) -> bytes:
        """
        Serialize a pytorch tensor to bytes. The serialized bytes should contain
        both the data and the metadata (shape, dtype, etc.) of the tensor.

        Input:
            t: the input pytorch tensor, can be on any device, in any shape,
               with any dtype
        
        Returns:
            bytes: the serialized bytes
        """
        # TODO: permute is expensive here, need a better way to do it at lower level
        if self.fmt == "huggingface":
            tensor = tensor.permute(0, 1, 3, 2, 4)

        ''' expecting a tensor of shape [num_layers, 2, num_tokens, num_heads, head_size] '''
        ntokens = tensor.shape[2]
        output_dict = encode_function(tensor, self.cachegen_config, ntokens)
        return output_dict.to_bytes()
