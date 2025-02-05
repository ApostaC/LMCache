import io
import pickle
from dataclasses import dataclass
import threading
from typing import List, Tuple

import torch
from transformers import AutoConfig

from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256


@dataclass
class QuantizationSpec:
    start_layer: int
    end_layer: int
    bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenConfig:
    # TODO: move this class to another file like "cachegen_basics.py"
    nlayers: int
    kspecs: List[QuantizationSpec]
    vspecs: List[QuantizationSpec]

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = [
            "mistralai/Mistral-7B-Instruct-v0.2", "lmsys/longchat-7b-16k",
            "Qwen/Qwen-7B"
        ]
        family_8b = ["meta-llama/Llama-3.1-8B-Instruct"]
        family_9b = ["THUDM/glm-4-9b-chat"]
        if model_name in family_7b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        elif model_name in family_8b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        # TODO(Jiayi): needs tuning for better quality
        elif model_name in family_9b:
            return CacheGenConfig(
                nlayers=40,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=40, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=40, bins=16),
                ],
            )
        else:
            try:
                config = AutoConfig.from_pretrained(model_name)
                # Default name caught by num_hidden_layers
                if config.num_hidden_layers is None:
                    raise ValueError(
                        f"num_hidden_layers is None for model {model_name}")
                if config.num_hidden_layers < 10:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32),
                        ],
                        vspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32),
                        ],
                    )
                else:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(start_layer=0,
                                             end_layer=10,
                                             bins=32),
                            QuantizationSpec(
                                start_layer=10,
                                end_layer=config.num_hidden_layers,
                                bins=16),
                        ],
                        vspecs=[
                            QuantizationSpec(start_layer=0,
                                             end_layer=2,
                                             bins=32),
                            QuantizationSpec(
                                start_layer=2,
                                end_layer=config.num_hidden_layers,
                                bins=16),
                        ],
                    )
            except Exception as e:
                raise ValueError(
                    f"Model {model_name} not supported by CacheGenConfig"
                ) from e


@dataclass
class CacheGenEncoderOutput:
    # TODO: maybe use numpy array so that we can directly tobytes() and
    # frombuffer() to have a better performance
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """Save the output to a file"""
        with io.BytesIO() as f:
            # torch.save(self, f)
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)


@dataclass
class CacheGenGPUBytestream:
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor  # [nlayers, nchannels, bytestream_length]
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def equals(self, other: "CacheGenGPUBytestream") -> bool:
        if self.ntokens != other.ntokens:
            print(f"Ntokens not match: {self.ntokens} vs {other.ntokens}")
            return False
        if not torch.equal(self.bytestream, other.bytestream):
            print("Bytestream not equal")
            return False
        if not torch.equal(self.bytestream_lengths, other.bytestream_lengths):
            print("Bytestream_lengths not equal")
            return False
        return torch.equal(self.bytestream, other.bytestream) and \
               torch.equal(self.bytestream_lengths, other.bytestream_lengths) and \
               self.ntokens == other.ntokens


@dataclass
class CacheGenGPUEncoderOutput:
    data_chunks: List[
        CacheGenGPUBytestream]  # [nlayers, num_heads * head_size, Lp]
    cdf: torch.Tensor  # [2 * nlayers, ntokens, 1]
    max_tensors_key: torch.Tensor  # [nlayers, ntokens, 1]
    max_tensors_value: torch.Tensor  # [nlayers, ntokens, 1]
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @_lmcache_nvtx_annotate
    def to_bytes(self) -> bytes:
        """Save the output to a file"""
        logger.warning("This function is very slow and should only "
                       "be used for debugging purposes.")
        return CacheGenEncoderOutputSerializer(50 * 1024 *
                                               1024).serialize(self)
        #with io.BytesIO() as f:
        #    pickle.dump(self, f)
        #    return f.getvalue()

    @staticmethod
    @_lmcache_nvtx_annotate
    def from_bytes(bs: bytes) -> "CacheGenGPUEncoderOutput":
        logger.warning("This function is very slow and should only "
                       "be used for debugging purposes.")
        return CacheGenEncoderOutputSerializer(50 * 1024 *
                                               1024).deserialize(bs)
        #with io.BytesIO(bs) as f:
        #    return pickle.load(f)

    def equals(self, other: "CacheGenGPUEncoderOutput") -> bool:
        if self.num_heads != other.num_heads:
            print(
                f"Num_heads not match: {self.num_heads} vs {other.num_heads}")
            return False
        if self.head_size != other.head_size:
            print(
                f"Head_size not match: {self.head_size} vs {other.head_size}")
            return False
        if not torch.equal(self.cdf, other.cdf):
            print("CDF not equal")
            return False
        if not torch.equal(self.max_tensors_key, other.max_tensors_key):
            print("Max_tensors_key not equal")
            return False
        if not torch.equal(self.max_tensors_value, other.max_tensors_value):
            print("Max_tensors_value not equal")
            return False
        if len(self.data_chunks) != len(other.data_chunks):
            print(
                f"Data_chunks length not match: {len(self.data_chunks)} vs {len(other.data_chunks)}"
            )
            return False
        for i in range(len(self.data_chunks)):
            if not self.data_chunks[i].equals(other.data_chunks[i]):
                print(f"Data_chunks {i} not equal")
                return False
        return all([self.data_chunks[i].equals(other.data_chunks[i])
                    for i in range(len(self.data_chunks))]) and \
               torch.equal(self.cdf, other.cdf) and \
               torch.equal(self.max_tensors_key, other.max_tensors_key) and \
               torch.equal(self.max_tensors_value, other.max_tensors_value) and \
               self.num_heads == other.num_heads and \
               self.head_size == other.head_size

    def debug_print_device(self):
        logger.debug(
            f"bytestream device: {self.data_chunks[0].bytestream.device}")
        logger.debug(f"bytestream_lengths device: "
                     f"{self.data_chunks[0].bytestream_lengths.device}")
        logger.debug(f"cdf device: {self.cdf.device}")
        logger.debug(f"max_tensors_key device: {self.max_tensors_key.device}")
        logger.debug(
            f"max_tensors_value device: {self.max_tensors_value.device}")


class CacheGenEncoderOutputSerializer:
    """
    Format:
    - First number:
      - metadata_len
      - data_len
    - Metadata part, all are ints
      - nlayers, nheads, head_size, max_tokens, Lp
      - num_data_chunks
      - [ntokens] * num_data_chunks
      - [bs_len] * num_data_chunks
      - device: -1 means cpu, x>=0 means gpu with id x
      - max_tensors_dtype: 0: bfloat16, 1: half, 2: float
    - Data part
     - max_tensors_key: [nlayers, max_tokens, 1], bfloat16
     - max_tensors_value: [nlayers, max_tokens, 1], bfloat16
     - cdf: [2 * nlayers, nheads * head_size, lp], int16
     - [bytestream_lengths] * num_data_chunks
       - shape: [nlayers, nheads * head_size], int
     - [bytestream] * num_data_chunks
       - shape: [bs_len], uint8
    """

    def __init__(self, max_length):
        self.buffer = torch.zeros(max_length,
                                  dtype=torch.uint8,
                                  device="cpu",
                                  pin_memory=True)
        self.buffer_lock = threading.Lock()
        self.dtype_to_int = {
            torch.bfloat16: 0,
            torch.half: 1,
            torch.float: 2,
        }
        self.int_to_dtype = {
            0: torch.bfloat16,
            1: torch.half,
            2: torch.float,
        }

    def _metadata_len(self, encoder_output: CacheGenGPUEncoderOutput) -> int:
        overall_metadata_len = 24
        device_len = 4

        num_data_chunks = len(encoder_output.data_chunks)
        data_chunks_meta_len = 8 * num_data_chunks

        return overall_metadata_len + data_chunks_meta_len + device_len

    def _tensor_size(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _data_len(self, encoder_output: CacheGenGPUEncoderOutput) -> int:
        data_len = 0
        data_len += self._tensor_size(encoder_output.max_tensors_key)
        data_len += self._tensor_size(encoder_output.max_tensors_value)
        data_len += self._tensor_size(encoder_output.cdf)
        for data_chunk in encoder_output.data_chunks:
            data_len += self._tensor_size(data_chunk.bytestream_lengths)
            data_len += self._tensor_size(data_chunk.bytestream)
        return data_len

    def _write_int(self, value: int, offset: int):
        """
        Returns the new offset
        """
        self.buffer[offset:offset + 4].copy_(
            torch.tensor(value, dtype=torch.int32))
        return offset + 4

    def _write_tensor(self, tensor: torch.Tensor, offset: int):
        """
        Write a tensor to the output buffer
        Returns the new offset
        """
        tensor_bytes = tensor.view(dtype=torch.uint8).flatten()
        tensor_len = tensor_bytes.numel()
        self.buffer[offset:offset + tensor_len].copy_(tensor_bytes)
        return offset + tensor_len

    def _read_int(self, offset: int) -> Tuple[int, int]:
        """
        read an int from output buffer
        returns the int and the new offset
        """
        return int(self.buffer[offset:offset+4].view(dtype=torch.int).item()), \
                offset + 4

    def _read_tensor(self, offset: int, tensor: torch.Tensor) -> int:
        """
        read a new tensor from output buffer
        returns new int
        """
        tensor_bytes = tensor.view(dtype=torch.uint8).flatten()
        tensor_len = tensor_bytes.numel()
        tensor_bytes.copy_(self.buffer[offset:offset + tensor_len])
        return offset + tensor_len

    @_lmcache_nvtx_annotate
    def serialize(self, encoder_output: CacheGenGPUEncoderOutput) -> bytes:
        metadata_len = self._metadata_len(encoder_output)
        data_len = self._data_len(encoder_output)
        if metadata_len + data_len + 8 > self.buffer.numel():
            raise ValueError("Buffer size is too small")

        if encoder_output.max_tensors_key.dtype not in self.dtype_to_int:
            raise ValueError(
                f"Unsupported dtype: {encoder_output.max_tensors_key.dtype}")

        device_id = encoder_output.max_tensors_key.device.index
        s_dtype = self.dtype_to_int[encoder_output.max_tensors_key.dtype]

        offset = 0
        self.buffer_lock.acquire()

        # First two numbers
        offset = self._write_int(metadata_len, offset)
        offset = self._write_int(data_len, offset)

        # Metadata
        max_tokens = encoder_output.max_tensors_key.shape[1]
        metadatas = [
            encoder_output.max_tensors_key.shape[0],
            encoder_output.num_heads,
            encoder_output.head_size,
            max_tokens,
            encoder_output.cdf.shape[2],
            len(encoder_output.data_chunks),
        ]

        metadatas.extend(
            [data_chunk.ntokens for data_chunk in encoder_output.data_chunks])

        metadatas.extend([
            data_chunk.bytestream.numel()
            for data_chunk in encoder_output.data_chunks
        ])

        metadatas.append(device_id)
        metadatas.append(s_dtype)

        offset = self._write_tensor(torch.tensor(metadatas, dtype=torch.int32),
                                    offset)

        # Real data
        offset = self._write_tensor(encoder_output.max_tensors_key, offset)
        offset = self._write_tensor(encoder_output.max_tensors_value, offset)
        offset = self._write_tensor(encoder_output.cdf, offset)
        for data_chunk in encoder_output.data_chunks:
            offset = self._write_tensor(data_chunk.bytestream_lengths, offset)
            offset = self._write_tensor(data_chunk.bytestream, offset)

        final_bytestream = self.buffer[:offset].cpu().numpy().tobytes()
        self.buffer_lock.release()
        return final_bytestream

    @_lmcache_nvtx_annotate
    def deserialize(self, bs: bytes) -> CacheGenGPUEncoderOutput:
        offset = 0
        self.buffer_lock.acquire()
        self.buffer[:len(bs)].copy_(torch.frombuffer(bs, dtype=torch.uint8))

        # First two numbers
        metadata_len, offset = self._read_int(offset)
        data_len, offset = self._read_int(offset)

        # Metadata
        metadatas = self.buffer[offset:offset +
                                24].view(dtype=torch.int32).tolist()
        nlayers, nheads, head_size, max_tokens, lp, num_data_chunks = metadatas
        offset = offset + 24

        tokens_per_chunk = []
        bs_len = []
        for i in range(num_data_chunks):
            tpc, offset = self._read_int(offset)
            tokens_per_chunk.append(tpc)
        for i in range(num_data_chunks):
            bsl, offset = self._read_int(offset)
            bs_len.append(bsl)

        device_id, offset = self._read_int(offset)
        s_dtype, offset = self._read_int(offset)
        d_dtype = self.int_to_dtype[s_dtype]

        # Real data
        max_tensors_key = torch.zeros((nlayers, max_tokens, 1),
                                      dtype=d_dtype,
                                      device=f"cuda:{device_id}")
        max_tensors_value = torch.zeros((nlayers, max_tokens, 1),
                                        dtype=d_dtype,
                                        device=f"cuda:{device_id}")
        cdf = torch.zeros((2 * nlayers, nheads * head_size, lp),
                          dtype=torch.int16,
                          device=f"cuda:{device_id}")

        offset = self._read_tensor(offset, max_tensors_key)
        offset = self._read_tensor(offset, max_tensors_value)
        offset = self._read_tensor(offset, cdf)

        data_chunks = []
        for i in range(num_data_chunks):
            bytestream_lengths = torch.zeros((2 * nlayers, nheads * head_size),
                                             dtype=torch.int32,
                                             device=f"cuda:{device_id}")
            bytestream = torch.zeros(bs_len[i],
                                     dtype=torch.uint8,
                                     device=f"cuda:{device_id}")
            offset = self._read_tensor(offset, bytestream_lengths)
            offset = self._read_tensor(offset, bytestream)
            data_chunks.append(
                CacheGenGPUBytestream(bytestream, bytestream_lengths,
                                      tokens_per_chunk[i]))

        encoder_output = CacheGenGPUEncoderOutput(data_chunks, cdf,
                                                  max_tensors_key,
                                                  max_tensors_value, nheads,
                                                  head_size)

        self.buffer_lock.release()
        return encoder_output
