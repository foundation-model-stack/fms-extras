import abc
import dataclasses
import queue
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
from torch._dynamo import mark_static_address
from torch._inductor.virtualized import V

from fms_extras.paged_c import attn_ops, cache_ops  # type: ignore


KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key cache, value cache)

# adding paged attention to the torch namespace in order to support torch compile
lib = torch.library.Library("paged_attention", "FRAGMENT")

lib.define(
    "reshape_and_cache(Tensor key, Tensor value, Tensor key_cache, Tensor value_cache, Tensor slot_mapping) -> (Tensor, Tensor)"
)


# needed for compile
@torch.library.impl(lib, "reshape_and_cache", "Meta")
def _reshape_and_cache_meta(key, value, key_cache, value_cache, slot_mapping):
    return key_cache, value_cache


@torch.library.impl(lib, "reshape_and_cache", "CUDA")
def _reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    return key_cache, value_cache


lowering.fallbacks.add(torch.ops.paged_attention.reshape_and_cache)


@lowering.register_lowering(
    torch.ops.paged_attention.reshape_and_cache, type_promotion_kind=None
)
def _reshape_and_cache_lowering(key, value, key_cache, value_cache, slot_mapping):
    PagedAttnKernel.create(
        torch.ops.paged_attention.reshape_and_cache.default,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        mutated_inputs=[key_cache, value_cache],
    )
    return key_cache, value_cache


lib.define(
    "paged_attention_v2(Tensor out, Tensor exp_sums, Tensor max_logits, Tensor tmp_out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
)


@torch.library.impl(lib, "paged_attention_v2", "Meta")
def _paged_attention_v2_meta(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    return out


@torch.library.impl(lib, "paged_attention_v2", "CUDA")
def _paged_attention_v2(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    attn_ops.paged_attention_v2(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    return out


lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v2)


@lowering.register_lowering(
    torch.ops.paged_attention.paged_attention_v2, type_promotion_kind=None
)
def _paged_attention_v2_lowering(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    PagedAttnKernel.create(
        torch.ops.paged_attention.paged_attention_v2.default,
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        mutated_inputs=[out],
    )
    return out


lib.define(
    "paged_attention_v1(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
)


@torch.library.impl(lib, "paged_attention_v1", "Meta")
def _paged_attention_v1_meta(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    return out


@torch.library.impl(lib, "paged_attention_v1", "CUDA")
def _paged_attention_v1(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    attn_ops.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    return out


lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v1)


@lowering.register_lowering(
    torch.ops.paged_attention.paged_attention_v1, type_promotion_kind=None
)
def _paged_attention_v1_lowering(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    PagedAttnKernel.create(
        torch.ops.paged_attention.paged_attention_v1.default,
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        mutated_inputs=[out],
    )
    return out


class PagedAttnKernel(ir.FallbackKernel):
    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    @classmethod
    def create(cls, kernel, *args, mutated_inputs=[], **kwargs) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            ) = cls.process_kernel(kernel, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        packed = cls(
            ir.NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        # Mark inplace inputs as mutated
        for kernel_input in mutated_inputs:
            V.graph.mark_buffer_mutated(kernel_input.get_name())
            ir.MutationOutput(kernel_input.layout, kernel_input, packed)


@dataclasses.dataclass
class PagedAttentionCacheDataLayer:
    """
    Dataclass responsible for storing keys and values in a single layer of cache data
    """

    data_layer: Tuple[torch.Tensor, torch.Tensor]
    max_sequence_length: int
    context_lengths: Optional[torch.Tensor]
    slot_mapping: torch.Tensor
    block_mapping: torch.Tensor
    block_size: int
    scale: float
    num_heads: int
    kv_heads: int
    head_size: int
    is_generating: bool

    def get_cache_type(self) -> str:
        return "paged-attention"

    def store(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Store the computed keys and values in the cache data layer

        Parameters
        ----------
        key: torch.Tensor
            the keys to store in this cache layer
        value: torch.Tensor
            the values to store in this cache layer

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            the updated keys and values to be passed in to attention computation
        """
        key_to_cache = keys.view(-1, self.kv_heads, self.head_size)
        value_to_cache = values.view(-1, self.kv_heads, self.head_size)

        self.data_layer = torch.ops.paged_attention.reshape_and_cache(
            key_to_cache,
            value_to_cache,
            self.data_layer[0],
            self.data_layer[1],
            self.slot_mapping,
        )

        return keys, values

    def attend(
        self,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Perform paged attention on this layer of the Cache

        Parameters
        ----------
        query: torch.Tensor
            the query tensor

        Returns
        -------
        torch.Tensor
            the output attention computation
        """
        query = query.view(-1, self.num_heads, self.head_size)

        # Pre-allocate the output tensor.
        attn = torch.empty_like(query)

        num_seqs, num_heads, head_size = query.shape
        _PARTITION_SIZE = 512
        max_num_partitions = (
            self.max_sequence_length + _PARTITION_SIZE - 1
        ) // _PARTITION_SIZE

        use_v1 = self.max_sequence_length <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512
        )

        # from vLLM (woosuk) - Tune this heuristic
        # We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        if use_v1:
            attn = torch.ops.paged_attention.paged_attention_v1(
                attn,
                # num_sequences x num_heads x head_size
                query,
                self.data_layer[0],
                self.data_layer[1],
                self.kv_heads,
                self.scale,
                self.block_mapping,
                self.context_lengths,
                self.block_size,
                self.max_sequence_length,
                None,
            )
        else:
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=attn.dtype,
                device=attn.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=attn.device,
            )
            max_logits = torch.empty_like(exp_sums)

            attn = torch.ops.paged_attention.paged_attention_v2(
                attn,
                exp_sums,
                max_logits,
                tmp_output,
                # num_sequences x num_heads x head_size
                query,
                self.data_layer[0],
                self.data_layer[1],
                self.kv_heads,
                self.scale,
                self.block_mapping,
                self.context_lengths,
                self.block_size,
                self.max_sequence_length,
                None,
            )
        return attn

    def is_filled(self) -> bool:
        """
        Denotes whether this cache data layer is in post-fill stage

        Returns
        -------
        bool
            True if cache data layer is currently in the post-fill stage, otherwise False and cache data layer is being
            pre-filled
        """
        return self.is_generating


@dataclasses.dataclass
class PagedAttentionCacheData:
    """
    Dataclass responsible for holding raw cache data
    """

    data: List[Tuple[torch.Tensor, torch.Tensor]]
    max_sequence_length: int
    context_lengths: Optional[torch.Tensor]
    slot_mapping: torch.Tensor
    block_mapping: torch.Tensor
    block_size: int
    scale: float
    num_heads: int
    kv_heads: int
    head_size: int
    is_generating: bool
    sequence_ids: List[int]

    def get_layer(self, layer_index: int) -> PagedAttentionCacheDataLayer:
        """
        Get a single layer of the cache data

        Parameters
        ----------
        layer_index: int
            index of layer

        Returns
        -------
        PagedAttentionCacheDataLayer
            a single layer of the cache data as a dataclass
        """
        return PagedAttentionCacheDataLayer(
            data_layer=self.data[layer_index],
            max_sequence_length=self.max_sequence_length,
            context_lengths=self.context_lengths,
            slot_mapping=self.slot_mapping,
            block_mapping=self.block_mapping,
            block_size=self.block_size,
            scale=self.scale,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            head_size=self.head_size,
            is_generating=self.is_generating,
        )

    def is_filled(self) -> bool:
        """
        Determines if the cache has been filled with the prompt, or is completely empty for this piece of cache data

        Returns
        -------
        bool
            True if the keys and values for the prompt have been set, otherwise False.
        """
        return self.is_generating

    def compute_position_ids(self, num_tokens_per_sequence: List[int]) -> torch.Tensor:
        """Compute position ids based on the current context lengths and the new tokens to add

        Parameters
        ----------
        num_tokens_per_sequence: List[int]
            number of tokens to be added to each sequence

        Returns
        -------
        torch.Tensor
            the position ids for each sequence
        """
        device = self.data[0][0].device
        max_tokens = max(num_tokens_per_sequence)
        position_ids = []
        for seq_i, num_tokens in enumerate(num_tokens_per_sequence):
            start = (
                0
                if self.context_lengths is None
                else self.context_lengths[seq_i].item() - num_tokens
            )
            pads = torch.zeros(max_tokens - num_tokens, dtype=torch.long, device=device)
            positions = torch.arange(
                start, start + num_tokens, dtype=torch.long, device=device
            )
            position_ids.append(torch.cat((pads, positions)))
        return torch.stack(position_ids)


def get_cache_block_size(block_size, head_size, num_heads, num_layers, dtype) -> int:
    kv_cache_block_size = block_size * num_heads * head_size * 2  # 2 for k and v

    total_size = num_layers * kv_cache_block_size
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    return dtype_size * total_size


# TODO: This can be improved to profile a forward pass with batch and max length
def get_max_gpu_blocks_available(
    block_size: int,
    emb_dim: int,
    nheads: int,
    nlayers: int,
    gpu_memory_utilization: float,
    dtype,
) -> int:
    """
    gets the max number of gpu blocks available within some gpu memory utilization.

    Note:This will capture whatever is currently being used in GPU to determine the number of blocks to be allocated

    Args:
        block_size: int
            the number of tokens in each block
        emb_dim: int
            the embedding dimension of the model
        nheads: int
            the number of heads in the model
        nlayers: int
            the number of layers in the model
        gpu_memory_utilization: float
            the max gpu memory utilization target
        dtype: dtype
            value type to store in cache

    Returns:
    int
        the number of gpu blocks that can be allocated to stay under the given gpu utilization
    """
    # Calculate the number of blocks that can be allocated with the
    # profiled peak memory.
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    total_gpu_memory = torch.cuda.get_device_properties("cuda").total_memory

    # get the total size of a cache block
    cache_block_size = get_cache_block_size(
        block_size, emb_dim // nheads, nheads, nlayers, dtype
    )

    # determine the number of gpu blocks that can be allocated with the remaining memory
    num_gpu_blocks = int(
        (total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size
    )
    num_gpu_blocks = max(num_gpu_blocks, 0)
    torch.cuda.empty_cache()
    return num_gpu_blocks


class CacheBlock:
    """
    CacheBlock is a logical construct to denote a single block in GPU memory. CacheBlock contains a block_number,
    block_size, and num_tokens.

    block_number is used to determine its physical address in GPU memory
    block_size is the maximum number of slots allowed to be stored in the block (each slot holds a single token)
    num_tokens is the current number of tokens that reside in the CacheBlock
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ):
        self.block_number = block_number
        self.block_size = block_size
        self.num_tokens = 0

    def num_available_slots(self) -> int:
        """Get the total remaining number of slots available in this CacheBlock

        Returns:
        int
            the number of unused slots
        """
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        """Denotes whether all slots are occupied

        Returns:
        bool
            True if all slots are occupied, otherwise False
        """
        return self.num_available_slots() == 0

    def append_num_tokens(self, num_tokens: int):
        """Logically adds num_tokens tokens to the CacheBlock
        Args:
            num_tokens: int
                number of tokens to add
        """
        self.num_tokens += num_tokens

    def subtract_num_tokens(self, num_tokens: int):
        """Logically subtracts num_tokens tokens from the CacheBlock
        Args:
            num_tokens: int
                number of tokens to add
        """
        self.num_tokens -= num_tokens

    def __repr__(self):
        return f"CacheBlock(block_number={self.block_number}, block_size={self.block_size}, num_tokens={self.num_tokens})"


class CacheBlockGroup(List[CacheBlock]):
    """
    CacheBlockGroup is a logical construct to denote a List of CacheBlocks. A CacheBlockGroup consists of a sequence_id,
    block_size, prefix, and ref_count

    sequence_id is used to denote which sequence this CacheBlockGroup belongs to
    block_size is the maximum number of slots allowed to be stored in a single CacheBlock (each slot holds a single
    token)
    prefix is used when the given CacheBlockGroup is referencing another CacheBlockGroup (not new memory)
    ref_count is the count of CacheBlockGroups referencing the current CacheBlockGroup and is used in garbage collection
    """

    def __init__(self, sequence_id: int, block_size: int):
        super().__init__()
        self.sequence_id = sequence_id
        self.block_size = block_size
        self._is_generating = False
        self._is_initialized_with_prompt = False
        self.prefix: Optional[CacheBlockGroup] = None
        self.ref_count = 0

    @classmethod
    def from_prefix(cls, sequence_id: int, prefix: "CacheBlockGroup"):
        """Create a CacheBlockGroup from another CacheBlockGroup (reference) as its prefix

        Args:
            sequence_id: int
                the sequence id for this new CacheBlockGroup
            prefix: CacheBlockGroup
                the prefix for this CacheBlockGroup

        Returns:
        CacheBlockGroup
            a new CacheBlockGroup who's prefixed by the given prefix
        """
        cbg = cls(sequence_id, prefix.block_size)
        cbg._is_generating = True
        cbg._is_initialized_with_prompt = True

        # add duplicate blocks
        for cb in prefix:
            cbg.append(cb)

        # set the prefix
        cbg.prefix = prefix
        # update the reference count of the prefix
        prefix.ref_count += 1

        return cbg

    def remove_tokens(self, num_tokens: int) -> List[CacheBlock]:
        """Logically remove tokens from a CacheBlockGroup

        Args:
            num_tokens: int
                number of tokens to remove

        Returns:
        List[CacheBlock]
            the list of cache blocks that are to be freed
        """
        # remove tokens and return the blocks to be freed
        prefix_sequence_length = (
            0 if self.prefix is None else self.prefix.get_sequence_length()
        )

        if num_tokens > (self.get_sequence_length() - prefix_sequence_length):
            raise ValueError(
                "the number of tokens to remove is greater than what exists in this cache block group not including the prefix"
            )
        num_tokens_to_remove = num_tokens
        blocks_to_free = []
        for cb in reversed(self):
            # if we have more tokens to remove than exist in the cache block, we can simply remove that cache block
            if cb.num_tokens < num_tokens_to_remove:
                num_tokens_to_remove -= cb.num_tokens
                cb.num_tokens = 0
                blocks_to_free.append(cb)
            # if we have more tokens in the cache block than we are trying to remove, we just remove them from the
            # current cache block
            else:
                # if its equal to num tokens, we don't need to free the block as it can just be re-used
                cb.subtract_num_tokens(num_tokens_to_remove)
                break
        # remove the blocks from this CacheBlockGroup that are to be freed in the cache manager
        for _ in blocks_to_free:
            self.pop()
        return blocks_to_free

    def is_initialized_with_prompt(self) -> bool:
        """Denotes whether this CacheBlockGroup has already been initialized with the prompt

        Returns:
        bool
            returns True if the CacheBlockGroup has already gone through a single forward pass where the keys and values
            have been store, otherwise returns False
        """
        return self._is_initialized_with_prompt

    def is_generating(self) -> bool:
        """Denotes whether this CacheBlockGroup is currently being used in generation (decode)

        Returns:
        bool
            Returns True if this CacheBlockGroup is in decode phase, otherwise False
        """
        return self._is_generating

    def last_cache_block_is_full(self) -> bool:
        """Denotes whether the last cache block in this group is full

        Returns:
        bool
            Returns True if the last cache block is full, otherwise False
        """
        return self[-1].is_full()

    def get_sequence_length(self) -> int:
        """Gets the sequence length of this CacheBlockGroup

        Returns:
        int
            the sequence length of this CacheBlockGroup
        """
        if len(self) == 0:
            return 0
        else:
            return sum([cb.num_tokens for cb in self])

    def get_cache_block(self, position: int) -> CacheBlock:
        """Get a single CacheBlock in this CacheBlockGroup

        Args:
            position: int
                the token position

        Returns:
        CacheBlock
            a single CacheBlock which contains a token at the given position
        """
        return self[position // self.block_size]

    def get_slot_mapping(self, position: Optional[int] = None) -> List[int]:
        """Get the slot mapping from the given position til the end of the sequence

        Args:
            position: int, optional
                the start token position. If None, will be 0 (default is None)
        Returns:
        List[int]
            a list containing the slot position for each token from position to sequence length
        """
        slot_mapping = []
        start = position if position else 0
        for position_i in range(start, self.get_sequence_length()):
            block_number = self.get_cache_block(position_i).block_number
            block_offset = position_i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
        return slot_mapping

    def get_block_mapping(self) -> List[int]:
        """gets a list of the block numbers of all CacheBlocks in this CacheBlockGroup
        Returns:
        List[int]
            a list of the block numbers of all CacheBlocks in this CacheBlockGroup
        """
        return [cb.block_number for cb in self]


class PagedKVCacheManager:
    """
    PagedKVCacheManager is used for management of the kv-cache when using the Paged Attention kernels.

    This class is responsible for:

    - allocation of new sequences in the kv-cache
    - allocation/de-allocation of tokens for each sequence
    - garbage collection of all sequences in the kv-cache
    - contains the physical cache (a list of key and value tensors)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        emb_dim: int,
        kv_heads: int = 0,
        total_num_gpu_blocks: Optional[int] = None,
        block_size: int = 16,
        tensor_parallel_size: int = 1,
        device: Optional[Union[str, torch.device]] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        self.block_size = block_size
        self.cache: List[KVCache] = []
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.device = device

        if kv_heads == 0:
            kv_heads = num_heads

        self.kv_heads = kv_heads // tensor_parallel_size if kv_heads > 1 else kv_heads
        self.num_heads = (
            num_heads // tensor_parallel_size if num_heads > 1 else num_heads
        )
        self.emb_dim = emb_dim

        if not total_num_gpu_blocks:
            total_num_gpu_blocks = get_max_gpu_blocks_available(
                block_size,
                emb_dim // tensor_parallel_size,
                self.kv_heads,
                num_layers,
                0.8,
                dtype,
            )
        self.total_num_gpu_blocks = total_num_gpu_blocks

        self.head_size = emb_dim // num_heads

        x = self.block_size // element_size
        key_block_shape = (
            self.kv_heads,
            self.head_size // x,
            block_size,
            x,
        )
        value_block_shape = (
            self.kv_heads,
            self.head_size,
            block_size,
        )
        for _ in range(num_layers):
            key_blocks = torch.empty(
                size=(total_num_gpu_blocks, *key_block_shape),
                dtype=dtype,
                device=self.device,
            )
            value_blocks = torch.empty(
                size=(total_num_gpu_blocks, *value_block_shape),
                dtype=dtype,
                device=self.device,
            )
            mark_static_address(key_blocks)
            mark_static_address(value_blocks)
            self.cache.append((key_blocks, value_blocks))

        self.free_blocks: List[CacheBlock] = []
        self.unused_keys: queue.Queue[int] = queue.Queue(len(self.free_blocks))
        for i in range(total_num_gpu_blocks):
            self.free_blocks.append(CacheBlock(i, block_size))
            self.unused_keys.put_nowait(i)

        # each sequence will be mapped to a cache block group
        # for now this will just assume we always have the same sequences in batch
        self.cbg_map: Dict[int, CacheBlockGroup] = {}

    def get_max_sequence_length(self, sequence_ids: List[int]) -> int:
        return max(
            [self.cbg_map[seq_id].get_sequence_length() for seq_id in sequence_ids]
        )

    def _allocate_block(self) -> CacheBlock:
        return self.free_blocks.pop()

    @staticmethod
    def __pad_to_max_left(x: List[int], max_len: int, pad: int) -> List[int]:
        return [pad] * (max_len - len(x)) + x

    @staticmethod
    def __pad_to_max_right(x: List[int], max_len: int, pad: int) -> List[int]:
        return x + [pad] * (max_len - len(x))

    def is_generating(self, sequence_ids: List[int]):
        for sequence_id in sequence_ids:
            if (
                sequence_id not in self.cbg_map
                or not self.cbg_map[sequence_id].is_generating()
            ):
                return False
        return True

    def is_initialized_with_prompt(self, sequence_ids: List[int]):
        for sequence_id in sequence_ids:
            if (
                sequence_id not in self.cbg_map
                or not self.cbg_map[sequence_id].is_initialized_with_prompt()
            ):
                return False
        return True

    def free(self, sequence_id: int):
        if sequence_id not in self.cbg_map:
            return
        cbg = self.cbg_map[sequence_id]

        if cbg.ref_count != 0:
            raise ValueError(
                f"This sequence id is being reference by other sequences and cannot be freed"
            )

        # remove a reference count from all cache block groups that was a prefix as part of this sequence
        if cbg.prefix is not None:
            cbg.prefix.ref_count -= 1
            prefix_block_numbers = set(cbg.prefix.get_block_mapping())

        for cb in cbg:
            if cbg.prefix is None or (
                cbg.prefix is not None and cb.block_number not in prefix_block_numbers
            ):
                cb.num_tokens = 0
                self.free_blocks.append(cb)
        self.unused_keys.put_nowait(sequence_id)
        del self.cbg_map[sequence_id]

    def free_sequences(self, sequence_ids: List[int], recursive: bool = False):
        """
        free the given sequence ids from the kv-cache

        Parameters
        ----------
        sequence_ids: List[int]
            list of sequence ids to free
        recursive: bool
            if True, will free all sequences this sequence is referencing, directly or indirectly, otherwise will only
            free this sequence alone.
        """
        for seq_id in sequence_ids:
            prefix_cbg = self.cbg_map[seq_id].prefix if recursive else None
            self.free(seq_id)
            while prefix_cbg is not None:
                # only free a prefix if its refcount is 0, otherwise it is still being used by another sequence
                if prefix_cbg.ref_count == 0:
                    self.free(prefix_cbg.sequence_id)
                prefix_cbg = prefix_cbg.prefix

    def _get_unassigned_sequence_id(self) -> int:
        return self._get_unassigned_sequence_ids(1)[0]

    def _get_unassigned_sequence_ids(self, num_sequences: int) -> List[int]:
        return [self.unused_keys.get_nowait() for _ in range(num_sequences)]

    def _get_cache_metadata(
        self,
        sequence_ids: List[int],
        is_prompt: bool,
        num_tokens_per_sequence: Optional[List[int]] = None,
    ) -> PagedAttentionCacheData:
        slot_mapping = []
        block_tables = []
        context_lengths = []
        max_sequence_length = self.get_max_sequence_length(sequence_ids)
        remainder = max_sequence_length % self.block_size
        max_num_blocks = max_sequence_length // self.block_size
        max_num_tokens_per_sequence = (
            max(num_tokens_per_sequence) if num_tokens_per_sequence else None
        )
        if remainder != 0:
            max_num_blocks += 1
        i = 0
        for sequence_id in sequence_ids:
            cbg = self.cbg_map[sequence_id]

            context_length = cbg.get_sequence_length()
            if is_prompt:
                slot = cbg.get_slot_mapping()
                slot = self.__pad_to_max_left(slot, max_sequence_length, -1)
            else:
                num_tokens = num_tokens_per_sequence[i]  # type: ignore
                start = context_length - num_tokens
                slot = self.__pad_to_max_left(
                    cbg.get_slot_mapping(start), max_num_tokens_per_sequence, -1  # type: ignore
                )
                i += 1

            block_mapping = cbg.get_block_mapping()
            block_mapping = self.__pad_to_max_right(block_mapping, max_num_blocks, 0)

            slot_mapping.append(slot)
            block_tables.append(block_mapping)
            context_lengths.append(context_length)

        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        )
        block_tables_tensor = torch.tensor(
            block_tables, dtype=torch.int, device=self.device
        )
        context_lengths_tensor = torch.tensor(
            context_lengths, dtype=torch.int, device=self.device
        )

        return PagedAttentionCacheData(
            data=self.cache,
            max_sequence_length=max_sequence_length,
            context_lengths=None if is_prompt else context_lengths_tensor,
            slot_mapping=slot_mapping_tensor,
            block_mapping=block_tables_tensor,
            block_size=self.block_size,
            scale=self.head_size**-0.5,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            head_size=self.head_size,
            is_generating=not is_prompt,
            sequence_ids=sequence_ids,
        )

    def allocate_tokens(
        self,
        num_tokens_per_sequence: List[int],
        sequence_ids: Optional[List[int]] = None,
    ) -> PagedAttentionCacheData:
        """
        allocate tokens in the kv-cache. If sequence ids are not given, this will be considered pre-fill for the prompt,
        and sequence ids will be generated. If sequence ids are given, this will be considered allocating tokens for
        generation and paged attention will be used

        Parameters
        ----------
        num_tokens_per_sequence: List[int]
            the number of tokens per sequence to expand in the kv-cache. This should correspond index-to-index with the
            given sequence_ids
        sequence_ids: List[int], optional
            a list of sequence ids that will be expanded in the cache with generated tokens. If no sequence ids are
            given, a new sequence id will be generated and allocation will be considered for the prompt
            (default is None)

        Returns
        -------
        PagedAttentionCacheData
            a cache data object that includes metadata associated with it based on the current state of the
            PagedKVCacheManager for the given sequence ids.
        """
        if sequence_ids is None:
            return self._allocate_prompt_tokens(num_tokens_per_sequence)
        else:
            return self._allocate_generated_tokens(
                sequence_ids, num_tokens_per_sequence
            )

    def _allocate_prompt_tokens(
        self, num_tokens_per_sequence: List[int]
    ) -> PagedAttentionCacheData:
        sequence_ids = self._get_unassigned_sequence_ids(len(num_tokens_per_sequence))

        for seq_id, num_tokens in zip(sequence_ids, num_tokens_per_sequence):
            self._allocate_prompt_sequence(seq_id, num_tokens)

        return self._get_cache_metadata(sequence_ids, is_prompt=True)

    def _allocate_generated_tokens(
        self, sequence_ids: List[int], num_tokens_per_sequence: List[int]
    ) -> PagedAttentionCacheData:
        for seq_id, num_tokens in zip(sequence_ids, num_tokens_per_sequence):
            cache_block_group = self.cbg_map[seq_id]
            cache_block_group._is_generating = True

            for i in range(num_tokens):
                if cache_block_group.last_cache_block_is_full():
                    last_block = self._allocate_block()
                    last_block.append_num_tokens(1)
                    cache_block_group.append(last_block)
                else:
                    cache_block_group[-1].append_num_tokens(1)

        return self._get_cache_metadata(
            sequence_ids,
            is_prompt=False,
            num_tokens_per_sequence=num_tokens_per_sequence,
        )

    def _allocate_prompt_sequence(self, seq_id: int, num_tokens: int):
        cache_block_group: CacheBlockGroup = CacheBlockGroup(seq_id, self.block_size)

        # one block allocation will happen automatically as the group always starts empty
        last_cache_block = self._allocate_block()

        cursor = 0
        while cursor < num_tokens:
            tokens_to_append = (
                min(num_tokens, cursor + last_cache_block.num_available_slots())
                - cursor
            )
            last_cache_block.append_num_tokens(tokens_to_append)
            cursor += tokens_to_append

            if cursor >= num_tokens:
                # we are done, so we need to append but not allocate
                cache_block_group.append(last_cache_block)
            elif last_cache_block.is_full():
                # if the block is full we can append it
                cache_block_group.append(last_cache_block)
                # because the other condition did not hold, we can allocate a new block
                last_cache_block = self._allocate_block()

        cache_block_group._is_initialized_with_prompt = True
        self.cbg_map[seq_id] = cache_block_group

    def add_child_sequence(self, parent_sequence_id: int) -> int:
        parent_cbg = self.cbg_map[parent_sequence_id]

        child_sequence_id = self._get_unassigned_sequence_id()
        child_cbg = CacheBlockGroup.from_prefix(child_sequence_id, parent_cbg)
        key_caches = [key_cache for key_cache, _ in self.cache]
        value_caches = [value_cache for _, value_cache in self.cache]

        if not parent_cbg.last_cache_block_is_full():
            new_block_to_copy = self._allocate_block()
            cache_ops.copy_blocks(
                key_caches,
                value_caches,
                {parent_cbg[-1].block_number: [new_block_to_copy.block_number]},
            )
            new_block_to_copy.append_num_tokens(parent_cbg[-1].num_tokens)
            child_cbg.pop()
            child_cbg.append(new_block_to_copy)

        self.cbg_map[child_sequence_id] = child_cbg

        return child_sequence_id

    def add_child_sequences(
        self, parent_sequence_id: int, num_sequences: int
    ) -> list[int]:
        child_sequence_ids = []
        for _ in range(num_sequences):
            child_sequence_ids.append(self.add_child_sequence(parent_sequence_id))
        return child_sequence_ids

    def remove_tokens(self, sequence_id: int, num_tokens: int):
        blocks_to_free = self.cbg_map[sequence_id].remove_tokens(num_tokens)
        for cb in blocks_to_free:
            self.free_blocks.append(cb)
