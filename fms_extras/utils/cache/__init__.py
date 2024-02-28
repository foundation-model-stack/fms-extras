import abc
import dataclasses
from typing import List, Optional, Tuple

import torch


def compute_position_ids(
    num_tokens_per_sequence: List[int], context_lengths: Optional[List[int]] = None
) -> List[List[int]]:
    """Compute position ids based on the current context lengths and the new tokens to add

    Parameters
    ----------
    num_tokens_per_sequence: List[int]
        number of tokens to be added to each sequence
    context_lengths: List[int], optional
        optional list of current context lengths per sequence. If none, will assume no context length and starting
        position will be 0 (default is None)

    Returns
    -------
    List[List[int]]
        the position ids for each sequence
    """

    max_tokens = max(num_tokens_per_sequence)
    position_ids = []
    for seq_i, num_tokens in enumerate(num_tokens_per_sequence):
        if context_lengths is None:
            start = 0
        else:
            start = context_lengths[seq_i] - 1
        position_ids_i = [0 for _ in range(max_tokens - num_tokens)] + [
            i for i in range(start, start + num_tokens)
        ]
        position_ids.append(position_ids_i)
    return position_ids


class AttentionComputationMixin(metaclass=abc.ABCMeta):
    """
    Include this mixin in a class to implement a custom version of attention
    """

    @abc.abstractmethod
    def attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        pass


@dataclasses.dataclass
class CacheDataLayer(metaclass=abc.ABCMeta):
    """
    Dataclass responsible for storing keys and values in a single layer of cache data

    Attributes
    ----------
    data_layer: Tuple[torch.Tensor, torch.Tensor]
        a tuple corresponding to the key block and value block
    """

    data_layer: Tuple[torch.Tensor, torch.Tensor]

    @abc.abstractmethod
    def get_cache_type(self) -> str:
        """
        Get the name associated with this cache data layer

        Returns
        -------
        str
            the name associated with this cache data layer
        """
        pass

    @abc.abstractmethod
    def store(
        self, key: torch.Tensor, value: torch.Tensor
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
        pass

    @abc.abstractmethod
    def is_filled(self) -> bool:
        """
        Denotes whether this cache data layer is in post-fill stage

        Returns
        -------
        bool
            True if cache data layer is currently in the post-fill stage, otherwise False and cache data layer is being
            pre-filled
        """
        pass


@dataclasses.dataclass
class CacheData(metaclass=abc.ABCMeta):
    """
    Dataclass responsible for holding raw cache data.

    Attributes
    ----------
    data: List[Tuple[torch.Tensor, torch.Tensor]]
        data is represented as a List of tuples of Tensors where each index in the list corresponds to a layer in the
        cache, and each position in the tuple corresponds to the key and value cache block for that layer
    max_sequence_length: int
        max sequence length of all sequences stored in this cache data
    """

    data: List[Tuple[torch.Tensor, torch.Tensor]]
    max_sequence_length: int

    @abc.abstractmethod
    def get_layer(self, layer_index: int) -> CacheDataLayer:
        """
        Get a single layer of the cache data

        Parameters
        ----------
        layer_index: int
            index of layer

        Returns
        -------
        CacheDataLayer
            a single layer of the cache data as a dataclass
        """
        pass

    @abc.abstractmethod
    def is_filled(self) -> bool:
        """
        Determines if the cache has been filled with the prompt, or is completely empty for this piece of cache data

        Returns
        -------
        bool
            True if the keys and values for the prompt have been set, otherwise False.
        """
        pass


@dataclasses.dataclass
class CacheDataWithMetadata(CacheData):
    """A special form of CacheData that includes some simple metadata associated with it

    Attributes
    ----------
    data: List[Tuple[torch.Tensor, torch.Tensor]]
        data is represented as a List of tuples of Tensors where each index in the list corresponds to a layer in the
        cache, and each position in the tuple corresponds to the key and value cache block for that layer
    max_sequence_length: int
        max sequence length of all sequences corresponding to the sequence ids in this cache data
    sequence_ids: List[int]
        the integer ids associated with each sequence, these will correspond by index with the input ids passed to the
        model
    context_lengths: torch.Tensor, optional
        a 1d tensor corresponding to the length of each sequence in the batch denoted by the sequence ids. If None,
        cache data is in pre-fill
    """

    data: List[Tuple[torch.Tensor, torch.Tensor]]
    max_sequence_length: int
    sequence_ids: List[int]
    context_lengths: Optional[torch.Tensor]

    def compute_position_ids(self, num_tokens_per_sequence: List[int]) -> torch.Tensor:
        position_ids_list = compute_position_ids(
            num_tokens_per_sequence,
            None
            if self.context_lengths is None
            else [
                r - l + 1
                for l, r in zip(num_tokens_per_sequence, self.context_lengths.tolist())
            ],
        )
        return torch.tensor(
            position_ids_list, dtype=torch.long, device=self.data[0][0].device
        )


class KVCacheManager(metaclass=abc.ABCMeta):
    """
    Simple interface for managing any arbitrary KV-Cache. The kv-cache manager is responsible for keeping track of the
    sequences that are being stored in the cache denoted by an integer ID.
    """

    @abc.abstractmethod
    def allocate_tokens(
        self,
        num_tokens_per_sequence: List[int],
        sequence_ids: Optional[List[int]] = None,
    ) -> CacheDataWithMetadata:
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
        CacheDataWithMetadata
            a cache data object that includes metadata associated with it based on the current state of the
            KVCacheManager for the given sequence ids.
        """
        pass

    @abc.abstractmethod
    def free_sequences(self, sequence_ids: List[int]):
        """
        free the given sequence ids from the kv-cache

        Parameters
        ----------
        sequence_ids: List[int]
            list of sequence ids to free
        """
        pass


KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key cache, value cache)


def flatten_batch(inp):
    """
    Takes a speculator suffix tree: a bsize x n_candidates x candidate_len rectangular batch
    of token indices, and flattens it while removing redundant tokens. For example, given:
    a b c
    a b d
    a e f
    Tokens 'a b' in line 2 and token 'a' in line 3 are functionally equivalent to 'a b' in
    line 1, so the flattened batch returns `a b c d e f`

    Returns:
    1) the flattened, pruned input
    2) a tensor, sized as input, mapping each input token to its slot in output
    3) a tensor, sized as output, mapping each output token to its slot in the flattened input
    """
    ind_out = torch.zeros_like(inp)
    inp = inp.tolist()
    out = []
    ind_flat = []
    batch_offset = 0
    for b, candidate_set in enumerate(inp):
        lineages = []
        for k, candidate in enumerate(candidate_set):
            for n in range(len(candidate)):
                lineage = tuple(candidate[: n + 1])
                if lineage in lineages:
                    # Token is redundant
                    ind_out[b, k, n] = lineages.index(lineage) + batch_offset
                else:
                    # Token is not redundant
                    ind_out[b, k, n] = len(lineages) + batch_offset
                    lineages.append(lineage)
                    ind_flat.append(
                        b * len(inp[0]) * len(inp[0][0]) + k * len(inp[0][0]) + n
                    )
        out.append(
            torch.tensor(
                [lineage[-1] for lineage in lineages],
                device=ind_out.device,
                dtype=torch.int32,
            )
        )
        batch_offset += len(lineages)
    return (
        torch.cat(out),
        ind_out,
        torch.tensor(ind_flat, device=ind_out.device, dtype=torch.int32),
    )


def select_inflate_dim(inp, inds, dim=0):
    """
    Takes an input of size ([...], n, [...]), with n in slot corresponding to value of dim,
    and tensor of indices of size (a, ..., z). Using those indices we over/under sample the
    input on dimension n, to create output tensor with size ([...], (a, ..., z), [...]).
    
    i.e. if dim=0, inp has size (6,3,2), and inds has size (8,4), then:
    1) max(inds) < 6
    2) output has size (8,4,3,2)
    """
    inds_shape = inds.size()
    inp_shape = inp.size()
    out = inp.index_select(dim, inds.view(-1))
    return out.view(*inp_shape[:dim], *inds_shape, *inp_shape[dim + 1 :])
