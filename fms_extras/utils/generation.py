import functools
import time
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from fms_extras.utils.cache.paged import PagedKVCacheManager


def __right_pad_zeros(input_tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    return torch.stack(
        [
            F.pad(
                input_tensor[i],
                (0, max_length - input_tensor.size(1)),
            )
            for i in range(input_tensor.size(0))
        ]
    )


def __create_prefill_mask(
    model_input_lengths: List[int], device: Union[str, torch.device]
) -> torch.Tensor:
    max_tokens = max(model_input_lengths)

    is_pad_list = []
    for seq_len in model_input_lengths:
        pads = torch.zeros(max_tokens - seq_len, dtype=torch.bool, device=device)
        non_pads = torch.ones(seq_len, dtype=torch.bool, device=device)
        is_pad_list.append(torch.cat((pads, non_pads)))
    is_pad = torch.stack(is_pad_list)
    mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
    return mask.tril(diagonal=0)


def paged_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids_list: List[torch.Tensor],
    kv_cache_manager: PagedKVCacheManager,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None,
    # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
    cudagraphs: bool = False,
) -> Tuple[torch.Tensor, int, float, float]:
    """
    A trivial generate function that can be used for validation/testing generation using paged attention

    Args:
        model: Callable or nn.Module
            A function or nn.Module that takes a batch of input_ids and returns logits
        input_ids_list: List[torch.Tensor]
            A list of tensors, each tensor being for a single prompt
        kv_cache_manager: PagedKVCacheManager
            the paged KVCacheManager that handles management of the kv-cache for paged attention
        max_seq_len: int
            the max sequence length of the model
        max_new_tokens: int
            max tokens to generate
        temperature: float
            temperature of softmax when sampling
        top_k: int
            only search among top k tokens
        do_sample: bool
            multinomial sampling. False for greedy.
        decode_model: Callable or nn.Module, optional
            a model to used specifically for decode step. If not given, the model input will be used for decode step
        cudagraphs: bool
            if True, model is using cudagraphs and input will be padded, otherwise no cudagraphs is used and input is
            not padded

    Returns:
    Tuple[torch.Tensor, int, float, float]
        the resulting output tokens, the number of new tokens generated, the time to first token in seconds, and the
        cumulative time of all subsequent generation steps (forward pass, kv-cache management, sampling) in seconds
    """
    start_time = time.time()
    if decode_model is None:
        decode_model = model

    bsize = len(input_ids_list)

    if cudagraphs and bsize != 1:
        raise NotImplementedError(
            "cudagraphs is not yet supported for batch sizes greater than 1"
        )

    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids_list])
    model_input_lengths = [seq.size(0) for seq in input_ids_list]
    input_ids = torch.stack(
        [
            F.pad(input_ids_list[i], (max_len - model_input_lengths[i], 0))
            for i in range(bsize)
        ]
    )

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["cache_data"] = None
    kwargs["use_cache"] = True
    kwargs["mask"] = __create_prefill_mask(model_input_lengths, device=input_ids.device)
    sequence_ids: Optional[List[int]] = None
    block_mapping_max = ((max_len + max_new_tokens) // 16) + 1
    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        kwargs["cache_data"] = kv_cache_manager.allocate_tokens(
            model_input_lengths, sequence_ids
        )

        if i == 0:
            logits, _ = model(input_ids, **kwargs)
        else:
            # cudagraph requires static shapes
            if cudagraphs:
                kwargs["cache_data"].block_mapping = __right_pad_zeros(
                    kwargs["cache_data"].block_mapping, block_mapping_max
                )

            logits, _ = decode_model(input_ids, **kwargs)
        logits = logits[:, -1, :]

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        result = torch.cat((result, next_val), dim=-1)

        next_input = next_val

        if i == 0:
            ttft = time.time() - start_time
            start_time = time.time()
            kwargs["mask"] = None
            sequence_ids = kwargs["cache_data"].sequence_ids
            model_input_lengths = [1 for _ in range(bsize)]

    kv_cache_manager.free_sequences(sequence_ids)  # type: ignore
    return result, max_new_tokens, ttft, (time.time() - start_time)
