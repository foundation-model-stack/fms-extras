import functools
import time
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from fms_extras.models.speculator import (
    MLPSpeculator,
    flatten_batch,
    apply_index_map,
)
from fms_extras.utils.cache.paged import PagedAttentionCacheData, PagedKVCacheManager


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    speculator: MLPSpeculator,
    kv_cache_manager: PagedKVCacheManager,
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    n_candidates: int = 5,
    threshes=[5, 3, 2],
    flatting=True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None,
    # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
    cudagraphs: bool = False,
):
    """
    A reference implementation of speculative decoding generation.
    Returns at least the specified number of tokens - the speculator may return a
    few extra in the final step.
    If input is batched, continues generating until EVERY sequence has produced AT LEAST the required number of tokens.
    Input (and output) tokens beyond max_seq_len are simply dropped for a sliding-window approach.
    Currently reproduces behavior of greedy decoding only.
    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: A length n tensor of token IDs, or list of such tensors
        speculator: A function or nn.Module that takes a state vector and sampled token
            and returns a set of candidate suffixes
        kv_cache_manager: PagedKVCacheManager
            the paged kv-cache manager to be used in generation
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        n_candidates: only consider the top n most confident candidates from the speculator
        threshes: build candidate suffix trees by taking top-(threshes[n]) most confident values
            for head n. len(threshes) must equal speculator.n_predict. prod(threshes) must be greater
            than or equal to n_candidates (we cannot have more candidates than tree leaves).
        flatting: enable batch flattening / tree attention with redundant prefix removal when compression
            ratio is favorable. Adds extra overhead to shrink the token count in each batch.
        decode_model: TODO
        cudagraphs: TODO
    Returns:
        result: List of id tensors, possibly different lengths if batching.
        n_steps: Number of foward passes used to generate provided tokens.
    """
    start_time = time.time()
    # Construct batch(es) and initial inputs
    if decode_model is None:
        decode_model = model
    bsize = len(input_ids)

    if cudagraphs and (flatting or bsize != 1):
        raise NotImplementedError(
            "cudagraphs is not yet supported for batch sizes greater than 1 or flatting"
        )

    result = input_ids  # [b] n

    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    model_input_lengths = [seq.size(0) for seq in input_ids]
    inputs = torch.stack(
        [
            F.pad(input_ids[i], (max_len - model_input_lengths[i], 0))
            for i in range(bsize)
        ]
    )
    cache_data: PagedAttentionCacheData = kv_cache_manager.allocate_tokens(
        model_input_lengths
    )
    parent_sequence_ids = cache_data.sequence_ids

    # Build padded causal mask
    mask = __create_prefill_mask(model_input_lengths, inputs.device)

    # Build kv cache and get initial state vector
    inp_len = speculator.n_predict + 1
    inputs = inputs[:, -max_seq_len + inp_len :]
    output = model(
        inputs,
        mask=mask,
        cache_data=cache_data,
        return_embeds=True,
        use_cache=True,
    )
    logits, _, embeds = output
    embeds = embeds[:, -1:]  # b 1 d
    logits = logits[:, -1:]  # b 1 v
    ttft = time.time() - start_time

    start_time = time.time()
    n_gen = torch.zeros(bsize, device=inputs.device, dtype=torch.int)
    n_steps = 0
    input_ids = torch.argmax(logits, dim=2)  # b 1
    result = [
        torch.cat((line, input_id), dim=0)
        for line, input_id in zip(result, list(input_ids))
    ]
    block_mapping_max = ((max_len + new_tokens) // 16) + 1
    while min(n_gen) < new_tokens:
        n_steps += 1

        # create candidate sequences
        child_sequence_ids_list = []
        child_sequence_ids_flattened = []
        num_tokens_per_sequence = [
            inp_len for _ in range(input_ids.size(0) * n_candidates)
        ]
        # each parent will have n_candidates child sequences
        for parent_sequence_id in parent_sequence_ids:
            child_sequence_ids = kv_cache_manager.add_child_sequences(
                parent_sequence_id, n_candidates
            )
            child_sequence_ids_list.append(child_sequence_ids)
            child_sequence_ids_flattened.extend(child_sequence_ids)

        # add inp_len tokens to each candidate
        cache_data = kv_cache_manager.allocate_tokens(
            num_tokens_per_sequence, child_sequence_ids_flattened
        )

        # Get candidate set of speculations
        suffix_ids = speculator.generate_suffixes(
            embeds, input_ids, threshes, n_candidates
        )  # b k h
        input_ids = torch.cat(
            [input_ids.unsqueeze(1).expand(bsize, n_candidates, 1), suffix_ids], dim=-1
        ).int()  # b k 1+h

        # Apply batch flattening / tree attention if compression is good enough
        this_flatting = False
        if flatting:
            flat_inputs, unflat_indices, flat_indices = flatten_batch(
                input_ids
            )  # n', b k 1+h, n'
            compression = flat_inputs.numel() / input_ids.numel()
            if compression < 0.75:
                this_flatting = True
                flat_inputs = flat_inputs[None,]  # 1 n'
                cache_data.unflatten_indices = unflat_indices
                cache_data.flatten_indices = flat_indices
                cache_data.position_ids = apply_index_map(
                    cache_data.position_ids.view(-1), flat_indices
                )[
                    None,
                ]
        input_ids = input_ids.view(-1, inp_len)  # bk 1+h

        # Set up kv cache metadata for paged memory access over tokens/candidates with shared prefixes
        context_lengths = cache_data.context_lengths  # bk
        inflate_factor = (
            cache_data.query_length
            if cache_data.unflatten_indices is None
            else cache_data.unflatten_indices.size(-1)
        )
        # no reason to check type here as generation allocation always returns context_lengths
        context_lengths = context_lengths.unsqueeze(1).expand(  # type: ignore
            -1, inflate_factor
        )  # bk n
        # subtract arange(inp_len) to get lengths for each token in candidates
        context_lengths = (
            context_lengths.sub(context_lengths.sign().cumsum(1).flip([1]).sub(1))
            .int()
            .view(-1)
        )  # bkn
        block_mappings = cache_data.block_mapping.repeat_interleave(
            inflate_factor, dim=0
        )  # bkn n_blocks
        # If batch is flattened, flatten corresponding metadata too
        if cache_data.flatten_indices is not None:
            context_lengths = apply_index_map(
                context_lengths, cache_data.flatten_indices
            )  # n'
            block_mappings = apply_index_map(
                block_mappings, cache_data.flatten_indices
            )  # n' n_blocks

        # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
        if cudagraphs:
            # pad for cudagraphs
            block_mappings = torch.stack(
                [
                    F.pad(
                        block_mappings[i],
                        (0, block_mapping_max - block_mappings.size(1)),
                    )
                    for i in range(block_mappings.size(0))
                ]
            )

        cache_data.block_mapping = block_mappings
        cache_data.context_lengths = context_lengths

        input_ids_unflat = input_ids.view(bsize, n_candidates, inp_len)
        if this_flatting:
            input_ids = flat_inputs

        # Base model forward pass
        output = decode_model(
            input_ids,
            cache_data=cache_data,
            return_embeds=True,
            use_cache=True,
        )  # 1 n' v  OR  bk 1+h v
        logits, _, embeds = output  # 1 n' v, 1 n' d  OR  bk 1+h v, bk 1+h d
        next_vals = torch.argmax(logits, dim=-1)  # 1 n'  OR  bk 1+h

        # If we used batch flattening / tree attention, unflatten the outputs
        if this_flatting:
            next_vals = apply_index_map(next_vals[0], unflat_indices)  # b k 1+h
            embeds = apply_index_map(embeds[0], unflat_indices)  # b k 1+h d
        else:
            next_vals = next_vals.view(bsize, n_candidates, inp_len)  # b k 1+h
            embeds = embeds.view(
                bsize, n_candidates, inp_len, embeds.size(2)
            )  # b k 1+h d

        # Check correctness of speculator predictions
        test = input_ids_unflat[:, :, 1:].eq(next_vals[:, :, :-1]).cumprod(2)
        n_correct = test.sum(2).view(bsize, n_candidates)
        best_guess = n_correct.argmax(1)  # b
        best_guess_unflat = (
            best_guess.unsqueeze(1).expand(bsize, inp_len).unsqueeze(1)
        )  # b 1 1+h

        # Set global values to those of best guess
        next_vals = next_vals.gather(1, best_guess_unflat).squeeze(1)  # b 1+h
        n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
        embeds = embeds.gather(
            1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(3))
        ).squeeze(
            1
        )  # b 1+h d

        # free all non-best candidates and keep best candidates as parents
        parent_sequence_ids = []
        for parent_index, child_sequence_ids in enumerate(child_sequence_ids_list):
            best_index = int(best_guess[parent_index].item())

            # free all bad candidates
            kv_cache_manager.free_sequences(
                child_sequence_ids[:best_index] + child_sequence_ids[best_index + 1 :]
            )

            # decrease the context length of the sequence which used to be sequence length + inp_len by the number of incorrect tokens
            # for the best candidate
            best_sequence_id = child_sequence_ids[best_index]
            parent_sequence_ids.append(best_sequence_id)
            kv_cache_manager.remove_tokens(
                best_sequence_id, inp_len - n_correct[parent_index].item() - 1
            )

        # Remove any wrong speculator tokens from best candidate
        next_vals_split = list(next_vals)
        next_vals_split = [
            next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
        ]  # [b] h'
        n_gen += n_correct + 1
        embeds = embeds.gather(
            1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
        )  # Grab last correct embed

        # Update results
        result = [
            torch.cat((result[i], next_vals_split[i]), dim=0) for i in range(bsize)
        ]
        input_ids = torch.stack([line[-1:] for line in next_vals_split], dim=0)  # b 1

    kv_cache_manager.free_sequences(parent_sequence_ids, recursive=True)
    return result, n_steps, ttft, (time.time() - start_time)


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
