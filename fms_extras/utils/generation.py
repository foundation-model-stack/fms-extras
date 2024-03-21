import functools
import time
from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F

from fms_extras.models.speculator import (
    MLPSpeculator,
    flatten_batch,
    select_inflate_dim,
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
        n_candidates: only score the top k candidates from the speculator
        threshes: use top k predictions from each head to generate speculator candidate pool
        verbose_dict: Optional HF tokenizer vocab dict. If provided, runs verbosely and prints
            speculator behavior and scoring for each step
    Returns:
        result: List of id tensors, possibly different lengths if batching.
        n_steps: Number of foward passes used to generate provided tokens.
    """
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
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    n_pads = torch.tensor(n_pads_init).to(device=input_ids[0].device, dtype=torch.int)
    inputs = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(bsize)]
    )
    num_tokens_per_sequence = torch.count_nonzero(inputs.T, dim=0).tolist()
    cache_data: PagedAttentionCacheData = kv_cache_manager.allocate_tokens(
        num_tokens_per_sequence
    )
    parent_sequence_ids = cache_data.sequence_ids

    # Build padded causal mask
    mask = torch.ones(
        bsize,
        1,
        inputs.size(1),
        inputs.size(1),
        device=inputs.device,
    )
    mask = mask.tril()  # b 1 n n

    # Mask off any left-pads
    pad_mask = torch.arange(mask.size(3), device=mask.device).view(
        1, 1, 1, -1
    )  # 1 1 1 n
    pad_mask = pad_mask.expand(bsize, 1, 1, -1)  # b 1 1 n
    pad_mask = pad_mask.sub(n_pads.sub(1).view(-1, 1, 1, 1)).clamp(0, 1)
    eye = torch.eye(mask.size(3), device=mask.device)[None, None, :, :]  # 1 1 n n
    mask = mask.mul(pad_mask).logical_or(eye).log()  # b 1 n n

    # Handle position_ids
    pos_ids = torch.arange(mask.size(3), device=inputs.device).repeat(bsize, 1)  # b n
    pos_ids -= n_pads[:, None]

    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = True

    # Build kv cache and get initial state vector
    inp_len = speculator.n_predict + 1
    inputs = inputs[:, -max_seq_len + inp_len :]
    position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)
    output = model(
        inputs,
        position_ids=position_ids,
        mask=mask,
        cache_data=cache_data,
        return_embeds=True,
        **kwargs
    )
    logits, _, embeds = output
    embeds = embeds[:, -1:]  # b 1 d
    logits = logits[:, -1:]  # b 1 v

    n_gen = torch.zeros(bsize, device=inputs.device, dtype=torch.int)
    n_steps = 0
    input_ids = torch.argmax(logits, dim=2)  # b 1
    block_mapping_max = ((max_len + new_tokens) // 16) + 1
    start_time = time.time()
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
        position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)

        # Get candidate set of speculations
        input_ids = speculator.generate_suffixes(
            embeds, input_ids, threshes, n_candidates
        )  # b k h
        input_ids = torch.cat(
            [input_ids.unsqueeze(1).expand(bsize, n_candidates, 1), input_ids], dim=-1
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
                position_ids = select_inflate_dim(position_ids.view(-1), flat_indices)[
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
            context_lengths = select_inflate_dim(
                context_lengths, cache_data.flatten_indices
            )  # n'
            block_mappings = select_inflate_dim(
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

        input_ids_unflat = input_ids
        if this_flatting:
            input_ids = flat_inputs

        # Base model forward pass
        output = decode_model(
            input_ids,
            position_ids=position_ids,
            cache_data=cache_data,
            return_embeds=True,
            **kwargs
        )  # 1 n' v  OR  bk 1+h v
        logits, _, embeds = output  # 1 n' v, 1 n' d  OR  bk 1+h v, bk 1+h d
        next_vals = torch.argmax(logits, dim=-1)  # 1 n'  OR  bk 1+h

        # If we used batch flattening / tree attention, unflatten the outputs
        if this_flatting:
            next_vals = select_inflate_dim(next_vals[0], unflat_indices)  # b k 1+h
            embeds = select_inflate_dim(embeds[0], unflat_indices)  # b k 1+h d
        else:
            next_vals = next_vals.view(bsize, n_candidates, inp_len)  # b k 1+h
            embeds = embeds.view(
                bsize, n_candidates, inp_len, embeds.size(2)
            )  # b k 1+h d

        # Check correctness of speculator predictions
        test = input_ids_unflat[:, :-1].eq(next_vals[:, 1:]).cumprod(1)
        n_correct = test.sum(1).view(bsize, n_candidates)
        best_guess = n_correct.argmax(1)  # b
        best_guess_unflat = (
            best_guess.unsqueeze(1).expand(bsize, inp_len).unsqueeze(1)
        )  # b 1 1+h

        # Set global values to those of best guess
        next_vals = next_vals.gather(1, best_guess_unflat).squeeze(1)  # b 1+h
        n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
        embeds = embeds.gather(
            1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
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
    end_time = time.time()
    return result, n_steps, (end_time - start_time)


def paged_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    kv_cache_manager: PagedKVCacheManager,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None,
    # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
    cudagraphs: bool = False,
):
    """
    A trivial generate function that can be used for validation/testing generation using paged attention

    Args:
        model: Callable or nn.Module
            A function or nn.Module that takes a batch of input_ids and returns logits
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

    Returns:
    Tuple[torch.Tensor, int, int]
        the resulting output tokens, the number of new tokens generated, and the time it took per token in seconds
    """
    if decode_model is None:
        decode_model = model

    bsize = len(input_ids)

    if cudagraphs and bsize != 1:
        raise NotImplementedError(
            "cudagraphs is not yet supported for batch sizes greater than 1"
        )

    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    input_ids = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(bsize)]
    )

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["cache_data"] = None
    kwargs["use_cache"] = True
    sequence_ids: Optional[List[int]] = None
    block_mapping_max = ((max_len + max_new_tokens) // 16) + 1
    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # compute the mask
        if i == 0:
            is_pad = input_ids == 0
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            kwargs["mask"] = mask.tril(diagonal=0)
        else:
            if i == 1:
                start_time = time.time()
            kwargs["mask"] = None

        # get the cache data and position ids if using cache
        if sequence_ids is None:
            num_tokens_per_sequence = torch.count_nonzero(input_ids.T, dim=0).tolist()
        else:
            num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]

        cache_data = kv_cache_manager.allocate_tokens(
            num_tokens_per_sequence, sequence_ids
        )

        sequence_ids = cache_data.sequence_ids

        kwargs["cache_data"] = cache_data
        kwargs["position_ids"] = cache_data.compute_position_ids(
            num_tokens_per_sequence
        )

        if i == 0:
            logits, _ = model(input_ids, **kwargs)
        else:
            # cudagraph requires static shapes
            if cudagraphs:
                block_mapping = kwargs["cache_data"].block_mapping
                block_mapping = torch.stack(
                    [
                        F.pad(
                            block_mapping[i],
                            (0, block_mapping_max - block_mapping.size(1)),
                        )
                        for i in range(block_mapping.size(0))
                    ]
                )
                kwargs["cache_data"].block_mapping = block_mapping

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

    kv_cache_manager.free_sequences(sequence_ids)  # type: ignore
    end_time = time.time()
    return result, max_new_tokens, (end_time - start_time)