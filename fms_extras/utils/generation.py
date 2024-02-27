import time
from typing import Union, Callable, List, MutableMapping, Any, Optional

import torch
import torch.nn.functional as F
from torch import distributed as dist
from fms_extras.models.speculator import MLPSpeculator
from fms_extras.utils.cache import select_inflate_dim, flatten_batch, KVCacheManager
from fms_extras.utils.cache.expandable import ExpandableKVCacheManager
from fms_extras.utils.cache.paged import PagedKVCacheManager


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    speculator: MLPSpeculator,
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    top_k: int = 5,
    threshes=[5, 3, 2],
    kv_cache_manager: PagedKVCacheManager = None,
    flatting = True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None
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
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        top_k: only score the top k candidates from the speculator
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
    result = input_ids  # [b] n
    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    n_pads = torch.Tensor(n_pads_init).to(device=input_ids[0].device, dtype=torch.int)
    inputs = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(bsize)]
    )
    num_tokens_per_sequence = torch.count_nonzero(
        inputs[:, :-1].T, dim=0
    ).tolist()
    cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence)
    parent_sequence_ids = cache_data.sequence_ids
    # Build padded causal mask
    mask = torch.ones(
        bsize,
        1,
        inputs.size(1) - 1,
        inputs.size(1) - 1,
        device=inputs.device,
    )
    mask = mask.tril()  # b 1 n-1 n-1
    # Mask off any left-pads
    pad_mask = torch.arange(mask.size(3), device=mask.device).view(
        1, 1, 1, -1
    )  # 1 1 1 n-1
    pad_mask = pad_mask.expand(bsize, 1, 1, -1)  # b 1 1 n-1
    pad_mask = pad_mask.sub(n_pads.sub(1).view(-1, 1, 1, 1)).clamp(0, 1)
    eye = torch.eye(mask.size(3), device=mask.device)[None, None, :, :]  # 1 1 n-1 n-1
    mask = mask.mul(pad_mask).logical_or(eye).log()  # b 1 n-1 n-1
    # Handle position_ids
    pos_ids = torch.arange(mask.size(3), device=inputs.device).repeat(bsize, 1)  # b n-1
    pos_ids -= n_pads[:, None]

    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = True

    # Build kv cache and get initial state vector
    n_adds = speculator.n_predict + 1
    inputs = inputs[:, -max_seq_len + n_adds :]
    position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)
    output = model(
        inputs[:, :-1],
        position_ids=position_ids,
        mask=mask,
        cache_data=cache_data,
        return_embeds=True,
        **kwargs
    )
    _, embeds = output
    embeds = embeds[:, -1:]

    n_gen = torch.zeros(bsize, device=inputs.device, dtype=torch.int)
    n_steps = 0
    input_ids = inputs[:, -1:]
    start_time = time.time()
    while min(n_gen) < new_tokens:
        n_steps += 1

        # create candidate sequences
        child_sequence_ids_list = []
        child_sequence_ids_flattened = []
        num_tokens_per_sequence = [n_adds for _ in range(input_ids.size(0) * top_k)]
        # each parent will have top_k child sequences
        for parent_sequence_id in parent_sequence_ids:
            child_sequence_ids = kv_cache_manager.add_child_sequences(parent_sequence_id, top_k)
            child_sequence_ids_list.append(child_sequence_ids)
            child_sequence_ids_flattened.extend(child_sequence_ids)

        # add n_adds tokens to each candidate
        cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence, child_sequence_ids_flattened)
        position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)

        # Get candidate set of speculations
        adds = speculator.generate_suffixes(embeds, input_ids, threshes, top_k)  # b k h
        input_ids = torch.cat(
            [input_ids.unsqueeze(1).expand(bsize, top_k, 1), adds], dim=-1
        ).int()  # b k 1+h

        this_flatting = False
        if flatting:
            flat_inputs, unflat_indices, flat_indices = flatten_batch(input_ids) # b', b k 1+h, b'
            compression = flat_inputs.numel() / input_ids.numel()
            if compression < .75:
                this_flatting = True
                flat_inputs = flat_inputs[None,] # 1 b'
                cache_data.unflatten_indices = unflat_indices
                cache_data.flatten_indices = flat_indices
                position_ids = select_inflate_dim(position_ids.view(-1), flat_indices)[None,]
        input_ids = input_ids.view(-1, n_adds)  # bk 1+h

        context_lengths = cache_data.context_lengths  # bk
        inflate_factor = (
            cache_data.query_length
            if cache_data.unflatten_indices is None
            else cache_data.unflatten_indices.size(-1)
        )
        context_lengths = context_lengths.unsqueeze(1).expand(
            -1, inflate_factor
        )  # bk n
        context_lengths = (
            context_lengths.sub(context_lengths.sign().cumsum(1).flip([1]).sub(1))
            .int()
            .view(-1)
        )  # bkn
        block_mappings = cache_data.block_mapping.repeat_interleave(
            inflate_factor, dim=0
        )  # bkn n_blocks
        if cache_data.flatten_indices is not None:
            context_lengths = select_inflate_dim(
                context_lengths, cache_data.flatten_indices
            )  # n'
            block_mappings = select_inflate_dim(
                block_mappings, cache_data.flatten_indices
            )  # n' n_blocks

        # pad for cudagraphs
        cache_data.block_mapping = block_mappings
        cache_data.context_lengths = context_lengths

        input_ids_unflat = input_ids
        if this_flatting:
            input_ids = flat_inputs

        # Base model forward pass
        logits, embeds = decode_model(
            input_ids, position_ids=position_ids, cache_data=cache_data, return_embeds=True, **kwargs
        ) # 1 n' v, 1 n' d
        next_vals = torch.argmax(logits, dim=-1)  # 1 n'

        if this_flatting:
            unflat_indices = unflat_indices.view(-1, unflat_indices.size(2))
            next_vals = select_inflate_dim(next_vals[0], unflat_indices) # bk 1+h
            embeds = select_inflate_dim(embeds[0], unflat_indices) # bk 1+h d
            # TODO: make more efficient by best guessing out of unflat indices rather than from here directly
        else:
            next_vals = next_vals.view(-1, n_adds)
            embeds = embeds.view(next_vals.size(0), n_adds, -1)

        # Check correctness of speculator predictions
        test = input_ids_unflat.roll(-1, 1).eq(next_vals).cumprod(1)
        n_correct = (
            test.sum(1).clamp(0, n_adds - 1).view(bsize, top_k)
        )  # clamp in case pred[0]==targ[-1]
        best_guess = n_correct.argmax(1)  # b
        best_guess_unflat = (
            best_guess.unsqueeze(1).expand(bsize, n_adds).unsqueeze(1)
        )  # b 1 1+h

        # Set global values to those of best guess
        next_vals = next_vals.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)  # b 1+h
        n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
        embeds = embeds.view(bsize, top_k, *embeds.size()[1:]).gather(
            1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
        ).squeeze(1)  # b 1+h d

        # free all worst candidates and keep best candidates as parents
        parent_sequence_ids = []
        for parent_index, child_sequence_ids in enumerate(child_sequence_ids_list):
            best_index = best_guess[parent_index].item()

            # free all bad candidates
            kv_cache_manager.free_sequences(child_sequence_ids[:best_index] + child_sequence_ids[best_index + 1:])

            # decrease the context length of the sequence which used to be sequence length + n_adds by the number of incorrect tokens
            # for the correct candidate
            best_sequence_id = child_sequence_ids[best_index]
            parent_sequence_ids.append(best_sequence_id)
            kv_cache_manager.remove_tokens(best_sequence_id, n_adds - n_correct[parent_index].item() - 1)

        # Toss any wrong speculator tokens
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
    kv_cache_manager: Optional[PagedKVCacheManager],
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement batching nor beam search, but those could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        prefix: A tensor of token IDs.
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
    """
    if decode_model is None:
        decode_model = model

    bsize = len(input_ids)
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
            num_tokens_per_sequence = torch.count_nonzero(
                input_ids.T, dim=0
            ).tolist()
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
            logits = model(input_ids, **kwargs)
        else:
            logits = decode_model(input_ids, **kwargs)
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
