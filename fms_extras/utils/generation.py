import functools
import time
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms_extras.models.speculator import MLPSpeculator, apply_index_map, flatten_batch
from fms_extras.utils.cache.paged import PagedAttentionCacheData, PagedKVCacheManager


def __execute_prefill(
    model: Union[nn.Module, Callable],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    cache_data: PagedAttentionCacheData,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute the prefill and return information needed for first prediction.

    By prefill, we are referring to the initial forward pass on the prompt tokens prior
    to any computed keys/values added to the cache

    Args:
        model: Union[nn.Module, Callable]
            A function or nn.Module that takes a batch of input_ids and returns logits
        input_ids: Union[torch.Tensor, List[torch.Tensor]]
            list of prompts where each tensor in the list corresponds to a single prompt
            in the batch
        cache_data: PagedAttentionCacheData
            the paged attention kv-cache data for prefill
        max_seq_len: int
            max length that base model can handle accounting for upcoming speculated
            tokens

    Returns:
    Tuple[torch.Tensor, torch.Tensor]
        the output logits and embedding vector for the predicted token at end of prompt
    """

    # get the last tokens if past max length in model
    input_ids = [seq[-max_seq_len:] for seq in input_ids]
    model_input_lengths = [seq.size(0) for seq in input_ids]
    max_len = max(model_input_lengths)

    # pad the inputs
    inputs = torch.stack(
        [
            F.pad(input_ids[i], (max_len - model_input_lengths[i], 0))
            for i in range(len(input_ids))
        ]
    )

    # Build padded causal mask
    mask = __create_prefill_mask(model_input_lengths, inputs.device)

    logits, _, embeds = model(
        inputs,
        mask=mask,
        cache_data=cache_data,
        return_embeds=True,
        use_cache=True,
    )
    embeds = embeds[:, -1:]  # b 1 d
    logits = logits[:, -1:]  # b 1 v

    return logits, embeds


def __prepare_candidate_sequences_cache_data(
    kv_cache_manager: PagedKVCacheManager,
    parent_sequence_ids: List[int],
    model_input_lengths: List[int],
    num_candidates_per_sequence: int,
) -> Tuple[PagedAttentionCacheData, List[List[int]]]:
    """
    Speculative generate produces suffix candidates for each sequence. This method
    allocates these candidates in the kv-cache as child sequences (sequences
    referencing their parents as a prefix optimized for better memory efficiency
    with less duplication)

    Args:
        kv_cache_manager: PagedKVCacheManager
            the paged attention kv-cache manager
        parent_sequence_ids: List[int]
            the parent sequence ids associated with each prompt in the batch
        model_input_lengths: List[int]
            list where each value corresponds the number of tokens in the prompt at
            that index
        num_candidates_per_sequence: int
            the number of child candidates to create for each sequence

    Returns:
    Tuple[PagedAttentionCacheData, List[List[int]]]
        the cache data created after allocation, and the list of child sequences
        per parent
    """
    child_sequence_ids_list = []
    child_sequence_ids_flattened = []
    # each parent will have n_candidates child sequences
    for parent_sequence_id in parent_sequence_ids:
        child_sequence_ids = kv_cache_manager.add_child_sequences(
            parent_sequence_id, num_candidates_per_sequence
        )
        child_sequence_ids_list.append(child_sequence_ids)
        child_sequence_ids_flattened.extend(child_sequence_ids)

    # add inp_len tokens to each candidate
    cache_data = kv_cache_manager.allocate_tokens(
        model_input_lengths, child_sequence_ids_flattened
    )

    # Set up kv cache metadata for paged memory access over tokens/candidates with shared prefixes
    cache_data.apply_batch_expansion()

    return cache_data, child_sequence_ids_list


def __maybe_flatten(
    input_ids: torch.Tensor,
    cache_data: PagedAttentionCacheData,
    flattening: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove redundant tokens if compression ratio of data is below 0.75

    Args:
        input_ids: torch.Tensor
            the input ids for a given decode step
        cache_data: PagedAttentionCacheData
            the paged-attention cache data
        flattening: bool
            denotes whether we want to attempt flattening

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            a tensor of the input_ids to be sent to decode model, an optional tensor
            of the indices for performing the un-flattening operation if redundant
            tokens are being removed, an optional tensor of indices for performing
            the flattening operation if redundant tokens are being removed
    """
    if not flattening:
        return input_ids.view(-1, input_ids.size(2)), None, None

    # flatten the batch
    flat_inputs, unflat_indices, flat_indices = flatten_batch(
        input_ids
    )  # n', b k 1+h, n'

    # check if high enough level of compression to perform batch flattening
    compression = flat_inputs.numel() / input_ids.numel()
    perform_batch_flattening = compression < 0.75

    # set the correct input_ids to return if compression ratio satisfied
    if perform_batch_flattening:
        result_input_ids = flat_inputs[None,]  # 1 n'
        cache_data.apply_batch_flattening(unflat_indices, flat_indices)
    else:
        result_input_ids = input_ids.view(-1, input_ids.size(2))
        unflat_indices = None  # type: ignore
        flat_indices = None  # type: ignore

    return result_input_ids, unflat_indices, flat_indices


def __free_incorrect_tokens_and_sequences(
    kv_cache_manager: PagedKVCacheManager,
    child_sequence_ids_list: List[List[int]],
    best_guess: torch.Tensor,
    n_correct: torch.Tensor,
    decode_seq_length: int,
) -> List[int]:
    """
    Free all non-best candidate sequences from the kv-cache and for best candidate
    sequences, logically remove tokens based on the number of incorrect tokens
    generated

    Args:
        kv_cache_manager: PagedKVCacheManager
            the paged attention kv-cache manager
        child_sequence_ids_list: List[List[int]]
            the list of child sequences per parent
        best_guess: torch.Tensor
            a tensor containing the best candidate per sequence in the batch
        n_correct: torch.Tensor
            a tensor containing the number of correct tokens best candidate in the
            batch
        decode_seq_length: int
            the model input length at decode step

    Returns:
    List[int]
        a list of the new parent sequence ids for next decode step
    """
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
            best_sequence_id, decode_seq_length - n_correct[parent_index].item() - 1  # type: ignore
        )
    return parent_sequence_ids


def __get_best_candidates(
    input_ids: torch.Tensor,
    next_vals: torch.Tensor,
    embeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the candidates with the best speculator predictions and get the indices of the
    best candidates, the number of tokens correct, and the base model output values
    for that candidate (tokens and embeddings)

    Args:
        input_ids: torch.Tensor
            the original input ids unflattened
        next_vals: torch.Tensor
            a tensor of the next tokens
        embeds: torch.Tensor
            the output embeddings for the best guesses

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        a tensor of the next tokens per best guess (b x 1+h), a tensor of the embeds
        per best guess (b x 1+h x d), a tensor of the number of correct tokens per
        best guess (b), and a tensor containing the best guess amongst all candidates
        per sequence (b)
    """
    batch_size, num_candidates_per_sequence, decode_seq_length = input_ids.shape

    # Check correctness of speculator predictions
    test = input_ids[:, :, 1:].eq(next_vals[:, :, :-1]).cumprod(2)
    n_correct = test.sum(2).view(batch_size, num_candidates_per_sequence)
    best_guess = n_correct.argmax(1)  # b
    best_guess_unflat = (
        best_guess.unsqueeze(1).expand(batch_size, decode_seq_length).unsqueeze(1)
    )  # b 1 1+h

    # Set global values to those of best guess
    next_vals = next_vals.gather(1, best_guess_unflat).squeeze(1)  # b 1+h
    n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
    embeds = embeds.gather(
        1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(3))
    ).squeeze(
        1
    )  # b 1+h d
    return next_vals, embeds, n_correct, best_guess


def __get_correct_tokens(
    next_vals: torch.Tensor, n_correct: torch.Tensor, embeds: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    extract the correct tokens and the last correct embedding from each candidate, to
    be used to start the next set of speculative candidates

    Args:
        next_vals: torch.Tensor
            a tensor of the next tokens per best guess
        n_correct: torch.Tensor
            a tensor of the number of correct tokens per best guess
        embeds: torch.Tensor
            a tensor of the embeds per best guess

    Returns:
    Tuple[List[torch.Tensor], torch.Tensor]
        a list of tensor of the correct tokens, and a tensor of the correct embeddings
    """
    next_vals_split = list(next_vals)
    next_vals_split = [
        next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
    ]  # [b] h'
    embeds = embeds.gather(
        1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
    )  # Grab last correct embed
    return next_vals_split, embeds


def __prune_candidates(
    input_ids: torch.Tensor,
    next_vals: torch.Tensor,
    embeds: torch.Tensor,
    kv_cache_manager: PagedKVCacheManager,
    child_sequence_ids_list: List[List[int]],
) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    """
    Finds the correct set of candidates as well and get their respective next tokens
    and embeds

    Args:
        input_ids: torch.Tensor
            the original input ids unflattened
        next_vals: torch.Tensor
            a tensor of the next tokens
        embeds: torch.Tensor
            the output embeddings for the best guesses
        kv_cache_manager: PagedKVCacheManager
            the paged kv-cache manager
        child_sequence_ids_list: List[List[int]]
            the list of child sequences per parent

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, List[int]]
            a list of tensor of the correct tokens per sequence, and a tensor of the
            correct embeddings, and a list of the best candidate id per sequence
    """
    # get the best candidates
    next_vals, embeds, n_correct, best_guess = __get_best_candidates(
        input_ids, next_vals, embeds
    )

    # free all non-best candidates and keep best candidates as parents
    parent_sequence_ids = __free_incorrect_tokens_and_sequences(
        kv_cache_manager,
        child_sequence_ids_list,
        best_guess,
        n_correct,
        input_ids.size(2),
    )

    # Remove any wrong speculator tokens from best candidate
    next_vals_split, embeds = __get_correct_tokens(next_vals, n_correct, embeds)
    return next_vals_split, embeds, parent_sequence_ids


def __extract_decode_output(
    model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unflat_indices: Optional[torch.Tensor],
    batch_size: int,
    n_candidates: int,
    decode_seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the unflattened next tokens and embeds from the model outputs

    Args:
        model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the output from the model (logits, cache tensor, embeds)
        unflat_indices: torch.Tensor, optional
            the indices used for unflattening if flattening was applied
        batch_size: int
            the original batch size
        n_candidates: int
            the number of candidates used
        decode_seq_length: int
            the model input length at decode step

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
            the un-flattened logit scores per token per candidate per sequence,
            and the un-flattened output embedding vectors
    """
    logits, _, embeds = model_output  # 1 n' v, 1 n' d  OR  bk 1+h v, bk 1+h d

    # If we used batch flattening / tree attention, unflatten the outputs
    if unflat_indices is not None:
        logits = apply_index_map(logits[0], unflat_indices)  # b k 1+h v
        embeds = apply_index_map(embeds[0], unflat_indices)  # b k 1+h d
    else:
        logits = logits.view(
            batch_size, n_candidates, decode_seq_length, logits.size(2)
        )  # b k 1+h v
        embeds = embeds.view(
            batch_size, n_candidates, decode_seq_length, embeds.size(2)
        )  # b k 1+h d
    return logits, embeds


def __generate_targets(
    logits: torch.Tensor,
    do_sample: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 5,
) -> torch.Tensor:
    """
    Extracts ground-truth tokens from a set of logits. If performing greedy decoding,
    simply returns the most confident tokens. Otherwise, implements consistent multinomial
    sampling - two identical distributions will always produce the same (randomized) sample.
    Thus by induction, two candidates with identical prefixes will receive the same ground
    truth sample up to the point their inputs diverge. This allows us to ensure that at least
    one candidate will be accepted, so long as the candidate set covers the top_k options.

    For example, if the base model predicts tokens A and B with equal 50% probability, and the
    speculator produces one candidate with A and another with B, with independent sampling there's
    a 25% chance of rejecting both, even though one must be correct. Consistent sampling allows us
    to avoid this.

    Args:
        logits: torch.Tensor
            Probability logits for a set of candidate sequences. Expects size
            bsize x n_candidates x seq_len x vocab_size
        do_sample: torch.Tensor
            A tensor of booleans enabling/disabling non-greedy decoding with consistent
            sampling, for each of bsize input sequences
        temperature: float
            Degree of smoothing on softmax sampling distribution
        top_k: int
            Sample only among the top_k most confident tokens

    Returns:
        torch.Tensor
            Tensor of chosen token values for each sequence
    """

    # Get sample distributions
    logits = logits / temperature
    v, _ = logits.topk(top_k)
    logits[logits < v[:, :, :, [-1]]] = -float("inf")
    probs = logits.softmax(-1)  # b k 1+h v

    # Sample candidate-consistent ground truths: partition number line in [0,1]
    # according to given multinomial distribution. Pick a random location
    # on that line, return interval containing that location.
    key = torch.rand(1, 1, logits.size(2), 1, device=probs.device)
    a = (
        probs.cumsum(3).sub(key).sign()
    )  # Sign flips on probability interval containing key
    samples = a.sub(1).div(-2).sum(3)  # Get index of sign-flip

    # Composite greedy and non greedy outputs
    greedy = logits.argmax(-1)
    mask = do_sample[:, None, None].int()
    return samples * mask + (1 - mask) * greedy


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    speculator: MLPSpeculator,
    kv_cache_manager: PagedKVCacheManager,
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    n_candidates: int = 5,
    threshes=[5, 3, 2],
    flattening: bool = True,
    decode_model: Optional[Union[Callable, torch.nn.Module]] = None,
    # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
    cudagraphs: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 5,
):
    """
    A reference implementation of speculative decoding generation.
    Returns at least the specified number of tokens - the speculator may return a
    few extra in the final step.
    If input is batched, continues generating until EVERY sequence has produced
    AT LEAST the required number of tokens. Input (and output) tokens beyond
    max_seq_len are simply dropped for a sliding-window approach. Currently
    reproduces behavior of greedy decoding only.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: A length n tensor of token IDs, or list of such tensors
        speculator: A function or nn.Module that takes a state vector and
            sampled token and returns a set of candidate suffixes
        kv_cache_manager: PagedKVCacheManager
            the paged kv-cache manager to be used in generation
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        n_candidates: only consider the top n most confident candidates
            from the speculator
        threshes: build candidate suffix trees by taking top-(threshes[n]) most
            confident values for head n. len(threshes) must equal speculator.n_predict.
            prod(threshes) must be greater than or equal to n_candidates (we cannot
            have more candidates than tree leaves).
        flattening: enable batch flattening / tree attention with redundant prefix
            removal when compression ratio is favorable. Adds extra overhead to
            shrink the token count in each batch.
        decode_model: nn.Module, optional
            an optional model that performs the decode forward step. If None,
            will use the function or nn.Module provided to model param to perform
            decode forward. This parameter is intended to be used when the compile
            flags for the model doing prefill and the model generating subsequent
            tokens are different (default is None). When we refer to prefill,
            this is typically the first model pass with the prompt, where the cache
            has not been filled with any tokens yet. When we refer to decode step,
            this is typically the forward pass where the cache has been filled with
            the prompt and is now generating new tokens.
        cudagraphs: bool
            if True, cudagraphs is used and all metadata will be padded, otherwise
            metadata will not be padded unless required. Note: This is a WIP and
            only works for batch_size=1
        do_sample: bool
            non-deterministic, multinomial output sampling. False for greedy.
            Provides output diversity, but lowers speculative decoding speedup.
        temperature: float
            temperature of softmax when sampling. Lowering this should provide
            better speculative decoding speedup when do_sample=True.
        top_k: int
            only search among top k tokens. Lowering this should provide
            better speculative decoding speedup when do_sample=True.
    Returns:
        result: List of id tensors, possibly different lengths if batching.
        n_steps: Number of foward passes used to generate provided tokens.
    """
    start_time = time.time()
    # Construct batch(es) and initial inputs
    if decode_model is None:
        decode_model = model
    bsize = len(input_ids)

    if cudagraphs and (flattening or bsize != 1):
        raise NotImplementedError(
            "cudagraphs is not yet supported for batch sizes greater than 1 or flatting"
        )

    result = input_ids  # [b] n
    inp_len = speculator.n_predict + 1
    prompt_lengths = [min(seq.size(0), max_seq_len) for seq in input_ids]

    # reserve blocks in the cache for the prompts for prefill
    cache_data: PagedAttentionCacheData = kv_cache_manager.allocate_tokens(
        prompt_lengths
    )

    # execute the prefill step
    # logits -  b 1 d
    # embeds - b 1 v
    logits, embeds = __execute_prefill(
        model, input_ids, cache_data, max_seq_len - inp_len
    )

    # get the time to first token
    ttft = time.time() - start_time

    # setting local variables for re-use prior to decode loop
    start_time = time.time()
    parent_sequence_ids = cache_data.sequence_ids
    n_gen = [0] * bsize
    n_steps = 0
    input_ids = torch.argmax(logits, dim=2)  # b 1
    result = [
        torch.cat((line, input_id), dim=0)
        for line, input_id in zip(result, list(input_ids))
    ]
    # block mapping max is the max blocks required for the longest prompt in the batch
    # this is used in padding the block mapping when using cudagraphs because cudagraphs required static shapes
    block_mapping_max = ((max(prompt_lengths) + new_tokens - 1) // 16) + 1
    model_input_lengths = [inp_len] * (input_ids.size(0) * n_candidates)

    # perform decode step
    while min(n_gen) < new_tokens:
        n_steps += 1

        # allocate candidate sequences in cache and get the sequence ids
        cache_data, child_sequence_ids_list = __prepare_candidate_sequences_cache_data(
            kv_cache_manager, parent_sequence_ids, model_input_lengths, n_candidates
        )

        # Get candidate set of speculations
        suffix_ids = speculator.generate_suffixes(
            embeds, input_ids, threshes, n_candidates
        )  # b k h
        # attach speculator output to original token inputs
        input_ids = torch.cat(
            [input_ids.unsqueeze(1).expand(bsize, n_candidates, 1), suffix_ids], dim=-1
        ).int()  # b k 1+h

        # Apply batch flattening / tree attention if compression is good enough
        model_input_ids, unflat_indices, flat_indices = __maybe_flatten(
            input_ids, cache_data, flattening
        )

        # todo: This is a WIP to enable cudagraphs, currently its only for batch_size=1
        # cudagraph requires static shapes
        if cudagraphs:
            cache_data.apply_right_padding_to_block_mapping(block_mapping_max)

        # Base model forward pass
        output = decode_model(
            model_input_ids,
            cache_data=cache_data,
            return_embeds=True,
            use_cache=True,
        )  # 1 n' v  OR  bk 1+h v

        logits, embeds = __extract_decode_output(
            output, unflat_indices, bsize, n_candidates, inp_len
        )

        if do_sample:
            do_sample_vector = torch.ones(bsize, device=logits.device)
        else:
            do_sample_vector = torch.zeros(bsize, device=logits.device)
        next_vals = __generate_targets(
            logits, do_sample_vector, temperature=temperature, top_k=top_k
        )

        next_vals_list, embeds, parent_sequence_ids = __prune_candidates(
            input_ids, next_vals, embeds, kv_cache_manager, child_sequence_ids_list
        )

        # Update results
        result = [torch.cat(x, dim=0) for x in zip(result, next_vals_list)]
        input_ids = torch.stack([line[-1:] for line in next_vals_list], dim=0)  # b 1

        # update number of generated tokens per sequence
        n_gen = [
            n_gen_i + next_vals_i.size(0)
            for n_gen_i, next_vals_i in zip(n_gen, next_vals_list)
        ]

    # free final parent sequences from the kv-cache
    kv_cache_manager.free_sequences(parent_sequence_ids, recursive=True)

    return result, n_steps, ttft, (time.time() - start_time)


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
                kwargs["cache_data"].apply_right_padding_to_block_mapping(
                    block_mapping_max
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
