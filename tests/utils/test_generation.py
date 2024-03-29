import random
import re
import tempfile
from typing import List

import pytest
import torch
from fms.models import get_model
from fms.utils import serialization
from fms.utils.generation import generate

from fms_extras.models.speculator import MLPSpeculator


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="must have cuda to run paged llama generation test",
)
def test_paged_generate():
    from fms_extras.models import paged_llama
    from fms_extras.utils.cache.paged import PagedKVCacheManager
    from fms_extras.utils.generation import paged_generate

    torch.set_grad_enabled(False)

    llama = get_model("llama", "micro", device_type="cuda", nheads=2)

    with tempfile.TemporaryDirectory() as workdir:
        sd_path = f"{workdir}/model.pth"
        torch.save(llama.state_dict(), sd_path)

        paged_llama = get_model(
            "paged_llama",
            "micro",
            model_path=sd_path,
            source="fms_llama",
            device_type="cuda",
            nheads=2,
        )

    kv_cache_manager = PagedKVCacheManager(
        paged_llama.config.nlayers,
        paged_llama.config.nheads,
        paged_llama.config.emb_dim,
        kv_heads=paged_llama.config.kvheads,
        dtype=torch.get_default_dtype(),
        total_num_gpu_blocks=100,
    )

    input_ids = torch.tensor(
        [1] + [i for i in range(5, 25)], dtype=torch.long, device="cuda"
    )

    paged_result, _, _, _ = paged_generate(
        paged_llama, [input_ids], kv_cache_manager, do_sample=False
    )

    result = generate(llama, input_ids.unsqueeze(0), do_sample=False)

    torch.testing.assert_close(paged_result, result)


class MockSpeculator(MLPSpeculator):
    def __init__(self, candidates_per_step: List[List[List[List[int]]]]):
        # candidates_per_step: decode_steps x batch x num_candidates x num_predictions
        super().__init__()
        self.n_predict = len(candidates_per_step[0][0][0])
        self.step = 0
        self.guesses = candidates_per_step

    def generate_suffixes(
        self,
        state: torch.Tensor,
        ind: torch.Tensor,
        topk: List[int] = [5, 4, 3],
        n: int = 5,
    ) -> torch.Tensor:
        guess = self.guesses[self.step]
        self.step += 1
        return torch.tensor(guess, device="cuda").int()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="must have cuda to run paged llama generation test",
)
def test_speculative_generate():
    from fms_extras.models import paged_llama
    from fms_extras.utils.cache.paged import PagedKVCacheManager
    from fms_extras.utils.generation import paged_generate, speculative_generate

    torch.set_grad_enabled(False)

    paged_llama = get_model(
        "paged_llama",
        "micro",
        device_type="cuda",
        nheads=2,
    )

    kv_cache_manager = PagedKVCacheManager(
        paged_llama.config.nlayers,
        paged_llama.config.nheads,
        paged_llama.config.emb_dim,
        kv_heads=paged_llama.config.kvheads,
        dtype=torch.get_default_dtype(),
        total_num_gpu_blocks=100,
    )

    input_ids1 = torch.tensor(
        [1] + [i for i in range(5, 25)], dtype=torch.long, device="cuda"
    )

    input_ids2 = torch.tensor(
        [1] + [i for i in range(30, 35)], dtype=torch.long, device="cuda"
    )
    input_ids_list = [input_ids1, input_ids2]
    max_prompt = max([input_ids.size(0) for input_ids in input_ids_list])
    max_new_tokens = 20

    paged_result, paged_n_steps, _, _ = paged_generate(
        paged_llama,
        input_ids_list,
        kv_cache_manager,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    # running tests for different prediction lengths and number of candidates
    n_predict_list = [2, 3, 4]
    num_candidates_list = [1, 3, 5]

    for n_predict in n_predict_list:
        for num_candidates in num_candidates_list:
            # randomly generate the correct number of guesses per step to mock the speculator
            candidates_per_step = []
            # starting needle at 1 since first token is free
            needles = [1, 1]
            # need to continue til both sequences have completed max_new_tokens
            while any(needle <= max_new_tokens for needle in needles):
                candidates = []

                # get candidates for each sequence
                for i, result_i in enumerate(paged_result):
                    # offsetting by max_prompt to only include generated tokens
                    tokens = result_i.tolist()[max_prompt:]
                    candidates_per_sequence = []
                    max_num_correct = -1
                    # adding a max so we reduce chance of all correct
                    n_correct_max = random.randint(0, n_predict)

                    # get each candidate of variable correctness
                    for _ in range(num_candidates):
                        n_correct = random.randint(0, n_correct_max)
                        candidate = tokens[needles[i] : needles[i] + n_predict]

                        # inject a wrong token if needed
                        if n_correct < len(candidate):
                            candidate[n_correct] = (
                                candidate[n_correct] - 1
                            ) % paged_llama.config.src_vocab_size

                        # pad if not enough ground truth tokens left
                        if len(candidate) < n_predict:
                            candidate = candidate + ([0] * (n_predict - len(candidate)))

                        candidates_per_sequence.append(candidate)
                        max_num_correct = max(max_num_correct, n_correct)
                    candidates.append(candidates_per_sequence)

                    # +1 for one free token
                    needles[i] += max_num_correct + 1
                candidates_per_step.append(candidates)

            speculator = MockSpeculator(candidates_per_step)
            speculative_result, speculative_n_steps, _, _ = speculative_generate(
                paged_llama,
                input_ids_list,
                speculator,
                kv_cache_manager,
                new_tokens=max_new_tokens,
                n_candidates=num_candidates,
            )

            # test that we actually were able to perform speculative decoding in the correct number of steps
            assert speculative_n_steps == len(candidates_per_step)

            # test for correctness of output
            for paged_single, speculative_single, prompt_ids in zip(
                paged_result, speculative_result, input_ids_list
            ):
                paged_single = paged_single[max_prompt:]
                num_pads = max_prompt - prompt_ids.size(0)
                speculative_single = speculative_single[
                    max_prompt - num_pads : max_prompt - num_pads + paged_single.size(0)
                ]
                torch.testing.assert_close(paged_single, speculative_single)
