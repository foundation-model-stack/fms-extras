import pytest
from fms.models import get_model
from fms.models.hf import to_hf_api

from fms_extras.models.hf import register_fms_models


@pytest.mark.slow
def test_sphinx_equivalence():
    pytest.importorskip("megatron_models")
    # TODO: model_gpt_bigcode required a change on line 999 past_length = past_key_values[0][0].shape[2] for generate to work
    import torch
    from fms.testing.comparison import (
        HFModelSignatureParams,
        ModelSignatureParams,
        compare_model_signatures,
    )
    from megatron_models import GPTMegatronForCausalLM
    from transformers import AutoTokenizer, pipeline

    register_fms_models()

    device = "cpu"
    # path to GPTMegatronForCausalLM weights
    path = ""
    # 1b, 8b, or 13b
    variant = "1b"

    gpt_megatron_model = GPTMegatronForCausalLM.from_pretrained(path, device_map=device)
    sphinx_model = get_model(
        "sphinx", variant, path, source="megatron", device_type=device
    )

    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(gpt_megatron_model) == count_parameters(sphinx_model)

    inp = torch.arange(5, 15).unsqueeze(0)
    params_mega = HFModelSignatureParams(
        model=gpt_megatron_model, params=["input_ids"], inp=inp
    )
    params_fms = ModelSignatureParams(model=sphinx_model, params=1, inp=inp)

    compare_model_signatures(params_mega, params_fms)

    # huggingface model backed by fms internals
    sphinx_hf_model = to_hf_api(
        sphinx_model,
        pad_token_id=gpt_megatron_model.config.pad_token_id,
        bos_token_id=gpt_megatron_model.config.bos_token_id,
        eos_token_id=gpt_megatron_model.config.eos_token_id,
    )

    # generate some text -- the first time will be slow since the model needs to be compiled, but subsequent generations should be faster.
    tokenizer = AutoTokenizer.from_pretrained(path)
    sphinx_generator = pipeline(
        task="text-generation", model=sphinx_hf_model, tokenizer=tokenizer
    )
    sphinx_out = sphinx_generator(
        """q: how are you? a: I am good. How about you? q: What is the weather like today? a:""",
        max_new_tokens=25,
    )
    print(sphinx_out)

    gpt_megatron_generator = pipeline(
        task="text-generation", model=gpt_megatron_model, tokenizer=tokenizer
    )
    gpt_megatron_out = gpt_megatron_generator(
        """q: how are you? a: I am good. How about you? q: What is the weather like today? a:""",
        max_new_tokens=25,
    )
    print(gpt_megatron_out)
    assert sphinx_out == gpt_megatron_out
