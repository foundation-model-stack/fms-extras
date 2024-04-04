import tempfile

import torch

from fms_extras.models.hf.modeling_mlp_speculator import (
    MLPSpeculatorConfig,
    MLPSpeculatorPreTrainedModel,
    load_from_fms_weights,
)
from fms_extras.models.speculator import MLPSpeculator


def __test_speculator_equivalence(
    speculator_1, speculator_2, emb_dim, top_k_tokens_per_head, n_candidates
):
    sd1 = speculator_1.state_dict()
    sd2 = speculator_2.state_dict()
    # make sure state dicts are same
    assert len(sd1) == len(sd2)
    for k in sd1.keys():
        torch.testing.assert_close(sd1[k], sd2[k])

    # make sure generate_suffixes produce same output
    state = torch.rand(4, 1, emb_dim)
    ind = torch.randint(low=0, high=10, size=(4, 1))

    speculator1_out = speculator_1.generate_suffixes(
        state, ind, top_k_tokens_per_head, n_candidates
    )
    speculator2_out = speculator_2.generate_suffixes(
        state, ind, top_k_tokens_per_head, n_candidates
    )
    torch.testing.assert_close(speculator1_out, speculator2_out)


def test_load_from_fms_weights():
    vocab_size = 256
    emb_dim = 64
    inner_dim = 32
    n_predict = 4
    speculator = MLPSpeculator(
        emb_dim=emb_dim, vocab_size=vocab_size, inner_dim=inner_dim, n_predict=n_predict
    )
    speculator.reset_parameters()
    speculator.eval()
    original_sd = speculator.state_dict()

    top_k_tokens_per_head = [5, 3, 2, 2]
    n_candidates = 5
    hf_speculator = load_from_fms_weights(
        original_sd, n_predict, top_k_tokens_per_head, n_candidates, emb_dim
    )
    hf_speculator.eval()

    __test_speculator_equivalence(
        speculator, hf_speculator, emb_dim, top_k_tokens_per_head, n_candidates
    )


def test_round_trip():
    vocab_size = 256
    emb_dim = 64
    inner_dim = 32
    n_predict = 4
    top_k_tokens_per_head = [5, 3, 2, 2]
    n_candidates = 5
    config = MLPSpeculatorConfig(
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        inner_dim=inner_dim,
        n_predict=n_predict,
        top_k_tokens_per_head=top_k_tokens_per_head,
        n_candidates=n_candidates,
    )
    original_model = MLPSpeculatorPreTrainedModel(config)
    original_model.reset_parameters()
    original_model.eval()

    with tempfile.TemporaryDirectory() as workdir:
        hf_path = f"{workdir}/hf_speculator_out.pth"
        original_model.save_pretrained(hf_path)

        loaded_model = MLPSpeculatorPreTrainedModel.from_pretrained(hf_path)
        loaded_model.eval()

    __test_speculator_equivalence(
        original_model, loaded_model, emb_dim, top_k_tokens_per_head, n_candidates
    )
