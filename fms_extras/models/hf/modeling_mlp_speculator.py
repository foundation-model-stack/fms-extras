import os
from typing import List, OrderedDict

import torch
from transformers import PretrainedConfig, PreTrainedModel

from fms_extras.models.speculator import MLPSpeculator


def load_from_fms_weights(
    weights: OrderedDict,
    n_predict: int,
    top_k_tokens_per_head: List[int],
    n_candidates: int,
    emb_dim: int = 0,
) -> "MLPSpeculatorPreTrainedModel":
    """
    Convenience function for loading from fms MLPSpeculator

    Args:
        weights: OrderedDict
            the weights
        n_predict: int
            number of lookaheads for speculator
        top_k_tokens_per_head: List[int]
            Number of tokens to consider from each head when forming the candidate tree.
            For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
        n_candidates: int
            number of child candidates to create per sequence
        emb_dim: int
            if 0 will set emb_dim to inner_dim, otherwise emb_dim and inner_dim will differ in model

    Returns:
        MLPSpeculatorPreTrainedModel
            a huggingface MLPSpeculator
    """

    vocab_size = weights["emb.0.weight"].size(0)
    inner_dim = weights["emb.0.weight"].size(1)
    if emb_dim == 0:
        emb_dim = inner_dim

    config = MLPSpeculatorConfig(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        inner_dim=inner_dim,
        n_predict=n_predict,
        top_k_tokens_per_head=top_k_tokens_per_head,
        n_candidates=n_candidates,
    )

    hf_mlp_speculator = MLPSpeculatorPreTrainedModel(config)
    hf_mlp_speculator.load_state_dict(weights)
    return hf_mlp_speculator


class MLPSpeculatorConfig(PretrainedConfig):
    model_type = "mlp_speculator"

    attribute_map = {
        "hidden_size": "emb_dim",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: List[int] = [5, 4, 3],
        n_candidates: int = 5,
        **kwargs
    ):
        """
        Initialize an MLPSpeculatorConfig

        Args:
            vocab_size: int
                the model vocab size
            emb_dim: int
                the model embedding dimension
            inner_dim: int
                the inner dimension of the model. If 0, will be the emb_dim.
            n_predict: int
                the number of lookaheads for the speculator
            top_k_tokens_per_head: List[int]
                Number of tokens to consider from each head when forming the candidate tree.
                For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
            n_candidates: int
                number of child candidates to create per sequence
        """
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        super().__init__(**kwargs)


class MLPSpeculatorPreTrainedModel(MLPSpeculator, PreTrainedModel):
    """
    Hugginface MLPSpeculator which provides loading/saving in huggingface
    """

    config_class = MLPSpeculatorConfig

    def __init__(self, config: MLPSpeculatorConfig):
        super().__init__(
            config=config,
            emb_dim=config.emb_dim,
            inner_dim=config.inner_dim,
            vocab_size=config.vocab_size,
            n_predict=config.n_predict,
        )
