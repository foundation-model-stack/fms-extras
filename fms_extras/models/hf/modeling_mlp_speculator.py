from typing import List, Optional

import torch
from fms.models.hf import _fms_to_hf_adapt_map
from transformers import PretrainedConfig, PreTrainedModel

from fms_extras.models.speculator import MLPSpeculator


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
        tie_wts: bool = False,
        scale_input: bool = False,
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
        self.tie_wts = tie_wts
        self.scale_input = scale_input
        super().__init__(**kwargs)


class MLPSpeculatorPreTrainedModel(PreTrainedModel):
    """
    Huggingface MLPSpeculator which provides loading/saving in huggingface
    """

    config_class = MLPSpeculatorConfig

    def __init__(
        self, config: MLPSpeculatorConfig, speculator: Optional[MLPSpeculator] = None
    ):
        super().__init__(
            config=config,
            emb_dim=config.emb_dim,
            inner_dim=config.inner_dim,
            vocab_size=config.vocab_size,
            n_predict=config.n_predict,
            tie_wts=config.tie_wts,
            scale_input=config.scale_input,
        )
        if speculator is None:
            self.speculator = MLPSpeculator(
                config.emb_dim, config.inner_dim, config.vocab_size, config.n_predict, tie_wts=config.tie_wts, scale_input=config.scale_input
            )
            self.speculator.reset_parameters()
        else:
            self.speculator = speculator

    @classmethod
    def from_fms_model(
        cls,
        model: MLPSpeculator,
        top_k_tokens_per_head: List[int],
        n_candidates: int,
        tie_wts: bool = False,
        scale_input: bool = False,
        *args,
        **kwargs
    ):
        config = MLPSpeculatorConfig(
            vocab_size=model.vsize,
            emb_dim=model.emb_dim,
            inner_dim=model.inner_dim,
            n_predict=model.n_predict,
            top_k_tokens_per_head=top_k_tokens_per_head,
            n_candidates=n_candidates,
            tie_wts=tie_wts,
            scale_input=scale_input,
        )
        return cls(config, model)

    def generate_suffixes(
        self,
        state: torch.Tensor,
        ind: torch.Tensor,
        topk: List[int] = [5, 4, 3],
        n: int = 5,
    ) -> torch.Tensor:
        """
        FOR INFERENCE
        Generate tree of candidate sequences.
        ...
        Args
        ----
        state : torch.Tensor
            Most recent embedding vector from the base model (pre-classification head).
            Expects size [b 1 d] where b is batch size and d is model width.
        ind : torch.Tensor
            Token indices of the base model's most recent predicted token(s).
            Expects size [b 1] where b is batch size.
        topk : List(int)
            Number of tokens to consider from each head when forming the candidate tree.
            For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
        n : int
            Given the final tree of prod(topk) candidates, return only the top n most confident.
        ...
        Output : torch.Tensor
            The tensor of most likely candidate sequences.
            Has size [b n self.n_predict], where b is batch size and n is provided above.
        """
        return self.speculator.generate_suffixes(state, ind, topk, n)

    def forward(
        self,
        state: torch.Tensor,
        inds: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOR TRAINING
        A parallel forward pass on pre-existing ground-truth tokens in pretraining contexts.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive
        behavior for end positions.
        ...
        Args
        ----
        state : torch.Tensor
            Embedding vectors from the base model for a given sequence.
            Expects size [b n d] where b is batch size, n is seq len, and d is model width.
        inds : torch.Tensor
            Ground-truth token indices. inds[:,i] is the prediction coming from state[:,i]
            (or the legal fiction ground truth corresponding to that prediction).
            Expects size [b n+self.n_predict].
        ...
        Output : torch.Tensor
            Prediction logits at each position, for each head of the speculator.
            Has size [self.n_predict b n v] where v is vocab size.
        """
        return self.speculator(state, inds)

    def reset_parameters(self):
        self.speculator.reset_parameters()


_fms_to_hf_adapt_map[MLPSpeculator] = MLPSpeculatorPreTrainedModel
