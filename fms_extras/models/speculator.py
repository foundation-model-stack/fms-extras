import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fms.modules.layernorm import LayerNormParameterized


class MLPSpeculator(nn.Module):
    """
    This is a simple MLP-based speculator that functions similarly to Medusa
    (https://arxiv.org/abs/2401.10774), ingesting context via the final embedding
    vector from the base model. However, this model also conditions on previously
    predicted tokens, similarly to an RNN, allowing it to generate better-quality n-grams.

    The architecture is as flat and simple as possible: for each prediction head,
    the current state vector is projected into a new latent space and added to the
    previous token's embedding. This sum goes through layernorm and activation, forming
    the new state vector. This state predicts the next token (or set of candidate tokens)
    for the current head, and then is passed on to the next.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of the input vector from the base model.
    inner_dim : int
        Latent dimensionality of the speculator model.
    vocab_size : int
        Number of entries in the tokenizer associated with the base model.
    n_predict : int
        Number of heads / number of tokens to guess ahead. Model size and speed scale with this value.
    """

    def __init__(self, emb_dim=4096, inner_dim=0, vocab_size=32000, n_predict=3):
        super().__init__()
        self.n_predict = n_predict
        self.emb_dim = emb_dim
        inner_dim = inner_dim if inner_dim != 0 else emb_dim
        self.inner_dim = inner_dim
        self.vsize = vocab_size
        self.emb = nn.ModuleList(
            [nn.Embedding(vocab_size, inner_dim) for _ in range(n_predict)]
        )
        self.proj = nn.ModuleList(
            [
                nn.Linear((emb_dim if i == 0 else inner_dim), inner_dim, bias=False)
                for i in range(n_predict)
            ]
        )
        self.head = nn.ModuleList(
            [nn.Linear(inner_dim, vocab_size, bias=False) for _ in range(n_predict)]
        )
        self.ln = nn.ModuleList(
            [
                LayerNormParameterized(
                    inner_dim, elementwise_shift=True, elementwise_scale=True
                )
                for _ in range(n_predict)
            ]
        )
        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = 0.5 ** (0.5 / n_predict)
        self.emb_weight = math.sqrt(1 - self.state_weight**2)
        self.activation = nn.GELU()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 1 / math.sqrt(self.inner_dim))
            elif isinstance(m, LayerNormParameterized):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        # k indicates # of candidates
        # h indicates # of generated tokens
        b = state.size(0)
        out = torch.empty(b, 1, 0, device=state.device).int()  # b k h
        log_probs = torch.zeros(b, 1, device=state.device)  # b k
        assert (
            len(topk) == self.n_predict
        ), f"You must provide a topk number for each head ({self.n_predict} heads, {len(topk)} provided)"
        for i in range(self.n_predict):
            # Project and predict
            z = self.emb[i](ind)
            z = z.mul(self.emb_weight * math.sqrt(self.inner_dim / 2))  # b k d
            state = self.proj[i](state) * self.state_weight + z
            state = self.activation(self.ln[i](state))  # b k d
            probs = F.log_softmax(self.head[i](state), dim=2)  # b k v
            probs, preds = probs.topk(topk[i], dim=2)  # b k k'

            # Update candidate set with new predictions
            out = out.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' h
            out = torch.cat([out, preds.unsqueeze(3)], dim=3)  # b k k' h+1
            out = out.view(b, -1, i + 1)  # b kk' h+1

            # Update state, log_probs and ind for new predictions
            state = state.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' d
            state = state.reshape(b, -1, state.size(3))  # b kk' d
            ind = preds.view(b, -1)  # b kk'
            log_probs = log_probs.unsqueeze(2).expand(b, -1, topk[i])  # b k k'
            log_probs = log_probs.add(probs).reshape(b, -1)  # b kk'

        # Take only top n best guesses
        best_guesses = log_probs.topk(n, dim=1)[1]  # b k
        return out.gather(
            1, best_guesses.unsqueeze(2).expand(-1, -1, self.n_predict)
        )  # b n h

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
        out = []
        for i in range(self.n_predict):
            z = self.emb[i](inds[:, i : i + state.size(1)])
            z = z.mul(self.emb_weight * math.sqrt(self.inner_dim / 2))  # b n d
            state = self.proj[i](state) * self.state_weight + z
            state = self.activation(self.ln[i](state))  # b n d
            out.append(self.head[i](state))  # b n v
        return torch.stack(out, dim=0)  # h b n v


def apply_index_map(
    inp: torch.Tensor, inds: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    Applies index map to specified dimension of input tensor. Used for batch flattening/unflattening.
    
    More precisely, takes input of size ([...], n, [...]), with n in the dim-th dimension,
    and tensor of indices of size (a, ..., z). Using those indices we over/under sample the
    input on dimension dim, to create output tensor with size ([...], (a, ..., z), [...]).

    i.e. if dim=0, inp has size (6,3,2), and inds has size (8,4), then:
    1) max(inds) < 6
    2) output has size (8,4,3,2)

    Args:
        inp: torch.Tensor
            tensor of inputs
        inds: torch.Tensor
            tensor of indices
        dim: int
            dimension to sample on

    Returns:
        torch.Tensor
            output tensor with new size ([...], (a, ..., z), [...])
    """
    inds_shape = inds.size()
    inp_shape = inp.size()
    out = inp.index_select(dim, inds.view(-1))
    return out.view(*inp_shape[:dim], *inds_shape, *inp_shape[dim + 1 :])


def flatten_batch(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes a speculator suffix tree: a bsize x n_candidates x candidate_len rectangular batch
    of token indices, and flattens it while removing redundant tokens.

    For example, given:

    a b c
    a b d
    a e f

    Tokens 'a b' in line 2 and token 'a' in line 3 are functionally equivalent to 'a b' in
    line 1, so the flattened batch returns `a b c d e f`

    Args:
        inp: torch.Tensor
            speculator suffix tree

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            1) the flattened, pruned input
            2) a tensor, sized as input, mapping each input token to its slot in output
            3) a tensor, sized as output, mapping each output token to its slot in the flattened input
    """
    unflat_map = torch.zeros_like(inp)
    inp_list = inp.tolist()
    flat_map = []
    batch_offset = 0
    # Generate the flatten/unflatten maps
    for b, candidate_set in enumerate(inp_list):
        lineages: Dict[Tuple[List[int]]] = {} # Prefix : n unique prefixes observed so far
        for k, candidate in enumerate(candidate_set):
            for n in range(len(candidate)):
                lineage = tuple(candidate[: n + 1])
                if lineage in lineages:
                    # Token is redundant
                    unflat_map[b, k, n] = lineages[lineage] + batch_offset
                else:
                    # Token is not redundant
                    unflat_map[b, k, n] = len(lineages) + batch_offset
                    lineages[lineage] = len(lineages)
                    flat_map.append(
                        b * len(candidate_set) * len(candidate)
                        + k * len(candidate)
                        + n
                    )
        batch_offset += len(lineages)
    # Generate the flattened batch
    flat_map = torch.tensor(flat_map, device=unflat_map.device, dtype=torch.int32)
    out = apply_index_map(inp.view(-1), flat_map, 0)
    return out, unflat_map, flat_map
