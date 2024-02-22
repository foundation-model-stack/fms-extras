import math
from typing import MutableMapping, Tuple, Optional

from fms.modules.positions import PositionEncoder
import torch

class RotaryEmbeddingContiguous(PositionEncoder):
    def __init__(
        self, dim: int, ratio: int = 10_000, max_seq_len=2048, ntk_scaling=False
    ):
        """
        Note: This is an implementation improvement over what is in fms which fixes some issues with discontiguous
        tensors

        This implementation of Rotary Position Embeddings (RoPE) avoids
        complex numbers, and so can be used with torch.compile.

        https://arxiv.org/abs/2104.09864

        ...
        Args
        ----
        dim : int
            Per-head embedding dimension
        max_seq_len : int
            Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed
        ratio: int
            The ratio for the geometric progression to compute the rotation angles
        """
        super(RotaryEmbeddingContiguous, self).__init__()
        self.dim = dim
        self.ratio = ratio
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}
        self.ntk_scaling = ntk_scaling
        self.max_seq_len = max_seq_len
        self.cached_alpha = None

    def _alpha(self, seq_len) -> int:
        if not self.ntk_scaling:
            return 1
        else:
            alpha = seq_len / self.max_seq_len
            alpha = math.ceil(alpha)
            # for some reason math.log2 didn't `torch.compile` but
            # `math.log` does
            alpha = math.log(alpha) / math.log(2)
            alpha = math.ceil(alpha)
            alpha = 2**alpha
            alpha = int(alpha)
            return alpha

    def compute_freqs_cis(self, device, max_seq_len=2048):
        # NTK scaling.
        # https://arxiv.org/abs/2306.15595
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        #
        # we'll store the freqs for each alpha value. This means that for
        # shorter sequences, we preserve the original scale.
        # To limit the number of multiples to store we'll maintain alphas for
        # `2**i` where i is the ratio of actual vs initial max seq len. (i.e. 2,
        # 4, 8, ... as needed)
        alpha = self._alpha(max_seq_len)
        dev_idx = device.index

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        # This condition can be combined with the model using Rotary calling this method
        # on model init when device is known to avoid a graph break (see llama.py)
        if self.ntk_scaling:
            max_seq_len = max(max_seq_len, self.max_seq_len * alpha)
        else:
            if self.max_seq_len_cached[dev_idx] > 0:
                return alpha
            max_seq_len = max(max_seq_len, self.max_seq_len)

        if (
            alpha in self.cached_freqs[dev_idx]
            and max_seq_len <= self.max_seq_len_cached[dev_idx]
        ):
            return alpha

        ratio = self.ratio
        dim = self.dim

        if self.ntk_scaling:
            ratio = ratio * alpha ** (dim / (dim - 2))

        freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )

        t = torch.arange(max_seq_len, device=device, dtype=freqs.dtype)
        freqs = torch.outer(t, freqs).float()
        self.max_seq_len_cached[dev_idx] = max_seq_len
        self.cached_freqs[dev_idx][alpha] = torch.stack(
            [
                torch.cos(freqs),
                -torch.sin(freqs),
                torch.sin(freqs),
                torch.cos(freqs),
            ],
            dim=2,
        ).reshape(*freqs.size(), 2, 2)

        return alpha

    def reshape_for_broadcast(self, x: torch.Tensor, cur_freqs):
        ndim = x.ndim
        assert 1 < ndim, ndim
        assert cur_freqs.size()[:2] == (
            x.size(2),
            x.size(-2),
        ), f"for {cur_freqs.size()} and {x.size()}"
        shape = [d if i == 2 or i >= ndim - 2 else 1 for i, d in enumerate(x.size())]
        return cur_freqs.view(*shape, 2)

    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        position_ids : Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. This is important in
            kv-caching and left-padding situations, for which the rotation to be applied might
            not always be the pre-cached position 0...S. For kv-caching without dynamic batching
            or variable per-row left padding position_ids is shared for all the batch.
        """
        assert len(q.size()) == 4
        assert len(k.size()) == 4

        if position_ids is None:
            # Compute position_ids based on cache config
            position_ids = torch.arange(
                0, q.size(2), dtype=torch.long, device=q.device
            ).repeat(q.size(0), 1)
        seq_len = q.size(2)

        q_ = q.float().view(*q.size()[:-1], -1, 2)  # B H L D/2 2
        k_ = k.float().view(*k.size()[:-1], -1, 2)  # B H L D/2 2

        # the max start position should be based on the max first position of each sequence
        max_start_pos = torch.max(position_ids[:, 0])
        alpha = self.compute_freqs_cis(q.device, max_start_pos + seq_len)
        freqs = self.cached_freqs[q.device.index][alpha][position_ids]

        freqs = freqs.float()  # 1 1 L D/2 2 2

        q_out = (
            freqs[:, -q.size(1):, None, :, :, :].mul(q_.unsqueeze(-2)).sum(5).flatten(3)
        ).type_as(q)
        k_out = (
            freqs[:, -k.size(1):, None, :, :, :].mul(k_.unsqueeze(-2)).sum(5).flatten(3)
        ).type_as(k)

        return q_out.view_as(q), k_out.view_as(k)