from typing import Optional
import torch
import torch.nn.functional as F
from fms.modules.attention import MultiHeadAttention
from fms.modules.positions import PositionEncoder
from fms_extras.utils.cache import AttentionComputationMixin


class PagedMultiHeadAttention(MultiHeadAttention):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.

    Note: this class extends MultiHeadAttention to enable tensor parallel support
    """

    def __init__(self, emb_dim, emb_kq, emb_v, nheads, kvheads, p_dropout=None, use_bias=False,
                 position_encoder: Optional[PositionEncoder] = None, gain=1):

        super().__init__(emb_dim, emb_kq, emb_v, nheads, kvheads, p_dropout, use_bias, position_encoder, gain)

    def forward(
        self,
        q,
        k,
        v,
        mask: Optional[torch.Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        cache_data_layer=None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        cache_data_layer: CacheDataLayer, optional
            A single layer of the cache (default is None)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved
        is_self: bool
            if True, this will perform self attention, otherwise this will perform cross attention. Note: This will
            only be used in the case that use_cache=True. This may be removed in future

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """

        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()

        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )
        keys = self.key(k).view(
            batch_size, q_len, self.kvheads, self.emb_kq_per_head
        )
        values = self.value(v).view(
            batch_size, q_len, self.kvheads, self.emb_v_per_head
        )

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries,
                keys,
                position_ids,
                None,
                True,
            )

        # store the values in kv-cache
        if use_cache and cache_data_layer:
            keys, values = cache_data_layer.store(keys, values)

        custom_attention = (
            use_cache
            and cache_data_layer
            and isinstance(cache_data_layer, AttentionComputationMixin)
        )

        # Provide a method for a user to perform their own implementation of attention in the cache case if required
        if custom_attention and cache_data_layer.is_filled():
            attn = cache_data_layer.attend(queries, keys, values)
        # otherwise we always fall back into SDPA as this is either a prompt or it is a single contiguous cache
        else:
            queries = queries.transpose(2, 1)
            keys = keys.transpose(2, 1)
            values = values.transpose(2, 1)

            # Merge rel pos bias and mask into single float mask
            if mask is not None:
                # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
                # we need to create the nheads dimension
                while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                    mask = mask.unsqueeze(1)

            if self.position_encoder is not None:
                attn_mask = self.position_encoder.adjusted_mask(
                    mask, queries, keys, position_ids, use_cache
                )
            else:
                attn_mask = mask

            # Expand kv so black-box attn will work
            expansion = self.nheads // self.kvheads
            # k/v: b h l d
            if expansion != 1:
                keys_e = (
                    keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
                )
                values_e = (
                    values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
                )
            else:
                keys_e = keys
                values_e = values

            if attn_algorithm:
                # Pick which fused attn kernels will run.
                use_flash = attn_algorithm == "flash"
                use_mem_efficient = attn_algorithm == "mem"
                use_math = attn_algorithm == "math"

                torch.backends.cuda.enable_flash_sdp(use_flash)
                torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)
                torch.backends.cuda.enable_math_sdp(use_math)

            attn = F.scaled_dot_product_attention(
                queries,
                keys_e,
                values_e,
                attn_mask=attn_mask,
                dropout_p=self.p_dropout if self.training else 0.0,
                is_causal=is_causal_mask,
            )

            if attn_algorithm:
                torch.backends.cuda.enable_flash_sdp(self.previous_flash)
                torch.backends.cuda.enable_mem_efficient_sdp(
                    self.previous_mem_efficient
                )
                torch.backends.cuda.enable_math_sdp(self.previous_math)

            # attn: bs x seq_len x nheads*emb_v_per_head
            # attn: b x h x qlen x ds
            # attn after permute: b x qlen x h x ds
            # b x qlen x (d)
            attn = attn.transpose(2, 1).contiguous()

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)

        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            # note: needed to add this check to return the data_layer as it fails compile otherwise
            return out, cache_data_layer.data_layer if custom_attention else (
                keys,
                values,
            )
        else:
            return out