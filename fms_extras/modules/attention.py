from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from fms import distributed, models
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule
from torch._C._distributed_c10d import ProcessGroup

from fms_extras.utils.cache.paged import PagedAttentionCacheDataLayer


class PagedMultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.

    Note: this class extends MultiHeadAttention to enable tensor parallel support
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        gain=1,
    ):
        super(PagedMultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )

        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = nn.Linear(
            self.emb_dim,
            sum(self.splits),
            bias=use_bias,
        )

        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # Avoiding graph breaks
        self.previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
        self.previous_mem_efficient: bool = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        self.previous_math: bool = torch.backends.cuda.math_sdp_enabled()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPPagedMultiHeadAttention":
        return TPPagedMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
        cache_data_layer: Optional[PagedAttentionCacheDataLayer] = None,
        use_cache: bool = False,
        is_self: bool = True,
        is_causal_mask: bool = False,
    ):
        """
        cache_data_layer: PagedAttentionCacheDataLayer, optional
            A single layer of the cache (default is None)
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
        position_ids = (
            None if cache_data_layer is None else cache_data_layer.position_ids
        )

        queries, keys, values = self.qkv_fused(q).split(self.splits, dim=-1)

        queries = queries.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = keys.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = values.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries,
                keys,
                position_ids,  # type: ignore
                None,
                use_cache,
            )

        # store the values in kv-cache
        if use_cache and cache_data_layer:
            keys, values = cache_data_layer.store(keys, values)

        if use_cache and cache_data_layer and cache_data_layer.is_filled():
            attn = cache_data_layer.attend(queries)
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
                    mask, queries, keys, position_ids, use_cache  # type: ignore
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
        if use_cache and cache_data_layer:
            # note: needed to add this check to return the data_layer as it fails compile otherwise
            return out, cache_data_layer.data_layer
        else:
            return out


class TPPagedMultiHeadAttention(PagedMultiHeadAttention, TPModule):
    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        gain=1,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert (
            nheads % world_size == 0
        ), "The number of heads must be divisible by world size"
        PagedMultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads > 1 else kvheads,
            p_dropout,
            use_bias,
            position_encoder,
            gain,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        qkv_weight = self._get_sd_weight(
            tensor_values, used_keys, ["qkv_fused", "weight"]
        )
        dense_weight = self._get_sd_weight(
            tensor_values, used_keys, ["dense", "weight"]
        )
        if self.use_bias:
            qkv_bias = self._get_sd_weight(
                tensor_values, used_keys, ["qkv_fused", "bias"]
            )
            dense_bias = self._get_sd_weight(
                tensor_values, used_keys, ["dense", "bias"]
            )

        # 2. Raise exceptions
        if len(tensor_values) > (4 if self.use_bias else 2):
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        # The number in max_partition_sizes will signify the largest world size
        # til we need to duplicate.  For instance if we have nheads=16 and
        # world_size=32, then first 2 ranks will get first 1/16th of query
        self.sharded_copy(
            self.qkv_fused.weight,
            qkv_weight,
            0,
            [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
        )
        self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
        if self.use_bias:
            self.sharded_copy(
                self.qkv_fused.bias,
                qkv_bias,
                0,
                [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
            )
            self.sharded_copy(self.dense.bias, dense_bias, 1, [self.world_size], False)

    @staticmethod
    def import_module(
        mha: PagedMultiHeadAttention, group: ProcessGroup
    ) -> "TPPagedMultiHeadAttention":
        tp_mha = TPPagedMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
        )
        return tp_mha

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
        cache_data_layer: Optional[PagedAttentionCacheDataLayer] = None,
        use_cache: bool = False,
        is_self: bool = True,
        is_causal_mask: bool = False,
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par = copy_to_tensor_model_parallel_region(q)
        k_par = copy_to_tensor_model_parallel_region(k)
        v_par = copy_to_tensor_model_parallel_region(v)

        out_par = PagedMultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            mask,
            attn_algorithm,
            cache_data_layer,
            use_cache,
            is_self,
            is_causal_mask,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.world_size)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.world_size)
            return out
