import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fms import distributed, models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    UniformModelParallelStrategy,
)
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import PositionEncoder, RotaryEmbedding
from fms.modules.tp import TPModule
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.serialization import _legacy_mlp_glu_unfused_to_fused_adapter
from torch._C._distributed_c10d import ProcessGroup

from fms_extras.utils.cache.paged import (
    PagedAttentionCacheData,
    PagedAttentionCacheDataLayer,
)


# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


@dataclass
class PagedLLaMAConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    rope_theta: int = 10_000


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
            out = reduce_from_tensor_model_parallel_region(out_par[0])
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par)
            return out


class PagedLLaMABlock(nn.Module):
    def __init__(self, config: PagedLLaMAConfig, rotary_emb: RotaryEmbedding):
        super(PagedLLaMABlock, self).__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = PagedMultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        cache_data_layer: Optional[PagedAttentionCacheDataLayer] = None,
        use_cache: bool = False,
        is_causal_mask: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            k=x,
            v=x,
            mask=mask,
            attn_algorithm=attn_algorithm,
            cache_data_layer=cache_data_layer,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class PagedLLaMAHeadless(nn.Module):
    def __init__(
        self, config: PagedLLaMAConfig, distributed_strategy: DistributedStrategy
    ):
        super(PagedLLaMAHeadless, self).__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy
        shared = WordEmbedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
            abs_pos=False,
            reversible=True,
            tie_weights=False,
            bias=False,
        )
        self.shared = self.distributed_strategy.distribute_module(shared)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )
        if isinstance(self.distributed_strategy, UniformModelParallelStrategy):
            for dev_idx in set(self.distributed_strategy.layer_to_device):
                self.rot_emb.compute_freqs_cis(
                    torch.device("cuda", dev_idx), self.config.max_expected_seq_len
                )
        else:
            self.rot_emb.compute_freqs_cis(
                self.shared.emb.weight.device, self.config.max_expected_seq_len
            )

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = PagedLLaMABlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x_in: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache_data: Optional[PagedAttentionCacheData] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        qlen = x_in.size(1)
        filled_cache = False

        if use_cache:
            if cache_data:
                filled_cache = cache_data.is_filled()

        # if mask is none, we need to specify causal mask
        if mask is None:
            # we are caching and can assume all 1s in the mask
            if use_cache and filled_cache and qlen == 1:
                # b x h x qlen x kvlen
                is_causal_mask = False
            else:
                is_causal_mask = True
        else:
            is_causal_mask = False

        x_in = self.shared(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                mask=mask,
                cache_data_layer=None
                if cache_data is None
                else cache_data.get_layer(i),
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class PagedLLaMA(nn.Module):
    def __init__(
        self,
        config: Optional[PagedLLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(PagedLLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PagedLLaMAConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.headless_model = PagedLLaMAHeadless(self.config, distributed_strategy)
        # todo: head should be separated from shared for this, but keeping here for now as we would need to add tp
        self.head = self.headless_model.shared

    def get_config(self) -> PagedLLaMAConfig:
        return self.config

    @classmethod
    def from_config(cls, config: PagedLLaMAConfig) -> "PagedLLaMA":
        return cls(config)

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, PagedMultiHeadAttention)
                or isinstance(m, WordEmbedding)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache_data: Optional[PagedAttentionCacheData] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None,
        return_embeds: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        main forward pass for a paged llama model

        Args:
            x: torch.Tensor
                the input ids as a long type
            mask: torch.Tensor, optional
                an optional mask to be used in SDPA. If None, no mask will be used
            cache_data: PagedAttentionCacheData, optional
                the optional cache data used in paged attention. If None is given, SDPA will always be used
            use_cache: bool
                denotes whether a cache should be used. If True, the cache will be returned as well as the logits
            only_last_token: bool
                only return the last token from embedding output
            attn_algorithm: str, optional
                string to denote which attention algorithm to use
            return_embeds: bool
                If True, will return the embeddings, otherwise wil not return embeddings

        Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            if use_cache and return_embeds is True => (logits, cache, embeds)
            if use_cache is True => (logits, cache)
            if return_embeds is True => (logits, embeds)
            otherwise => logits
        """
        embeds, cache = self.headless_model(
            x,
            mask,
            cache_data,
            use_cache,
            attn_algorithm,
        )

        if only_last_token:
            embeds = embeds[:, -1, :]

        preds = self.head(embeds, reverse=True)

        out = [preds]
        if use_cache:
            out.append(cache)
        if return_embeds:
            out.append(embeds)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


# Register common LLaMA variants with the model registration API

# a micro llama model to use with a char-level tokenizer
_micro_char_config = PagedLLaMAConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

_7b_config = PagedLLaMAConfig()
_ibm_7b_instruct_lab_config = PagedLLaMAConfig(src_vocab_size=32008)
_13b_config = PagedLLaMAConfig(emb_dim=5120, nheads=40, nlayers=40)
_13b_code_config = PagedLLaMAConfig(
    emb_dim=5120,
    nheads=40,
    nlayers=40,
    src_vocab_size=32016,
    max_expected_seq_len=16384,
    rope_theta=1_000_000,
)
# todo: add 35B config

_70b_config = PagedLLaMAConfig(
    emb_dim=8192,
    multiple_of=4096,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=(1.3 * 8 / 3),
)

_architecture_name = "paged_llama"


def _llama_factory_factory(config):
    def factory(**kwargs):
        return PagedLLaMA(config, **kwargs)

    return factory


# llama2

models.register_model(
    _architecture_name, "micro", _llama_factory_factory(_micro_char_config)
)
models.register_model(
    _architecture_name,
    "7b.ibm_instruct_lab",
    _llama_factory_factory(_ibm_7b_instruct_lab_config),
)

models.register_model(_architecture_name, "7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "13b", _llama_factory_factory(_13b_config))
models.register_model(
    _architecture_name, "13b.code", _llama_factory_factory(_13b_code_config)
)
models.register_model(_architecture_name, "70b", _llama_factory_factory(_70b_config))

# llama3

_8b_llama3_config = PagedLLaMAConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=32,
    hidden_grow_factor=3.5,
    multiple_of=1024,
    max_expected_seq_len=8192,
)

models.register_model(
    _architecture_name, "llama3.8b", _llama_factory_factory((_8b_llama3_config))
)


def _rename_weights_to_fms(orig_sd):
    replacements = [
        (r"^tok_embeddings", "headless_model.shared.emb"),
        (r"^norm", "headless_model.dec_norm"),
        (r"^output", "head.head"),
        (r"^layers", "headless_model.layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.query"),
        (r"attn\.wk", "attn.key"),
        (r"attn\.wv", "attn.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # llama in meta has unfused qkv attn weights, so these weights must be converted to fused weights in fms
        if (
            "attn.query" in new_name
            or "attn.key" in new_name
            or "attn.value" in new_name
        ):
            unfused_weights = [
                re.sub(r"w[qkv]", "wq", name),
                re.sub(r"w[qkv]", "wk", name),
                re.sub(r"w[qkv]", "wv", name),
            ]

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([orig_sd[w] for w in unfused_weights], dim=0)

    new_sd = _legacy_mlp_glu_unfused_to_fused_adapter(new_sd)

    return new_sd


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"^lm_head.weight", "headless_model.shared.head.weight"),
        (r"^model.embed_tokens.weight", "headless_model.shared.emb.weight"),
        (r"^model.norm", "headless_model.dec_norm"),
        (r"^model.layers", "headless_model.layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}

    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # llama in hf has unfused qkv attn weights, so these weights must be converted to fused weights in fms
        if (
            "attn.query" in new_name
            or "attn.key" in new_name
            or "attn.value" in new_name
        ):
            del new_sd[new_name]
            unfused_weights = [
                re.sub(r"self_attn\.[kvq]_proj", "self_attn.q_proj", name),
                re.sub(r"self_attn\.[kvq]_proj", "self_attn.k_proj", name),
                re.sub(r"self_attn\.[kvq]_proj", "self_attn.v_proj", name),
            ]
            missing_weights = [w for w in unfused_weights if w not in hf_sd.keys()]
            if len(missing_weights) != 0:
                raise ValueError(
                    f"The following weights are required for properly fusing: {missing_weights}"
                )

            raw_mapping = {w: hf_sd[w] for w in unfused_weights}

            # q=0, k=1
            for unfused_weight_key in unfused_weights[:2]:
                temp = raw_mapping[unfused_weight_key]
                # nheads is used in the transformation required for hf->fms
                # here we are using 128 as this value fits with all popular models
                #   7B, 13B, 70B to recover the number of heads
                nheads = int(temp.size(0) / 128)

                temp = (
                    temp.view(nheads, 2, -1, temp.size(1))
                    .transpose(1, 2)
                    .reshape(*temp.size())
                )

                raw_mapping[unfused_weight_key] = temp

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([raw_mapping[w] for w in unfused_weights], dim=0)

    new_sd = _legacy_mlp_glu_unfused_to_fused_adapter(new_sd)

    return new_sd


def _rename_fms_weights_to_fms_paged(orig_sd):
    replacements = [
        (r"attn\.in_proj\.qkv_fused\.weight", "attn.qkv_fused.weight"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = f"headless_model.{name}"
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


serialization.register_adapter(_architecture_name, "meta", _rename_weights_to_fms)
serialization.register_adapter(_architecture_name, "hf", _hf_sd_to_fms_sd)
serialization.register_adapter(
    _architecture_name, "fms_llama", _rename_fms_weights_to_fms_paged
)
