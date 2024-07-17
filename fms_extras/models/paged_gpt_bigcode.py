import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
from fms import models
from fms.modules.feedforward import FeedForwardBlock
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy

from fms_extras.modules.attention import PagedMultiHeadAttention
from fms_extras.utils.cache.paged import (
    PagedAttentionCacheData,
    PagedAttentionCacheDataLayer,
)


@dataclass
class PagedGPTBigCodeConfig(ModelConfig):
    src_vocab_size: int = 49157  # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    emb_dim: int = 2048  # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    nheads: int = 12
    nlayers: int = 12
    pad_id: int = 0
    max_pos: int = 512
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu-tanh"
    p_dropout: float = 0.0
    emb_dropout: float = 0.0
    multiquery_attn: bool = True
    ln_eps: float = 1e-5


class PagedGPTBigCodeBlock(nn.Module):
    def __init__(self, config: PagedGPTBigCodeConfig):
        super().__init__()
        self.config = config

        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)

        self.attn = PagedMultiHeadAttention(
            self.config.emb_dim,
            self.config.emb_dim // self.config.nheads,
            self.config.emb_dim // self.config.nheads,
            self.config.nheads,
            kvheads=1 if self.config.multiquery_attn else self.config.nheads,
            p_dropout=self.config.p_dropout,
            use_bias=True,
        )

        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=True,
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
        # self attention
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
            return x, cache
        else:
            return x


class PagedGPTBigCodeHeadless(nn.Module):
    def __init__(self, config: PagedGPTBigCodeConfig, distributed_strategy: DistributedStrategy):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        layers = []
        for i in range(self.config.nlayers):
            block = PagedGPTBigCodeBlock(self.config)
            block_module = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block_module)
        self.layers = nn.ModuleList(layers)

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)
        self.position_embedding = nn.Embedding(self.config.max_pos, self.config.emb_dim)

        self.dec_norm = self.distributed_strategy.distribute_module(
            nn.LayerNorm(self.config.emb_dim, eps=self.config.ln_eps), final_layers=True
        )

        if self.config.emb_dropout:
            self.emb_dropout = nn.Dropout(self.config.emb_dropout)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        cache_data: Optional[PagedAttentionCacheData] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len

        qlen = x.size(1)
        filled_cache = False

        # if we are using the cache, the key length needs to be extended with the past keys length
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

        x_emb = self.embedding(x)

        # if pad_id exists
        #   is_pad will be a BoolTensor
        #   otherwise pad_id will not be taken into account
        if self.config.pad_id is None:
            is_pad = torch.zeros_like(x, dtype=bool, device=x.device)
        else:
            is_pad = x == self.config.pad_id

        if cache_data is None or cache_data.position_ids is None:
            position_ids = ((~is_pad).cumsum(1) - 1).clamp(min=0)

            if cache_data is not None:
                cache_data.position_ids = position_ids
        else:
            position_ids = cache_data.position_ids

        # look up position embeddings
        position_out = self.position_embedding(position_ids)

        # zero out the associated position embeddings
        if self.config.pad_id is not None:
            position_out = position_out.mul(~is_pad.unsqueeze(-1))

        # perform absolute position embedding
        x = x_emb + position_out

        # apply dropout to embeddings
        if self.config.emb_dropout:
            x = self.emb_dropout(x)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x,
                mask=mask,
                cache_data_layer=None
                if cache_data is None
                else cache_data.get_layer(i),
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
            )

            if use_cache:
                x, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x = output

        dec_out = self.dec_norm(x)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


# Implements the decoder-only PagedGPTBigCodeModel
class PagedGPTBigCode(nn.Module):
    def __init__(
        self,
        config: Optional[PagedGPTBigCodeConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(PagedGPTBigCode, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PagedGPTBigCodeConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = PagedGPTBigCodeHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

        # this model ties weights, so we tie here
        self.head.weight = self.base_model.embedding.weight

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: PagedGPTBigCodeConfig) -> "PagedGPTBigCode":
        return cls(config)

    def get_config(self) -> PagedGPTBigCodeConfig:
        return self.config

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if isinstance(m, PagedMultiHeadAttention) or isinstance(
                m, FeedForwardBlock
            ):
                m.reset_parameters()

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        cache_data: Optional[PagedAttentionCacheData] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
        return_embeds: bool = False,
    ):
        embeds, cache = self.base_model(
            x,
            mask,
            cache_data=cache_data,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        preds = self.head(embeds)

        out = [preds]
        if use_cache:
            out.append(cache)
        if return_embeds:
            out.append(embeds)

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


_santacoder_config = PagedGPTBigCodeConfig(
    src_vocab_size=49280,
    emb_dim=2048,
    nheads=16,
    nlayers=24,
    pad_id=-1,
    max_pos=2048,
    p_dropout=0.1,
    emb_dropout=0.1,
)

_13b_config = PagedGPTBigCodeConfig(
    src_vocab_size=50304,
    emb_dim=5632,
    nheads=44,
    nlayers=40,
    pad_id=50280,
    max_pos=8192,
    hidden_grow_factor=4.0,
    p_dropout=0.1,
    emb_dropout=0.1,
    ln_eps=1e-5,
)
_20b_config = PagedGPTBigCodeConfig(
    src_vocab_size=49152,
    emb_dim=6144,
    nheads=48,
    nlayers=52,
    pad_id=0,
    max_pos=8192,
    hidden_grow_factor=4.0,
    p_dropout=0.1,
    emb_dropout=0.1,
    ln_eps=1e-5,
)


_architecture_name = "paged_gpt_bigcode"


def _gpt_bigcode_factory_factory(config):
    def factory(**kwargs):
        return PagedGPTBigCode(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "santacoder", _gpt_bigcode_factory_factory(_santacoder_config)
)
models.register_model(
    _architecture_name, "ibm.13b", _gpt_bigcode_factory_factory(_13b_config)
)
models.register_model(
    _architecture_name, "ibm.20b", _gpt_bigcode_factory_factory(_20b_config)
)


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    import re

    replacements = [
        ("lm_head.weight", "head.weight"),
        (r"^transformer.wte.weight", "base_model.embedding.weight"),
        (r"^transformer.wpe.weight", "base_model.position_embedding.weight"),
        (r"^transformer.ln_f", "base_model.dec_norm"),
        (r"^transformer.h", "base_model.layers"),
        (r"attn\.c_attn", "attn.qkv_fused"),
        (r"attn\.c_proj", "attn.dense"),
        (r"mlp\.c_fc", "ff_sub_layer.w1"),
        (r"mlp\.c_proj", "ff_sub_layer.w2"),
        (r"ln_1", "ln"),
        (r"ln_2", "ff_ln"),
    ]

    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        new_sd[new_name] = param

    return new_sd


serialization.register_adapter(_architecture_name, "hf", _hf_sd_to_fms_sd)
