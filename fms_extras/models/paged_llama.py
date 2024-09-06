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

from fms_extras.modules.attention import PagedMultiHeadAttention
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
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False


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
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
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
            tie_weights=self.config.tie_heads,
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

_70b_llama3_config = PagedLLaMAConfig(
    src_vocab_size=128256,
    emb_dim=8192,
    norm_eps=1e-5,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=3.5,
    multiple_of=4096,
    max_expected_seq_len=8192,
    rope_theta=500000,
)

models.register_model(
    _architecture_name, "llama3.8b", _llama_factory_factory((_8b_llama3_config))
)

models.register_model(
    _architecture_name, "llama3.70b", _llama_factory_factory((_70b_llama3_config))
)

# calico

_8b_calico_code_config = PagedLLaMAConfig(
    src_vocab_size=49152,
    emb_dim=4096,
    nheads=32,
    kvheads=8,
    nlayers=36,
    pad_id=0,
    hidden_grow_factor=14336 / 4096,
    multiple_of=1,
    max_expected_seq_len=4096,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

models.register_model(
    _architecture_name,
    "calico.8b.code",
    _llama_factory_factory((_8b_calico_code_config)),
)

_3b_calico_code_config = PagedLLaMAConfig(
    src_vocab_size=49152,
    emb_dim=2560,
    nheads=32,
    kvheads=32,
    nlayers=32,
    pad_id=0,
    hidden_grow_factor=10240 / 2560,
    multiple_of=1,
    max_expected_seq_len=2048,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

models.register_model(
    _architecture_name,
    "calico.3b.code",
    _llama_factory_factory((_3b_calico_code_config)),
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
        ) and "weight" in new_name:
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

                emb_dim = param.size(1)
                if emb_dim == 2560:
                    head_size = 80
                else:
                    head_size = 128

                # nheads is used in the transformation required for hf->fms
                # here we are using 128 as this value fits with all popular models
                #   7B, 13B, 70B to recover the number of heads
                nheads = int(temp.size(0) / head_size)

                temp = (
                    temp.view(nheads, 2, -1, temp.size(1))
                    .transpose(1, 2)
                    .reshape(*temp.size())
                )

                raw_mapping[unfused_weight_key] = temp

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([raw_mapping[w] for w in unfused_weights], dim=0)
        elif (
            "attn.query" in new_name
            or "attn.key" in new_name
            or "attn.value" in new_name
        ) and "bias" in new_name:
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

                weight_name = name.replace("bias", "weight")
                emb_dim = hf_sd[weight_name].size(1)
                if emb_dim == 2560:
                    head_size = 80
                else:
                    head_size = 128

                # nheads is used in the transformation required for hf->fms
                # here we are using 128 as this value fits with all popular models
                #   7B, 13B, 70B to recover the number of heads
                nheads = int(temp.size(0) / head_size)

                temp = temp.view(nheads, 2, -1).transpose(1, 2).reshape(*temp.size())

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
