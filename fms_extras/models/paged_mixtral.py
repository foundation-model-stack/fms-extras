import math
import re
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    UniformModelParallelStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import MOEFeedForward
from fms.modules.head import LinearClassificationHead
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.config import ModelConfig
from sympy import Union

from fms_extras.modules.attention import PagedMultiHeadAttention
from fms_extras.utils.cache.paged import PagedAttentionCacheData, PagedAttentionCacheDataLayer


@dataclass
class PagedMixtralConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 8
    nlayers: int = 32
    hidden_dim: int = 14336
    p_dropout: float = 0.0
    num_experts: int = 8
    top_k_experts: int = 2
    max_expected_seq_len: int = 32768
    rope_base: float = 1000000.0
    ntk_scaling: bool = False


class PagedMixtralBlock(nn.Module):
    def __init__(self, config: PagedMixtralConfig, rotary_emb: RotaryEmbedding):
        super(PagedMixtralBlock, self).__init__()
        self.config = config
        emb_kq = self.config.dim // self.config.nheads
        emb_v = self.config.dim // self.config.nheads

        self.ln = LayerNormParameterized(
            self.config.dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.dim,
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
            self.config.dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
        )
        self.ff_sub_layer = MOEFeedForward(
            self.config.num_experts,
            self.config.top_k_experts,
            self.config.dim,
            self.config.hidden_dim,
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


class PagedMixtralHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[PagedMixtralConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(PagedMixtralHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PagedMixtralConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.dim
        self.max_expected_seq_len = self.config.max_expected_seq_len

        embedding = nn.Embedding(self.config.src_vocab_size, self.config.dim)
        self.embedding = self.distributed_strategy.distribute_module(embedding)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.dim // self.config.nheads,
            ratio=self.config.rope_base,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
        )

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = PagedMixtralBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.dim,
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

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.dim**-0.5
        )

        # RoPE init
        if isinstance(self.distributed_strategy, UniformModelParallelStrategy):
            for dev_idx in set(self.distributed_strategy.layer_to_device):
                self.rot_emb.compute_freqs_cis(
                    torch.device("cuda", dev_idx), self.config.max_expected_seq_len
                )
        else:
            self.rot_emb.compute_freqs_cis(
                self.embedding.weight.device, self.config.max_expected_seq_len
            )

        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, MOEFeedForward)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

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

        x = self.embedding(x_in)

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
                x, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x = output

        dec_out = x
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class PagedMixtral(nn.Module):
    def __init__(
        self,
        config: Optional[PagedMixtralConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(PagedMixtral, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PagedMixtralConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = PagedMixtralHeadless(self.config, self.distributed_strategy)
        head = LinearClassificationHead(
            self.config.dim, self.config.src_vocab_size, bias=False
        )
        self.head = self.distributed_strategy.distribute_module(head)

    def get_config(self) -> PagedMixtralConfig:
        return self.config

    @classmethod
    def from_config(cls, config: PagedMixtralConfig) -> "PagedMixtral":
        return cls(config)

    def reset_parameters(self):
        # We're just going to down-scale the final prediction head to be
        # mixed-fan (inputs and gradients scale to the same inverse factors) if it isn't tied
        self.head.weight.data.normal_(
            0, 1 / math.sqrt(math.sqrt(self.config.dim * self.config.src_vocab_size))
        )

        # Call reset_parameters for relevant sub-layers
        self.base_model.reset_parameters()

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
        main forward pass for a paged mixtral model

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
        embeds, cache = self.base_model(
            x,
            mask,
            cache_data,
            use_cache,
            attn_algorithm,
        )

        if only_last_token:
            embeds = embeds[:, -1, :]

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

# Register common Mixtral variants with the model registration API
_8x7b_config = PagedMixtralConfig()

_architecture_name = "paged_mixtral"


def _mixtral_factory_factory(config):
    def factory(**kwargs):
        return PagedMixtral(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "8x7b", _mixtral_factory_factory(_8x7b_config)
)

_convert_to_fused_qkv = serialization._legacy_attn_unfused_to_fused_adapter


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"output.weight", "head.weight"),
        (r"tok_embeddings.weight", "base_model.embedding.weight"),
        (r"^norm", "base_model.dec_norm"),
        (r"^layers", "base_model.layers"),
        (r"attention\.wk", "attn.key"),
        (r"attention\.wv", "attn.value"),
        (r"attention\.wq", "attn.query"),
        (r"attention\.wo", "attn.dense"),
        (r"block_sparse_moe\.w1", "ff_sub_layer.cond_ffn.w1"),
        (r"block_sparse_moe\.w2", "ff_sub_layer.cond_ffn.w2"),
        (r"block_sparse_moe\.w3", "ff_sub_layer.cond_ffn.w3"),
        (r"block_sparse_moe\.gate", "ff_sub_layer.gate"),
        (r"attention_norm", "ln"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}

    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

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

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([raw_mapping[w] for w in unfused_weights], dim=0)

        if "gate" in new_name:
            weight_name = name.replace("gate", "w1")[:-7]
            if weight_name not in hf_sd:
                missing_weights = [
                    name.replace("gate", "w1")[:-7],
                    name.replace("gate", "w2")[:-7],
                    name.replace("gate", "w3")[:-7],
                ]
                raise ValueError(f"Missing {missing_weights}")

        if "w1" in new_name or "w2" in new_name or "w3" in new_name:
            gate_name = re.sub(r"w\d", "gate", name) + ".weight"
            if gate_name not in hf_sd:
                missing_weights = [
                    gate_name,
                    re.sub(r"w\d", "w1", name),
                    re.sub(r"w\d", "w2", name),
                    re.sub(r"w\d", "w3", name),
                ]
                missing_weights = [w for w in missing_weights if w != name]
                raise ValueError(f"Missing {missing_weights}")
            num_experts = hf_sd[gate_name].size(0)
            temp = new_sd[new_name]
            new_sd[new_name] = temp.reshape(
                num_experts, temp.size(0) // num_experts, temp.size(1)
            ).contiguous()

    for key in list(new_sd.keys()):
        if key not in new_sd:
            continue
        if "gate" in key:
            new_sd[key] = new_sd[key].contiguous()
        if "w1" in key:
            w3_weight = key.replace("w1", "w3")
            fused_name = key.replace("w1", "w13")
            new_sd[fused_name] = torch.cat([new_sd[key], new_sd[w3_weight]], dim=1)
            del new_sd[key]
            del new_sd[w3_weight]
        if "w2" in key:
            new_sd[key] = new_sd[key].transpose(1, 2).contiguous()

    return new_sd


serialization.register_adapter("mixtral", "hf", _hf_sd_to_fms_sd)