import re
import tempfile

import torch
from fms.models import get_model
from fms.testing.comparison import get_signature
from fms.utils import serialization

from fms_extras.models import paged_llama


def __rename_weights_to_paged(orig_sd):
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = f"headless_model.{name}"
        new_sd[new_name] = param

        # llama in meta has unfused qkv attn weights, so these weights must be converted to fused weights in fms
        if (
            "attn.query" in new_name
            or "attn.key" in new_name
            or "attn.value" in new_name
        ):
            unfused_weights = [
                re.sub(r"attn.(query|key|value)", "attn.query", name),
                re.sub(r"attn.(query|key|value)", "attn.key", name),
                re.sub(r"attn.(query|key|value)", "attn.value", name),
            ]
            missing_weights = [w for w in unfused_weights if w not in orig_sd.keys()]
            if len(missing_weights) != 0:
                raise serialization.FusableWeightsMissingError(missing_weights)

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([orig_sd[w] for w in unfused_weights], dim=0)

    return new_sd


serialization.register_adapter("paged_llama", "fms_llama", __rename_weights_to_paged)


def test_llama_and_paged_llama_equivalency():
    llama = get_model("llama", "micro")

    with tempfile.TemporaryDirectory() as workdir:
        sd_path = f"{workdir}/model.pth"
        torch.save(llama.state_dict(), sd_path)

        paged_llama = get_model(
            "paged_llama", "micro", model_path=sd_path, source="fms_llama"
        )

    llama_signature = torch.tensor(get_signature(llama))
    paged_llama_signature = torch.tensor(get_signature(paged_llama))
    torch.testing.assert_close(llama_signature, paged_llama_signature)
