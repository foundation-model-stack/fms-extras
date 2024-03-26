import re
import tempfile

import pytest
import torch
from fms.models import get_model
from fms.utils import serialization
from fms.utils.generation import generate


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="must have cuda to run paged llama generation test",
)
def test_paged_generate():
    from fms_extras.models import paged_llama
    from fms_extras.utils.cache.paged import PagedKVCacheManager
    from fms_extras.utils.generation import paged_generate

    torch.set_grad_enabled(False)

    llama = get_model("llama", "micro", device_type="cuda", nheads=2)

    with tempfile.TemporaryDirectory() as workdir:
        sd_path = f"{workdir}/model.pth"
        torch.save(llama.state_dict(), sd_path)

        paged_llama = get_model(
            "paged_llama",
            "micro",
            model_path=sd_path,
            source="fms_llama",
            device_type="cuda",
            nheads=2,
        )

    kv_cache_manager = PagedKVCacheManager(
        paged_llama.config.nlayers,
        paged_llama.config.nheads,
        paged_llama.config.emb_dim,
        kv_heads=paged_llama.config.kvheads,
        dtype=torch.get_default_dtype(),
        total_num_gpu_blocks=100,
    )

    input_ids = torch.tensor(
        [1] + [i for i in range(5, 25)], dtype=torch.long, device="cuda"
    )

    paged_result, _, _, _ = paged_generate(
        paged_llama, [input_ids], kv_cache_manager, do_sample=False
    )

    result = generate(llama, input_ids.unsqueeze(0), do_sample=False)

    torch.testing.assert_close(paged_result, result)
