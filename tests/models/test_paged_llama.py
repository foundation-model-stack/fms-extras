import re
import tempfile

import pytest
import torch
from fms.models import get_model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="must have cuda to run paged llama equivalency test",
)
def test_llama_and_paged_llama_equivalency():
    from fms_extras.models import paged_llama
    from fms_extras.utils.cache.paged import PagedKVCacheManager

    # note: changed micro to have nheads=2 to increase head size for paged attention kernels
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
    torch.set_grad_enabled(False)
    llama.eval()
    paged_llama.eval()

    kv_cache_manager = PagedKVCacheManager(
        paged_llama.config.nlayers,
        paged_llama.config.nheads,
        paged_llama.config.emb_dim,
        kv_heads=paged_llama.config.kvheads,
        dtype=torch.get_default_dtype(),
        total_num_gpu_blocks=100,
    )
    input_ids = torch.arange(0, 16, device="cuda").unsqueeze(0)
    cache_data = kv_cache_manager.allocate_tokens([input_ids.size(1)])
    position_ids = cache_data.compute_position_ids([input_ids.size(1)])

    prefill_llama, prefill_cache = llama.forward(
        input_ids, position_ids=position_ids, use_cache=True
    )
    prefill_paged_llama, _ = paged_llama.forward(
        input_ids, position_ids=position_ids, use_cache=True, cache_data=cache_data
    )

    torch.testing.assert_close(prefill_llama, prefill_paged_llama)

    input_ids = torch.argmax(prefill_llama[:, -1, :], dim=-1).unsqueeze(0).t()
    cache_data = kv_cache_manager.allocate_tokens([1], cache_data.sequence_ids)
    position_ids = cache_data.compute_position_ids([1])

    decode_llama, _ = llama.forward(
        input_ids,
        position_ids=position_ids,
        use_cache=True,
        past_key_value_states=prefill_cache,
    )
    decode_paged_llama, _ = paged_llama.forward(
        input_ids, position_ids=position_ids, use_cache=True, cache_data=cache_data
    )

    torch.testing.assert_close(decode_llama, decode_paged_llama)
