# fms-extras

This is a repo as part of the foundation-model-stack organization which is used for new features staged to be integrated 
with [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack). This repo is the home
for extensions, research and/or in-development work, and fms-based models trained by IBM.

## Installation

### Local

```bash
pip install -e .
```

## Notable Features

1. `MLPSpeculator`: a lightweight speculator model that can be used along-side a generative model to speed up inference (currently deployed in IBM TGIS with training in [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp))
2. `PagedKVCacheManager`: an implementation of kv-cache management that provides a user with the proper input to use paged-attention with their own models (currently deployed in IBM TGIS)
3. `PagedLLaMA`: a LLaMA implementation that uses paged-attention in Multi-Head Attention. This model is compilable without graph breaks.
4. `speculative generation`: a reference implementation of speculative generate using PagedKVCacheManager and MLPSpeculator

## Structure and contents of this Repository

This repo follows a similar structure to that of [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack)

* `fms_extras/models/` - Pure pytorch implementations of popular model architectures, without requiring any specific common interface beyond `nn.Module`. Each model configuration is registered with `fms.models.register_model()` so that instances can be obtained through `fms.models.get_model('architecture', 'variant', '/path/to/data')`. Each model can also register sources/formats/versions of data to load (e.g. checkpoints provided by meta, HF, or trained from this repo).
* `fms_extras/models/hf/` - Adapters that compose our native PyTorch FMS model architecture implementations in HF-compatible wrapper interfaces. Each FMS model implements an adapter, and adapted instances are obtained via `fms.models.hf.to_hf_api(model)`
* `fms_extras/utils/` - Other operators useful in working with LLMs. These include a `speculative_generate()` function, `PagedKVCacheManager` class for easy-to-use kv-cache management with paged attention kernels, etc.
* `scripts/` - Various scripts for inference (paged generation and speculative generation)
* `csrc/` - Custom kernels used in fms-extra, currently related to paged-attention

## References

- Huggingface TGI: https://github.com/huggingface/text-generation-inference
- IBM TGIS: https://github.com/IBM/text-generation-inference
