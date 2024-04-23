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

1. `MLPSpeculator`: a lightweight speculator model that can be used along-side a generative model to speed up inference (currently deployed in IBM TGIS with training in https://github.com/foundation-model-stack/fms-fsdp)
2. `PagedKVCacheManager`: an implementation of kv-cache management that provides a user with the proper input to use paged-attention with their own models (currently deployed in IBM TGIS)
3. `PagedLLaMA`: a LLaMA implementation that uses paged-attention in Multi-Head Attention. This model is compilable without graph breaks.
4. `speculative generation`: a reference implementation of speculative generate using PagedKVCacheManager and MLPSpeculator

## Structure and contents of this Repository

This repo follows the same structure to that of [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack)

* `fms_extras/models/` - Pure pytorch implementations of popular model architectures, without requiring any specific common interface beyond `nn.Module`. Each model configuration is registered with `fms.models.register_model()` so that instances can be obtained through `fms.models.get_model('architecture', 'variant', '/path/to/data')`. Each model can also register sources/formats/versions of data to load (e.g. checkpoints provided by meta, HF, or trained from this repo). Users of the repo (e.g. `fms-extras`) can register their own model architectures as well.
* `fms_extras/models/hf/` - Adapters that compose our native PyTorch FMS model architecture implementations in HF-compatible wrapper interfaces. Each FMS model implements an adapter, and adapted instances are obtained via `fms.models.hf.to_hf_api(model)`
* `fms_extras/utils/` - Other operators useful in working with LLMs. These include a `generate()` function, `Tensor` subclasses, code for dealing with LLM checkpoints that might be saved/sharded in a variety of formats, tokenization code, and various other useful helper functions.
* `scripts/` - Various scripts for inference, benchmarking, and evaluation, as well as an entry-point for tuning/training.

## References

Huggingface TGI: https://github.com/huggingface/text-generation-inference
IBM TGIS: https://github.com/IBM/text-generation-inference