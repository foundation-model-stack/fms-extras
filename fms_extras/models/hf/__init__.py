def register_fms_models():
    from fms.models.hf import _causal_lm_models, _fms_to_hf_adapt_map, _headless_models
    from fms.models.hf.utils import register_fms_models

    from fms_extras.models.hf.modeling_sphinx_hf import (
        HFAdaptedSphinxForCausalLM,
        HFAdaptedSphinxHeadless,
    )
    from fms_extras.models.sphinx import Sphinx

    # todo: should have a better registration method than this
    if HFAdaptedSphinxHeadless not in _headless_models:
        _headless_models.append(HFAdaptedSphinxHeadless)

    if HFAdaptedSphinxForCausalLM not in _causal_lm_models:
        _causal_lm_models.append(HFAdaptedSphinxForCausalLM)

    if Sphinx not in _fms_to_hf_adapt_map:
        _fms_to_hf_adapt_map[Sphinx] = HFAdaptedSphinxForCausalLM

    register_fms_models()
