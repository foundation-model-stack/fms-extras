def register_fms_models():
    from fms.models.hf import _causal_lm_models, _fms_to_hf_adapt_map, _headless_models
    from fms.models.hf.utils import register_fms_models

    from fms_extras.models.calico import Calico
    from fms_extras.models.hf.modeling_calico_hf import (
        HFAdaptedCalicoForCausalLM,
        HFAdaptedCalicoHeadless,
    )

    # todo: should have a better registration method than this
    if HFAdaptedCalicoHeadless not in _headless_models:
        _headless_models.append(HFAdaptedCalicoHeadless)

    if HFAdaptedCalicoForCausalLM not in _causal_lm_models:
        _causal_lm_models.append(HFAdaptedCalicoForCausalLM)

    if Calico not in _fms_to_hf_adapt_map:
        _fms_to_hf_adapt_map[Calico] = HFAdaptedCalicoForCausalLM

    register_fms_models()
