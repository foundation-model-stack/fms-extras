
#include "../machete_prepack_launcher.cuh"

namespace machete {
using PrepackBDispatcher_ = PrepackBDispatcher<
  cutlass::bfloat16_t, // ElementA
  cutlass::vllm_uint8b128_t, // ElementB
  cutlass::bfloat16_t, // ElementD
  float, // Accumulator
  cutlass::bfloat16_t, // Scales
  cutlass::bfloat16_t>; // Zeropoints

using PrepackedLayoutB = PrepackedLayoutBTemplate<
  cutlass::bfloat16_t, // ElementA
  cutlass::vllm_uint8b128_t, // ElementB
  cutlass::bfloat16_t, // ElementD
  float, // Accumulator
  cutlass::layout::ColumnMajor,
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>;

template <>
torch::Tensor PrepackBDispatcher_::dispatch(torch::Tensor B) {
  return prepack_impl<PrepackedLayoutB>(B);
}
}; // namespace machete