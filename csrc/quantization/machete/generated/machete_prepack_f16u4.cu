
#include "../machete_prepack_launcher.cuh"

namespace machete {
using PrepackBDispatcher_ = PrepackBDispatcher<
  cutlass::half_t, // ElementA
  cutlass::uint4b_t, // ElementB
  cutlass::half_t, // ElementD
  float, // Accumulator
  cutlass::half_t, // Scales
  cutlass::half_t>; // Zeropoints

using PrepackedLayoutB = PrepackedLayoutBTemplate<
  cutlass::half_t, // ElementA
  cutlass::uint4b_t, // ElementB
  cutlass::half_t, // ElementD
  float, // Accumulator
  cutlass::layout::ColumnMajor,
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>;

template <>
torch::Tensor PrepackBDispatcher_::dispatch(torch::Tensor B) {
  return prepack_impl<PrepackedLayoutB>(B);
}
}; // namespace machete