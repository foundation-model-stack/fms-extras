
#include "../machete_mm_launcher.cuh"

namespace machete {
using GemmDispatcher_ = GemmDispatcher<
    cutlass::half_t,  // ElementA
    uint8_t,  // ElementB
    cutlass::half_t,  // ElementD
    float, // Accumulator
    cutlass::half_t, // Scales
    cutlass::half_t>; // Zeropoints

extern torch::Tensor 
impl_f16u8f16f32f16f16_sch_128x16_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args);
extern torch::Tensor 
impl_f16u8f16f32f16f16_sch_128x32_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args);
extern torch::Tensor 
impl_f16u8f16f32f16f16_sch_128x64_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args);
extern torch::Tensor 
impl_f16u8f16f32f16f16_sch_128x128_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args);

template <>
torch::Tensor GemmDispatcher_::dispatch(PyTorchArguments args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.schedule) {
    if (M > 64)
        return impl_f16u8f16f32f16f16_sch_128x128_1x1x1_TmaMI_TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u8f16f32f16f16_sch_128x64_1x1x1_TmaMI_TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u8f16f32f16f16_sch_128x32_1x1x1_TmaMI_TmaCoop_streamK(args);
    else
        return impl_f16u8f16f32f16f16_sch_128x16_1x1x1_TmaMI_TmaCoop_streamK(args);
  }

  
  if (*args.schedule == "128x16_1x1x1_TmaMI_TmaCoop_streamK") {
    return impl_f16u8f16f32f16f16_sch_128x16_1x1x1_TmaMI_TmaCoop_streamK(args);
  }
  
  if (*args.schedule == "128x32_1x1x1_TmaMI_TmaCoop_streamK") {
    return impl_f16u8f16f32f16f16_sch_128x32_1x1x1_TmaMI_TmaCoop_streamK(args);
  }
  
  if (*args.schedule == "128x64_1x1x1_TmaMI_TmaCoop_streamK") {
    return impl_f16u8f16f32f16f16_sch_128x64_1x1x1_TmaMI_TmaCoop_streamK(args);
  }
  
  if (*args.schedule == "128x128_1x1x1_TmaMI_TmaCoop_streamK") {
    return impl_f16u8f16f32f16f16_sch_128x128_1x1x1_TmaMI_TmaCoop_streamK(args);
  }
  
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.schedule);
}

template <>
std::vector<std::string> GemmDispatcher_::supported_schedules() {
  return { 
    "128x16_1x1x1_TmaMI_TmaCoop_streamK",
    "128x32_1x1x1_TmaMI_TmaCoop_streamK",
    "128x64_1x1x1_TmaMI_TmaCoop_streamK",
    "128x128_1x1x1_TmaMI_TmaCoop_streamK"
  };
}

}; // namespace machete