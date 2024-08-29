
#include "../machete_mm_launcher.cuh"

namespace machete {
template <typename Config, bool with_C, bool with_scales, bool with_zeropoints>
using Kernel = MacheteKernelTemplate<
    cutlass::bfloat16_t,  // ElementA
    cutlass::vllm_uint4b8_t,  // ElementB
    cutlass::bfloat16_t,  // ElementD
    float, // Accumulator
    cutlass::bfloat16_t, // Scales
    cutlass::bfloat16_t, // Zeropoints
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput,
    Config, with_C, with_scales, with_zeropoints>;


struct sch_128x16_1x1x1_TmaMI_TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

torch::Tensor 
impl_bf16u4b8bf16f32bf16bf16_sch_128x16_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args) {
  bool with_C = args.C.has_value(), with_scales = args.scales.has_value(),
       with_zeropoints = args.zeros.has_value();

  
  if (with_C == false
      && with_zeropoints == false
      && with_scales == true) {
      return run_impl<Kernel<sch_128x16_1x1x1_TmaMI_TmaCoop_streamK, false,
        true, false>>(args);
  }

  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "for the sake of compile times and binary size machete_mm(..) is "
      " not implemented for with_C=", with_C, ", with_scales=", with_scales, 
      ", with_zeropoints=", with_zeropoints, 
      " (for bf16u4b8bf16f32bf16bf16_sch_128x16_1x1x1_TmaMI_TmaCoop_streamK)");
}

struct sch_128x32_1x1x1_TmaMI_TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

torch::Tensor 
impl_bf16u4b8bf16f32bf16bf16_sch_128x32_1x1x1_TmaMI_TmaCoop_streamK(PyTorchArguments args) {
  bool with_C = args.C.has_value(), with_scales = args.scales.has_value(),
       with_zeropoints = args.zeros.has_value();

  
  if (with_C == false
      && with_zeropoints == false
      && with_scales == true) {
      return run_impl<Kernel<sch_128x32_1x1x1_TmaMI_TmaCoop_streamK, false,
        true, false>>(args);
  }

  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "for the sake of compile times and binary size machete_mm(..) is "
      " not implemented for with_C=", with_C, ", with_scales=", with_scales, 
      ", with_zeropoints=", with_zeropoints, 
      " (for bf16u4b8bf16f32bf16bf16_sch_128x32_1x1x1_TmaMI_TmaCoop_streamK)");
}


}; // namespace machete