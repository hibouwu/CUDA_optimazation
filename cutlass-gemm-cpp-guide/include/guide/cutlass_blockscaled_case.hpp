// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/block_scale_reference.hpp"
#include "guide/case_runner.hpp"
#include "guide/dense_reference.hpp"
#include "guide/device_buffer.hpp"
#include "guide/scale_layout.hpp"
#include "guide/test_support.hpp"

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace guide {

template <class ElementA_, class ElementB_, class ElementD_,
          class MmaTileShape_, class ClusterShape_,
          class MainloopSchedule_, class EpilogueSchedule_,
          int AlignmentA_, int AlignmentB_, int AlignmentD_>
struct BlockScaledGemmConfig {
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = void;
  using ElementD = ElementD_;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = AlignmentA_;
  static constexpr int AlignmentB = AlignmentB_;
  static constexpr int AlignmentC = 1;
  static constexpr int AlignmentD = AlignmentD_;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule_>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
  using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

template <class Config>
VerificationResult verify_blockscaled_cutlass(CaseDescriptor const& descriptor,
                                              std::uint64_t seed,
                                              double atol, double rtol) {
  VerificationResult result = require_sm110_device();
  if (result.status == Status::skip) return result;

  using Gemm = typename Config::Gemm;
  using DataA = typename Config::ElementA::DataType;
  using DataB = typename Config::ElementB::DataType;
  using ScaleA = typename Config::ElementA::ScaleFactorType;
  using ScaleB = typename Config::ElementB::ScaleFactorType;
  using ElementD = typename Config::ElementD;

  int const m = descriptor.problem_mnkl[0];
  int const n = descriptor.problem_mnkl[1];
  int const k = descriptor.problem_mnkl[2];
  int constexpr sv = Config::ScaleConfig::SFVecSize;
  int const sfk = (k + sv - 1) / sv;

  auto stride_a = cutlass::make_cute_packed_stride(typename Config::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(typename Config::StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(typename Config::StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(typename Config::StrideD{}, {m, n, 1});
  auto layout_a = cute::make_layout(cute::make_shape(m, k, 1), stride_a);
  auto layout_b = cute::make_layout(cute::make_shape(n, k, 1), stride_b);
  auto layout_d = cute::make_layout(cute::make_shape(m, n, 1), stride_d);
  auto problem = cute::make_shape(m, n, k, 1);
  auto layout_sfa = Config::ScaleConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = Config::ScaleConfig::tile_atom_to_shape_SFB(problem);

  cutlass::HostTensor<DataA, cutlass::layout::PackedVectorLayout> block_a;
  cutlass::HostTensor<DataB, cutlass::layout::PackedVectorLayout> block_b;
  cutlass::HostTensor<ScaleA, cutlass::layout::PackedVectorLayout> block_sfa;
  cutlass::HostTensor<ScaleB, cutlass::layout::PackedVectorLayout> block_sfb;
  block_a.reset(cutlass::make_Coord(cute::size(layout_a)));
  block_b.reset(cutlass::make_Coord(cute::size(layout_b)));
  block_sfa.reset(cutlass::make_Coord(cute::size(cute::filter_zeros(layout_sfa))));
  block_sfb.reset(cutlass::make_Coord(cute::size(cute::filter_zeros(layout_sfb))));

  auto tensor_a = cute::make_tensor(cute::recast_ptr<DataA>(block_a.host_data()), layout_a);
  auto tensor_b = cute::make_tensor(cute::recast_ptr<DataB>(block_b.host_data()), layout_b);
  auto tensor_sfa = cute::make_tensor(block_sfa.host_data(), layout_sfa);
  auto tensor_sfb = cute::make_tensor(block_sfb.host_data(), layout_sfb);
  std::vector<float> logical_a(static_cast<std::size_t>(m) * k);
  std::vector<float> logical_b(static_cast<std::size_t>(n) * k);
  std::vector<float> logical_sfa(static_cast<std::size_t>(m) * sfk);
  std::vector<float> logical_sfb(static_cast<std::size_t>(n) * sfk);

  for (int row = 0; row < m; ++row) {
    for (int kk = 0; kk < k; ++kk) {
      DataA value(deterministic_value(row, kk, seed));
      tensor_a(row, kk, 0) = value;
      logical_a[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(value);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      DataB value(deterministic_value(col + 1009, kk, seed ^ 0xbb67ae85ULL));
      tensor_b(col, kk, 0) = value;
      logical_b[static_cast<std::size_t>(col) * k + kk] = static_cast<float>(value);
    }
  }
  for (int row = 0; row < m; ++row) {
    for (int sf = 0; sf < sfk; ++sf) {
      float requested = std::ldexp(1.0f, ((row + sf) % 3) - 1);
      ScaleA scale(requested);
      int kk = sf * sv;
      std::size_t closed_form = sm1xx_scale_offset(row, kk, m, k, sv);
      std::size_t cute_offset = static_cast<std::size_t>(layout_sfa(row, kk, 0));
      if (closed_form != cute_offset) {
        result.status = Status::fail;
        result.message = "independent SFA offset disagrees with CUTLASS layout";
        return result;
      }
      tensor_sfa(row, kk, 0) = scale;
      logical_sfa[static_cast<std::size_t>(row) * sfk + sf] = static_cast<float>(scale);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int sf = 0; sf < sfk; ++sf) {
      float requested = std::ldexp(1.0f, ((col + 2 * sf) % 3) - 1);
      ScaleB scale(requested);
      int kk = sf * sv;
      std::size_t closed_form = sm1xx_scale_offset(col, kk, n, k, sv);
      std::size_t cute_offset = static_cast<std::size_t>(layout_sfb(col, kk, 0));
      if (closed_form != cute_offset) {
        result.status = Status::fail;
        result.message = "independent SFB offset disagrees with CUTLASS layout";
        return result;
      }
      tensor_sfb(col, kk, 0) = scale;
      logical_sfb[static_cast<std::size_t>(col) * sfk + sf] = static_cast<float>(scale);
    }
  }

  block_a.sync_device();
  block_b.sync_device();
  block_sfa.sync_device();
  block_sfb.sync_device();
  auto expected = block_scaled_reference_mnk(logical_a, logical_b, logical_sfa, logical_sfb,
                                             m, n, k, sv);

  std::size_t d_storage = static_cast<std::size_t>(cute::cosize(layout_d));
  constexpr std::size_t guard = 32;
  ElementD sentinel(31.0f);
  std::vector<ElementD> host_d_guarded(d_storage + 2 * guard, sentinel);
  DeviceBuffer<ElementD> device_d(host_d_guarded.size());
  device_d.copy_from_host(host_d_guarded.data(), host_d_guarded.size());

  cutlass::KernelHardwareInfo hardware_info;
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  hardware_info.device_id = device;
  hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {block_a.device_data(), stride_a,
       block_b.device_data(), stride_b,
       block_sfa.device_data(), layout_sfa,
       block_sfb.device_data(), layout_sfb},
      {{1.0f, 0.0f}, nullptr, stride_c, device_d.get() + guard, stride_d},
      hardware_info};

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    result.status = Status::fail;
    result.message = "Gemm::can_implement: " + cutlass_status_message(status);
    return result;
  }
  DeviceBuffer<std::uint8_t> workspace(Gemm::get_workspace_size(arguments));
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    result.status = Status::fail;
    result.message = "Gemm::initialize: " + cutlass_status_message(status);
    return result;
  }
  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    result.status = Status::fail;
    result.message = "Gemm::run: " + cutlass_status_message(status);
    return result;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  device_d.copy_to_host(host_d_guarded.data(), host_d_guarded.size());

  for (std::size_t i = 0; i < guard; ++i) {
    if (static_cast<float>(host_d_guarded[i]) != static_cast<float>(sentinel) ||
        static_cast<float>(host_d_guarded[guard + d_storage + i]) != static_cast<float>(sentinel)) {
      result.status = Status::fail;
      result.message = "output canary was modified";
      return result;
    }
  }
  std::vector<ElementD> observed(static_cast<std::size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      observed[static_cast<std::size_t>(row) * n + col] =
          host_d_guarded[guard + static_cast<std::size_t>(layout_d(row, col, 0))];
    }
  }
  ErrorMetrics metrics = compare_full(expected, observed);
  result.max_abs_error = metrics.max_abs;
  result.max_rel_error = metrics.max_rel;
  if (!metrics.finite || (metrics.max_abs > atol && metrics.max_rel > rtol)) {
    std::ostringstream message;
    message << "full-output mismatch at logical index " << metrics.max_abs_index
            << ", max_abs=" << metrics.max_abs << ", max_rel=" << metrics.max_rel;
    result.status = Status::fail;
    result.message = message.str();
    return result;
  }
  result.status = Status::pass;
  result.message = "logical-scale CPU oracle, physical layout, full output and canaries passed";
  return result;
}

}  // namespace guide
