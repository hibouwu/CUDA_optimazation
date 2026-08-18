// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/case_runner.hpp"
#include "guide/dense_reference.hpp"
#include "guide/device_buffer.hpp"
#include "guide/test_support.hpp"

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace guide {

template <class ElementA_, class ElementB_, class ElementD_,
          class MmaTileShape_, class ClusterShape_,
          class MainloopSchedule_, class EpilogueSchedule_,
          int AlignmentA_, int AlignmentB_, int AlignmentD_>
struct DenseGemmConfig {
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
  using OperatorClass = cutlass::arch::OpClassTensorOp;
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

  static auto stride_a(int m, int, int k) {
    return cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  }
  static auto stride_b(int, int n, int k) {
    return cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  }
  static auto stride_c(int m, int n, int) {
    return cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  }
  static auto stride_d(int m, int n, int) {
    return cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  }
};

template <class Config>
VerificationResult verify_dense_cutlass(CaseDescriptor const& descriptor,
                                        std::uint64_t seed,
                                        double atol, double rtol) {
  VerificationResult result = require_sm110_device();
  if (result.status == Status::skip) return result;

  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementD = typename Config::ElementD;
  using Gemm = typename Config::Gemm;

  int const m = descriptor.problem_mnkl[0];
  int const n = descriptor.problem_mnkl[1];
  int const k = descriptor.problem_mnkl[2];

  auto stride_a = Config::stride_a(m, n, k);
  auto stride_b = Config::stride_b(m, n, k);
  auto stride_c = Config::stride_c(m, n, k);
  auto stride_d = Config::stride_d(m, n, k);
  auto layout_a = cute::make_layout(cute::make_shape(m, k, 1), stride_a);
  auto layout_b = cute::make_layout(cute::make_shape(n, k, 1), stride_b);
  auto layout_d = cute::make_layout(cute::make_shape(m, n, 1), stride_d);

  std::size_t const a_storage = static_cast<std::size_t>(cute::cosize(layout_a));
  std::size_t const b_storage = static_cast<std::size_t>(cute::cosize(layout_b));
  std::size_t const d_storage = static_cast<std::size_t>(cute::cosize(layout_d));
  std::vector<ElementA> host_a(a_storage);
  std::vector<ElementB> host_b(b_storage);
  std::vector<float> logical_a(static_cast<std::size_t>(m) * k);
  std::vector<float> logical_b(static_cast<std::size_t>(n) * k);

  for (int row = 0; row < m; ++row) {
    for (int kk = 0; kk < k; ++kk) {
      ElementA value(deterministic_value(row, kk, seed));
      host_a[static_cast<std::size_t>(layout_a(row, kk, 0))] = value;
      logical_a[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(value);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      ElementB value(deterministic_value(col + 1009, kk, seed ^ 0x6a09e667ULL));
      host_b[static_cast<std::size_t>(layout_b(col, kk, 0))] = value;
      logical_b[static_cast<std::size_t>(col) * k + kk] = static_cast<float>(value);
    }
  }

  auto expected = dense_reference_mnk(logical_a, logical_b, m, n, k);

  DeviceBuffer<ElementA> device_a(a_storage);
  DeviceBuffer<ElementB> device_b(b_storage);
  constexpr std::size_t guard = 32;
  ElementD sentinel(31.0f);
  std::vector<ElementD> host_d_guarded(d_storage + 2 * guard, sentinel);
  DeviceBuffer<ElementD> device_d(host_d_guarded.size());
  device_a.copy_from_host(host_a.data(), host_a.size());
  device_b.copy_from_host(host_b.data(), host_b.size());
  device_d.copy_from_host(host_d_guarded.data(), host_d_guarded.size());

  cutlass::KernelHardwareInfo hardware_info;
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  hardware_info.device_id = device;
  hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {device_a.get(), stride_a, device_b.get(), stride_b},
      {{1.0f, 0.0f}, nullptr, stride_c, device_d.get() + guard, stride_d},
      hardware_info};

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    result.status = Status::fail;
    result.message = "Gemm::can_implement: " + cutlass_status_message(status);
    return result;
  }
  std::size_t workspace_size = Gemm::get_workspace_size(arguments);
  DeviceBuffer<std::uint8_t> workspace(workspace_size);
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
  result.message = "full-output independent CPU reference and canaries passed";
  return result;
}

}  // namespace guide
