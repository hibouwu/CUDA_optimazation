// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/case_runner.hpp"
#include "guide/dense_reference.hpp"
#include "guide/device_buffer.hpp"
#include "guide/test_support.hpp"

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/half.h>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

namespace guide {

struct BiasReluF16Config {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementBias = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementBias, ElementC>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 8,
      ElementD, LayoutD, 8,
      cutlass::epilogue::TmaWarpSpecialized1Sm,
      FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

inline VerificationResult verify_bias_relu_f16(CaseDescriptor const& descriptor,
                                               std::uint64_t seed) {
  VerificationResult result = require_sm110_device();
  if (result.status == Status::skip) return result;
  using Config = BiasReluF16Config;
  using Gemm = Config::Gemm;
  using StrideA = typename Config::GemmKernel::StrideA;
  using StrideB = typename Config::GemmKernel::StrideB;
  using StrideC = typename Config::GemmKernel::StrideC;
  using StrideD = typename Config::GemmKernel::StrideD;
  int m = descriptor.problem_mnkl[0], n = descriptor.problem_mnkl[1], k = descriptor.problem_mnkl[2];

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_a = cute::make_layout(cute::make_shape(m, k, 1), stride_a);
  auto layout_b = cute::make_layout(cute::make_shape(n, k, 1), stride_b);
  auto layout_c = cute::make_layout(cute::make_shape(m, n, 1), stride_c);
  auto layout_d = cute::make_layout(cute::make_shape(m, n, 1), stride_d);

  std::vector<Config::ElementA> host_a(static_cast<std::size_t>(cute::cosize(layout_a)));
  std::vector<Config::ElementB> host_b(static_cast<std::size_t>(cute::cosize(layout_b)));
  std::vector<Config::ElementC> host_c(static_cast<std::size_t>(cute::cosize(layout_c)), Config::ElementC(0));
  std::vector<Config::ElementBias> host_bias(m);
  std::vector<float> logical_a(static_cast<std::size_t>(m) * k);
  std::vector<float> logical_b(static_cast<std::size_t>(n) * k);
  for (int row = 0; row < m; ++row) {
    host_bias[row] = Config::ElementBias((row % 7 - 3) * 0.25f);
    for (int kk = 0; kk < k; ++kk) {
      Config::ElementA value(deterministic_value(row, kk, seed));
      host_a[static_cast<std::size_t>(layout_a(row, kk, 0))] = value;
      logical_a[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(value);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      Config::ElementB value(deterministic_value(col + 1009, kk, seed ^ 0x3c6ef372ULL));
      host_b[static_cast<std::size_t>(layout_b(col, kk, 0))] = value;
      logical_b[static_cast<std::size_t>(col) * k + kk] = static_cast<float>(value);
    }
  }
  auto expected = dense_reference_mnk(logical_a, logical_b, m, n, k);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      auto index = static_cast<std::size_t>(row) * n + col;
      expected[index] = std::max(0.0f, expected[index] + static_cast<float>(host_bias[row]));
    }
  }

  DeviceBuffer<Config::ElementA> device_a(host_a.size());
  DeviceBuffer<Config::ElementB> device_b(host_b.size());
  DeviceBuffer<Config::ElementC> device_c(host_c.size());
  DeviceBuffer<Config::ElementBias> device_bias(host_bias.size());
  DeviceBuffer<Config::ElementD> device_d(static_cast<std::size_t>(cute::cosize(layout_d)));
  device_a.copy_from_host(host_a.data(), host_a.size());
  device_b.copy_from_host(host_b.data(), host_b.size());
  device_c.copy_from_host(host_c.data(), host_c.size());
  device_bias.copy_from_host(host_bias.data(), host_bias.size());

  typename Config::CollectiveEpilogue::Arguments epilogue{};
  epilogue.thread.alpha = 1.0f;
  epilogue.thread.beta = 0.0f;
  epilogue.thread.bias_ptr = device_bias.get();
  epilogue.ptr_C = device_c.get();
  epilogue.dC = stride_c;
  epilogue.ptr_D = device_d.get();
  epilogue.dD = stride_d;

  cutlass::KernelHardwareInfo hardware_info;
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  hardware_info.device_id = device;
  hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {device_a.get(), stride_a, device_b.get(), stride_b},
      epilogue,
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
  if (status == cutlass::Status::kSuccess) status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    result.status = Status::fail;
    result.message = "CUTLASS fused epilogue: " + cutlass_status_message(status);
    return result;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  std::vector<Config::ElementD> observed_storage(static_cast<std::size_t>(cute::cosize(layout_d)));
  device_d.copy_to_host(observed_storage.data(), observed_storage.size());
  std::vector<Config::ElementD> observed(static_cast<std::size_t>(m) * n);
  for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
    observed[static_cast<std::size_t>(row) * n + col] =
        observed_storage[static_cast<std::size_t>(layout_d(row, col, 0))];
  }
  auto metrics = compare_full(expected, observed);
  result.max_abs_error = metrics.max_abs;
  result.max_rel_error = metrics.max_rel;
  if (!metrics.finite || (metrics.max_abs > 0.05 && metrics.max_rel > 0.02)) {
    result.status = Status::fail;
    result.message = "fused bias+ReLU full-output mismatch";
    return result;
  }
  result.status = Status::pass;
  result.message = "fused per-row bias + ReLU matched the independent CPU oracle";
  return result;
}

}  // namespace guide
