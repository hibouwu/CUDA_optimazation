#pragma once

#include "gemm_common.cuh"
#include "tc3_gemm_kernel.cuh"

#include <cuda/barrier>
#include <cuda_fp16.h>

// tc4(sm110) = Blackwell mainloop rewrite scaffold.
//
// This file is intentionally a design skeleton only. The real launch path is
// not enabled yet. GEMMsm110 uses it to keep the tc4-only experiment tree
// separate from the broader GEMM directory.

enum class Tc4Sm110WarpRole : int {
  kMma = 0,
  kScheduler = 1,
  kMainloopLoad = 2,
  kEpilogueLoad = 3,
  kEpilogue0 = 4,
  kEpilogue1 = 5,
  kEpilogue2 = 6,
  kEpilogue3 = 7,
};

struct Tc4Sm110Shape {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 64;
  static constexpr int kBlockK = 32;
  static constexpr int kStages = 3;
  static constexpr int kWarps = 8;
};

template <typename Shape = Tc4Sm110Shape>
constexpr size_t tc4_sm110_pipeline_smem_bytes() {
  const size_t mainloop_smem =
      static_cast<size_t>(Shape::kStages) * Shape::kBlockK *
      (Shape::kBlockM + Shape::kBlockN) * sizeof(half);
  const size_t epilogue_smem =
      static_cast<size_t>(Shape::kBlockM) * Shape::kBlockN * sizeof(float);
  return mainloop_smem + 2 * epilogue_smem;
}

__host__ __device__ constexpr bool tc4_sm110_launch_available() {
  return false;
}
