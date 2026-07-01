#pragma once

// Stage 0 baseline: CUDA WMMA Tensor Core GEMM.
//
// This backend intentionally does not use CuTe, TMA descriptors, TCGen05
// intrinsics, or TMEM allocation. It is the project-owned correctness and
// performance baseline before the SM110-specific programming model is added.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

namespace tc0_detail {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

__global__ void tc0_wmma_gemm_kernel(const half* a, const half* b, float* d,
                                     int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda;

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int tiles_n = n / kWmmaN;
  const int tile_id =
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  const int tile_m = tile_id / tiles_n;
  const int tile_n = tile_id - tile_m * tiles_n;

  if (tile_m >= m / kWmmaM) {
    return;
  }

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k_begin = 0; k_begin < k; k_begin += kWmmaK) {
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                   wmma::row_major>
        b_fragment;

    const half* a_tile =
        a + static_cast<size_t>(tile_m * kWmmaM) * k + k_begin;
    const half* b_tile =
        b + static_cast<size_t>(k_begin) * n + tile_n * kWmmaN;
    wmma::load_matrix_sync(a_fragment, a_tile, k);
    wmma::load_matrix_sync(b_fragment, b_tile, n);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }

  float* d_tile =
      d + static_cast<size_t>(tile_m * kWmmaM) * n + tile_n * kWmmaN;
  wmma::store_matrix_sync(d_tile, accumulator, n, wmma::mem_row_major);
#else
  (void)a;
  (void)b;
  (void)d;
  (void)m;
  (void)n;
  (void)k;
#endif
}

inline void check_cuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                 cudaGetErrorString(status));
    std::abort();
  }
}

}  // namespace tc0_detail

class Tc0Runner {
 public:
  Tc0Runner(const half* a, const half* b, float* d, int m, int n, int k)
      : a_(a), b_(b), d_(d), m_(m), n_(n), k_(k) {
    if (m % tc0_detail::kWmmaM != 0 ||
        n % tc0_detail::kWmmaN != 0 ||
        k % tc0_detail::kWmmaK != 0) {
      std::fprintf(stderr,
                   "tc0 WMMA requires M, N, and K to be multiples of 16\n");
      std::abort();
    }
  }

  void launch() {
    const int output_tiles =
        (m_ / tc0_detail::kWmmaM) * (n_ / tc0_detail::kWmmaN);
    const int blocks =
        (output_tiles + tc0_detail::kWarpsPerBlock - 1) /
        tc0_detail::kWarpsPerBlock;
    tc0_detail::tc0_wmma_gemm_kernel<<<blocks,
                                      tc0_detail::kThreadsPerBlock>>>(
        a_, b_, d_, m_, n_, k_);
    tc0_detail::check_cuda(cudaGetLastError(),
                           "tc0_wmma_gemm_kernel launch");
  }

 private:
  const half* a_;
  const half* b_;
  float* d_;
  int m_;
  int n_;
  int k_;
};

}  // namespace gemm_sm110::backends
