#pragma once

#include "../cublaslt_reference.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::backends {

namespace shapeopt_detail {

using gemm_sm110::references::EpilogueMode;

__device__ __forceinline__ float apply_epilogue(
    float value, int row, int col, int n, EpilogueMode mode,
    const float* bias, const float* residual) {
  switch (mode) {
    case EpilogueMode::kBias:
      return value + bias[col];
    case EpilogueMode::kRelu:
      return value > 0.0f ? value : 0.0f;
    case EpilogueMode::kGelu: {
      constexpr float kInvSqrt2 = 0.7071067811865475f;
      return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
    }
    case EpilogueMode::kResidual:
      return value + residual[static_cast<size_t>(row) * n + col];
    case EpilogueMode::kNone:
      return value;
  }
  return value;
}

template <int ColumnsPerBlock, int ThreadsPerBlock>
__global__ __launch_bounds__(ThreadsPerBlock)
void small_m_gemv_like_kernel(const half* a, const half* b_nk, float* d,
                              int m, int n, int k, EpilogueMode epilogue,
                              const float* bias, const float* residual) {
  const int row = static_cast<int>(blockIdx.y);
  const int col_base = static_cast<int>(blockIdx.x) * ColumnsPerBlock;
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= m) return;

  float accum[ColumnsPerBlock];
#pragma unroll
  for (int local_col = 0; local_col < ColumnsPerBlock; ++local_col) {
    accum[local_col] = 0.0f;
  }

  const half* row_a = a + static_cast<size_t>(row) * k;
  for (int kk = tid; kk < k; kk += ThreadsPerBlock) {
    const float a_value = __half2float(row_a[kk]);
#pragma unroll
    for (int local_col = 0; local_col < ColumnsPerBlock; ++local_col) {
      const int col = col_base + local_col;
      if (col < n) {
        const half* col_b = b_nk + static_cast<size_t>(col) * k;
        accum[local_col] += a_value * __half2float(col_b[kk]);
      }
    }
  }

  __shared__ float partial[ColumnsPerBlock][ThreadsPerBlock];
#pragma unroll
  for (int local_col = 0; local_col < ColumnsPerBlock; ++local_col) {
    partial[local_col][tid] = accum[local_col];
  }
  __syncthreads();

  for (int stride = ThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int local_col = 0; local_col < ColumnsPerBlock; ++local_col) {
        partial[local_col][tid] += partial[local_col][tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int local_col = 0; local_col < ColumnsPerBlock; ++local_col) {
      const int col = col_base + local_col;
      if (col < n) {
        const float value = apply_epilogue(
            partial[local_col][0], row, col, n, epilogue, bias, residual);
        d[static_cast<size_t>(row) * n + col] = value;
      }
    }
  }
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void small_m_warp_per_output_kernel(const half* a, const half* b_nk, float* d,
                                    int m, int n, int k,
                                    EpilogueMode epilogue, const float* bias,
                                    const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int row = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
  if (row >= m || col >= n) return;

  const half* row_a = a + static_cast<size_t>(row) * k;
  const half* col_b = b_nk + static_cast<size_t>(col) * k;
  float accum = 0.0f;
  const bool use_wide_unroll = (k & 3) == 0 && !(m == 1 && n > 128);
  if (use_wide_unroll) {
    for (int kk = lane * 4; kk < k; kk += 128) {
      const half2 a_values_0 =
          *reinterpret_cast<const half2*>(row_a + kk);
      const half2 b_values_0 =
          *reinterpret_cast<const half2*>(col_b + kk);
      const half2 a_values_1 =
          *reinterpret_cast<const half2*>(row_a + kk + 2);
      const half2 b_values_1 =
          *reinterpret_cast<const half2*>(col_b + kk + 2);
      const float2 a_float_0 = __half22float2(a_values_0);
      const float2 b_float_0 = __half22float2(b_values_0);
      const float2 a_float_1 = __half22float2(a_values_1);
      const float2 b_float_1 = __half22float2(b_values_1);
      accum += a_float_0.x * b_float_0.x + a_float_0.y * b_float_0.y +
               a_float_1.x * b_float_1.x + a_float_1.y * b_float_1.y;
    }
  } else if ((k & 1) == 0) {
    for (int kk = lane * 2; kk < k; kk += 64) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(row_a + kk);
      const half2 b_values =
          *reinterpret_cast<const half2*>(col_b + kk);
      const float2 a_float = __half22float2(a_values);
      const float2 b_float = __half22float2(b_values);
      accum += a_float.x * b_float.x + a_float.y * b_float.y;
    }
  } else {
    for (int kk = lane; kk < k; kk += 32) {
      accum += __half2float(row_a[kk]) * __half2float(col_b[kk]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum += __shfl_down_sync(0xffffffff, accum, offset);
  }

  if (lane == 0) {
    d[static_cast<size_t>(row) * n + col] =
        apply_epilogue(accum, row, col, n, epilogue, bias, residual);
  }
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void m1_warp_two_outputs_kernel(const half* a, const half* b_nk, float* d,
                                int n, int k, EpilogueMode epilogue,
                                const float* bias, const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int col0 =
      (static_cast<int>(blockIdx.x) * WarpsPerBlock + warp) * 2;
  const int col1 = col0 + 1;
  if (col0 >= n) return;

  const half* row_a = a;
  const half* col0_b = b_nk + static_cast<size_t>(col0) * k;
  const half* col1_b = col1 < n ? b_nk + static_cast<size_t>(col1) * k
                                : col0_b;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  if ((k & 1) == 0) {
    for (int kk = lane * 2; kk < k; kk += 64) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(row_a + kk);
      const half2 b0_values =
          *reinterpret_cast<const half2*>(col0_b + kk);
      const half2 b1_values =
          *reinterpret_cast<const half2*>(col1_b + kk);
      const float2 a_float = __half22float2(a_values);
      const float2 b0_float = __half22float2(b0_values);
      const float2 b1_float = __half22float2(b1_values);
      accum0 += a_float.x * b0_float.x + a_float.y * b0_float.y;
      accum1 += a_float.x * b1_float.x + a_float.y * b1_float.y;
    }
  } else {
    for (int kk = lane; kk < k; kk += 32) {
      const float a_value = __half2float(row_a[kk]);
      accum0 += a_value * __half2float(col0_b[kk]);
      accum1 += a_value * __half2float(col1_b[kk]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum0 += __shfl_down_sync(0xffffffff, accum0, offset);
    accum1 += __shfl_down_sync(0xffffffff, accum1, offset);
  }

  if (lane == 0) {
    d[col0] = apply_epilogue(accum0, 0, col0, n, epilogue, bias, residual);
    if (col1 < n) {
      d[col1] =
          apply_epilogue(accum1, 0, col1, n, epilogue, bias, residual);
    }
  }
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void m1_warp_four_outputs_kernel(const half* a, const half* b_nk, float* d,
                                 int n, int k, EpilogueMode epilogue,
                                 const float* bias, const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int col_base =
      (static_cast<int>(blockIdx.x) * WarpsPerBlock + warp) * 4;
  if (col_base >= n) return;

  const half* row_a = a;
  const half* col_b[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = col_base + i;
    col_b[i] = b_nk + static_cast<size_t>(col < n ? col : col_base) * k;
  }

  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if ((k & 1) == 0) {
    for (int kk = lane * 2; kk < k; kk += 64) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(row_a + kk);
      const float2 a_float = __half22float2(a_values);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const half2 b_values =
            *reinterpret_cast<const half2*>(col_b[i] + kk);
        const float2 b_float = __half22float2(b_values);
        accum[i] += a_float.x * b_float.x + a_float.y * b_float.y;
      }
    }
  } else {
    for (int kk = lane; kk < k; kk += 32) {
      const float a_value = __half2float(row_a[kk]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accum[i] += a_value * __half2float(col_b[i][kk]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      accum[i] += __shfl_down_sync(0xffffffff, accum[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = col_base + i;
      if (col < n) {
        d[col] = apply_epilogue(accum[i], 0, col, n, epilogue, bias,
                                residual);
      }
    }
  }
}

template <int OutputsPerBlock>
__global__ __launch_bounds__(OutputsPerBlock * 2 * 32)
void m1_two_warps_per_output_kernel(const half* a, const half* b_nk,
                                    float* d, int n, int k,
                                    EpilogueMode epilogue, const float* bias,
                                    const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int output_in_block = warp >> 1;
  const int k_partition = warp & 1;
  const int col = static_cast<int>(blockIdx.x) * OutputsPerBlock +
                  output_in_block;
  if (col >= n) return;

  const half* row_a = a;
  const half* col_b = b_nk + static_cast<size_t>(col) * k;
  float accum = 0.0f;
  if ((k & 1) == 0) {
    for (int kk = (k_partition * 32 + lane) * 2; kk < k; kk += 128) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(row_a + kk);
      const half2 b_values =
          *reinterpret_cast<const half2*>(col_b + kk);
      const float2 a_float = __half22float2(a_values);
      const float2 b_float = __half22float2(b_values);
      accum += a_float.x * b_float.x + a_float.y * b_float.y;
    }
  } else {
    for (int kk = k_partition * 32 + lane; kk < k; kk += 64) {
      accum += __half2float(row_a[kk]) * __half2float(col_b[kk]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum += __shfl_down_sync(0xffffffff, accum, offset);
  }

  __shared__ float partial[OutputsPerBlock][2];
  if (lane == 0) {
    partial[output_in_block][k_partition] = accum;
  }
  __syncthreads();

  if (lane == 0 && k_partition == 0) {
    const float value = partial[output_in_block][0] +
                        partial[output_in_block][1];
    d[col] = apply_epilogue(value, 0, col, n, epilogue, bias, residual);
  }
}

template <int OutputsPerBlock>
__global__ __launch_bounds__(OutputsPerBlock * 2 * 32)
void m1_two_warps_shared_a_kernel(const half* a, const half* b_nk,
                                  float* d, int n, int k,
                                  EpilogueMode epilogue, const float* bias,
                                  const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int output_in_block = warp >> 1;
  const int k_partition = warp & 1;
  const int col = static_cast<int>(blockIdx.x) * OutputsPerBlock +
                  output_in_block;
  if (col >= n) return;

  extern __shared__ half shared_a[];
  for (int kk = tid; kk < k; kk += OutputsPerBlock * 2 * 32) {
    shared_a[kk] = a[kk];
  }
  __syncthreads();

  const half* col_b = b_nk + static_cast<size_t>(col) * k;
  float accum = 0.0f;
  if ((k & 1) == 0) {
    for (int kk = (k_partition * 32 + lane) * 2; kk < k; kk += 128) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(shared_a + kk);
      const half2 b_values =
          *reinterpret_cast<const half2*>(col_b + kk);
      const float2 a_float = __half22float2(a_values);
      const float2 b_float = __half22float2(b_values);
      accum += a_float.x * b_float.x + a_float.y * b_float.y;
    }
  } else {
    for (int kk = k_partition * 32 + lane; kk < k; kk += 64) {
      accum += __half2float(shared_a[kk]) * __half2float(col_b[kk]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum += __shfl_down_sync(0xffffffff, accum, offset);
  }

  __shared__ float partial[OutputsPerBlock][2];
  if (lane == 0) {
    partial[output_in_block][k_partition] = accum;
  }
  __syncthreads();

  if (lane == 0 && k_partition == 0) {
    const float value = partial[output_in_block][0] +
                        partial[output_in_block][1];
    d[col] = apply_epilogue(value, 0, col, n, epilogue, bias, residual);
  }
}

template <int OutputsPerBlock>
__global__ __launch_bounds__(OutputsPerBlock * 4 * 32)
void m1_four_warps_per_output_kernel(const half* a, const half* b_nk,
                                     float* d, int n, int k,
                                     EpilogueMode epilogue, const float* bias,
                                     const float* residual) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int output_in_block = warp >> 2;
  const int k_partition = warp & 3;
  const int col = static_cast<int>(blockIdx.x) * OutputsPerBlock +
                  output_in_block;
  if (col >= n) return;

  const half* row_a = a;
  const half* col_b = b_nk + static_cast<size_t>(col) * k;
  float accum = 0.0f;
  if ((k & 1) == 0) {
    for (int kk = (k_partition * 32 + lane) * 2; kk < k; kk += 256) {
      const half2 a_values =
          *reinterpret_cast<const half2*>(row_a + kk);
      const half2 b_values =
          *reinterpret_cast<const half2*>(col_b + kk);
      const float2 a_float = __half22float2(a_values);
      const float2 b_float = __half22float2(b_values);
      accum += a_float.x * b_float.x + a_float.y * b_float.y;
    }
  } else {
    for (int kk = k_partition * 32 + lane; kk < k; kk += 128) {
      accum += __half2float(row_a[kk]) * __half2float(col_b[kk]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum += __shfl_down_sync(0xffffffff, accum, offset);
  }

  __shared__ float partial[OutputsPerBlock][4];
  if (lane == 0) {
    partial[output_in_block][k_partition] = accum;
  }
  __syncthreads();

  if (lane == 0 && k_partition == 0) {
    const float value = partial[output_in_block][0] +
                        partial[output_in_block][1] +
                        partial[output_in_block][2] +
                        partial[output_in_block][3];
    d[col] = apply_epilogue(value, 0, col, n, epilogue, bias, residual);
  }
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void small_m_wmma_kernel(const half* a, const half* b, float* d, int m, int n,
                         int k, EpilogueMode epilogue, const float* bias,
                         const float* residual) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  constexpr int kTile = 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int tile_n = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
  const int col_base = tile_n * kTile;
  if (col_base >= n) return;

  __shared__ half shared_a[WarpsPerBlock][kTile * kTile];
  __shared__ half shared_b[WarpsPerBlock][kTile * kTile];
  __shared__ float shared_c[WarpsPerBlock][kTile * kTile];

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k_base = 0; k_base < k; k_base += kTile) {
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
      const int tile_row = idx / kTile;
      const int tile_col = idx - tile_row * kTile;
      const int a_global_k = k_base + tile_col;
      shared_a[warp][idx] =
          (tile_row < m && a_global_k < k)
              ? a[static_cast<size_t>(tile_row) * k + a_global_k]
              : __float2half(0.0f);
      const int b_global_k = k_base + tile_row;
      const int col = col_base + tile_col;
      shared_b[warp][idx] =
          (b_global_k < k && col < n)
              ? b[static_cast<size_t>(b_global_k) * n + col]
              : __float2half(0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                   wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, shared_a[warp], kTile);
    wmma::load_matrix_sync(b_frag, shared_b[warp], kTile);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncwarp();
  }

  wmma::store_matrix_sync(shared_c[warp], acc, kTile, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane; idx < kTile * kTile; idx += 32) {
    const int row = idx / kTile;
    const int col = col_base + idx - row * kTile;
    if (row < m && col < n) {
      d[static_cast<size_t>(row) * n + col] =
          apply_epilogue(shared_c[warp][idx], row, col, n, epilogue, bias,
                         residual);
    }
  }
#else
  (void)a;
  (void)b;
  (void)d;
  (void)m;
  (void)n;
  (void)k;
  (void)epilogue;
  (void)bias;
  (void)residual;
#endif
}

inline void check_cuda(cudaError_t status, const char* where) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
               cudaGetErrorString(status));
  std::abort();
}

__global__ void transpose_row_major_kernel(const float* src, float* dst,
                                           int src_rows, int src_cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= src_rows || col >= src_cols) return;
  dst[static_cast<size_t>(col) * src_rows + row] =
      src[static_cast<size_t>(row) * src_cols + col];
}

inline void launch_transpose_row_major(const float* src, float* dst,
                                       int src_rows, int src_cols) {
  dim3 block(16, 16, 1);
  dim3 grid(ceil_div(src_cols, static_cast<int>(block.x)),
            ceil_div(src_rows, static_cast<int>(block.y)), 1);
  transpose_row_major_kernel<<<grid, block>>>(src, dst, src_rows, src_cols);
  check_cuda(cudaGetLastError(), "shapeopt transpose_row_major_kernel launch");
}

__global__ void pad_k_major_rows_kernel(const half* src, half* dst, int rows,
                                        int k, int padded_k) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || col >= padded_k) return;
  const half value =
      col < k ? src[static_cast<size_t>(row) * k + col] : __float2half(0.0f);
  dst[static_cast<size_t>(row) * padded_k + col] = value;
}

inline void launch_pad_k_major_rows(const half* src, half* dst, int rows,
                                    int k, int padded_k) {
  dim3 block(32, 8, 1);
  dim3 grid(ceil_div(padded_k, static_cast<int>(block.x)),
            ceil_div(rows, static_cast<int>(block.y)), 1);
  pad_k_major_rows_kernel<<<grid, block>>>(src, dst, rows, k, padded_k);
  check_cuda(cudaGetLastError(), "shapeopt pad_k_major_rows_kernel launch");
}

__global__ void pad_rows_kernel(const half* src, half* dst, int rows,
                                int padded_rows, int cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= padded_rows || col >= cols) return;
  const half value =
      row < rows ? src[static_cast<size_t>(row) * cols + col]
                 : __float2half(0.0f);
  dst[static_cast<size_t>(row) * cols + col] = value;
}

inline void launch_pad_rows(const half* src, half* dst, int rows,
                            int padded_rows, int cols) {
  dim3 block(32, 8, 1);
  dim3 grid(ceil_div(cols, static_cast<int>(block.x)),
            ceil_div(padded_rows, static_cast<int>(block.y)), 1);
  pad_rows_kernel<<<grid, block>>>(src, dst, rows, padded_rows, cols);
  check_cuda(cudaGetLastError(), "shapeopt pad_rows_kernel launch");
}

__global__ void pad_2d_rows_cols_kernel(const half* src, half* dst, int rows,
                                        int cols, int padded_rows,
                                        int padded_cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= padded_rows || col >= padded_cols) return;
  const half value =
      row < rows && col < cols
          ? src[static_cast<size_t>(row) * cols + col]
          : __float2half(0.0f);
  dst[static_cast<size_t>(row) * padded_cols + col] = value;
}

inline void launch_pad_2d_rows_cols(const half* src, half* dst, int rows,
                                    int cols, int padded_rows,
                                    int padded_cols) {
  dim3 block(32, 8, 1);
  dim3 grid(ceil_div(padded_cols, static_cast<int>(block.x)),
            ceil_div(padded_rows, static_cast<int>(block.y)), 1);
  pad_2d_rows_cols_kernel<<<grid, block>>>(
      src, dst, rows, cols, padded_rows, padded_cols);
  check_cuda(cudaGetLastError(),
             "shapeopt pad_2d_rows_cols_kernel launch");
}

__global__ void ragged_half2_dot_kernel(const half* a, const half* b_nk,
                                        float* d, int m, int n, int k) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  const half* row_a = a + static_cast<size_t>(row) * k;
  const half* col_b = b_nk + static_cast<size_t>(col) * k;
  float accum = 0.0f;
  int kk = 0;
  for (; kk + 1 < k; kk += 2) {
    const half2 a_values = *reinterpret_cast<const half2*>(row_a + kk);
    const half2 b_values = *reinterpret_cast<const half2*>(col_b + kk);
    const float2 a_float = __half22float2(a_values);
    const float2 b_float = __half22float2(b_values);
    accum += a_float.x * b_float.x + a_float.y * b_float.y;
  }
  if (kk < k) {
    accum += __half2float(row_a[kk]) * __half2float(col_b[kk]);
  }
  d[static_cast<size_t>(row) * n + col] = accum;
}

inline void launch_ragged_half2_dot(const half* a, const half* b_nk,
                                    float* d, int m, int n, int k) {
  dim3 block(16, 16, 1);
  dim3 grid(ceil_div(n, static_cast<int>(block.x)),
            ceil_div(m, static_cast<int>(block.y)), 1);
  ragged_half2_dot_kernel<<<grid, block>>>(a, b_nk, d, m, n, k);
  check_cuda(cudaGetLastError(), "shapeopt ragged_half2_dot_kernel launch");
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void ragged_padded_wmma_kernel(const half* a_padded,
                               const half* b_padded_row_major, float* d,
                               int m, int n, int padded_n, int padded_k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  constexpr int kTile = 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int tiles_n = padded_n / kTile;
  const int tile_id =
      static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
  const int tile_m = tile_id / tiles_n;
  const int tile_n = tile_id - tile_m * tiles_n;
  const int row_base = tile_m * kTile;
  const int col_base = tile_n * kTile;
  if (row_base >= m || col_base >= n) return;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k_base = 0; k_base < padded_k; k_base += kTile) {
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                   wmma::row_major>
        b_frag;
    const half* a_tile =
        a_padded + static_cast<size_t>(row_base) * padded_k + k_base;
    const half* b_tile =
        b_padded_row_major + static_cast<size_t>(k_base) * padded_n +
        col_base;
    wmma::load_matrix_sync(a_frag, a_tile, padded_k);
    wmma::load_matrix_sync(b_frag, b_tile, padded_n);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
  }

  if (col_base + kTile <= n) {
    wmma::store_matrix_sync(d + static_cast<size_t>(row_base) * n + col_base,
                            acc, n, wmma::mem_row_major);
  } else {
    __shared__ float tail_tile[WarpsPerBlock][kTile * kTile];
    wmma::store_matrix_sync(tail_tile[warp], acc, kTile,
                            wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
      const int row = idx / kTile;
      const int col = idx - row * kTile;
      if (row_base + row < m && col_base + col < n) {
        d[static_cast<size_t>(row_base + row) * n + col_base + col] =
            tail_tile[warp][idx];
      }
    }
  }
#else
  (void)a_padded;
  (void)b_padded_row_major;
  (void)d;
  (void)m;
  (void)n;
  (void)padded_n;
  (void)padded_k;
#endif
}

inline void launch_ragged_padded_wmma(const half* a_padded,
                                      const half* b_padded_row_major,
                                      float* d, int m, int n, int padded_n,
                                      int padded_k) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kTile = 16;
  const int tiles = ceil_div(m, kTile) * ceil_div(padded_n, kTile);
  dim3 block(kWarpsPerBlock * 32, 1, 1);
  dim3 grid(ceil_div(tiles, kWarpsPerBlock), 1, 1);
  ragged_padded_wmma_kernel<kWarpsPerBlock><<<grid, block>>>(
      a_padded, b_padded_row_major, d, m, n, padded_n, padded_k);
  check_cuda(cudaGetLastError(),
             "shapeopt ragged_padded_wmma_kernel launch");
}

__global__ __launch_bounds__(256)
void wmma_m64n32_shared_kernel(const half* a, const half* b_row_major,
                               float* d, int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 32;
  constexpr int kTile = 16;
  constexpr int kWarps = 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int block_m = static_cast<int>(blockIdx.y) * kBlockM;
  const int block_n = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ half shared_a[kBlockM * kTile];
  __shared__ half shared_b[kTile * kBlockN];

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;

  for (int k_base = 0; k_base < k; k_base += kTile) {
    for (int idx = tid; idx < kBlockM * kTile; idx += kWarps * 32) {
      const int row = idx / kTile;
      const int col = idx - row * kTile;
      shared_a[idx] =
          a[static_cast<size_t>(block_m + row) * k + k_base + col];
    }
    for (int idx = tid; idx < kTile * kBlockN; idx += kWarps * 32) {
      const int row = idx / kBlockN;
      const int col = idx - row * kBlockN;
      shared_b[idx] =
          b_row_major[static_cast<size_t>(k_base + row) * n +
                      block_n + col];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                   wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag,
                           shared_a + warp_m * kTile * kTile, kTile);
    wmma::load_matrix_sync(b_frag, shared_b + warp_n * kTile, kBlockN);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(
      d + static_cast<size_t>(block_m + warp_m * kTile) * n +
          block_n + warp_n * kTile,
      acc, n, wmma::mem_row_major);
#else
  (void)a;
  (void)b_row_major;
  (void)d;
  (void)m;
  (void)n;
  (void)k;
#endif
}

inline void launch_wmma_m64n32_shared(const half* a,
                                      const half* b_row_major, float* d,
                                      int m, int n, int k) {
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 32;
  dim3 block(256, 1, 1);
  dim3 grid(ceil_div(n, kBlockN), ceil_div(m, kBlockM), 1);
  wmma_m64n32_shared_kernel<<<grid, block>>>(a, b_row_major, d, m, n, k);
  check_cuda(cudaGetLastError(),
             "shapeopt wmma_m64n32_shared_kernel launch");
}

}  // namespace shapeopt_detail

class ShapeOptSpecializedRunner {
 public:
  ShapeOptSpecializedRunner(const half* a, const half* b_nk, float* d,
                            const half* b_row_major, int m, int n, int k,
                            shapeopt_detail::EpilogueMode epilogue,
                            const float* bias, const float* residual)
      : a_(a),
        b_nk_(b_nk),
        b_row_major_(b_row_major),
        d_(d),
        bias_(bias),
        residual_(residual),
        m_(m),
        n_(n),
        k_(k),
        epilogue_(epilogue) {
    if (!supports(m_, n_, k_, epilogue_, bias_, residual_)) {
      std::fprintf(stderr,
                   "ShapeOptSpecializedRunner does not support M=%d N=%d "
                   "K=%d epilogue=%d\n",
                   m_, n_, k_, static_cast<int>(epilogue_));
      std::abort();
    }
  }

  static bool supports(int m, int n, int k,
                       shapeopt_detail::EpilogueMode epilogue,
                       const float* bias, const float* residual) {
    if (m <= 0 || n <= 0 || k <= 0) return false;
    if (m > 16) return false;
    if (m != 1 && n > 128) return false;
    if (k < 512) return false;
    if (epilogue == shapeopt_detail::EpilogueMode::kBias && bias == nullptr) {
      return false;
    }
    if (epilogue == shapeopt_detail::EpilogueMode::kResidual &&
        residual == nullptr) {
      return false;
    }
    return true;
  }

  void launch() {
    if (m_ == 1 && n_ > 128) {
      if (n_ <= 4096) {
        int variant = 0;
        if (const char* env = std::getenv("SHAPEOPT_M1_VARIANT")) {
          variant = std::atoi(env);
        }
        switch (variant) {
          case 1:
            launch_warp_per_output<8>();
            break;
          case 2:
            launch_m1_two_outputs<16>();
            break;
          case 3:
            launch_m1_four_outputs<8>();
            break;
          case 4:
            launch_m1_four_warps_per_output<4>();
            break;
          case 5:
            launch_m1_two_warps_per_output<4>();
            break;
          case 6:
            launch_m1_two_warps_per_output<16>();
            break;
          case 7:
            launch_m1_two_warps_shared_a<8>();
            break;
          case 8:
            launch_m1_two_warps_shared_a<4>();
            break;
          case 9:
            launch_m1_two_warps_per_output<2>();
            break;
          case 10:
            launch_m1_two_warps_per_output<1>();
            break;
          case 11:
            launch_m1_two_warps_per_output<3>();
            break;
          case 12:
            launch_m1_two_warps_per_output<6>();
            break;
          default:
            launch_m1_two_warps_per_output<1>();
            break;
        }
      } else {
        launch_m1_two_outputs<16>();
      }
    } else {
      launch_warp_per_output<8>();
    }
  }

  const char* label() const {
    if (m_ == 1 && n_ > 128) {
      return n_ <= 4096 ? "ShapeOpt custom M1 two-warp GEMV"
                        : "ShapeOpt custom M1 two-output warp GEMV";
    }
    return "ShapeOpt custom small-M warp-output GEMV";
  }

 private:
  template <int WarpsPerBlock>
  void launch_m1_two_outputs() {
    dim3 block(WarpsPerBlock * 32, 1, 1);
    dim3 grid(ceil_div(n_, WarpsPerBlock * 2), 1, 1);
    shapeopt_detail::m1_warp_two_outputs_kernel<WarpsPerBlock>
        <<<grid, block>>>(a_, b_nk_, d_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(), "shapeopt m1_warp_two_outputs_kernel launch");
  }

  template <int WarpsPerBlock>
  void launch_m1_four_outputs() {
    dim3 block(WarpsPerBlock * 32, 1, 1);
    dim3 grid(ceil_div(n_, WarpsPerBlock * 4), 1, 1);
    shapeopt_detail::m1_warp_four_outputs_kernel<WarpsPerBlock>
        <<<grid, block>>>(a_, b_nk_, d_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(), "shapeopt m1_warp_four_outputs_kernel launch");
  }

  template <int OutputsPerBlock>
  void launch_m1_two_warps_per_output() {
    dim3 block(OutputsPerBlock * 2 * 32, 1, 1);
    dim3 grid(ceil_div(n_, OutputsPerBlock), 1, 1);
    shapeopt_detail::m1_two_warps_per_output_kernel<OutputsPerBlock>
        <<<grid, block>>>(a_, b_nk_, d_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(),
        "shapeopt m1_two_warps_per_output_kernel launch");
  }

  template <int OutputsPerBlock>
  void launch_m1_two_warps_shared_a() {
    dim3 block(OutputsPerBlock * 2 * 32, 1, 1);
    dim3 grid(ceil_div(n_, OutputsPerBlock), 1, 1);
    shapeopt_detail::m1_two_warps_shared_a_kernel<OutputsPerBlock>
        <<<grid, block, static_cast<size_t>(k_) * sizeof(half)>>>(
            a_, b_nk_, d_, n_, k_, epilogue_, bias_, residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(),
        "shapeopt m1_two_warps_shared_a_kernel launch");
  }

  template <int OutputsPerBlock>
  void launch_m1_four_warps_per_output() {
    dim3 block(OutputsPerBlock * 4 * 32, 1, 1);
    dim3 grid(ceil_div(n_, OutputsPerBlock), 1, 1);
    shapeopt_detail::m1_four_warps_per_output_kernel<OutputsPerBlock>
        <<<grid, block>>>(a_, b_nk_, d_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(),
        "shapeopt m1_four_warps_per_output_kernel launch");
  }

  template <int WarpsPerBlock>
  void launch_wmma() {
    dim3 block(WarpsPerBlock * 32, 1, 1);
    dim3 grid(ceil_div(n_, WarpsPerBlock * 16), 1, 1);
    shapeopt_detail::small_m_wmma_kernel<WarpsPerBlock>
        <<<grid, block>>>(a_, b_row_major_, d_, m_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(cudaGetLastError(),
                                "shapeopt small_m_wmma_kernel launch");
  }

  template <int WarpsPerBlock>
  void launch_warp_per_output() {
    dim3 block(WarpsPerBlock * 32, 1, 1);
    dim3 grid(ceil_div(n_, WarpsPerBlock), m_, 1);
    shapeopt_detail::small_m_warp_per_output_kernel<WarpsPerBlock>
        <<<grid, block>>>(a_, b_nk_, d_, m_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(),
        "shapeopt small_m_warp_per_output_kernel launch");
  }

  template <int ColumnsPerBlock>
  void launch_tiled() {
    constexpr int kThreads = 256;
    dim3 block(kThreads, 1, 1);
    dim3 grid(ceil_div(n_, ColumnsPerBlock), m_, 1);
    shapeopt_detail::small_m_gemv_like_kernel<ColumnsPerBlock, kThreads>
        <<<grid, block>>>(a_, b_nk_, d_, m_, n_, k_, epilogue_, bias_,
                          residual_);
    shapeopt_detail::check_cuda(
        cudaGetLastError(), "shapeopt small_m_gemv_like_kernel launch");
  }

  const half* a_ = nullptr;
  const half* b_nk_ = nullptr;
  const half* b_row_major_ = nullptr;
  float* d_ = nullptr;
  const float* bias_ = nullptr;
  const float* residual_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  shapeopt_detail::EpilogueMode epilogue_ =
      shapeopt_detail::EpilogueMode::kNone;
};

}  // namespace gemm_sm110::backends
