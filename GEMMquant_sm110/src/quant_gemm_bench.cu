#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                << " code=" << static_cast<int>(err__)                       \
                << " name=" << cudaGetErrorName(err__)                       \
                << " msg=" << cudaGetErrorString(err__) << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                << " status=" << static_cast<int>(status__) << std::endl;    \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

float gflops(int n, float ms) {
  return 2.0f * static_cast<float>(n) * n * n / (ms * 1.0e6f);
}

void fill_float_inputs(std::vector<float>& a, std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.0625f;
  }
}

std::vector<__nv_fp8_e4m3> to_fp8_e4m3(
    const std::vector<float>& input) {
  std::vector<__nv_fp8_e4m3> out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = __nv_fp8_e4m3(input[i]);
  }
  return out;
}

std::vector<int8_t> make_int8_input(size_t elements, int modulus,
                                    int offset) {
  std::vector<int8_t> out(elements);
  for (size_t i = 0; i < elements; ++i) {
    out[i] = static_cast<int8_t>(static_cast<int>(i % modulus) - offset);
  }
  return out;
}

std::vector<int8_t> transpose_int8_square(const std::vector<int8_t>& input,
                                          int n) {
  std::vector<int8_t> out(input.size());
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      out[static_cast<size_t>(col) * n + row] =
          input[static_cast<size_t>(row) * n + col];
    }
  }
  return out;
}

__global__ void fp8_naive_kernel(const __nv_fp8_e4m3* a,
                                 const __nv_fp8_e4m3* b, float* c, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= n) return;

  float acc = 0.0f;
  for (int kk = 0; kk < n; ++kk) {
    const float av = static_cast<float>(a[static_cast<size_t>(row) * n + kk]);
    const float bv = static_cast<float>(b[static_cast<size_t>(kk) * n + col]);
    acc += av * bv;
  }
  c[static_cast<size_t>(row) * n + col] = acc;
}

__global__ void int8_naive_kernel(const int8_t* a, const int8_t* b,
                                  int32_t* c, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= n) return;

  int32_t acc = 0;
  for (int kk = 0; kk < n; ++kk) {
    const int32_t av = static_cast<int32_t>(a[static_cast<size_t>(row) * n + kk]);
    const int32_t bv = static_cast<int32_t>(b[static_cast<size_t>(kk) * n + col]);
    acc += av * bv;
  }
  c[static_cast<size_t>(row) * n + col] = acc;
}

__global__ void fp8_vec4cols_kernel(const __nv_fp8_e4m3* a,
                                    const __nv_fp8_e4m3* b, float* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (row >= n || col_base >= n) return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int kk = 0; kk < n; ++kk) {
    const float av = static_cast<float>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    acc0 += av * static_cast<float>(b[b_base]);
    if (col_base + 1 < n) acc1 += av * static_cast<float>(b[b_base + 1]);
    if (col_base + 2 < n) acc2 += av * static_cast<float>(b[b_base + 2]);
    if (col_base + 3 < n) acc3 += av * static_cast<float>(b[b_base + 3]);
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  c[c_base] = acc0;
  if (col_base + 1 < n) c[c_base + 1] = acc1;
  if (col_base + 2 < n) c[c_base + 2] = acc2;
  if (col_base + 3 < n) c[c_base + 3] = acc3;
}

__global__ void int8_vec4cols_kernel(const int8_t* a, const int8_t* b,
                                     int32_t* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (row >= n || col_base >= n) return;

  int32_t acc0 = 0;
  int32_t acc1 = 0;
  int32_t acc2 = 0;
  int32_t acc3 = 0;
  for (int kk = 0; kk < n; ++kk) {
    const int32_t av =
        static_cast<int32_t>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    acc0 += av * static_cast<int32_t>(b[b_base]);
    if (col_base + 1 < n) acc1 += av * static_cast<int32_t>(b[b_base + 1]);
    if (col_base + 2 < n) acc2 += av * static_cast<int32_t>(b[b_base + 2]);
    if (col_base + 3 < n) acc3 += av * static_cast<int32_t>(b[b_base + 3]);
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  c[c_base] = acc0;
  if (col_base + 1 < n) c[c_base + 1] = acc1;
  if (col_base + 2 < n) c[c_base + 2] = acc2;
  if (col_base + 3 < n) c[c_base + 3] = acc3;
}

__global__ void fp8_vec8cols_kernel(const __nv_fp8_e4m3* a,
                                    const __nv_fp8_e4m3* b, float* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (row >= n || col_base >= n) return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;
  for (int kk = 0; kk < n; ++kk) {
    const float av = static_cast<float>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    acc0 += av * static_cast<float>(b[b_base]);
    if (col_base + 1 < n) acc1 += av * static_cast<float>(b[b_base + 1]);
    if (col_base + 2 < n) acc2 += av * static_cast<float>(b[b_base + 2]);
    if (col_base + 3 < n) acc3 += av * static_cast<float>(b[b_base + 3]);
    if (col_base + 4 < n) acc4 += av * static_cast<float>(b[b_base + 4]);
    if (col_base + 5 < n) acc5 += av * static_cast<float>(b[b_base + 5]);
    if (col_base + 6 < n) acc6 += av * static_cast<float>(b[b_base + 6]);
    if (col_base + 7 < n) acc7 += av * static_cast<float>(b[b_base + 7]);
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  c[c_base] = acc0;
  if (col_base + 1 < n) c[c_base + 1] = acc1;
  if (col_base + 2 < n) c[c_base + 2] = acc2;
  if (col_base + 3 < n) c[c_base + 3] = acc3;
  if (col_base + 4 < n) c[c_base + 4] = acc4;
  if (col_base + 5 < n) c[c_base + 5] = acc5;
  if (col_base + 6 < n) c[c_base + 6] = acc6;
  if (col_base + 7 < n) c[c_base + 7] = acc7;
}

__global__ void int8_vec8cols_kernel(const int8_t* a, const int8_t* b,
                                     int32_t* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (row >= n || col_base >= n) return;

  int32_t acc0 = 0;
  int32_t acc1 = 0;
  int32_t acc2 = 0;
  int32_t acc3 = 0;
  int32_t acc4 = 0;
  int32_t acc5 = 0;
  int32_t acc6 = 0;
  int32_t acc7 = 0;
  for (int kk = 0; kk < n; ++kk) {
    const int32_t av =
        static_cast<int32_t>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    acc0 += av * static_cast<int32_t>(b[b_base]);
    if (col_base + 1 < n) acc1 += av * static_cast<int32_t>(b[b_base + 1]);
    if (col_base + 2 < n) acc2 += av * static_cast<int32_t>(b[b_base + 2]);
    if (col_base + 3 < n) acc3 += av * static_cast<int32_t>(b[b_base + 3]);
    if (col_base + 4 < n) acc4 += av * static_cast<int32_t>(b[b_base + 4]);
    if (col_base + 5 < n) acc5 += av * static_cast<int32_t>(b[b_base + 5]);
    if (col_base + 6 < n) acc6 += av * static_cast<int32_t>(b[b_base + 6]);
    if (col_base + 7 < n) acc7 += av * static_cast<int32_t>(b[b_base + 7]);
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  c[c_base] = acc0;
  if (col_base + 1 < n) c[c_base + 1] = acc1;
  if (col_base + 2 < n) c[c_base + 2] = acc2;
  if (col_base + 3 < n) c[c_base + 3] = acc3;
  if (col_base + 4 < n) c[c_base + 4] = acc4;
  if (col_base + 5 < n) c[c_base + 5] = acc5;
  if (col_base + 6 < n) c[c_base + 6] = acc6;
  if (col_base + 7 < n) c[c_base + 7] = acc7;
}

__global__ void fp8_vec16cols_kernel(const __nv_fp8_e4m3* a,
                                     const __nv_fp8_e4m3* b, float* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (row >= n || col_base >= n) return;

  float acc[16];
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    acc[j] = 0.0f;
  }
  for (int kk = 0; kk < n; ++kk) {
    const float av = static_cast<float>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (col_base + j < n) {
        acc[j] += av * static_cast<float>(b[b_base + j]);
      }
    }
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    if (col_base + j < n) {
      c[c_base + j] = acc[j];
    }
  }
}

__device__ __forceinline__ uint32_t load_fp8_byte(
    const __nv_fp8_e4m3* ptr, size_t index) {
  return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t*>(ptr)[index]);
}

__global__ void fp8_mma_m16n8k32_global_kernel(const __nv_fp8_e4m3* a,
                                               const __nv_fp8_e4m3* b,
                                               float* c, int n) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 64;
  const int tile_n = blockIdx.x * 64;
  const int warp_m = tile_m + warp_id * 16;

  float d[8][4];
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < n; k0 += 32) {
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_a0 = k0 + thread_in_group * 4 + i;
      const int k_a1 = k_a0 + 16;
      a0 |= load_fp8_byte(a, static_cast<size_t>(warp_m + group) * n + k_a0)
            << (8 * i);
      a1 |= load_fp8_byte(a, static_cast<size_t>(warp_m + group + 8) * n +
                                 k_a0)
            << (8 * i);
      a2 |= load_fp8_byte(a, static_cast<size_t>(warp_m + group) * n + k_a1)
            << (8 * i);
      a3 |= load_fp8_byte(a, static_cast<size_t>(warp_m + group + 8) * n +
                                 k_a1)
            << (8 * i);
    }

    #pragma unroll
    for (int col_tile = 0; col_tile < 8; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k_b0 = k0 + thread_in_group * 4 + i;
        const int k_b1 = k_b0 + 16;
        const int col = tile_n + col_tile * 8 + group;
        b0 |= load_fp8_byte(b, static_cast<size_t>(k_b0) * n + col)
              << (8 * i);
        b1 |= load_fp8_byte(b, static_cast<size_t>(k_b1) * n + col)
              << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
          "f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
            "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }

  const int row0 = warp_m + group;
  const int row1 = warp_m + group + 8;
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    c[static_cast<size_t>(row0) * n + col0] = d[col_tile][0];
    c[static_cast<size_t>(row0) * n + col0 + 1] = d[col_tile][1];
    c[static_cast<size_t>(row1) * n + col0] = d[col_tile][2];
    c[static_cast<size_t>(row1) * n + col0 + 1] = d[col_tile][3];
  }
}

__global__ void fp8_mma_m16n8k32_smem64_kernel(const __nv_fp8_e4m3* a,
                                               const __nv_fp8_e4m3* b,
                                               float* c, int n) {
  __shared__ alignas(16) uint8_t a_smem[64 * 32];
  __shared__ alignas(16) uint8_t b_smem[32 * 64];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 64;
  const int tile_n = blockIdx.x * 64;
  const int warp_m = warp_id * 16;

  float d[8][4];
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < n; k0 += 32) {
    for (int idx = tid; idx < 64 * 32; idx += blockDim.x) {
      const int row = idx / 32;
      const int kk = idx - row * 32;
      a_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              a, static_cast<size_t>(tile_m + row) * n + k0 + kk));
    }
    for (int idx = tid; idx < 32 * 64; idx += blockDim.x) {
      const int kk = idx / 64;
      const int col = idx - kk * 64;
      b_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              b, static_cast<size_t>(k0 + kk) * n + tile_n + col));
    }
    __syncthreads();

    const int smem_row = warp_m + group;
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_a0 = thread_in_group * 4 + i;
      const int k_a1 = k_a0 + 16;
      a0 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a0])
            << (8 * i);
      a1 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a0])
            << (8 * i);
      a2 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a1])
            << (8 * i);
      a3 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a1])
            << (8 * i);
    }

    #pragma unroll
    for (int col_tile = 0; col_tile < 8; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k_b0 = thread_in_group * 4 + i;
        const int k_b1 = k_b0 + 16;
        const int col = col_tile * 8 + group;
        b0 |= static_cast<uint32_t>(b_smem[k_b0 * 64 + col]) << (8 * i);
        b1 |= static_cast<uint32_t>(b_smem[k_b1 * 64 + col]) << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
          "f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
            "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }

  const int row0 = tile_m + warp_m + group;
  const int row1 = row0 + 8;
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    c[static_cast<size_t>(row0) * n + col0] = d[col_tile][0];
    c[static_cast<size_t>(row0) * n + col0 + 1] = d[col_tile][1];
    c[static_cast<size_t>(row1) * n + col0] = d[col_tile][2];
    c[static_cast<size_t>(row1) * n + col0 + 1] = d[col_tile][3];
  }
}

__global__ void fp8_mma_m16n8k32_smem64x128_kernel(const __nv_fp8_e4m3* a,
                                                   const __nv_fp8_e4m3* b,
                                                   float* c, int n) {
  __shared__ alignas(16) uint8_t a_smem[64 * 32];
  __shared__ alignas(16) uint8_t b_smem[32 * 128];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 64;
  const int tile_n = blockIdx.x * 128;
  const int warp_m = warp_id * 16;

  float d[16][4];
  #pragma unroll
  for (int col_tile = 0; col_tile < 16; ++col_tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < n; k0 += 32) {
    for (int idx = tid; idx < 64 * 32; idx += blockDim.x) {
      const int row = idx / 32;
      const int kk = idx - row * 32;
      a_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              a, static_cast<size_t>(tile_m + row) * n + k0 + kk));
    }
    for (int idx = tid; idx < 32 * 128; idx += blockDim.x) {
      const int kk = idx / 128;
      const int col = idx - kk * 128;
      b_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              b, static_cast<size_t>(k0 + kk) * n + tile_n + col));
    }
    __syncthreads();

    const int smem_row = warp_m + group;
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_a0 = thread_in_group * 4 + i;
      const int k_a1 = k_a0 + 16;
      a0 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a0])
            << (8 * i);
      a1 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a0])
            << (8 * i);
      a2 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a1])
            << (8 * i);
      a3 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a1])
            << (8 * i);
    }

    #pragma unroll
    for (int col_tile = 0; col_tile < 16; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k_b0 = thread_in_group * 4 + i;
        const int k_b1 = k_b0 + 16;
        const int col = col_tile * 8 + group;
        b0 |= static_cast<uint32_t>(b_smem[k_b0 * 128 + col]) << (8 * i);
        b1 |= static_cast<uint32_t>(b_smem[k_b1 * 128 + col]) << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
          "f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
            "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }

  const int row0 = tile_m + warp_m + group;
  const int row1 = row0 + 8;
  #pragma unroll
  for (int col_tile = 0; col_tile < 16; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    c[static_cast<size_t>(row0) * n + col0] = d[col_tile][0];
    c[static_cast<size_t>(row0) * n + col0 + 1] = d[col_tile][1];
    c[static_cast<size_t>(row1) * n + col0] = d[col_tile][2];
    c[static_cast<size_t>(row1) * n + col0 + 1] = d[col_tile][3];
  }
}

__global__ void fp8_mma_m16n8k32_smem128x64_kernel(const __nv_fp8_e4m3* a,
                                                   const __nv_fp8_e4m3* b,
                                                   float* c, int n) {
  __shared__ alignas(16) uint8_t a_smem[128 * 32];
  __shared__ alignas(16) uint8_t b_smem[32 * 64];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 128;
  const int tile_n = blockIdx.x * 64;
  const int warp_m = warp_id * 16;

  float d[8][4];
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < n; k0 += 32) {
    for (int idx = tid; idx < 128 * 32; idx += blockDim.x) {
      const int row = idx / 32;
      const int kk = idx - row * 32;
      a_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              a, static_cast<size_t>(tile_m + row) * n + k0 + kk));
    }
    for (int idx = tid; idx < 32 * 64; idx += blockDim.x) {
      const int kk = idx / 64;
      const int col = idx - kk * 64;
      b_smem[idx] =
          static_cast<uint8_t>(load_fp8_byte(
              b, static_cast<size_t>(k0 + kk) * n + tile_n + col));
    }
    __syncthreads();

    const int smem_row = warp_m + group;
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_a0 = thread_in_group * 4 + i;
      const int k_a1 = k_a0 + 16;
      a0 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a0])
            << (8 * i);
      a1 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a0])
            << (8 * i);
      a2 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a1])
            << (8 * i);
      a3 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a1])
            << (8 * i);
    }

    #pragma unroll
    for (int col_tile = 0; col_tile < 8; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k_b0 = thread_in_group * 4 + i;
        const int k_b1 = k_b0 + 16;
        const int col = col_tile * 8 + group;
        b0 |= static_cast<uint32_t>(b_smem[k_b0 * 64 + col]) << (8 * i);
        b1 |= static_cast<uint32_t>(b_smem[k_b1 * 64 + col]) << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
          "f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(d[col_tile][0]), "+f"(d[col_tile][1]),
            "+f"(d[col_tile][2]), "+f"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }

  const int row0 = tile_m + warp_m + group;
  const int row1 = row0 + 8;
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    c[static_cast<size_t>(row0) * n + col0] = d[col_tile][0];
    c[static_cast<size_t>(row0) * n + col0 + 1] = d[col_tile][1];
    c[static_cast<size_t>(row1) * n + col0] = d[col_tile][2];
    c[static_cast<size_t>(row1) * n + col0 + 1] = d[col_tile][3];
  }
}

__global__ void int8_vec16cols_kernel(const int8_t* a, const int8_t* b,
                                      int32_t* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (row >= n || col_base >= n) return;

  int32_t acc[16];
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    acc[j] = 0;
  }
  for (int kk = 0; kk < n; ++kk) {
    const int32_t av =
        static_cast<int32_t>(a[static_cast<size_t>(row) * n + kk]);
    const size_t b_base = static_cast<size_t>(kk) * n + col_base;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (col_base + j < n) {
        acc[j] += av * static_cast<int32_t>(b[b_base + j]);
      }
    }
  }
  const size_t c_base = static_cast<size_t>(row) * n + col_base;
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    if (col_base + j < n) {
      c[c_base + j] = acc[j];
    }
  }
}

__device__ __forceinline__ uint32_t load_int8_byte(const int8_t* ptr,
                                                   size_t index) {
  return static_cast<uint32_t>(
      static_cast<uint8_t>(ptr[index]));
}

__global__ void int8_mma_m16n8k32_smem64_kernel(const int8_t* a,
                                                const int8_t* b,
                                                int32_t* c, int n) {
  __shared__ alignas(16) uint8_t a_smem[64 * 32];
  __shared__ alignas(16) uint8_t b_smem[32 * 64];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 64;
  const int tile_n = blockIdx.x * 64;
  const int warp_m = warp_id * 16;

  int32_t d[8][4];
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[col_tile][i] = 0;
    }
  }

  for (int k0 = 0; k0 < n; k0 += 32) {
    for (int idx = tid; idx < 64 * 32; idx += blockDim.x) {
      const int row = idx / 32;
      const int kk = idx - row * 32;
      a_smem[idx] =
          static_cast<uint8_t>(
              load_int8_byte(a, static_cast<size_t>(tile_m + row) * n +
                                    k0 + kk));
    }
    for (int idx = tid; idx < 32 * 64; idx += blockDim.x) {
      const int kk = idx / 64;
      const int col = idx - kk * 64;
      b_smem[idx] =
          static_cast<uint8_t>(
              load_int8_byte(b, static_cast<size_t>(k0 + kk) * n +
                                    tile_n + col));
    }
    __syncthreads();

    const int smem_row = warp_m + group;
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_a0 = thread_in_group * 4 + i;
      const int k_a1 = k_a0 + 16;
      a0 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a0])
            << (8 * i);
      a1 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a0])
            << (8 * i);
      a2 |= static_cast<uint32_t>(a_smem[smem_row * 32 + k_a1])
            << (8 * i);
      a3 |= static_cast<uint32_t>(a_smem[(smem_row + 8) * 32 + k_a1])
            << (8 * i);
    }

    #pragma unroll
    for (int col_tile = 0; col_tile < 8; ++col_tile) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k_b0 = thread_in_group * 4 + i;
        const int k_b1 = k_b0 + 16;
        const int col = col_tile * 8 + group;
        b0 |= static_cast<uint32_t>(b_smem[k_b0 * 64 + col]) << (8 * i);
        b1 |= static_cast<uint32_t>(b_smem[k_b1 * 64 + col]) << (8 * i);
      }

      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+r"(d[col_tile][0]), "+r"(d[col_tile][1]),
            "+r"(d[col_tile][2]), "+r"(d[col_tile][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }

  const int row0 = tile_m + warp_m + group;
  const int row1 = row0 + 8;
  #pragma unroll
  for (int col_tile = 0; col_tile < 8; ++col_tile) {
    const int col0 = tile_n + col_tile * 8 + thread_in_group * 2;
    c[static_cast<size_t>(row0) * n + col0] = d[col_tile][0];
    c[static_cast<size_t>(row0) * n + col0 + 1] = d[col_tile][1];
    c[static_cast<size_t>(row1) * n + col0] = d[col_tile][2];
    c[static_cast<size_t>(row1) * n + col0 + 1] = d[col_tile][3];
  }
}

__global__ void int8_wmma_m16n16k16_kernel(const int8_t* a, const int8_t* b,
                                           int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 16;
  const int col = tile_col * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    wmma::load_matrix_sync(b_frag, b + static_cast<size_t>(k0) * n + col, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag, n,
                          wmma::mem_row_major);
}

__global__ void int8_wmma_m16n16k16_8warp_kernel(const int8_t* a,
                                                 const int8_t* b, int32_t* c,
                                                 int n) {
  const int warp_id = threadIdx.x / 32;
  const int tile_col = blockIdx.x * 8 + warp_id;
  const int tile_row = blockIdx.y;
  const int tiles = n / 16;
  if (tile_col >= tiles) return;

  const int row = tile_row * 16;
  const int col = tile_col * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    wmma::load_matrix_sync(b_frag, b + static_cast<size_t>(k0) * n + col, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag, n,
                          wmma::mem_row_major);
}

__global__ void int8_wmma_m32n8k16_kernel(const int8_t* a, const int8_t* b,
                                          int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 32;
  const int col = tile_col * 8;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    wmma::load_matrix_sync(b_frag, b + static_cast<size_t>(k0) * n + col, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag, n,
                          wmma::mem_row_major);
}

__global__ void int8_wmma_m8n32k16_kernel(const int8_t* a, const int8_t* b,
                                          int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 8;
  const int col = tile_col * 32;

  wmma::fragment<wmma::matrix_a, 8, 32, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 8, 32, 16, signed char, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 8, 32, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    wmma::load_matrix_sync(b_frag, b + static_cast<size_t>(k0) * n + col, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag, n,
                          wmma::mem_row_major);
}

__global__ void int8_wmma_m32n64k16_smem_kernel(const int8_t* a,
                                                const int8_t* b, int32_t* c,
                                                int n) {
  __shared__ int8_t tile_a[32 * 16];
  __shared__ int8_t tile_b[16 * 64];

  const int warp_id = threadIdx.x / 32;
  const int lane_group = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 32;
  const int col = lane_group * 64 + warp_id * 8;
  const int linear_thread = threadIdx.x;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    for (int idx = linear_thread; idx < 32 * 16; idx += blockDim.x) {
      const int local_row = idx / 16;
      const int local_k = idx % 16;
      tile_a[idx] = a[static_cast<size_t>(row + local_row) * n + k0 + local_k];
    }
    for (int idx = linear_thread; idx < 16 * 64; idx += blockDim.x) {
      const int local_k = idx / 64;
      const int local_col = idx % 64;
      tile_b[idx] =
          b[static_cast<size_t>(k0 + local_k) * n + lane_group * 64 + local_col];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, tile_a, 16);
    wmma::load_matrix_sync(b_frag, tile_b + warp_id * 8, 64);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag, n,
                          wmma::mem_row_major);
}

__global__ void int8_wmma_m32n32k16_reuse_a_kernel(const int8_t* a,
                                                   const int8_t* b,
                                                   int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 32;
  const int col = tile_col * 32;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag0;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag1;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag2;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frag3;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag0;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag1;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag2;
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frag3;
  wmma::fill_fragment(acc_frag0, 0);
  wmma::fill_fragment(acc_frag1, 0);
  wmma::fill_fragment(acc_frag2, 0);
  wmma::fill_fragment(acc_frag3, 0);

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    wmma::load_matrix_sync(b_frag0, b + static_cast<size_t>(k0) * n + col, n);
    wmma::load_matrix_sync(b_frag1,
                           b + static_cast<size_t>(k0) * n + col + 8, n);
    wmma::load_matrix_sync(b_frag2,
                           b + static_cast<size_t>(k0) * n + col + 16, n);
    wmma::load_matrix_sync(b_frag3,
                           b + static_cast<size_t>(k0) * n + col + 24, n);
    wmma::mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
    wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
    wmma::mma_sync(acc_frag2, a_frag, b_frag2, acc_frag2);
    wmma::mma_sync(acc_frag3, a_frag, b_frag3, acc_frag3);
  }

  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc_frag0, n,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + 8,
                          acc_frag1, n, wmma::mem_row_major);
  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + 16,
                          acc_frag2, n, wmma::mem_row_major);
  wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + 24,
                          acc_frag3, n, wmma::mem_row_major);
}

__global__ void int8_wmma_m32n64k16_reuse_a_kernel(const int8_t* a,
                                                   const int8_t* b,
                                                   int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 32;
  const int col = tile_col * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b + static_cast<size_t>(k0) * n + col + i * 8,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m32n128k16_reuse_a_kernel(const int8_t* a,
                                                    const int8_t* b,
                                                    int32_t* c, int n) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * 32;
  const int col = tile_col * 128;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frags[16];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b + static_cast<size_t>(k0) * n + col + i * 8,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m128n64k16_4warp_reuse_a_kernel(const int8_t* a,
                                                           const int8_t* b,
                                                           int32_t* c,
                                                           int n) {
  const int warp_id = threadIdx.x / 32;
  const int row = blockIdx.y * 128 + warp_id * 32;
  const int col = blockIdx.x * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b + static_cast<size_t>(k0) * n + col + i * 8,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m128n128k16_8warp_reuse_a_kernel(const int8_t* a,
                                                           const int8_t* b,
                                                           int32_t* c,
                                                           int n) {
  const int warp_id = threadIdx.x / 32;
  const int row_group = warp_id & 3;
  const int col_group = warp_id >> 2;
  const int row = blockIdx.y * 128 + row_group * 32;
  const int col = blockIdx.x * 128 + col_group * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b + static_cast<size_t>(k0) * n + col + i * 8,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m256n64k16_8warp_reuse_a_kernel(const int8_t* a,
                                                           const int8_t* b,
                                                           int32_t* c,
                                                           int n) {
  const int warp_id = threadIdx.x / 32;
  const int row = blockIdx.y * 256 + warp_id * 32;
  const int col = blockIdx.x * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::row_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b + static_cast<size_t>(k0) * n + col + i * 8,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m128n64k16_4warp_reuse_a_bcol_kernel(
    const int8_t* a, const int8_t* b_t, int32_t* c, int n) {
  const int warp_id = threadIdx.x / 32;
  const int row = blockIdx.y * 128 + warp_id * 32;
  const int col = blockIdx.x * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::col_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b_t + static_cast<size_t>(col + i * 8) * n + k0,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m128n128k16_8warp_reuse_a_bcol_kernel(
    const int8_t* a, const int8_t* b_t, int32_t* c, int n) {
  const int warp_id = threadIdx.x / 32;
  const int row_group = warp_id & 3;
  const int col_group = warp_id >> 2;
  const int row = blockIdx.y * 128 + row_group * 32;
  const int col = blockIdx.x * 128 + col_group * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::col_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b_t + static_cast<size_t>(col + i * 8) * n + k0,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

__global__ void int8_wmma_m256n64k16_8warp_reuse_a_bcol_kernel(
    const int8_t* a, const int8_t* b_t, int32_t* c, int n) {
  const int warp_id = threadIdx.x / 32;
  const int row = blockIdx.y * 256 + warp_id * 32;
  const int col = blockIdx.x * 64;

  wmma::fragment<wmma::matrix_a, 32, 8, 16, signed char, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, signed char, wmma::col_major>
      b_frags[8];
  wmma::fragment<wmma::accumulator, 32, 8, 16, int> acc_frags[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::fill_fragment(acc_frags[i], 0);
  }

  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * n + k0, n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::load_matrix_sync(b_frags[i],
                             b_t + static_cast<size_t>(col + i * 8) * n + k0,
                             n);
      wmma::mma_sync(acc_frags[i], a_frag, b_frags[i], acc_frags[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    wmma::store_matrix_sync(c + static_cast<size_t>(row) * n + col + i * 8,
                            acc_frags[i], n, wmma::mem_row_major);
  }
}

template <typename Launch>
float benchmark_cuda_launch(Launch launch) {
  for (int i = 0; i < kWarmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < kRepeat; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / kRepeat;
}

cublasLtMatrixLayout_t make_col_major_layout(cudaDataType_t type,
                                             uint64_t rows, uint64_t cols,
                                             int64_t ld) {
  cublasLtMatrixLayout_t layout = nullptr;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
  return layout;
}

float benchmark_cublaslt_fp8_reference(int n, const __nv_fp8_e4m3* d_a,
                                       const __nv_fp8_e4m3* d_b,
                                       float* d_c) {
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  void* workspace = nullptr;
  constexpr uint64_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;

  CHECK_CUBLAS(cublasLtCreate(&lt));
  CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  // Use column-major descriptors over row-major storage:
  // C(row-major MxN) is the same bytes as (B^T * A^T)(column-major NxM).
  a_desc = make_col_major_layout(CUDA_R_8F_E4M3, n, n, n);
  b_desc = make_col_major_layout(CUDA_R_8F_E4M3, n, n, n);
  c_desc = make_col_major_layout(CUDA_R_32F, n, n, n);
  d_desc = make_col_major_layout(CUDA_R_32F, n, n, n);

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      lt, op_desc, a_desc, b_desc, c_desc, d_desc, pref, 1, &heuristic,
      &returned));
  if (returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLASLt FP8 heuristic returned no runnable algorithm\n";
    std::exit(EXIT_FAILURE);
  }

  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto launch = [&]() {
    CHECK_CUBLAS(cublasLtMatmul(
        lt, op_desc, &alpha, d_b, a_desc, d_a, b_desc, &beta, d_c, c_desc,
        d_c, d_desc, &heuristic.algo, workspace, kWorkspaceBytes, 0));
  };
  const float avg_ms = benchmark_cuda_launch(launch);

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
  CHECK_CUBLAS(cublasLtDestroy(lt));
  return avg_ms;
}

float benchmark_cublas_int8_reference(int n, const int8_t* d_a,
                                      const int8_t* d_b, int32_t* d_c) {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  const int32_t alpha = 1;
  const int32_t beta = 0;

  auto launch = [&]() {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, CUDA_R_8I,
        n, d_a, CUDA_R_8I, n, &beta, d_c, CUDA_R_32I, n, CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT));
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUBLAS(cublasDestroy(handle));
  return avg_ms;
}

bool compare_fp8_outputs(const std::vector<float>& ref,
                         const std::vector<float>& got) {
  int errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i] - got[i]);
    const float tol = 5.0e-1f + 5.0e-2f * std::fabs(ref[i]);
    if (diff > tol && ++errors <= 5) {
      std::cerr << "FP8 mismatch at " << i << ": ref=" << ref[i]
                << ", got=" << got[i] << ", diff=" << diff
                << ", tol=" << tol << '\n';
    }
  }
  return errors == 0;
}

bool compare_int8_outputs(const std::vector<int32_t>& ref,
                          const std::vector<int32_t>& got) {
  int errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i] && ++errors <= 5) {
      std::cerr << "INT8 mismatch at " << i << ": ref=" << ref[i]
                << ", got=" << got[i] << '\n';
    }
  }
  return errors == 0;
}

void write_csv_row(std::ofstream& csv, const std::string& backend_id,
                   const std::string& version, int n,
                   const std::string& precision,
                   const std::string& reference, float avg_ms, float perf,
                   float reference_perf, bool matched) {
  const float ratio = reference_perf > 0.0f ? perf / reference_perf : 0.0f;
  std::cout << backend_id << ": " << avg_ms << " ms, " << perf
            << " GFLOP/s, reference=" << reference_perf
            << " GFLOP/s, ratio=" << ratio
            << "x, matched=" << matched << '\n';
  csv << backend_id << ',' << version << ',' << n << ',' << precision << ','
      << reference << ',' << avg_ms << ',' << perf << ',' << ratio << ','
      << (matched ? 1 : 0) << '\n';
}

void run_fp8_q0(int n, std::ofstream& csv) {
  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
  auto launch = [&]() {
    fp8_naive_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q0_cuda_naive", "FP8 q0 naive CUDA baseline", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q1(int n, std::ofstream& csv) {
  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x * 4), ceil_div(n, block.y));
  auto launch = [&]() {
    fp8_vec4cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q1_cuda_vec4cols",
                "FP8 q1 register-tiled four-column CUDA baseline", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q2(int n, std::ofstream& csv) {
  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x * 8), ceil_div(n, block.y));
  auto launch = [&]() {
    fp8_vec8cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q2_cuda_vec8cols",
                "FP8 q2 register-tiled eight-column CUDA baseline", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q3(int n, std::ofstream& csv) {
  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(8, 16);
  dim3 grid(ceil_div(n, block.x * 16), ceil_div(n, block.y));
  auto launch = [&]() {
    fp8_vec16cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q3_cuda_vec16cols",
                "FP8 q3 register-tiled sixteen-column CUDA baseline", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q4(int n, std::ofstream& csv) {
  if ((n % 64) != 0) {
    std::cerr << "fp8_q4_mma_m16n8k32_global requires N divisible by 64\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(ceil_div(n, 64), ceil_div(n, 64));
  auto launch = [&]() {
    fp8_mma_m16n8k32_global_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q4_mma_m16n8k32_global",
                "FP8 q4 warp MMA M16N8K32 global-load Tensor Core", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q5(int n, std::ofstream& csv) {
  if ((n % 64) != 0) {
    std::cerr << "fp8_q5_mma_m16n8k32_smem64 requires N divisible by 64\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(ceil_div(n, 64), ceil_div(n, 64));
  auto launch = [&]() {
    fp8_mma_m16n8k32_smem64_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q5_mma_m16n8k32_smem64",
                "FP8 q5 warp MMA M16N8K32 shared 64x64 tile", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q6(int n, std::ofstream& csv) {
  if ((n % 128) != 0) {
    std::cerr << "fp8_q6_mma_m16n8k32_smem64x128 requires N divisible by 128\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(ceil_div(n, 128), ceil_div(n, 64));
  auto launch = [&]() {
    fp8_mma_m16n8k32_smem64x128_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q6_mma_m16n8k32_smem64x128",
                "FP8 q6 warp MMA M16N8K32 shared 64x128 tile", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q7(int n, std::ofstream& csv) {
  if ((n % 128) != 0) {
    std::cerr << "fp8_q7_mma_m16n8k32_smem128x64 requires N divisible by 128\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(ceil_div(n, 64), ceil_div(n, 128));
  auto launch = [&]() {
    fp8_mma_m16n8k32_smem128x64_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q7_mma_m16n8k32_smem128x64",
                "FP8 q7 warp MMA M16N8K32 shared 128x64 tile", n,
                "fp8_e4m3->fp32", "cuBLASLt FP8 E4M3", avg_ms,
                gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fp8_q8(int n, std::ofstream& csv) {
  std::vector<float> h_a_float(static_cast<size_t>(n) * n);
  std::vector<float> h_b_float(static_cast<size_t>(n) * n);
  fill_float_inputs(h_a_float, h_b_float);
  std::vector<__nv_fp8_e4m3> h_a = to_fp8_e4m3(h_a_float);
  std::vector<__nv_fp8_e4m3> h_b = to_fp8_e4m3(h_b_float);
  std::vector<float> h_ref(static_cast<size_t>(n) * n);
  std::vector<float> h_out(static_cast<size_t>(n) * n);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_out = nullptr;
  const size_t fp8_bytes = h_a.size() * sizeof(__nv_fp8_e4m3);
  const size_t out_bytes = h_ref.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_a, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, fp8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), fp8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLASLt FP8 E4M3 reference: " << ref_ms << " ms, "
            << ref_perf << " GFLOP/s\n";

  const float avg_ms = benchmark_cublaslt_fp8_reference(n, d_a, d_b, d_out);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_fp8_outputs(h_ref, h_out);
  write_csv_row(csv, "fp8_q8_cublaslt_matmul",
                "FP8 q8 cuBLASLt matmul backend", n, "fp8_e4m3->fp32",
                "cuBLASLt FP8 E4M3", avg_ms, gflops(n, avg_ms), ref_perf,
                matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q0(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
  auto launch = [&]() {
    int8_naive_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q0_cuda_naive", "INT8 q0 naive CUDA baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q1(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x * 4), ceil_div(n, block.y));
  auto launch = [&]() {
    int8_vec4cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q1_cuda_vec4cols",
                "INT8 q1 register-tiled four-column CUDA baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q2(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, block.x * 8), ceil_div(n, block.y));
  auto launch = [&]() {
    int8_vec8cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q2_cuda_vec8cols",
                "INT8 q2 register-tiled eight-column CUDA baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q3(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(8, 16);
  dim3 grid(ceil_div(n, block.x * 16), ceil_div(n, block.y));
  auto launch = [&]() {
    int8_vec16cols_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q3_cuda_vec16cols",
                "INT8 q3 register-tiled sixteen-column CUDA baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q4(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 16, n / 16);
  auto launch = [&]() {
    int8_wmma_m16n16k16_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q4_wmma_m16n16k16",
                "INT8 q4 WMMA M16N16K16 Tensor Core baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q5(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(ceil_div(n / 16, 8), n / 16);
  auto launch = [&]() {
    int8_wmma_m16n16k16_8warp_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q5_wmma_m16n16k16_8warp",
                "INT8 q5 WMMA M16N16K16 8-warps-per-block baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q6(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 8, n / 32);
  auto launch = [&]() {
    int8_wmma_m32n8k16_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q6_wmma_m32n8k16",
                "INT8 q6 WMMA M32N8K16 Tensor Core baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q7(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 32, n / 8);
  auto launch = [&]() {
    int8_wmma_m8n32k16_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q7_wmma_m8n32k16",
                "INT8 q7 WMMA M8N32K16 Tensor Core baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q8(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(n / 64, n / 32);
  auto launch = [&]() {
    int8_wmma_m32n64k16_smem_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q8_wmma_m32n64k16_smem",
                "INT8 q8 WMMA M32N64K16 shared-memory staged baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q9(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 32, n / 32);
  auto launch = [&]() {
    int8_wmma_m32n32k16_reuse_a_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q9_wmma_m32n32k16_reuse_a",
                "INT8 q9 WMMA M32N32K16 reuse-A baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q10(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 64, n / 32);
  auto launch = [&]() {
    int8_wmma_m32n64k16_reuse_a_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q10_wmma_m32n64k16_reuse_a",
                "INT8 q10 WMMA M32N64K16 reuse-A baseline", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q11(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(32);
  dim3 grid(n / 128, n / 32);
  auto launch = [&]() {
    int8_wmma_m32n128k16_reuse_a_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q11_wmma_m32n128k16_reuse_a",
                "INT8 q11 WMMA M32N128K16 reuse-A pressure test", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q12(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(n / 64, n / 128);
  auto launch = [&]() {
    int8_wmma_m128n64k16_4warp_reuse_a_kernel<<<grid, block>>>(d_a, d_b,
                                                               d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q12_wmma_m128n64k16_4warp_reuse_a",
                "INT8 q12 WMMA M128N64K16 4-warp reuse-A CTA", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q13(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(n / 128, n / 128);
  auto launch = [&]() {
    int8_wmma_m128n128k16_8warp_reuse_a_kernel<<<grid, block>>>(d_a, d_b,
                                                                d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q13_wmma_m128n128k16_8warp_reuse_a",
                "INT8 q13 WMMA M128N128K16 8-warp reuse-A CTA", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q14(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(n / 64, n / 256);
  auto launch = [&]() {
    int8_wmma_m256n64k16_8warp_reuse_a_kernel<<<grid, block>>>(d_a, d_b,
                                                               d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q14_wmma_m256n64k16_8warp_reuse_a",
                "INT8 q14 WMMA M256N64K16 8-warp reuse-A CTA", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q15(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int8_t> h_b_t = transpose_int8_square(h_b, n);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int8_t* d_b_t = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_t, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_t, h_b_t.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(n / 64, n / 128);
  auto launch = [&]() {
    int8_wmma_m128n64k16_4warp_reuse_a_bcol_kernel<<<grid, block>>>(
        d_a, d_b_t, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol",
                "INT8 q15 WMMA M128N64K16 4-warp reuse-A B-col layout", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_b_t));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q16(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int8_t> h_b_t = transpose_int8_square(h_b, n);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int8_t* d_b_t = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_t, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_t, h_b_t.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(n / 128, n / 128);
  auto launch = [&]() {
    int8_wmma_m128n128k16_8warp_reuse_a_bcol_kernel<<<grid, block>>>(
        d_a, d_b_t, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol",
                "INT8 q16 WMMA M128N128K16 8-warp reuse-A B-col layout", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_b_t));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q17(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int8_t> h_b_t = transpose_int8_square(h_b, n);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int8_t* d_b_t = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_t, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_t, h_b_t.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(256);
  dim3 grid(n / 64, n / 256);
  auto launch = [&]() {
    int8_wmma_m256n64k16_8warp_reuse_a_bcol_kernel<<<grid, block>>>(
        d_a, d_b_t, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol",
                "INT8 q17 WMMA M256N64K16 8-warp reuse-A B-col layout", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_b_t));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q18(int n, std::ofstream& csv) {
  if ((n % 64) != 0) {
    std::cerr << "int8_q18_mma_m16n8k32_smem64 requires N divisible by 64\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  dim3 block(128);
  dim3 grid(ceil_div(n, 64), ceil_div(n, 64));
  auto launch = [&]() {
    int8_mma_m16n8k32_smem64_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const float avg_ms = benchmark_cuda_launch(launch);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q18_mma_m16n8k32_smem64",
                "INT8 q18 inline MMA M16N8K32 shared 64x64 tile", n,
                "int8->int32", "cuBLAS INT8", avg_ms, gflops(n, avg_ms),
                ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

void run_int8_q19(int n, std::ofstream& csv) {
  std::vector<int8_t> h_a =
      make_int8_input(static_cast<size_t>(n) * n, 17, 8);
  std::vector<int8_t> h_b =
      make_int8_input(static_cast<size_t>(n) * n, 13, 6);
  std::vector<int32_t> h_ref(static_cast<size_t>(n) * n);
  std::vector<int32_t> h_out(static_cast<size_t>(n) * n);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int32_t* d_ref = nullptr;
  int32_t* d_out = nullptr;
  const size_t int8_bytes = h_a.size() * sizeof(int8_t);
  const size_t out_bytes = h_ref.size() * sizeof(int32_t);
  CHECK_CUDA(cudaMalloc(&d_a, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, int8_bytes));
  CHECK_CUDA(cudaMalloc(&d_ref, out_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), int8_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_ref, 0, out_bytes));
  CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));

  const float ref_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_ref);
  const float ref_perf = gflops(n, ref_ms);
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, out_bytes, cudaMemcpyDeviceToHost));
  std::cout << "cuBLAS INT8 reference: " << ref_ms << " ms, " << ref_perf
            << " GFLOP/s\n";

  const float avg_ms = benchmark_cublas_int8_reference(n, d_a, d_b, d_out);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  const bool matched = compare_int8_outputs(h_ref, h_out);
  write_csv_row(csv, "int8_q19_cublas_gemmex",
                "INT8 q19 cuBLAS GemmEx backend", n, "int8->int32",
                "cuBLAS INT8", avg_ms, gflops(n, avg_ms), ref_perf, matched);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_out));
}

bool wants_backend(const std::string& filter, const std::string& id) {
  return filter == "all" || filter == id;
}

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " [N] [all|fp8_q0_cuda_naive|fp8_q1_cuda_vec4cols|"
               "fp8_q2_cuda_vec8cols|fp8_q3_cuda_vec16cols|"
               "fp8_q4_mma_m16n8k32_global|"
               "fp8_q5_mma_m16n8k32_smem64|"
               "fp8_q6_mma_m16n8k32_smem64x128|"
               "fp8_q7_mma_m16n8k32_smem128x64|"
               "fp8_q8_cublaslt_matmul|"
               "int8_q0_cuda_naive|int8_q1_cuda_vec4cols|"
               "int8_q2_cuda_vec8cols|int8_q3_cuda_vec16cols|"
               "int8_q4_wmma_m16n16k16|int8_q5_wmma_m16n16k16_8warp|"
               "int8_q6_wmma_m32n8k16|int8_q7_wmma_m8n32k16|"
               "int8_q8_wmma_m32n64k16_smem|"
               "int8_q9_wmma_m32n32k16_reuse_a|"
               "int8_q10_wmma_m32n64k16_reuse_a|"
               "int8_q11_wmma_m32n128k16_reuse_a|"
               "int8_q12_wmma_m128n64k16_4warp_reuse_a|"
               "int8_q13_wmma_m128n128k16_8warp_reuse_a|"
               "int8_q14_wmma_m256n64k16_8warp_reuse_a|"
               "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol|"
               "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol|"
               "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol|"
               "int8_q18_mma_m16n8k32_smem64|"
               "int8_q19_cublas_gemmex]\n";
}

}  // namespace

int main(int argc, char** argv) {
  int n = 1024;
  std::string backend = "all";
  if (argc >= 2) {
    n = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    backend = argv[2];
  }
  if (argc > 3 || n <= 0 ||
      !(backend == "all" || backend == "fp8_q0_cuda_naive" ||
        backend == "fp8_q1_cuda_vec4cols" ||
        backend == "fp8_q2_cuda_vec8cols" ||
        backend == "fp8_q3_cuda_vec16cols" ||
        backend == "fp8_q4_mma_m16n8k32_global" ||
        backend == "fp8_q5_mma_m16n8k32_smem64" ||
        backend == "fp8_q6_mma_m16n8k32_smem64x128" ||
        backend == "fp8_q7_mma_m16n8k32_smem128x64" ||
        backend == "fp8_q8_cublaslt_matmul" ||
        backend == "int8_q0_cuda_naive" ||
        backend == "int8_q1_cuda_vec4cols" ||
        backend == "int8_q2_cuda_vec8cols" ||
        backend == "int8_q3_cuda_vec16cols" ||
        backend == "int8_q4_wmma_m16n16k16" ||
        backend == "int8_q5_wmma_m16n16k16_8warp" ||
        backend == "int8_q6_wmma_m32n8k16" ||
        backend == "int8_q7_wmma_m8n32k16" ||
        backend == "int8_q8_wmma_m32n64k16_smem" ||
        backend == "int8_q9_wmma_m32n32k16_reuse_a" ||
        backend == "int8_q10_wmma_m32n64k16_reuse_a" ||
        backend == "int8_q11_wmma_m32n128k16_reuse_a" ||
        backend == "int8_q12_wmma_m128n64k16_4warp_reuse_a" ||
        backend == "int8_q13_wmma_m128n128k16_8warp_reuse_a" ||
        backend == "int8_q14_wmma_m256n64k16_8warp_reuse_a" ||
        backend == "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol" ||
        backend == "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol" ||
        backend == "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol" ||
        backend == "int8_q18_mma_m16n8k32_smem64" ||
        backend == "int8_q19_cublas_gemmex")) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  std::cout << "N=" << n << ", backend=" << backend << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat << '\n';

  std::ofstream csv("quant_sm110_benchmark.csv");
  csv << "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,"
         "RatioToReference,Matched\n";

  if (wants_backend(backend, "fp8_q0_cuda_naive")) {
    run_fp8_q0(n, csv);
  }
  if (wants_backend(backend, "fp8_q1_cuda_vec4cols")) {
    run_fp8_q1(n, csv);
  }
  if (wants_backend(backend, "fp8_q2_cuda_vec8cols")) {
    run_fp8_q2(n, csv);
  }
  if (wants_backend(backend, "fp8_q3_cuda_vec16cols")) {
    run_fp8_q3(n, csv);
  }
  if (wants_backend(backend, "fp8_q4_mma_m16n8k32_global")) {
    run_fp8_q4(n, csv);
  }
  if (wants_backend(backend, "fp8_q5_mma_m16n8k32_smem64")) {
    run_fp8_q5(n, csv);
  }
  if (wants_backend(backend, "fp8_q6_mma_m16n8k32_smem64x128")) {
    run_fp8_q6(n, csv);
  }
  if (wants_backend(backend, "fp8_q7_mma_m16n8k32_smem128x64")) {
    run_fp8_q7(n, csv);
  }
  if (wants_backend(backend, "fp8_q8_cublaslt_matmul")) {
    run_fp8_q8(n, csv);
  }
  if (wants_backend(backend, "int8_q0_cuda_naive")) {
    run_int8_q0(n, csv);
  }
  if (wants_backend(backend, "int8_q1_cuda_vec4cols")) {
    run_int8_q1(n, csv);
  }
  if (wants_backend(backend, "int8_q2_cuda_vec8cols")) {
    run_int8_q2(n, csv);
  }
  if (wants_backend(backend, "int8_q3_cuda_vec16cols")) {
    run_int8_q3(n, csv);
  }
  if (wants_backend(backend, "int8_q4_wmma_m16n16k16")) {
    run_int8_q4(n, csv);
  }
  if (wants_backend(backend, "int8_q5_wmma_m16n16k16_8warp")) {
    run_int8_q5(n, csv);
  }
  if (wants_backend(backend, "int8_q6_wmma_m32n8k16")) {
    run_int8_q6(n, csv);
  }
  if (wants_backend(backend, "int8_q7_wmma_m8n32k16")) {
    run_int8_q7(n, csv);
  }
  if (wants_backend(backend, "int8_q8_wmma_m32n64k16_smem")) {
    run_int8_q8(n, csv);
  }
  if (wants_backend(backend, "int8_q9_wmma_m32n32k16_reuse_a")) {
    run_int8_q9(n, csv);
  }
  if (wants_backend(backend, "int8_q10_wmma_m32n64k16_reuse_a")) {
    run_int8_q10(n, csv);
  }
  if (wants_backend(backend, "int8_q11_wmma_m32n128k16_reuse_a")) {
    run_int8_q11(n, csv);
  }
  if (wants_backend(backend, "int8_q12_wmma_m128n64k16_4warp_reuse_a")) {
    run_int8_q12(n, csv);
  }
  if (wants_backend(backend, "int8_q13_wmma_m128n128k16_8warp_reuse_a")) {
    run_int8_q13(n, csv);
  }
  if (wants_backend(backend, "int8_q14_wmma_m256n64k16_8warp_reuse_a")) {
    run_int8_q14(n, csv);
  }
  if (wants_backend(backend, "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol")) {
    run_int8_q15(n, csv);
  }
  if (wants_backend(backend, "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol")) {
    run_int8_q16(n, csv);
  }
  if (wants_backend(backend, "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol")) {
    run_int8_q17(n, csv);
  }
  if (wants_backend(backend, "int8_q18_mma_m16n8k32_smem64")) {
    run_int8_q18(n, csv);
  }
  if (wants_backend(backend, "int8_q19_cublas_gemmex")) {
    run_int8_q19(n, csv);
  }

  csv.close();
  return EXIT_SUCCESS;
}
