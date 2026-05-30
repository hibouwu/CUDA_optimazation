#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err__) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                << " - status " << status__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

constexpr int kWarmup = 5;
constexpr int kRepeat = 10;

float abs_float(float value) { return value < 0.0f ? -value : value; }

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void sgemm_v1_naive(int m, int n, int k, float alpha,
                               const float* a, const float* b, float beta,
                               float* c) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int p = 0; p < k; ++p) {
    acc += a[row * k + p] * b[p * n + col];
  }
  c[row * n + col] = alpha * acc + beta * c[row * n + col];
}

template <int TILE>
__global__ void sgemm_v2_smem(int m, int n, int k, float alpha, const float* a,
                              const float* b, float beta, float* c) {
  __shared__ float as[TILE][TILE];
  __shared__ float bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  for (int tile = 0; tile < k; tile += TILE) {
    as[ty][tx] = (row < m && tile + tx < k) ? a[row * k + tile + tx] : 0.0f;
    bs[ty][tx] = (tile + ty < k && col < n) ? b[(tile + ty) * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int p = 0; p < TILE; ++p) {
      acc += as[ty][p] * bs[p][tx];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = alpha * acc + beta * c[row * n + col];
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v3_thread_tile(int m, int n, int k, float alpha,
                                     const float* a, const float* b, float beta,
                                     float* c) {
  __shared__ float as[BM * BK];
  __shared__ float bs[BK * BN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  constexpr int threads_per_row = BN / TN;
  const int thread_row = tid / threads_per_row;
  const int thread_col = tid % threads_per_row;
  const int row_base = thread_row * TM;
  const int col_base = thread_col * TN;

  float acc[TM][TN] = {0.0f};

  for (int tile = 0; tile < k; tile += BK) {
    for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
      const int r = idx / BK;
      const int p = idx % BK;
      const int gr = block_row + r;
      const int gp = tile + p;
      as[idx] = (gr < m && gp < k) ? a[gr * k + gp] : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
      const int p = idx / BN;
      const int col = idx % BN;
      const int gp = tile + p;
      const int gc = block_col + col;
      bs[idx] = (gp < k && gc < n) ? b[gp * n + gc] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < BK; ++p) {
      float frag_a[TM];
      float frag_b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        frag_a[i] = as[(row_base + i) * BK + p];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        frag_b[j] = bs[p * BN + col_base + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += frag_a[i] * frag_b[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = block_row + row_base + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = block_col + col_base + j;
      if (row < m && col < n) {
        c[row * n + col] = alpha * acc[i][j] + beta * c[row * n + col];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4_vectorized(int m, int n, int k, float alpha,
                                    const float* a, const float* b, float beta,
                                    float* c) {
  __shared__ float as[BK * BM];
  __shared__ float bs[BK * BN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  constexpr int threads_per_row = BN / TN;
  const int thread_row = tid / threads_per_row;
  const int thread_col = tid % threads_per_row;
  const int row_base = thread_row * TM;
  const int col_base = thread_col * TN;

  float acc[TM][TN] = {0.0f};

  for (int tile = 0; tile < k; tile += BK) {
    for (int vec = tid; vec < (BM * BK) / 4; vec += blockDim.x) {
      const int linear = vec * 4;
      const int r = linear / BK;
      const int p = linear % BK;
      const float4 values =
          reinterpret_cast<const float4*>(&a[(block_row + r) * k + tile + p])[0];
      as[(p + 0) * BM + r] = values.x;
      as[(p + 1) * BM + r] = values.y;
      as[(p + 2) * BM + r] = values.z;
      as[(p + 3) * BM + r] = values.w;
    }

    for (int vec = tid; vec < (BK * BN) / 4; vec += blockDim.x) {
      const int linear = vec * 4;
      const int p = linear / BN;
      const int col = linear % BN;
      reinterpret_cast<float4*>(&bs[p * BN + col])[0] =
          reinterpret_cast<const float4*>(
              &b[(tile + p) * n + block_col + col])[0];
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < BK; ++p) {
      float frag_a[TM];
      float frag_b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        frag_a[i] = as[p * BM + row_base + i];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        frag_b[j] = bs[p * BN + col_base + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += frag_a[i] * frag_b[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = block_row + row_base + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = block_col + col_base + j;
      c[row * n + col] = alpha * acc[i][j] + beta * c[row * n + col];
    }
  }
}

template <int BM, int BN, int BK>
__device__ __forceinline__ void load_vectorized_tile(
    int tid, int block_row, int block_col, int n, int k, int tile,
    const float* a, const float* b, float* as_buffer, float* bs_buffer) {
  for (int vec = tid; vec < (BM * BK) / 4; vec += blockDim.x) {
    const int linear = vec * 4;
    const int r = linear / BK;
    const int p = linear % BK;
    const float4 values =
        reinterpret_cast<const float4*>(&a[(block_row + r) * k + tile + p])[0];
    as_buffer[(p + 0) * BM + r] = values.x;
    as_buffer[(p + 1) * BM + r] = values.y;
    as_buffer[(p + 2) * BM + r] = values.z;
    as_buffer[(p + 3) * BM + r] = values.w;
  }

  for (int vec = tid; vec < (BK * BN) / 4; vec += blockDim.x) {
    const int linear = vec * 4;
    const int p = linear / BN;
    const int col = linear % BN;
    reinterpret_cast<float4*>(&bs_buffer[p * BN + col])[0] =
        reinterpret_cast<const float4*>(
            &b[(tile + p) * n + block_col + col])[0];
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5_double_buffer(int m, int n, int k, float alpha,
                                       const float* a, const float* b,
                                       float beta, float* c) {
  __shared__ float as[2][BK * BM];
  __shared__ float bs[2][BK * BN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  constexpr int threads_per_row = BN / TN;
  const int thread_row = tid / threads_per_row;
  const int thread_col = tid % threads_per_row;
  const int row_base = thread_row * TM;
  const int col_base = thread_col * TN;

  float acc[TM][TN] = {0.0f};

  load_vectorized_tile<BM, BN, BK>(tid, block_row, block_col, n, k, 0, a, b,
                                   as[0], bs[0]);
  __syncthreads();

  for (int tile = 0, read_buffer = 0; tile < k; tile += BK, read_buffer ^= 1) {
    const int next_tile = tile + BK;
    const int write_buffer = read_buffer ^ 1;
    if (next_tile < k) {
      load_vectorized_tile<BM, BN, BK>(tid, block_row, block_col, n, k,
                                       next_tile, a, b, as[write_buffer],
                                       bs[write_buffer]);
    }

    float frag_a[2][TM];
    float frag_b[2][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      frag_a[0][i] = as[read_buffer][row_base + i];
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      frag_b[0][j] = bs[read_buffer][col_base + j];
    }

#pragma unroll
    for (int p = 0; p < BK; ++p) {
      const int next_p = p + 1;
      if (next_p < BK) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          frag_a[next_p & 1][i] =
              as[read_buffer][next_p * BM + row_base + i];
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          frag_b[next_p & 1][j] =
              bs[read_buffer][next_p * BN + col_base + j];
        }
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += frag_a[p & 1][i] * frag_b[p & 1][j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = block_row + row_base + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = block_col + col_base + j;
      c[row * n + col] = alpha * acc[i][j] + beta * c[row * n + col];
    }
  }
}

constexpr int kWarpSize = 32;

template <int BM, int BN, int BK, int RowStrideA, int RowStrideB>
__device__ __forceinline__ void load_warp_tile_from_gmem(
    int n, int k, const float* a, const float* b, float* as, float* bs,
    int inner_row_a, int inner_col_a, int inner_row_b, int inner_col_b) {
  for (int offset = 0; offset + RowStrideA <= BM; offset += RowStrideA) {
    const float4 values = reinterpret_cast<const float4*>(
        &a[(inner_row_a + offset) * k + inner_col_a * 4])[0];
    as[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = values.x;
    as[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = values.y;
    as[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = values.z;
    as[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = values.w;
  }

  for (int offset = 0; offset + RowStrideB <= BK; offset += RowStrideB) {
    reinterpret_cast<float4*>(
        &bs[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
        reinterpret_cast<const float4*>(
            &b[(inner_row_b + offset) * n + inner_col_b * 4])[0];
  }
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER,
          int WSUBM, int WSUBN, int TM, int TN>
__device__ __forceinline__ void process_warp_tile_from_smem(
    float* reg_m, float* reg_n, float* thread_results, const float* as,
    const float* bs, int warp_row, int warp_col, int thread_row_in_warp,
    int thread_col_in_warp) {
  for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
#pragma unroll
    for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        reg_m[w_sub_row_idx * TM + i] =
            as[dot_idx * BM + warp_row * WM + w_sub_row_idx * WSUBM +
               thread_row_in_warp * TM + i];
      }
    }
#pragma unroll
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
#pragma unroll
      for (int i = 0; i < TN; ++i) {
        reg_n[w_sub_col_idx * TN + i] =
            bs[dot_idx * BN + warp_col * WN + w_sub_col_idx * WSUBN +
               thread_col_in_warp * TN + i];
      }
    }

#pragma unroll
    for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
#pragma unroll
      for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
#pragma unroll
        for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
#pragma unroll
          for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            thread_results[(w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                           w_sub_col_idx * TN + res_idx_n] +=
                reg_m[w_sub_row_idx * TM + res_idx_m] *
                reg_n[w_sub_col_idx * TN + res_idx_n];
          }
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NumThreads>
__global__ void __launch_bounds__(NumThreads)
    sgemm_v6_warp_tiling(int m, int n, int k, float alpha, const float* a,
                         const float* b, float beta, float* c) {
  const int c_row = blockIdx.y;
  const int c_col = blockIdx.x;

  const int warp_idx = threadIdx.x / kWarpSize;
  const int warp_col = warp_idx % (BN / WN);
  const int warp_row = warp_idx / (BN / WN);

  constexpr int WMITER = (WM * WN) / (kWarpSize * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int thread_idx_in_warp = threadIdx.x % kWarpSize;
  const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
  const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

  __shared__ float as[BM * BK];
  __shared__ float bs[BK * BN];

  a += c_row * BM * k;
  b += c_col * BN;
  c += (c_row * BM + warp_row * WM) * n + c_col * BN + warp_col * WN;

  const int inner_row_a = threadIdx.x / (BK / 4);
  const int inner_col_a = threadIdx.x % (BK / 4);
  constexpr int RowStrideA = (NumThreads * 4) / BK;
  const int inner_row_b = threadIdx.x / (BN / 4);
  const int inner_col_b = threadIdx.x % (BN / 4);
  constexpr int RowStrideB = NumThreads / (BN / 4);

  float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
  float reg_m[WMITER * TM] = {0.0f};
  float reg_n[WNITER * TN] = {0.0f};

  for (int bk_idx = 0; bk_idx < k; bk_idx += BK) {
    load_warp_tile_from_gmem<BM, BN, BK, RowStrideA, RowStrideB>(
        n, k, a, b, as, bs, inner_row_a, inner_col_a, inner_row_b,
        inner_col_b);
    __syncthreads();
    process_warp_tile_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM,
                                WSUBN, TM, TN>(
        reg_m, reg_n, thread_results, as, bs, warp_row, warp_col,
        thread_row_in_warp, thread_col_in_warp);
    a += BK;
    b += BK * n;
    __syncthreads();
  }

#pragma unroll
  for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
#pragma unroll
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
      float* c_interim =
          c + w_sub_row_idx * WSUBM * n + w_sub_col_idx * WSUBN;
#pragma unroll
      for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
#pragma unroll
        for (int res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
          float4 values = reinterpret_cast<float4*>(
              &c_interim[(thread_row_in_warp * TM + res_idx_m) * n +
                         thread_col_in_warp * TN + res_idx_n])[0];
          const int result_idx =
              (w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) +
              w_sub_col_idx * TN + res_idx_n;
          values.x = alpha * thread_results[result_idx + 0] + beta * values.x;
          values.y = alpha * thread_results[result_idx + 1] + beta * values.y;
          values.z = alpha * thread_results[result_idx + 2] + beta * values.z;
          values.w = alpha * thread_results[result_idx + 3] + beta * values.w;
          reinterpret_cast<float4*>(
              &c_interim[(thread_row_in_warp * TM + res_idx_m) * n +
                         thread_col_in_warp * TN + res_idx_n])[0] = values;
        }
      }
    }
  }
}

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>((i % 17) - 8) * 0.125f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>((i % 13) - 6) * 0.0625f;
  }
}

bool compare_result(const std::vector<float>& ref, const std::vector<float>& got,
                    float atol = 1e-2f, float rtol = 1e-3f) {
  int errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float diff = abs_float(ref[i] - got[i]);
    const float tol = atol + rtol * abs_float(ref[i]);
    if (diff > tol && ++errors <= 5) {
      std::cerr << "Mismatch at " << i << ": ref=" << ref[i]
                << ", got=" << got[i] << ", diff=" << diff << '\n';
    }
  }
  return errors == 0;
}

float gflops(int m, int n, int k, float ms) {
  return 2.0f * static_cast<float>(m) * n * k / (ms * 1.0e6f);
}

template <typename Launch>
float benchmark_kernel(const std::string& name, Launch launch, int m, int n,
                       int k, float* d_c, size_t c_bytes,
                       const std::vector<float>& ref,
                       std::vector<float>& out) {
  CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));
  for (int i = 0; i < kWarmup; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));

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
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const float avg_ms = total_ms / kRepeat;
  const bool ok = compare_result(ref, out);
  std::cout << name << ": " << avg_ms << " ms, " << gflops(m, n, k, avg_ms)
            << " GFLOPS, matched=" << ok << '\n';
  return avg_ms;
}

int main(int argc, char** argv) {
  int n = 1024;
  if (argc > 1) n = std::atoi(argv[1]);
  if (n <= 0) {
    std::cerr << "Usage: " << argv[0] << " [square_size]\n";
    return EXIT_FAILURE;
  }
  const int m = n;
  const int k = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);

  std::vector<float> h_a(static_cast<size_t>(m) * k);
  std::vector<float> h_b(static_cast<size_t>(k) * n);
  std::vector<float> h_ref(static_cast<size_t>(m) * n);
  std::vector<float> h_out(static_cast<size_t>(m) * n);
  fill_inputs(h_a, h_b);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&d_c, c_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  auto launch_cublas = [&]() {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_b, n, d_a, k, &beta, d_c, n));
  };

  CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));
  for (int i = 0; i < kWarmup; ++i) launch_cublas();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < kRepeat; ++i) {
    launch_cublas();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float cublas_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const float cublas_avg_ms = cublas_ms / kRepeat;
  std::cout << "N=" << n << '\n';
  std::cout << "cuBLAS: " << cublas_avg_ms << " ms, "
            << gflops(m, n, k, cublas_avg_ms) << " GFLOPS\n";

  dim3 v1_block(16, 16);
  dim3 v1_grid(ceil_div(n, v1_block.x), ceil_div(m, v1_block.y));
  benchmark_kernel(
      "v1 naive",
      [&]() {
        sgemm_v1_naive<<<v1_grid, v1_block>>>(m, n, k, alpha, d_a, d_b, beta,
                                              d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out);

  dim3 v2_block(32, 32);
  dim3 v2_grid(ceil_div(n, 32), ceil_div(m, 32));
  benchmark_kernel(
      "v2 shared-memory tile",
      [&]() {
        sgemm_v2_smem<32><<<v2_grid, v2_block>>>(m, n, k, alpha, d_a, d_b,
                                                 beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out);

  constexpr int V3_BM = 64;
  constexpr int V3_BN = 64;
  constexpr int V3_BK = 8;
  constexpr int V3_TM = 4;
  constexpr int V3_TN = 4;
  dim3 v3_grid(ceil_div(n, V3_BN), ceil_div(m, V3_BM));
  dim3 v3_block((V3_BM / V3_TM) * (V3_BN / V3_TN));
  benchmark_kernel(
      "v3 thread tile",
      [&]() {
        sgemm_v3_thread_tile<V3_BM, V3_BN, V3_BK, V3_TM, V3_TN>
            <<<v3_grid, v3_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out);

  if (n % 128 == 0) {
    constexpr int V4_BM = 128;
    constexpr int V4_BN = 128;
    constexpr int V4_BK = 8;
    constexpr int V4_TM = 8;
    constexpr int V4_TN = 8;
    dim3 v4_grid(n / V4_BN, m / V4_BM);
    dim3 v4_block((V4_BM / V4_TM) * (V4_BN / V4_TN));
    benchmark_kernel(
        "v4 vectorized",
        [&]() {
          sgemm_v4_vectorized<V4_BM, V4_BN, V4_BK, V4_TM, V4_TN>
              <<<v4_grid, v4_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out);

    benchmark_kernel(
        "v5 double buffer",
        [&]() {
          sgemm_v5_double_buffer<V4_BM, V4_BN, V4_BK, V4_TM, V4_TN>
              <<<v4_grid, v4_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out);

    constexpr int V6_NUM_THREADS = 128;
    constexpr int V6_BM = 128;
    constexpr int V6_BN = 128;
    constexpr int V6_BK = 16;
    constexpr int V6_WM = 64;
    constexpr int V6_WN = 64;
    constexpr int V6_WNITER = 4;
    constexpr int V6_TM = 8;
    constexpr int V6_TN = 4;
    static_assert((V6_BN % V6_WN == 0) && (V6_BM % V6_WM == 0));
    static_assert((V6_BN / V6_WN) * (V6_BM / V6_WM) ==
                  V6_NUM_THREADS / kWarpSize);
    static_assert((V6_NUM_THREADS * 4) % V6_BK == 0);
    static_assert((V6_NUM_THREADS * 4) % V6_BN == 0);
    static_assert((V6_BM * V6_BK) % (4 * V6_NUM_THREADS) == 0);
    static_assert((V6_BN * V6_BK) % (4 * V6_NUM_THREADS) == 0);
    dim3 v6_grid(n / V6_BN, m / V6_BM);
    dim3 v6_block(V6_NUM_THREADS);
    benchmark_kernel(
        "v6 warp tiling",
        [&]() {
          sgemm_v6_warp_tiling<V6_BM, V6_BN, V6_BK, V6_WM, V6_WN,
                               V6_WNITER, V6_TM, V6_TN, V6_NUM_THREADS>
              <<<v6_grid, v6_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out);
  } else {
    std::cout << "v4/v5/v6: skipped because N must be a multiple of 128\n";
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
