#pragma once

#include "gemm_common.cuh"

// v1: naive SGEMM，刻意保留非合并访存的线程映射。
// 连续 threadIdx.x 对应 C 的连续行，访问 B 时读同一列的不连续地址，
// 用来和 v2 的 coalesced 版本形成对照。
__global__ void sgemm_v1_naive_uncoalesced(int m, int n, int k, float alpha,
                                           const float* a, const float* b,
                                           float beta, float* c) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int p = 0; p < k; ++p) {
    acc += a[row * k + p] * b[p * n + col];
  }
  c[row * n + col] = alpha * acc + beta * c[row * n + col];
}

// v2: 合并访存 naive SGEMM。
// 连续 threadIdx.x 对应 C 的连续列，计算同一 p 时访问 B 的连续地址。
// 仍然没有 shared memory 复用，但 global memory 访问形态比 v1 规整很多。
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

// v3: shared memory tiling。
// 一个 block 计算 TILE x TILE 的 C 子块，每轮把 A/B 的一小块搬到 shared memory，
// 让同一个 tile 内的线程复用这批数据，减少 global memory 访问次数。
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

// v4: 1D block + shared memory padding。
// 用 1D thread block 覆盖 TILE x TILE 输出块，并给 A tile 每行加 1 个 padding，
// 避免某些访问模式下多个线程落到同一个 shared memory bank。
template <int TILE>
__global__ void sgemm_v4_smem_1d_padded(int m, int n, int k, float alpha,
                                        const float* a, const float* b,
                                        float beta, float* c) {
  __shared__ float as[TILE][TILE + 1];
  __shared__ float bs[TILE][TILE];

  const int tid = threadIdx.x;
  const int ty = tid / TILE;
  const int tx = tid % TILE;
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

// v6: thread coarsening / thread tile。
// 一个线程不再只算一个 C 元素，而是计算 TM x TN 个元素。
// 好处是 A/B 从 shared memory 读入寄存器后，可以在多个 FMA 中复用，提高计算密度。
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

// v9: float4 向量化搬运 + A tile 转置。
// A 从 global memory 读取时是连续的 float4，写入 shared memory 时转置成 BK x BM。
// 这样计算阶段读取 A 和 B 都更接近连续访问，减少访存指令和 shared memory 压力。
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

// v10 和 v5 共用的向量化 tile 加载函数。
// as_buffer 保存转置后的 A tile，bs_buffer 保存原布局的 B tile。
// 这里假设矩阵尺寸满足高性能路径的对齐约束，边界尺寸由 main 中跳过。
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

// v10: 双缓冲。
// shared memory 开两份 buffer：当前 buffer 用于计算，另一个 buffer 预取下一块 K tile。
// 同时在 BK 内部用两组寄存器 frag_a/frag_b 做一拍预取，尽量隐藏 shared memory 读取延迟。
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

// v5 的全局内存到 shared memory 搬运。
// 与 v4 一样，A tile 转置写入 shared memory，B tile 保持行主序。
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

// v5 的 warp 级计算核心。
// 一个 block tile 被分成多个 warp tile，每个 warp 负责 WM x WN 子块；
// warp 内线程再各自负责 TM x TN 的输出小块。
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

// v5: warp tiling。
// 三级划分：
// 1. block 处理 BM x BN；
// 2. warp 处理 WM x WN；
// 3. thread 处理 TM x TN。
// 这样可以提高 warp 级数据局部性，并让每个 warp 的工作更规则。
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
