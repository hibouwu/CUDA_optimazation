#pragma once

#include "reduce_common.cuh"

// v3 使用的最后一个 warp 手动展开版本。
// 当活跃线程数降到 32 以内时，线程天然在同一个 warp 内同步，
// 可以去掉 __syncthreads()，减少 barrier 和循环控制开销。
__device__ __forceinline__ void warp_reduce_volatile(volatile float* sdata,
                                                     unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// warp 内 shuffle 归约。数据直接在寄存器之间交换，不需要 shared memory。
__device__ __forceinline__ float warp_reduce_shuffle(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// v0: 最直观的交错寻址归约。
// 问题：tid % (2 * stride) 会造成 warp divergence，stride 变大后还容易出现 bank conflict。
__global__ void reduce_v0_interleaved(const float* input, float* partial,
                                      int n) {
  __shared__ float sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < static_cast<unsigned int>(n)) ? input[i] : 0.0f;
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
    if ((tid % (stride << 1)) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// v1: 顺序寻址归约。
// 每一轮让前 stride 个连续线程工作，避免 v0 中偶数/倍数线程工作导致的严重分化。
__global__ void reduce_v1_sequential(const float* input, float* partial,
                                     int n) {
  __shared__ float sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < static_cast<unsigned int>(n)) ? input[i] : 0.0f;
  __syncthreads();

  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// v2: 首轮每个线程加载两个元素并相加。
// 这样每个 block 处理 2 * blockDim.x 个元素，减少第一轮 partial 数量和全局写回。
__global__ void reduce_v2_first_add(const float* input, float* partial, int n) {
  __shared__ float sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

  float value = 0.0f;
  if (i < static_cast<unsigned int>(n)) value += input[i];
  if (i + blockDim.x < static_cast<unsigned int>(n)) {
    value += input[i + blockDim.x];
  }
  sdata[tid] = value;
  __syncthreads();

  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// v3: v2 + 最后一个 warp 手动展开。
// stride > 32 时仍使用 shared memory + __syncthreads()；进入最后 warp 后直接展开。
__global__ void reduce_v3_unroll_last_warp(const float* input, float* partial,
                                           int n) {
  __shared__ float sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

  float value = 0.0f;
  if (i < static_cast<unsigned int>(n)) value += input[i];
  if (i + blockDim.x < static_cast<unsigned int>(n)) {
    value += input[i + blockDim.x];
  }
  sdata[tid] = value;
  __syncthreads();

  for (unsigned int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) warp_reduce_volatile(sdata, tid);
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// v4: grid-stride loop + warp shuffle 两级归约。
// 第一级：每个线程用 grid-stride loop 累加多个元素。
// 第二级：warp 内用 shuffle 归约，每个 warp 只写一个结果到 shared memory。
__global__ void reduce_v4_shuffle(const float* input, float* partial, int n) {
  __shared__ float warp_sums[32];
  float sum = 0.0f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum += input[i];
  }

  sum = warp_reduce_shuffle(sum);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp_id] = sum;
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
  if (warp_id == 0) sum = warp_reduce_shuffle(sum);
  if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

// v5: v4 + float4 向量化读取。
// 用更宽的 global load 指令减少访存指令数量，尾部不足 4 个元素的部分单独处理。
__global__ void reduce_v5_vectorized(const float* input, float* partial, int n) {
  __shared__ float warp_sums[32];
  float sum = 0.0f;

  const int vector_n = n / 4;
  const float4* input4 = reinterpret_cast<const float4*>(input);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vector_n;
       i += gridDim.x * blockDim.x) {
    const float4 value = input4[i];
    sum += value.x + value.y + value.z + value.w;
  }

  for (int i = vector_n * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum += input[i];
  }

  sum = warp_reduce_shuffle(sum);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp_id] = sum;
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
  if (warp_id == 0) sum = warp_reduce_shuffle(sum);
  if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}
