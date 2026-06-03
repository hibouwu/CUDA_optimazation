#pragma once

#include "transformer_common.cuh"

// block 内 FP32 求和。LayerNorm 的 mean/variance 都是按一行 hidden 维度归约。
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float shared[kTransformerBlockSize];
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile float* vshared = shared;
    vshared[tid] += vshared[tid + 32];
    vshared[tid] += vshared[tid + 16];
    vshared[tid] += vshared[tid + 8];
    vshared[tid] += vshared[tid + 4];
    vshared[tid] += vshared[tid + 2];
    vshared[tid] += vshared[tid + 1];
  }
  __syncthreads();
  return shared[0];
}

// v1: naive LayerNorm。
// 一个线程处理一整行，串行计算 mean/variance 并写回。逻辑最直观，但没有并行归约。
__global__ void layernorm_v1_naive(const float* input, const float* gamma,
                                   const float* beta, float* output, int rows,
                                   int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;

  const float* x = input + static_cast<size_t>(row) * hidden;
  float* y = output + static_cast<size_t>(row) * hidden;

  float sum = 0.0f;
  for (int col = 0; col < hidden; ++col) {
    sum += x[col];
  }
  const float mean = sum / hidden;

  float variance_sum = 0.0f;
  for (int col = 0; col < hidden; ++col) {
    const float diff = x[col] - mean;
    variance_sum += diff * diff;
  }
  const float inv_std = rsqrtf(variance_sum / hidden + eps);

  for (int col = 0; col < hidden; ++col) {
    y[col] = (x[col] - mean) * inv_std * gamma[col] + beta[col];
  }
}

// v2: block reduction LayerNorm。
// 一个 block 处理一行，线程并行遍历 hidden 维度；mean 和 variance 用 shared memory 归约。
__global__ void layernorm_v2_block_reduce(const float* input,
                                          const float* gamma,
                                          const float* beta, float* output,
                                          int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const float* x = input + static_cast<size_t>(row) * hidden;
  float* y = output + static_cast<size_t>(row) * hidden;

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    local_sum += x[col];
  }
  const float mean = block_reduce_sum(local_sum) / hidden;
  __syncthreads();

  float local_var = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = x[col] - mean;
    local_var += diff * diff;
  }
  const float inv_std = rsqrtf(block_reduce_sum(local_var) / hidden + eps);
  __syncthreads();

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    y[col] = (x[col] - mean) * inv_std * gamma[col] + beta[col];
  }
}

// v3: vectorized LayerNorm。
// 在 v2 的基础上用 float4 读取和写回，减少 memory instruction 数量。
// hidden 不是 4 的倍数时，尾部仍用标量处理。
__global__ void layernorm_v3_vectorized(const float* input, const float* gamma,
                                        const float* beta, float* output,
                                        int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const float* x = input + static_cast<size_t>(row) * hidden;
  float* y = output + static_cast<size_t>(row) * hidden;
  const int hidden4 = hidden / 4;
  const float4* x4 = reinterpret_cast<const float4*>(x);

  float local_sum = 0.0f;
  for (int idx = threadIdx.x; idx < hidden4; idx += blockDim.x) {
    const float4 value = x4[idx];
    local_sum += value.x + value.y + value.z + value.w;
  }
  for (int col = hidden4 * 4 + threadIdx.x; col < hidden; col += blockDim.x) {
    local_sum += x[col];
  }
  const float mean = block_reduce_sum(local_sum) / hidden;
  __syncthreads();

  float local_var = 0.0f;
  for (int idx = threadIdx.x; idx < hidden4; idx += blockDim.x) {
    const float4 value = x4[idx];
    const float dx = value.x - mean;
    const float dy = value.y - mean;
    const float dz = value.z - mean;
    const float dw = value.w - mean;
    local_var += dx * dx + dy * dy + dz * dz + dw * dw;
  }
  for (int col = hidden4 * 4 + threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = x[col] - mean;
    local_var += diff * diff;
  }
  const float inv_std = rsqrtf(block_reduce_sum(local_var) / hidden + eps);
  __syncthreads();

  const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
  const float4* beta4 = reinterpret_cast<const float4*>(beta);
  float4* y4 = reinterpret_cast<float4*>(y);
  for (int idx = threadIdx.x; idx < hidden4; idx += blockDim.x) {
    const float4 value = x4[idx];
    const float4 g = gamma4[idx];
    const float4 b = beta4[idx];
    float4 out;
    out.x = (value.x - mean) * inv_std * g.x + b.x;
    out.y = (value.y - mean) * inv_std * g.y + b.y;
    out.z = (value.z - mean) * inv_std * g.z + b.z;
    out.w = (value.w - mean) * inv_std * g.w + b.w;
    y4[idx] = out;
  }
  for (int col = hidden4 * 4 + threadIdx.x; col < hidden; col += blockDim.x) {
    y[col] = (x[col] - mean) * inv_std * gamma[col] + beta[col];
  }
}
