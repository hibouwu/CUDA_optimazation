#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
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

constexpr int kBlockSize = 256;
constexpr int kWarmup = 5;
constexpr int kRepeat = 20;

float abs_float(float value) { return value < 0.0f ? -value : value; }

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

// CPU 参考结果，用于正确性校验。
float cpu_reduce(const std::vector<float>& data) {
  return std::accumulate(data.begin(), data.end(), 0.0f);
}

// 通用 benchmark：把任意 reduce kernel 循环调用，直到 partial 数组归约到 1 个元素。
// d_partial_a / d_partial_b 是 ping-pong buffer，避免为每一轮重新分配显存。
float benchmark(const std::string& name, void (*kernel)(const float*, float*, int),
                const float* d_input, float* d_partial_a, float* d_partial_b,
                int n, float expected, int first_pass_items,
                std::ofstream& csv) {
  auto launch_all = [&]() -> const float* {
    const float* in = d_input;
    float* out = d_partial_a;
    float* next_out = d_partial_b;
    int current_n = n;

    while (current_n > 1) {
      const int blocks = (current_n + first_pass_items - 1) / first_pass_items;
      kernel<<<blocks, kBlockSize>>>(in, out, current_n);
      CHECK_CUDA(cudaGetLastError());
      current_n = blocks;
      if (current_n == 1) return out;
      in = out;
      std::swap(out, next_out);
    }
    return in;
  };

  for (int i = 0; i < kWarmup; ++i) {
    launch_all();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  const float* result_ptr = nullptr;
  for (int i = 0; i < kRepeat; ++i) {
    result_ptr = launch_all();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  float actual = 0.0f;
  CHECK_CUDA(cudaMemcpy(&actual, result_ptr, sizeof(float),
                        cudaMemcpyDeviceToHost));

  const float avg_ms = ms / kRepeat;
  const float bandwidth =
      static_cast<float>(n) * sizeof(float) / (avg_ms * 1.0e6f);
  const bool ok =
      abs_float(actual - expected) <= std::max(1.0f, expected) * 1e-4f;

  std::cout << name << ": " << avg_ms << " ms, " << bandwidth
            << " GB/s, result=" << actual << ", matched=" << ok << '\n';
  csv << name << "," << n << "," << avg_ms << "," << bandwidth << ","
      << actual << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}

// CUB 生产级 baseline。它通常已经包含架构相关的高度优化策略，
// 用来判断手写版本离库实现还有多少差距。
float benchmark_cub(const float* d_input, float* d_output, int n,
                    float expected, std::ofstream& csv) {
  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  CHECK_CUDA(cub::DeviceReduce::Sum(temp_storage, temp_bytes, d_input, d_output,
                                    n));
  CHECK_CUDA(cudaMalloc(&temp_storage, temp_bytes));

  for (int i = 0; i < kWarmup; ++i) {
    CHECK_CUDA(cub::DeviceReduce::Sum(temp_storage, temp_bytes, d_input,
                                      d_output, n));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < kRepeat; ++i) {
    CHECK_CUDA(cub::DeviceReduce::Sum(temp_storage, temp_bytes, d_input,
                                      d_output, n));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  float actual = 0.0f;
  CHECK_CUDA(cudaMemcpy(&actual, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(temp_storage));

  const float avg_ms = ms / kRepeat;
  const float bandwidth =
      static_cast<float>(n) * sizeof(float) / (avg_ms * 1.0e6f);
  const bool ok =
      abs_float(actual - expected) <= std::max(1.0f, expected) * 1e-4f;

  std::cout << "CUB DeviceReduce::Sum: " << avg_ms << " ms, " << bandwidth
            << " GB/s, result=" << actual << ", matched=" << ok << '\n';
  csv << "CUB DeviceReduce::Sum," << n << "," << avg_ms << "," << bandwidth
      << "," << actual << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}

int main(int argc, char** argv) {
  int n = 1 << 24;
  if (argc > 1) n = std::atoi(argv[1]);
  if (n <= 0) {
    std::cerr << "Usage: " << argv[0] << " [num_elements]\n";
    return EXIT_FAILURE;
  }

  std::vector<float> h_input(n);
  for (int i = 0; i < n; ++i) {
    // 使用非纯 1.0 数据，避免某些错误实现因为输入过于简单而侥幸通过。
    h_input[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
  }
  const float expected = cpu_reduce(h_input);

  const int max_blocks = (n + kBlockSize - 1) / kBlockSize;
  float *d_input = nullptr, *d_partial_a = nullptr, *d_partial_b = nullptr;
  float* d_cub_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_a,
                        static_cast<size_t>(max_blocks) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_b,
                        static_cast<size_t>(max_blocks) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cub_output, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                        static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 同步输出 CSV，方便后续直接用 Python/表格画性能曲线。
  std::ofstream csv("reduce_benchmark.csv");
  csv << "Version,N,TimeMs,BandwidthGBps,Result,Matched\n";

  std::cout << "N=" << n << ", CPU result=" << expected << '\n';
  // 从最朴素版本一路跑到手写优化版和 CUB 基线，输出时间、带宽和校验结果。
  benchmark("v0 interleaved", reduce_v0_interleaved, d_input, d_partial_a,
            d_partial_b, n, expected, kBlockSize, csv);
  benchmark("v1 sequential", reduce_v1_sequential, d_input, d_partial_a,
            d_partial_b, n, expected, kBlockSize, csv);
  benchmark("v2 first-add", reduce_v2_first_add, d_input, d_partial_a,
            d_partial_b, n, expected, kBlockSize * 2, csv);
  benchmark("v3 unroll-last-warp", reduce_v3_unroll_last_warp, d_input,
            d_partial_a, d_partial_b, n, expected, kBlockSize * 2, csv);
  benchmark("v4 shuffle", reduce_v4_shuffle, d_input, d_partial_a, d_partial_b,
            n, expected, kBlockSize, csv);
  benchmark("v5 vectorized float4", reduce_v5_vectorized, d_input, d_partial_a,
            d_partial_b, n, expected, kBlockSize * 4, csv);
  benchmark_cub(d_input, d_cub_output, n, expected, csv);
  csv.close();

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_partial_a));
  CHECK_CUDA(cudaFree(d_partial_b));
  CHECK_CUDA(cudaFree(d_cub_output));
  return 0;
}
