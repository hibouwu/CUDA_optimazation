#pragma once

#include "reduce_common.cuh"

#include <algorithm>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// CPU 参考结果，用 double 顺序累加，避免大 N 下 FP32 小增量被舍入吞掉。
inline double cpu_reduce(const std::vector<float>& data) {
  return std::accumulate(data.begin(), data.end(), 0.0,
                         [](double sum, float value) {
                           return sum + static_cast<double>(value);
                         });
}

// GPU kernel 输出仍然是 FP32 归约结果；不同归约树会有轻微误差，按相对容差校验。
inline bool reduce_result_matched(float actual, double expected) {
  const double diff = abs_double(static_cast<double>(actual) - expected);
  const double scale = std::max(1.0, abs_double(expected));
  return diff <= scale * 1e-4;
}

// 通用 benchmark：把任意 reduce kernel 循环调用，直到 partial 数组归约到 1 个元素。
// d_partial_a / d_partial_b 是 ping-pong buffer，避免为每一轮重新分配显存。
inline float benchmark_reduce(const std::string& name,
                              void (*kernel)(const float*, float*, int),
                              const float* d_input, float* d_partial_a,
                              float* d_partial_b, int n, double expected,
                              int first_pass_items, std::ofstream& csv) {
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
  const bool ok = reduce_result_matched(actual, expected);

  std::cout << name << ": " << avg_ms << " ms, " << bandwidth
            << " GB/s, result=" << actual << ", matched=" << ok << '\n';
  csv << name << "," << n << "," << avg_ms << "," << bandwidth << ","
      << actual << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}

// CUB 生产级 baseline。它通常已经包含架构相关的高度优化策略，
// 用来判断手写版本离库实现还有多少差距。
inline float benchmark_cub(const float* d_input, float* d_output, int n,
                           double expected, std::ofstream& csv) {
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
  const bool ok = reduce_result_matched(actual, expected);

  std::cout << "CUB DeviceReduce::Sum: " << avg_ms << " ms, " << bandwidth
            << " GB/s, result=" << actual << ", matched=" << ok << '\n';
  csv << "CUB DeviceReduce::Sum," << n << "," << avg_ms << "," << bandwidth
      << "," << actual << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}
