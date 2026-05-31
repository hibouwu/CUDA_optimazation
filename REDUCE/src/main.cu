#include "reduce_benchmark.cuh"
#include "reduce_kernels.cuh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

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
  const double expected = cpu_reduce(h_input);

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
  benchmark_reduce("v0 interleaved", reduce_v0_interleaved, d_input,
                   d_partial_a, d_partial_b, n, expected, kBlockSize, csv);
  benchmark_reduce("v1 sequential", reduce_v1_sequential, d_input, d_partial_a,
                   d_partial_b, n, expected, kBlockSize, csv);
  benchmark_reduce("v2 first-add", reduce_v2_first_add, d_input, d_partial_a,
                   d_partial_b, n, expected, kBlockSize * 2, csv);
  benchmark_reduce("v3 unroll-last-warp", reduce_v3_unroll_last_warp, d_input,
                   d_partial_a, d_partial_b, n, expected, kBlockSize * 2, csv);
  benchmark_reduce("v4 shuffle", reduce_v4_shuffle, d_input, d_partial_a,
                   d_partial_b, n, expected, kBlockSize, csv);
  benchmark_reduce("v5 vectorized float4", reduce_v5_vectorized, d_input,
                   d_partial_a, d_partial_b, n, expected, kBlockSize * 4, csv);
  benchmark_cub(d_input, d_cub_output, n, expected, csv);
  csv.close();

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_partial_a));
  CHECK_CUDA(cudaFree(d_partial_b));
  CHECK_CUDA(cudaFree(d_cub_output));
  return 0;
}
