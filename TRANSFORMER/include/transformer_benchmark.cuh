#pragma once

#include "transformer_common.cuh"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline void fill_layernorm_inputs(std::vector<float>& input,
                                  std::vector<float>& gamma,
                                  std::vector<float>& beta) {
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i % 29) - 14) * 0.03125f;
  }
  for (size_t i = 0; i < gamma.size(); ++i) {
    gamma[i] = 0.75f + static_cast<float>(i % 7) * 0.03125f;
    beta[i] = static_cast<float>((i % 5) - 2) * 0.015625f;
  }
}

inline void cpu_layernorm(const std::vector<float>& input,
                          const std::vector<float>& gamma,
                          const std::vector<float>& beta,
                          std::vector<float>& output, int rows, int hidden,
                          float eps) {
  for (int row = 0; row < rows; ++row) {
    const size_t base = static_cast<size_t>(row) * hidden;
    double sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      sum += static_cast<double>(input[base + col]);
    }
    const double mean = sum / hidden;

    double var_sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      const double diff = static_cast<double>(input[base + col]) - mean;
      var_sum += diff * diff;
    }
    const float inv_std =
        1.0f / static_cast<float>(std::sqrt(var_sum / hidden + eps));

    for (int col = 0; col < hidden; ++col) {
      output[base + col] =
          (input[base + col] - static_cast<float>(mean)) * inv_std *
              gamma[col] +
          beta[col];
    }
  }
}

inline bool compare_layernorm(const std::vector<float>& ref,
                              const std::vector<float>& got,
                              float atol = 2e-4f, float rtol = 2e-3f) {
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

inline float layernorm_bandwidth_gbps(int rows, int hidden, float ms) {
  // input 读两遍，gamma/beta 各读一遍，output 写一遍。用于统一比较各版本。
  const size_t bytes =
      static_cast<size_t>(rows) * hidden * 5 * sizeof(float);
  return transformer_gbytes(bytes, ms);
}

template <typename Launch>
float benchmark_layernorm(const std::string& name, Launch launch, int rows,
                          int hidden, float* d_output, size_t output_bytes,
                          const std::vector<float>& ref,
                          std::vector<float>& out, std::ofstream& csv,
                          const TransformerShape& shape) {
  CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));
  for (int i = 0; i < kTransformerWarmup; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < kTransformerRepeat; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaMemcpy(out.data(), d_output, output_bytes,
                        cudaMemcpyDeviceToHost));

  const float avg_ms = total_ms / kTransformerRepeat;
  const float bandwidth = layernorm_bandwidth_gbps(rows, hidden, avg_ms);
  const bool ok = compare_layernorm(ref, out);

  std::cout << name << ": " << avg_ms << " ms, " << bandwidth
            << " GB/s, matched=" << ok << '\n';
  csv << "LayerNorm," << name << "," << shape.batch << "," << shape.seq_len
      << "," << shape.hidden << "," << shape.num_heads << ","
      << shape.head_dim << "," << avg_ms << "," << bandwidth << ","
      << (ok ? 1 : 0) << '\n';
  return avg_ms;
}
