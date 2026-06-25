#pragma once

#include "gemm_common.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// 构造稳定的测试输入。避免全部填 1 导致某些索引错误不容易暴露。
inline void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>((i % 17) - 8) * 0.125f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>((i % 13) - 6) * 0.0625f;
  }
}

inline std::vector<half> to_half_vector(const std::vector<float>& input) {
  std::vector<half> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = __float2half(input[i]);
  }
  return output;
}

inline std::vector<__nv_fp8_e4m3> to_fp8_e4m3_vector(
    const std::vector<float>& input) {
  std::vector<__nv_fp8_e4m3> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = __nv_fp8_e4m3(input[i]);
  }
  return output;
}

inline void cpu_gemm_fp8_e4m3_reference(
    int m, int n, int k, const std::vector<__nv_fp8_e4m3>& a,
    const std::vector<__nv_fp8_e4m3>& b, std::vector<float>& c) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<float>(a[row * k + kk]) *
               static_cast<float>(b[kk * n + col]);
      }
      c[row * n + col] = acc;
    }
  }
}

inline bool compare_gemm_fp8_e4m3_samples(
    int m, int n, int k, const std::vector<__nv_fp8_e4m3>& a,
    const std::vector<__nv_fp8_e4m3>& b, const std::vector<float>& got,
    int samples = 64, float atol = 5e-1f, float rtol = 5e-2f) {
  int errors = 0;
  for (int s = 0; s < samples; ++s) {
    const int row = (s * 131) % m;
    const int col = (s * 197) % n;
    float ref = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      ref += static_cast<float>(a[row * k + kk]) *
             static_cast<float>(b[kk * n + col]);
    }
    const float got_value = got[row * n + col];
    const float diff = abs_float(ref - got_value);
    const float tol = atol + rtol * abs_float(ref);
    if (diff > tol && ++errors <= 5) {
      std::cerr << "FP8 sample mismatch at (" << row << ", " << col
                << "): ref=" << ref << ", got=" << got_value
                << ", diff=" << diff << '\n';
    }
  }
  return errors == 0;
}

// 与 cuBLAS 结果对比。不同归约顺序会带来轻微浮点误差，所以使用相对/绝对容差。
inline bool compare_result(const std::vector<float>& ref,
                           const std::vector<float>& got, float atol = 1e-2f,
                           float rtol = 1e-3f) {
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

inline float gflops(int m, int n, int k, float ms) {
  return 2.0f * static_cast<float>(m) * n * k / (ms * 1.0e6f);
}

// 打印 benchmark 关键环境，避免把旧 cuBLAS / 新 GPU 的组合误解为算法结论。
inline void print_gemm_environment(cublasHandle_t handle) {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  int cublas_version = 0;
  CHECK_CUBLAS(cublasGetVersion(handle, &cublas_version));

  cublasMath_t math_mode;
  CHECK_CUBLAS(cublasGetMathMode(handle, &math_mode));

  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  std::cout << "cuBLAS version: " << cublas_version
            << ", math mode: " << static_cast<int>(math_mode) << '\n';
}

// 通用 kernel benchmark。
// 只把 kernel 调用放进 CUDA event 计时区；host/device 拷贝和校验不计入时间。
template <typename Launch>
float benchmark_kernel(const std::string& backend_id, const std::string& name,
                       const std::string& precision,
                       const std::string& reference_name, Launch launch, int m,
                       int n, int k, float* d_c, size_t c_bytes,
                       const std::vector<float>& ref,
                       std::vector<float>& out, std::ofstream& csv,
                       float reference_gflops, float atol = 1e-2f,
                       float rtol = 1e-3f) {
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
  const bool ok = compare_result(ref, out, atol, rtol);
  const float perf = gflops(m, n, k, avg_ms);
  std::cout << name << ": " << avg_ms << " ms, " << perf
            << " GFLOPS, matched=" << ok << '\n';
  csv << backend_id << "," << name << "," << n << "," << precision << ","
      << reference_name << "," << avg_ms << "," << perf << ","
      << perf / reference_gflops << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}
