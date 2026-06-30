#include "gemm_benchmark.cuh"
#include "cutlass_sm110_backends.cuh"
#include "custom_sm110_gemm.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kBackendUsage =
    "[all|cublas_tc|cutlass|tc3|tc4|tc5|tc5a|tc5b]";

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " [square_size] " << kBackendUsage
            << '\n';
}

bool is_valid_backend_filter(const std::string& filter) {
  return filter == "all" || filter == "cublas_tc" || filter == "cutlass" ||
         filter == "tc3" || filter == "tc4" || filter == "tc5" ||
         filter == "tc5a" || filter == "tc5b";
}

bool needs_cublas_reference(const std::string& filter) {
  (void)filter;
  return true;
}

bool wants_backend(const std::string& filter, const std::string& backend) {
  if (filter == "tc5") {
    return backend == "tc5a" || backend == "tc5b";
  }
  return filter == "all" || filter == backend;
}

template <typename Launch>
float benchmark_reference(Launch launch, float* d_c, size_t c_bytes,
                          std::vector<float>& h_ref) {
  CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));
  for (int i = 0; i < kWarmup; ++i) launch();
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
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
  return total_ms / kRepeat;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  int n = 1024;
  if (argc > 1) n = std::atoi(argv[1]);

  std::string backend_filter = "all";
  if (argc > 2) backend_filter = argv[2];

  if (n <= 0 || !is_valid_backend_filter(backend_filter)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const bool run_reference = needs_cublas_reference(backend_filter);
  const int m = n;
  const int k = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  const size_t d_c_elements = static_cast<size_t>(m) * n;
  const size_t d_c_bytes = d_c_elements * sizeof(float);

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  // CUDA 12+ initializes the runtime and primary context in cudaSetDevice.
  // Do not use the legacy cudaFree(0) initialization idiom: it returns
  // cudaErrorNotSupported on some Thor BSP/runtime combinations.
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp device_prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device));

  std::cout << "GPU: " << device_prop.name << " (sm_" << device_prop.major
            << device_prop.minor << ")\n";

  float* d_c = nullptr;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  if (d_c_bytes > 0) {
    CHECK_CUDA(cudaMalloc(&d_c, d_c_bytes));
  }

  const bool device_supports_tc3_sm110 = device_prop.major == 11;
  const bool device_supports_tc5_sm110 = device_prop.major == 11;

  std::ofstream csv("sgemm_sm110_benchmark.csv");
  csv << "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,"
         "RatioToReference,Matched\n";

  std::cout << "N=" << n << '\n';
  std::cout << "Backend filter=" << backend_filter << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat
            << " per backend before moving to the next backend\n";

  const size_t a_half_bytes = static_cast<size_t>(m) * k * sizeof(half);
  const size_t b_half_bytes = static_cast<size_t>(k) * n * sizeof(half);
  std::vector<float> h_a(static_cast<size_t>(m) * k);
  std::vector<float> h_b(static_cast<size_t>(k) * n);
  std::vector<float> h_ref_tc(static_cast<size_t>(m) * n);
  fill_inputs(h_a, h_b);

  std::vector<half> h_a_half = to_half_vector(h_a);
  std::vector<half> h_b_half = to_half_vector(h_b);

  CHECK_CUDA(cudaMalloc(&d_a_half, a_half_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_half, b_half_bytes));
  CHECK_CUDA(cudaMemcpy(d_a_half, h_a_half.data(), a_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half, h_b_half.data(), b_half_bytes,
                        cudaMemcpyHostToDevice));

  float cublas_tc_perf = 0.0f;
  if (run_reference) {
    cublasHandle_t tensor_core_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&tensor_core_handle));
    print_gemm_environment(tensor_core_handle);

    auto launch_cublas_tensor_core = [&]() {
      CHECK_CUBLAS(cublasGemmEx(
          tensor_core_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
          d_b_half, CUDA_R_16F, n, d_a_half, CUDA_R_16F, k, &beta, d_c,
          CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
    };

    const float cublas_avg_ms =
        benchmark_reference(launch_cublas_tensor_core, d_c, c_bytes, h_ref_tc);
    cublas_tc_perf = gflops(m, n, k, cublas_avg_ms);

    std::cout << "Tensor Core reference:\n";
    std::cout << "cuBLAS Tensor Core: " << cublas_avg_ms << " ms, "
              << cublas_tc_perf << " GFLOPS\n";
    csv << "cublas_tc,cuBLAS Tensor Core," << n
        << ",fp16->fp32,cuBLAS Tensor Core," << cublas_avg_ms << ","
        << cublas_tc_perf << ",1,1\n";
    CHECK_CUBLAS(cublasDestroy(tensor_core_handle));
  }

  if (wants_backend(backend_filter, "cutlass")) {
    if (!device_supports_tc3_sm110) {
      std::cout << "CUTLASS official Blackwell auto-schedule GEMM: skipped "
                   "because runtime GPU is not an sm110 family target\n";
      csv << "cutlass,CUTLASS official Blackwell auto-schedule GEMM skipped,"
          << n << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    } else {
      using CutlassOfficialRunner =
          gemm_sm110::cutlass_backend::Runner<
              gemm_sm110::cutlass_backend::CutlassOfficialConfig>;
      CutlassOfficialRunner cutlass_runner(d_a_half, d_b_half, d_c, m, n, k);
      auto launch_cutlass = [&]() { cutlass_runner.launch(); };
      std::vector<float> h_cutlass(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "cutlass", "CUTLASS official Blackwell auto-schedule GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch_cutlass, m, n, k, d_c,
          c_bytes, h_ref_tc, h_cutlass, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc3")) {
    if (!device_supports_tc3_sm110) {
      std::cout << "tc3 custom TCGen05 GEMM: skipped because runtime "
                   "GPU is not an sm110 family target with tcgen05 enabled\n";
      csv << "tc3,tc3 custom TCGen05 GEMM skipped," << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    } else {
      using Tc3Runner = gemm_sm110::custom_backend::Tc3Runner;
      Tc3Runner tc3_runner(d_a_half, d_b_half, d_c, m, n, k);
      auto launch_tc3 = [&]() {
        tc3_runner.launch();
      };
      std::vector<float> h_tc3(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc3", "tc3 custom cooperative-copy TCGen05 GEMM", "fp16->fp32",
          "cuBLAS Tensor Core", launch_tc3, m, n, k, d_c, c_bytes, h_ref_tc,
          h_tc3, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc4")) {
    if (!device_supports_tc3_sm110) {
      std::cout << "tc4 custom TMA TCGen05 GEMM: skipped because "
                   "runtime GPU is not an sm110 family target\n";
      csv << "tc4,tc4 custom TMA TCGen05 GEMM skipped," << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    } else {
      using Tc4Runner = gemm_sm110::custom_backend::Tc4Runner;
      Tc4Runner tc4_runner(d_a_half, d_b_half, d_c, m, n, k);
      auto launch_tc4 = [&]() { tc4_runner.launch(); };
      std::vector<float> h_tc4(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4", "tc4 custom TMA TCGen05 GEMM", "fp16->fp32",
          "cuBLAS Tensor Core", launch_tc4, m, n, k, d_c, c_bytes, h_ref_tc,
          h_tc4, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5a") ||
      wants_backend(backend_filter, "tc5b")) {
    if (!device_supports_tc5_sm110) {
      if (wants_backend(backend_filter, "tc5a")) {
        std::cout << "tc5a sm110 static persistent TCGen05 GEMM: "
                     "skipped because runtime GPU is not an sm110 family "
                     "target with tcgen05 enabled\n";
        csv << "tc5a,tc5a sm110 static persistent TCGen05 GEMM "
               "skipped,"
            << n << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "tc5b")) {
        std::cout << "tc5b sm110 dynamic CLC persistent TCGen05 GEMM: "
                     "skipped because runtime GPU is not an sm110 family "
                     "target with tcgen05 enabled\n";
        csv << "tc5b,tc5b sm110 dynamic CLC persistent TCGen05 GEMM "
               "skipped,"
            << n << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
      }
    } else {
      if (wants_backend(backend_filter, "tc5a")) {
        using Tc5StaticRunner =
            gemm_sm110::custom_backend::Tc5StaticRunner;
        Tc5StaticRunner tc5a_runner(d_a_half, d_b_half, d_c, m, n, k);
        auto launch_tc5a = [&]() { tc5a_runner.launch(); };
        std::vector<float> h_tc5a(static_cast<size_t>(m) * n);
        benchmark_kernel(
            "tc5a", "tc5a custom static persistent TMA TCGen05 GEMM",
            "fp16->fp32", "cuBLAS Tensor Core", launch_tc5a, m, n, k, d_c,
            c_bytes, h_ref_tc, h_tc5a, csv, cublas_tc_perf, 2e-2f, 2e-3f);
      }

      if (wants_backend(backend_filter, "tc5b")) {
        using Tc5DynamicRunner =
            gemm_sm110::custom_backend::Tc5ClcRunner;
        Tc5DynamicRunner tc5b_runner(d_a_half, d_b_half, d_c, m, n, k);
        auto launch_tc5b = [&]() { tc5b_runner.launch(); };
        std::vector<float> h_tc5b(static_cast<size_t>(m) * n);
        benchmark_kernel(
            "tc5b", "tc5b custom hardware CLC persistent TMA TCGen05 GEMM",
            "fp16->fp32", "cuBLAS Tensor Core", launch_tc5b, m, n, k, d_c,
            c_bytes, h_ref_tc, h_tc5b, csv, cublas_tc_perf, 2e-2f, 2e-3f);
      }
    }
  }

  csv.close();
  if (d_c != nullptr) CHECK_CUDA(cudaFree(d_c));
  if (d_a_half != nullptr) CHECK_CUDA(cudaFree(d_a_half));
  if (d_b_half != nullptr) CHECK_CUDA(cudaFree(d_b_half));
  return 0;
}
