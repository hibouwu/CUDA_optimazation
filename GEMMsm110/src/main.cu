#include "gemm_benchmark.cuh"
#include "backends/tc0_baseline.cuh"
#include "backends/tc1_tc2_tma.cuh"
#include "backends/tc3_pipeline.cuh"
#include "backends/tc4a_warp_specialized.cuh"
#include "backends/tc4bc_cluster.cuh"
#include "backends/tc5_persistent.cuh"
#include "cutlass_sm110_backends.cuh"
#include "sm110_backend_registry.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using gemm_sm110::wants_backend;

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " [square_size] "
            << gemm_sm110::kBackendUsage << '\n';
}

bool needs_cublas_reference(const std::string& filter) {
  (void)filter;
  return true;
}

void write_unavailable_backend(const gemm_sm110::BackendDescriptor& backend,
                               int n, std::ofstream& csv,
                               const char* reason = "not implemented yet") {
  std::cout << backend.id << " " << backend.label
            << ": unavailable (" << reason << ")\n";
  csv << backend.id << "," << backend.label << " unavailable," << n
      << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
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

  if (n <= 0 || !gemm_sm110::is_valid_backend_filter(backend_filter)) {
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
  half* d_b_half_nk = nullptr;
  if (d_c_bytes > 0) {
    CHECK_CUDA(cudaMalloc(&d_c, d_c_bytes));
  }

  const bool device_supports_tc3_sm110 = device_prop.major == 11;

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
  // Raw TCGen05 kernels use K-major operands, matching learn-cuda's
  // [N,K] storage for logical B[K,N].  Keep the original KxN allocation for
  // cuBLAS/CUTLASS and prepare this equivalent layout outside timed regions.
  std::vector<half> h_b_half_nk(static_cast<size_t>(n) * k);
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      h_b_half_nk[static_cast<size_t>(n_idx) * k + k_idx] =
          h_b_half[static_cast<size_t>(k_idx) * n + n_idx];
    }
  }

  CHECK_CUDA(cudaMalloc(&d_a_half, a_half_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_half, b_half_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_half_nk, b_half_bytes));
  CHECK_CUDA(cudaMemcpy(d_a_half, h_a_half.data(), a_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half, h_b_half.data(), b_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half_nk, h_b_half_nk.data(), b_half_bytes,
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

  if (wants_backend(backend_filter, "tc0")) {
    using Tc0Runner = gemm_sm110::backends::Tc0Runner;
    Tc0Runner tc0_runner(d_a_half, d_b_half, d_c, m, n, k);
    auto launch_tc0 = [&]() { tc0_runner.launch(); };
    std::vector<float> h_tc0(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc0", "tc0 CUDA WMMA Tensor Core baseline", "fp16->fp32",
        "cuBLAS Tensor Core", launch_tc0, m, n, k, d_c, c_bytes, h_ref_tc,
        h_tc0, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc1a")) {
    gemm_sm110::backends::Tc1aRunner runner(
        d_a_half, d_b_half, d_c, m, n, k);
    auto launch = [&]() { runner.launch(); };
    std::vector<float> output(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc1a", "tc1a 2D TMA linear-SMEM TCGen05 minimal", "fp16->fp32",
        "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
        output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc1b")) {
    gemm_sm110::backends::Tc1bRunner runner(
        d_a_half, d_b_half, d_c, m, n, k);
    auto launch = [&]() { runner.launch(); };
    std::vector<float> output(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc1b", "tc1b 3D TMA linear-SMEM TCGen05 minimal", "fp16->fp32",
        "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
        output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc2a")) {
    gemm_sm110::backends::Tc2aRunner runner(
        d_a_half, d_b_half, d_c, m, n, k);
    auto launch = [&]() { runner.launch(); };
    std::vector<float> output(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc2a", "tc2a 2D TMA SW128-SMEM TCGen05", "fp16->fp32",
        "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
        output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc2b")) {
    gemm_sm110::backends::Tc2bRunner runner(
        d_a_half, d_b_half, d_c, m, n, k);
    auto launch = [&]() { runner.launch(); };
    std::vector<float> output(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc2b", "tc2b 3D TMA SW128-SMEM TCGen05", "fp16->fp32",
        "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
        output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc3")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc3"), n, csv,
                                "requires an SM110-family target");
    } else {
      using Tc3Runner = gemm_sm110::backends::Tc3Runner;
      Tc3Runner tc3_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch_tc3 = [&]() { tc3_runner.launch(); };
      std::vector<float> h_tc3(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc3", "tc3 multi-stage 2D TMA SW128 TCGen05 pipeline",
          "fp16->fp32", "cuBLAS Tensor Core", launch_tc3, m, n, k, d_c,
          c_bytes, h_ref_tc, h_tc3, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc4a")) {
    gemm_sm110::backends::Tc4aRunner runner(
        d_a_half, d_b_half, d_c, m, n, k);
    auto launch = [&]() { runner.launch(); };
    std::vector<float> output(static_cast<size_t>(m) * n);
    benchmark_kernel(
        "tc4a", "tc4a warp-specialized TMA/TCGen05 pipeline",
        "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes,
        h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
  }

  if (wants_backend(backend_filter, "tc4b")) {
    if (m % 256 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc4b"), n, csv,
          "requires M a multiple of 256");
    } else {
      gemm_sm110::backends::Tc4bRunner runner(
          d_a_half, d_b_half, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4b", "tc4b 2-SM cluster TMA/TCGen05 pipeline", "fp16->fp32",
          "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
          output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc4c")) {
    if (m % 256 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc4c"), n, csv,
          "requires M a multiple of 256");
    } else {
      gemm_sm110::backends::Tc4cRunner runner(
          d_a_half, d_b_half, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4c", "tc4c warp-specialized 2-SM cluster pipeline",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes,
          h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5a")) {
    if (m % 256 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5a"), n, csv,
          "requires M%256=0, N%128=0, and K%64=0");
    } else {
      gemm_sm110::backends::Tc5aRunner runner(
          d_a_half, d_b_half, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5a",
          "tc5a static persistent TMEM-double-buffer 2-SM GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5b")) {
    if (m % 256 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5b"), n, csv,
          "requires M%256=0, N%128=0, and K%64=0");
    } else {
      gemm_sm110::backends::Tc5bRunner runner(
          d_a_half, d_b_half, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5b", "tc5b hardware CLC persistent 2-SM GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  csv.close();
  if (d_c != nullptr) CHECK_CUDA(cudaFree(d_c));
  if (d_a_half != nullptr) CHECK_CUDA(cudaFree(d_a_half));
  if (d_b_half != nullptr) CHECK_CUDA(cudaFree(d_b_half));
  if (d_b_half_nk != nullptr) CHECK_CUDA(cudaFree(d_b_half_nk));
  return 0;
}
