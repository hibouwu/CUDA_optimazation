#include "gemm_benchmark.cuh"
#include "tc3_gemm_kernel.cuh"
#include "tc4_gemm_kernel.cuh"
#include "tc5_gemm_kernel.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kBackendUsage = "[all|cublas_tc|tc3|tc4|tc5|tc5a|tc5b]";

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " [square_size] " << kBackendUsage
            << '\n';
}

bool is_valid_backend_filter(const std::string& filter) {
  return filter == "all" || filter == "cublas_tc" || filter == "tc3" ||
         filter == "tc4" || filter == "tc5" || filter == "tc5a" ||
         filter == "tc5b";
}

bool needs_cublas_reference(const std::string& filter) {
  return filter == "all" || filter == "cublas_tc";
}

bool wants_backend(const std::string& filter, const std::string& backend) {
  if (filter == "tc5") {
    return backend == "tc5a" || backend == "tc5b";
  }
  return filter == "all" || filter == backend;
}

bool wants_probe_backend(const std::string& filter) {
  return wants_backend(filter, "tc3") || wants_backend(filter, "tc5a") ||
         wants_backend(filter, "tc5b");
}

size_t probe_output_elements(const std::string& filter, int m, int n) {
  size_t elements = 1;
  if (wants_backend(filter, "tc3")) {
    elements = static_cast<size_t>(ceil_div(n, 64));
  }
  if (wants_backend(filter, "tc5a") || wants_backend(filter, "tc5b")) {
    const size_t tc5_tiles =
        static_cast<size_t>(ceil_div(m, Tc5Sm110Shape::kBlockM)) *
        ceil_div(n, Tc5Sm110Shape::kBlockN);
    if (tc5_tiles > elements) elements = tc5_tiles;
  }
  return elements;
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
  const bool run_probe = wants_probe_backend(backend_filter);
  const int m = n;
  const int k = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  const size_t d_c_elements =
      run_reference ? static_cast<size_t>(m) * n
                    : (run_probe ? probe_output_elements(backend_filter, m, n)
                                 : 0);
  const size_t d_c_bytes = d_c_elements * sizeof(float);

  CHECK_CUDA(cudaFree(0));
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp device_prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device));

  std::cout << "GPU: " << device_prop.name << " (sm_" << device_prop.major
            << device_prop.minor << ")\n";

  float* d_c = nullptr;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  int* d_tc5_work_counter = nullptr;
  if (d_c_bytes > 0) {
    CHECK_CUDA(cudaMalloc(&d_c, d_c_bytes));
  }
  if (wants_backend(backend_filter, "tc5b")) {
    CHECK_CUDA(cudaMalloc(&d_tc5_work_counter, sizeof(int)));
  }

  const bool device_supports_tc3_sm110 =
      device_prop.major == 11 && tc3_sm110_tcgen05_available();
  const bool device_supports_tc5_sm110 =
      device_prop.major == 11 && tc5_sm110_launch_available();

  std::ofstream csv("sgemm_sm110_benchmark.csv");
  csv << "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,"
         "RatioToReference,Matched\n";

  std::cout << "N=" << n << '\n';
  std::cout << "Backend filter=" << backend_filter << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat
            << " per backend before moving to the next backend\n";

  float cublas_tc_perf = 0.0f;
  if (run_reference) {
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

  if (wants_backend(backend_filter, "tc3")) {
    if (!device_supports_tc3_sm110) {
      std::cout << "tc3 sm110 tcgen05/tmem probe: skipped because runtime "
                   "GPU is not an sm110 family target with tcgen05 enabled\n";
      csv << "tc3,tc3 sm110 tcgen05/tmem probe skipped," << n
          << ",fp16->fp32,tcgen05 probe,0,0,0,0\n";
    } else {
      const int tiles = ceil_div(n, 64);
      dim3 tc3_grid(tiles);
      dim3 tc3_block(Tc3Sm110Shape::kThreads);
      auto launch_tc3 = [&]() {
        hgemm_tc3_sm110_tcgen05_tmem_probe<<<tc3_grid, tc3_block>>>(d_c,
                                                                    tiles);
        CHECK_CUDA(cudaGetLastError());
      };

      CHECK_CUDA(cudaMemset(d_c, 0, d_c_bytes));
      for (int i = 0; i < kWarmup; ++i) launch_tc3();
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < kRepeat; ++i) launch_tc3();
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));

      float probe_value = 0.0f;
      CHECK_CUDA(cudaMemcpy(&probe_value, d_c, sizeof(float),
                            cudaMemcpyDeviceToHost));
      const float avg_ms = total_ms / kRepeat;
      const bool matched = std::abs(probe_value - 1.0f) < 1e-6f;
      std::cout << "tc3 sm110 tcgen05/tmem probe: " << avg_ms
                << " ms, probe=" << probe_value << ", matched=" << matched
                << '\n';
      csv << "tc3,tc3 sm110 tcgen05/tmem probe," << n
          << ",fp16->fp32,tcgen05 probe," << avg_ms << ",0,0,"
          << (matched ? 1 : 0) << '\n';
    }
  }

  if (wants_backend(backend_filter, "tc4")) {
    std::cout << "tc4 sm110 blackwell ws pipeline: scaffold is aligned "
                 "with the work-tile pipeline diagram, but launch is "
                 "disabled until TCGen05 MMA/TMEM and TMA store are wired\n";
    csv << "tc4,tc4 sm110 blackwell ws pipeline scaffold," << n
        << ",fp16->fp32,none,0,0,0,0\n";
  }

  if (wants_backend(backend_filter, "tc5a") ||
      wants_backend(backend_filter, "tc5b")) {
    if (!device_supports_tc5_sm110) {
      if (wants_backend(backend_filter, "tc5a")) {
        std::cout << "tc5a sm110 static CLC persistent TCGen05/TMEM probe: "
                     "skipped because runtime GPU is not an sm110 family "
                     "target with tcgen05 enabled\n";
        csv << "tc5a,tc5a sm110 static CLC persistent TCGen05/TMEM probe "
               "skipped,"
            << n << ",fp16->fp32,tcgen05 probe,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "tc5b")) {
        std::cout << "tc5b sm110 dynamic CLC persistent TCGen05/TMEM probe: "
                     "skipped because runtime GPU is not an sm110 family "
                     "target with tcgen05 enabled\n";
        csv << "tc5b,tc5b sm110 dynamic CLC persistent TCGen05/TMEM probe "
               "skipped,"
            << n << ",fp16->fp32,tcgen05 probe,0,0,0,0\n";
      }
    } else {
      const int total_tiles =
          ceil_div(m, Tc5Sm110Shape::kBlockM) *
          ceil_div(n, Tc5Sm110Shape::kBlockN);
      int tc5_workers_per_sm = 1;
      if (const char* env_workers = std::getenv("TC5_SM110_WORKERS_PER_SM")) {
        const int parsed = std::atoi(env_workers);
        if (parsed > 0) tc5_workers_per_sm = parsed;
      }
      if (tc5_workers_per_sm > 8) tc5_workers_per_sm = 8;
      const int worker_count =
          total_tiles < device_prop.multiProcessorCount * tc5_workers_per_sm
              ? total_tiles
              : device_prop.multiProcessorCount * tc5_workers_per_sm;
      dim3 tc5_grid(worker_count);
      dim3 tc5_block(Tc5Sm110Shape::kThreads);

      if (wants_backend(backend_filter, "tc5a")) {
        auto launch_tc5a = [&]() {
          hgemm_tc5a_sm110_clc_static_tcgen05_tmem_persistent_probe
              <<<tc5_grid, tc5_block>>>(d_c, total_tiles);
          CHECK_CUDA(cudaGetLastError());
        };

        CHECK_CUDA(cudaMemset(d_c, 0, d_c_bytes));
        for (int i = 0; i < kWarmup; ++i) launch_tc5a();
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < kRepeat; ++i) launch_tc5a();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        float probe_value = 0.0f;
        CHECK_CUDA(cudaMemcpy(&probe_value, d_c, sizeof(float),
                              cudaMemcpyDeviceToHost));
        const float avg_ms = total_ms / kRepeat;
        const bool matched = std::abs(probe_value - 1.0f) < 1e-6f;
        std::cout << "tc5a sm110 static CLC persistent TCGen05/TMEM probe: "
                  << avg_ms << " ms, probe=" << probe_value
                  << ", matched=" << matched << ", workers=" << worker_count
                  << ", workers_per_sm=" << tc5_workers_per_sm << '\n';
        csv << "tc5a,tc5a sm110 static CLC persistent TCGen05/TMEM probe,"
            << n << ",fp16->fp32,tcgen05 probe," << avg_ms << ",0,0,"
            << (matched ? 1 : 0) << '\n';
      }

      if (wants_backend(backend_filter, "tc5b")) {
        auto launch_tc5b = [&]() {
          hgemm_tc5b_sm110_clc_dynamic_tcgen05_tmem_persistent_probe
              <<<tc5_grid, tc5_block>>>(d_c, total_tiles,
                                        d_tc5_work_counter);
          CHECK_CUDA(cudaGetLastError());
        };

        CHECK_CUDA(cudaMemset(d_c, 0, d_c_bytes));
        for (int i = 0; i < kWarmup; ++i) {
          CHECK_CUDA(cudaMemset(d_tc5_work_counter, 0, sizeof(int)));
          launch_tc5b();
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        float total_ms = 0.0f;
        for (int i = 0; i < kRepeat; ++i) {
          CHECK_CUDA(cudaMemset(d_tc5_work_counter, 0, sizeof(int)));
          CHECK_CUDA(cudaEventRecord(start));
          launch_tc5b();
          CHECK_CUDA(cudaEventRecord(stop));
          CHECK_CUDA(cudaEventSynchronize(stop));
          float iter_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
          total_ms += iter_ms;
        }
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        float probe_value = 0.0f;
        CHECK_CUDA(cudaMemcpy(&probe_value, d_c, sizeof(float),
                              cudaMemcpyDeviceToHost));
        const float avg_ms = total_ms / kRepeat;
        const bool matched = std::abs(probe_value - 1.0f) < 1e-6f;
        std::cout << "tc5b sm110 dynamic CLC persistent TCGen05/TMEM probe: "
                  << avg_ms << " ms, probe=" << probe_value
                  << ", matched=" << matched << ", workers=" << worker_count
                  << ", workers_per_sm=" << tc5_workers_per_sm << '\n';
        csv << "tc5b,tc5b sm110 dynamic CLC persistent TCGen05/TMEM probe,"
            << n << ",fp16->fp32,tcgen05 probe," << avg_ms << ",0,0,"
            << (matched ? 1 : 0) << '\n';
      }
    }
  }

  csv.close();
  if (d_c != nullptr) CHECK_CUDA(cudaFree(d_c));
  if (d_a_half != nullptr) CHECK_CUDA(cudaFree(d_a_half));
  if (d_b_half != nullptr) CHECK_CUDA(cudaFree(d_b_half));
  if (d_tc5_work_counter != nullptr) CHECK_CUDA(cudaFree(d_tc5_work_counter));
  return 0;
}
