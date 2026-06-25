#include "gemm_benchmark.cuh"
#include "sgemm_kernels.cuh"
#include "tc3_gemm_kernel.cuh"
#include "tc_gemm_kernels.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kBackendUsage =
    "[all|fp32|tensor_core|cublas|v1|v2|v3|v3a|v3b|v4|v5|v6|"
    "v7|v8a|v8b|v8c|cublas_tc|tc1|tc2|tc2p2|tc2p3|tc3]";

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " [square_size] " << kBackendUsage
            << '\n';
}

bool is_valid_backend_filter(const std::string& filter) {
  return filter == "all" || filter == "fp32" || filter == "tensor_core" ||
         filter == "cublas" || filter == "v1" || filter == "v2" ||
         filter == "v3" || filter == "v3a" || filter == "v3b" ||
         filter == "v4" || filter == "v5" || filter == "v6" ||
         filter == "v7" || filter == "v8a" || filter == "v8b" ||
         filter == "v8c" || filter == "cublas_tc" || filter == "tc1" ||
         filter == "tc2" || filter == "tc2p2" || filter == "tc2p3" ||
         filter == "tc3";
}

bool wants_fp32_reference(const std::string& filter) {
  return filter == "all" || filter == "fp32" || filter == "cublas" ||
         filter == "v1" || filter == "v2" || filter == "v3" ||
         filter == "v3a" || filter == "v3b" || filter == "v4" ||
         filter == "v5" || filter == "v6" || filter == "v7" ||
         filter == "v8a" || filter == "v8b" || filter == "v8c";
}

bool wants_tensor_core_reference(const std::string& filter) {
  return filter == "all" || filter == "tensor_core" ||
         filter == "cublas_tc" || filter == "tc1" || filter == "tc2" ||
         filter == "tc2p2" || filter == "tc2p3" || filter == "tc3";
}

bool wants_backend(const std::string& filter, const std::string& backend) {
  if (filter == "all" || filter == backend) return true;
  if (filter == "fp32") {
    return backend == "v1" || backend == "v2" || backend == "v3" ||
           backend == "v3a" || backend == "v3b" || backend == "v4" ||
           backend == "v5" || backend == "v6" || backend == "v7" ||
           backend == "v8a" || backend == "v8b" || backend == "v8c";
  }
  if (filter == "tensor_core") {
    return backend == "tc1" || backend == "tc2" || backend == "tc2p2" ||
           backend == "tc2p3";
  }
  return false;
}

template <int Stages, bool Persistent = false>
void configure_tc_tma_wmma_kernel() {
  CHECK_CUDA(cudaFuncSetAttribute(
      hgemm_tc_tma_wmma_128x64x32<Stages, Persistent>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(tc_tma_wmma_smem_bytes<Stages>())));
  CHECK_CUDA(cudaFuncSetAttribute(
      hgemm_tc_tma_wmma_128x64x32<Stages, Persistent>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
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

  const bool run_fp32 = wants_fp32_reference(backend_filter);
  const bool run_tensor_core = wants_tensor_core_reference(backend_filter);

  const int m = n;
  const int k = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  const size_t a_half_bytes = static_cast<size_t>(m) * k * sizeof(half);
  const size_t b_half_bytes = static_cast<size_t>(k) * n * sizeof(half);

  std::vector<float> h_a(static_cast<size_t>(m) * k);
  std::vector<float> h_b(static_cast<size_t>(k) * n);
  std::vector<float> h_ref(static_cast<size_t>(m) * n);
  std::vector<float> h_ref_tc(static_cast<size_t>(m) * n);
  std::vector<float> h_out(static_cast<size_t>(m) * n);
  fill_inputs(h_a, h_b);

  std::vector<half> h_a_half = to_half_vector(h_a);
  std::vector<half> h_b_half = to_half_vector(h_b);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUtensorMap* d_a_tc2_map = nullptr;
  CUtensorMap* d_b_tc2_map = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&d_c, c_bytes));
  CHECK_CUDA(cudaMalloc(&d_a_half, a_half_bytes));
  CHECK_CUDA(cudaMalloc(&d_b_half, b_half_bytes));
  CHECK_CUDA(cudaMalloc(&d_a_tc2_map, sizeof(CUtensorMap)));
  CHECK_CUDA(cudaMalloc(&d_b_tc2_map, sizeof(CUtensorMap)));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_a_half, h_a_half.data(), a_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half, h_b_half.data(), b_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CU(cuInit(0));
  alignas(64) CUtensorMap h_a_tc2_map{};
  alignas(64) CUtensorMap h_b_tc2_map{};
  tc_encode_rowmajor_tensor_map_2d(h_a_tc2_map, d_a_half, m, k, 128, 32);
  tc_encode_rowmajor_tensor_map_2d(h_b_tc2_map, d_b_half, k, n, 32, 64);
  CHECK_CUDA(cudaMemcpy(d_a_tc2_map, &h_a_tc2_map, sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_tc2_map, &h_b_tc2_map, sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice));
  configure_tc_tma_wmma_kernel<2>();
  configure_tc_tma_wmma_kernel<2, true>();
  configure_tc_tma_wmma_kernel<3, true>();

  cublasHandle_t handle;
  cublasHandle_t tensor_core_handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasCreate(&tensor_core_handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  print_gemm_environment(handle);
  cudaDeviceProp device_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
  const int sm_count = device_prop.multiProcessorCount;

  // cuBLAS uses column-major semantics. Swapping A/B computes row-major C=A@B.
  auto launch_cublas = [&]() {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_b, n, d_a, k, &beta, d_c, n));
  };
  auto launch_cublas_tensor_core = [&]() {
    CHECK_CUBLAS(cublasGemmEx(
        tensor_core_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b_half,
        CUDA_R_16F, n, d_a_half, CUDA_R_16F, k, &beta, d_c, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
  };

  std::ofstream csv("sgemm_benchmark.csv");
  csv << "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,"
         "RatioToReference,Matched\n";

  std::cout << "N=" << n << '\n';
  std::cout << "Backend filter=" << backend_filter << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat
            << " per backend before moving to the next backend\n";

  float cublas_perf = 0.0f;
  if (run_fp32) {
    const float cublas_avg_ms =
        benchmark_reference(launch_cublas, d_c, c_bytes, h_ref);
    cublas_perf = gflops(m, n, k, cublas_avg_ms);

    std::cout << "FP32 SIMT reference:\n";
    std::cout << "cuBLAS FP32 Pedantic: " << cublas_avg_ms << " ms, "
              << cublas_perf << " GFLOPS\n";
    csv << "cublas,cuBLAS FP32 Pedantic," << n
        << ",fp32,cuBLAS FP32 Pedantic," << cublas_avg_ms << ","
        << cublas_perf << ",1,1\n";

    dim3 v1_block(16, 16);
    dim3 v1_grid(ceil_div(m, v1_block.x), ceil_div(n, v1_block.y));
    if (wants_backend(backend_filter, "v1")) {
      benchmark_kernel(
          "v1", "v1 naive uncoalesced", "fp32", "cuBLAS FP32 Pedantic",
          [&]() {
            sgemm_v1_naive_uncoalesced<<<v1_grid, v1_block>>>(
                m, n, k, alpha, d_a, d_b, beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
    }

    dim3 v2_block(16, 16);
    dim3 v2_grid(ceil_div(n, v2_block.x), ceil_div(m, v2_block.y));
    if (wants_backend(backend_filter, "v2")) {
      benchmark_kernel(
          "v2", "v2 coalesced naive", "fp32", "cuBLAS FP32 Pedantic",
          [&]() {
            sgemm_v1_naive<<<v2_grid, v2_block>>>(m, n, k, alpha, d_a, d_b,
                                                  beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
    }

    dim3 v3_smem_block(32, 32);
    dim3 v3_smem_grid(ceil_div(n, 32), ceil_div(m, 32));
    if (wants_backend(backend_filter, "v3")) {
      benchmark_kernel(
          "v3", "v3 shared-memory tile", "fp32", "cuBLAS FP32 Pedantic",
          [&]() {
            sgemm_v2_smem<32><<<v3_smem_grid, v3_smem_block>>>(
                m, n, k, alpha, d_a, d_b, beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
    }

    constexpr int V3_BRANCH_TILE = 16;
    dim3 v3_branch_grid(ceil_div(n, V3_BRANCH_TILE),
                        ceil_div(m, V3_BRANCH_TILE));
    dim3 v3_branch_block(V3_BRANCH_TILE * V3_BRANCH_TILE);
    if (wants_backend(backend_filter, "v3a")) {
      benchmark_kernel(
          "v3a", "v3a shared-memory tile 1D", "fp32", "cuBLAS FP32 Pedantic",
          [&]() {
            sgemm_v3a_smem_1d<V3_BRANCH_TILE>
                <<<v3_branch_grid, v3_branch_block>>>(m, n, k, alpha, d_a, d_b,
                                                      beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
    }

    if (wants_backend(backend_filter, "v3b")) {
      benchmark_kernel(
          "v3b", "v3b shared-memory tile 1D padded", "fp32", "cuBLAS FP32 Pedantic",
          [&]() {
            sgemm_v4_smem_1d_padded<V3_BRANCH_TILE>
                <<<v3_branch_grid, v3_branch_block>>>(m, n, k, alpha, d_a, d_b,
                                                      beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
    }

    constexpr int V4_BM = 64;
    constexpr int V4_BN = 64;
    constexpr int V4_BK = 8;
    constexpr int V4_TM = 4;
    constexpr int V4_TN = 4;
    dim3 v4_grid(ceil_div(n, V4_BN), ceil_div(m, V4_BM));
    dim3 v4_block((V4_BM / V4_TM) * (V4_BN / V4_TN));
    auto launch_v4_thread_tile = [&]() {
      sgemm_v3_thread_tile<V4_BM, V4_BN, V4_BK, V4_TM, V4_TN>
          <<<v4_grid, v4_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
      CHECK_CUDA(cudaGetLastError());
    };
    if (wants_backend(backend_filter, "v4")) {
      benchmark_kernel("v4", "v4 thread tile", "fp32", "cuBLAS FP32 Pedantic",
                       launch_v4_thread_tile, m, n, k, d_c, c_bytes, h_ref,
                       h_out, csv, cublas_perf);
    }

    constexpr int V5_BM = 128;
    constexpr int V5_BN = 128;
    constexpr int V5_BK = 8;
    constexpr int V5_TM = 8;
    constexpr int V5_TN = 8;

    constexpr int V7_NUM_THREADS = 128;
    constexpr int V7_BM = 128;
    constexpr int V7_BN = 128;
    constexpr int V7_BK = 16;
    constexpr int V7_WM = 64;
    constexpr int V7_WN = 64;
    constexpr int V7_WNITER = 4;
    constexpr int V7_TM = 8;
    constexpr int V7_TN = 4;
    constexpr int V8C_BK = 8;

    // These divisibility checks describe only the currently compiled 128x128
    // baseline instances below. Future v7 tuning candidates must derive their
    // own M/N/K constraints from their BM/BN/BK parameters.
    const bool supports_128_tiled =
        (m % V5_BM == 0) && (n % V5_BN == 0) && (k % V7_BK == 0);
    if (supports_128_tiled) {
      dim3 v5_grid(n / V5_BN, m / V5_BM);
      dim3 v5_block((V5_BM / V5_TM) * (V5_BN / V5_TN));

      static_assert((V7_BN % V7_WN == 0) && (V7_BM % V7_WM == 0));
      static_assert((V7_BN / V7_WN) * (V7_BM / V7_WM) ==
                    V7_NUM_THREADS / kWarpSize);
      static_assert((V7_NUM_THREADS * 4) % V7_BK == 0);
      static_assert((V7_NUM_THREADS * 4) % V7_BN == 0);
      static_assert((V7_BM * V7_BK) % (4 * V7_NUM_THREADS) == 0);
      static_assert((V7_BN * V7_BK) % (4 * V7_NUM_THREADS) == 0);
      static_assert((V7_NUM_THREADS * 4) % V8C_BK == 0);
      static_assert((V7_BM * V8C_BK) % (4 * V7_NUM_THREADS) == 0);
      static_assert((V7_BN * V8C_BK) % (4 * V7_NUM_THREADS) == 0);
      dim3 v7_grid(n / V7_BN, m / V7_BM);
      dim3 v7_block(V7_NUM_THREADS);

      if (wants_backend(backend_filter, "v5")) {
        benchmark_kernel(
            "v5", "v5 vectorized load", "fp32", "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v4_vectorized<V5_BM, V5_BN, V5_BK, V5_TM, V5_TN>
                  <<<v5_grid, v5_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }

      if (wants_backend(backend_filter, "v6")) {
        benchmark_kernel(
            "v6", "v6 double buffer", "fp32", "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v5_double_buffer<V5_BM, V5_BN, V5_BK, V5_TM, V5_TN>
                  <<<v5_grid, v5_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }

      if (wants_backend(backend_filter, "v7")) {
        benchmark_kernel(
            "v7", "v7 warp tiling", "fp32", "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v7_warp_tiling_double_buffer<
                  V7_BM, V7_BN, V7_BK, V7_WM, V7_WN, V7_WNITER, V7_TM, V7_TN,
                  V7_NUM_THREADS><<<v7_grid, v7_block>>>(
                  m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }
      if (wants_backend(backend_filter, "v8a")) {
        benchmark_kernel(
            "v8a", "v8a cp.async B tile", "fp32", "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v8a_cp_async_b_tile<
                  V7_BM, V7_BN, V7_BK, V7_WM, V7_WN, V7_WNITER, V7_TM, V7_TN,
                  V7_NUM_THREADS><<<v7_grid, v7_block>>>(
                  m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }
      if (wants_backend(backend_filter, "v8b")) {
        benchmark_kernel(
            "v8b", "v8b cp.async A/B 2-stage", "fp32",
            "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v8b_cp_async_ab_2stage<
                  V7_BM, V7_BN, V7_BK, V7_WM, V7_WN, V7_WNITER, V7_TM, V7_TN,
                  V7_NUM_THREADS><<<v7_grid, v7_block>>>(
                  m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }
      if (wants_backend(backend_filter, "v8c")) {
        benchmark_kernel(
            "v8c", "v8c cp.async A/B 3-stage", "fp32",
            "cuBLAS FP32 Pedantic",
            [&]() {
              sgemm_v8c_cp_async_ab_3stage<
                  V7_BM, V7_BN, V8C_BK, V7_WM, V7_WN, V7_WNITER, V7_TM, V7_TN,
                  V7_NUM_THREADS><<<v7_grid, v7_block>>>(
                  m, n, k, alpha, d_a, d_b, beta, d_c);
              CHECK_CUDA(cudaGetLastError());
            },
            m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
      }
    } else if (wants_backend(backend_filter, "v5") ||
               wants_backend(backend_filter, "v6") ||
               wants_backend(backend_filter, "v7") ||
               wants_backend(backend_filter, "v8a") ||
               wants_backend(backend_filter, "v8b") ||
               wants_backend(backend_filter, "v8c")) {
      std::cout << "v5/v6/v7/v8: skipped because M/N must be multiples of "
                   "128 and K must be a multiple of 16\n";
      if (wants_backend(backend_filter, "v5")) {
        csv << "v5,v5 vectorized load skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "v6")) {
        csv << "v6,v6 double buffer skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "v7")) {
        csv << "v7,v7 warp tiling skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "v8a")) {
        csv << "v8a,v8a cp.async B tile skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "v8b")) {
        csv << "v8b,v8b cp.async A/B 2-stage skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
      if (wants_backend(backend_filter, "v8c")) {
        csv << "v8c,v8c cp.async A/B 3-stage skipped," << n
            << ",fp32,cuBLAS FP32 Pedantic,0,0,0,0\n";
      }
    }
  }

  if (run_tensor_core) {
    const float cublas_tc_avg_ms =
        benchmark_reference(launch_cublas_tensor_core, d_c, c_bytes, h_ref_tc);
    const float cublas_tc_perf = gflops(m, n, k, cublas_tc_avg_ms);

    std::cout << "Tensor Core reference:\n";
    std::cout << "cuBLAS Tensor Core: " << cublas_tc_avg_ms << " ms, "
              << cublas_tc_perf << " GFLOPS\n";
    csv << "cublas_tc,cuBLAS Tensor Core," << n
        << ",fp16->fp32,cuBLAS Tensor Core," << cublas_tc_avg_ms << ","
        << cublas_tc_perf << ",1,1\n";

    if (wants_backend(backend_filter, "tc1") && n % 16 == 0) {
      dim3 tc1_grid(n / 16, m / 16);
      dim3 tc1_block(kWarpSize);
      benchmark_kernel(
          "tc1", "tc1 wmma fp16 baseline", "fp16->fp32",
          "cuBLAS Tensor Core",
          [&]() {
            hgemm_tc1_wmma_16x16<<<tc1_grid, tc1_block>>>(
                m, n, k, alpha, d_a_half, d_b_half, beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref_tc, h_out, csv, cublas_tc_perf, 1e-1f,
          1e-2f);
    } else if (wants_backend(backend_filter, "tc1")) {
      std::cout << "tc1 wmma fp16 baseline: skipped because N must be a "
                   "multiple of 16\n";
      csv << "tc1,tc1 wmma fp16 baseline skipped," << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    }

    if (wants_backend(backend_filter, "tc2") && n % 128 == 0) {
      dim3 tc2_grid(n / 64, m / 128);
      dim3 tc2_block(8 * kWarpSize);
      benchmark_kernel(
          "tc2", "tc2 tma 2-stage wmma 128x64x32", "fp16->fp32",
          "cuBLAS Tensor Core",
          [&]() {
            hgemm_tc_tma_wmma_128x64x32<2>
                <<<tc2_grid, tc2_block, tc_tma_wmma_smem_bytes<2>()>>>(
                m, n, k, alpha, d_a_half, d_a_tc2_map, d_b_half, d_b_tc2_map,
                beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref_tc, h_out, csv, cublas_tc_perf, 1e-1f,
          1e-2f);
    } else if (wants_backend(backend_filter, "tc2")) {
      std::cout << "tc2 tma 2-stage wmma 128x64x32: skipped because N must be a "
                   "multiple of 128\n";
      csv << "tc2,tc2 tma 2-stage wmma 128x64x32 skipped," << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    }

    if (wants_backend(backend_filter, "tc3")) {
      if (!tc3_sm120a_narrow_mma_available()) {
        std::cout << "tc3 sm120a f8f6f4 mma probe: skipped because this "
                     "binary was not built with CUDA_ARCH=120a\n";
        csv << "tc3,tc3 sm120a f8f6f4 mma probe skipped," << n
            << ",narrow-mma,SM120a MMA probe,0,0,0,0\n";
      } else {
        const int probe_blocks = 256;
        dim3 tc3_grid(probe_blocks);
        dim3 tc3_block(kWarpSize);
        auto launch_tc3 = [&]() {
          hgemm_tc3_sm120a_f8f6f4_mma_probe<<<tc3_grid, tc3_block>>>(d_c);
          CHECK_CUDA(cudaGetLastError());
        };

        CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));
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
        std::cout << "tc3 sm120a f8f6f4 mma probe: " << avg_ms
                  << " ms, probe=" << probe_value
                  << ", matched=" << matched << '\n';
        csv << "tc3,tc3 sm120a f8f6f4 mma probe," << n
            << ",narrow-mma,SM120a MMA probe," << avg_ms << ",0,0,"
            << (matched ? 1 : 0) << '\n';
      }
    }

    if (wants_backend(backend_filter, "tc2p2") && n % 128 == 0) {
      const int tc2p_total_tiles = (m / 128) * (n / 64);
      const int tc2p2_blocks =
          tc2p_total_tiles < sm_count ? tc2p_total_tiles : sm_count;
      dim3 tc2p2_grid(tc2p2_blocks);
      dim3 tc2p2_block(8 * kWarpSize);
      benchmark_kernel(
          "tc2p2", "tc2p2 persistent tma 2-stage wmma 128x64x32",
          "fp16->fp32",
          "cuBLAS Tensor Core",
          [&]() {
            hgemm_tc_tma_wmma_128x64x32<2, true>
                <<<tc2p2_grid, tc2p2_block, tc_tma_wmma_smem_bytes<2>()>>>(
                m, n, k, alpha, d_a_half, d_a_tc2_map, d_b_half, d_b_tc2_map,
                beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref_tc, h_out, csv, cublas_tc_perf, 1e-1f,
          1e-2f);
    } else if (wants_backend(backend_filter, "tc2p2")) {
      std::cout << "tc2p2 persistent tma 2-stage wmma 128x64x32: skipped because N must be a "
                   "multiple of 128\n";
      csv << "tc2p2,tc2p2 persistent tma 2-stage wmma 128x64x32 skipped,"
          << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    }

    if (wants_backend(backend_filter, "tc2p3") && n % 128 == 0) {
      const int tc2p_total_tiles = (m / 128) * (n / 64);
      const int tc2p3_blocks =
          tc2p_total_tiles < sm_count ? tc2p_total_tiles : sm_count;
      dim3 tc2p3_grid(tc2p3_blocks);
      dim3 tc2p3_block(8 * kWarpSize);
      benchmark_kernel(
          "tc2p3", "tc2p3 persistent tma 3-stage wmma 128x64x32",
          "fp16->fp32",
          "cuBLAS Tensor Core",
          [&]() {
            hgemm_tc_tma_wmma_128x64x32<3, true>
                <<<tc2p3_grid, tc2p3_block, tc_tma_wmma_smem_bytes<3>()>>>(
                m, n, k, alpha, d_a_half, d_a_tc2_map, d_b_half, d_b_tc2_map,
                beta, d_c);
            CHECK_CUDA(cudaGetLastError());
          },
          m, n, k, d_c, c_bytes, h_ref_tc, h_out, csv, cublas_tc_perf, 1e-1f,
          1e-2f);
    } else if (wants_backend(backend_filter, "tc2p3")) {
      std::cout << "tc2p3 persistent tma 3-stage wmma 128x64x32: skipped because N must be a "
                   "multiple of 128\n";
      csv << "tc2p3,tc2p3 persistent tma 3-stage wmma 128x64x32 skipped,"
          << n
          << ",fp16->fp32,cuBLAS Tensor Core,0,0,0,0\n";
    }

  }

  csv.close();

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUBLAS(cublasDestroy(tensor_core_handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a_half));
  CHECK_CUDA(cudaFree(d_b_half));
  CHECK_CUDA(cudaFree(d_a_tc2_map));
  CHECK_CUDA(cudaFree(d_b_tc2_map));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
