#include "gemm_benchmark.cuh"
#include "sgemm_kernels.cuh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  int n = 1024;
  if (argc > 1) n = std::atoi(argv[1]);
  if (n <= 0) {
    std::cerr << "Usage: " << argv[0] << " [square_size]\n";
    return EXIT_FAILURE;
  }

  const int m = n;
  const int k = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);

  std::vector<float> h_a(static_cast<size_t>(m) * k);
  std::vector<float> h_b(static_cast<size_t>(k) * n);
  std::vector<float> h_ref(static_cast<size_t>(m) * n);
  std::vector<float> h_out(static_cast<size_t>(m) * n);
  fill_inputs(h_a, h_b);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&d_c, c_bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  print_gemm_environment(handle);

  // cuBLAS 默认按 column-major 理解矩阵。
  // 这里输入是 row-major A/B，通过交换 d_b 和 d_a，相当于计算 (B^T A^T)^T，
  // 最终得到 row-major 语义下的 C = A @ B。
  auto launch_cublas = [&]() {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_b, n, d_a, k, &beta, d_c, n));
  };

  CHECK_CUDA(cudaMemset(d_c, 0, c_bytes));
  for (int i = 0; i < kWarmup; ++i) launch_cublas();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < kRepeat; ++i) {
    launch_cublas();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float cublas_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const float cublas_avg_ms = cublas_ms / kRepeat;
  const float cublas_perf = gflops(m, n, k, cublas_avg_ms);

  // 同步写出 CSV，后续可以直接画 size/version/GFLOPS 曲线。
  std::ofstream csv("sgemm_benchmark.csv");
  csv << "Version,N,TimeMs,GFLOPS,RatioToCuBLAS,Matched\n";
  csv << "cuBLAS," << n << "," << cublas_avg_ms << "," << cublas_perf
      << ",1,1\n";

  std::cout << "N=" << n << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat
            << " per backend before moving to the next backend\n";
  std::cout << "cuBLAS: " << cublas_avg_ms << " ms, " << cublas_perf
            << " GFLOPS\n";

  // v1-v4 带边界判断，支持任意正方阵尺寸。
  dim3 v1_block(16, 16);
  dim3 v1_grid(ceil_div(m, v1_block.x), ceil_div(n, v1_block.y));
  benchmark_kernel(
      "v1 naive uncoalesced",
      [&]() {
        sgemm_v1_naive_uncoalesced<<<v1_grid, v1_block>>>(
            m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

  dim3 v2_block(16, 16);
  dim3 v2_grid(ceil_div(n, v2_block.x), ceil_div(m, v2_block.y));
  benchmark_kernel(
      "v2 coalesced naive",
      [&]() {
        sgemm_v1_naive<<<v2_grid, v2_block>>>(m, n, k, alpha, d_a, d_b, beta,
                                              d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

  dim3 v3_smem_block(32, 32);
  dim3 v3_smem_grid(ceil_div(n, 32), ceil_div(m, 32));
  benchmark_kernel(
      "v3 shared-memory tile",
      [&]() {
        sgemm_v2_smem<32><<<v3_smem_grid, v3_smem_block>>>(
            m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

  constexpr int V4_TILE = 16;
  dim3 v4_padded_grid(ceil_div(n, V4_TILE), ceil_div(m, V4_TILE));
  dim3 v4_padded_block(V4_TILE * V4_TILE);
  benchmark_kernel(
      "v4 1D block padded smem",
      [&]() {
        sgemm_v4_smem_1d_padded<V4_TILE>
            <<<v4_padded_grid, v4_padded_block>>>(m, n, k, alpha, d_a, d_b,
                                                  beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

  constexpr int V3_BM = 64;
  constexpr int V3_BN = 64;
  constexpr int V3_BK = 8;
  constexpr int V3_TM = 4;
  constexpr int V3_TN = 4;
  dim3 v3_grid(ceil_div(n, V3_BN), ceil_div(m, V3_BM));
  dim3 v3_block((V3_BM / V3_TM) * (V3_BN / V3_TN));
  auto launch_v6_thread_coarsening = [&]() {
    sgemm_v3_thread_tile<V3_BM, V3_BN, V3_BK, V3_TM, V3_TN>
        <<<v3_grid, v3_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
    CHECK_CUDA(cudaGetLastError());
  };

  // v5/v9/v10 是高性能路径，依赖 float4 对齐和整 tile 覆盖。
  // 为了让核心优化代码保持清晰，非 128 倍数尺寸直接跳过。
  if (n % 128 == 0) {
    constexpr int V4_BM = 128;
    constexpr int V4_BN = 128;
    constexpr int V4_BK = 8;
    constexpr int V4_TM = 8;
    constexpr int V4_TN = 8;
    dim3 v4_grid(n / V4_BN, m / V4_BM);
    dim3 v4_block((V4_BM / V4_TM) * (V4_BN / V4_TN));
    constexpr int V6_NUM_THREADS = 128;
    constexpr int V6_BM = 128;
    constexpr int V6_BN = 128;
    constexpr int V6_BK = 16;
    constexpr int V6_WM = 64;
    constexpr int V6_WN = 64;
    constexpr int V6_WNITER = 4;
    constexpr int V6_TM = 8;
    constexpr int V6_TN = 4;
    static_assert((V6_BN % V6_WN == 0) && (V6_BM % V6_WM == 0));
    static_assert((V6_BN / V6_WN) * (V6_BM / V6_WM) ==
                  V6_NUM_THREADS / kWarpSize);
    static_assert((V6_NUM_THREADS * 4) % V6_BK == 0);
    static_assert((V6_NUM_THREADS * 4) % V6_BN == 0);
    static_assert((V6_BM * V6_BK) % (4 * V6_NUM_THREADS) == 0);
    static_assert((V6_BN * V6_BK) % (4 * V6_NUM_THREADS) == 0);
    dim3 v6_grid(n / V6_BN, m / V6_BM);
    dim3 v6_block(V6_NUM_THREADS);
    benchmark_kernel(
        "v5 warp tiling",
        [&]() {
          sgemm_v6_warp_tiling<V6_BM, V6_BN, V6_BK, V6_WM, V6_WN,
                               V6_WNITER, V6_TM, V6_TN, V6_NUM_THREADS>
              <<<v6_grid, v6_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

    benchmark_kernel("v6 thread coarsening", launch_v6_thread_coarsening, m, n,
                     k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

    benchmark_kernel(
        "v9 vectorized",
        [&]() {
          sgemm_v4_vectorized<V4_BM, V4_BN, V4_BK, V4_TM, V4_TN>
              <<<v4_grid, v4_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);

    benchmark_kernel(
        "v10 double buffer",
        [&]() {
          sgemm_v5_double_buffer<V4_BM, V4_BN, V4_BK, V4_TM, V4_TN>
              <<<v4_grid, v4_block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
          CHECK_CUDA(cudaGetLastError());
        },
        m, n, k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
  } else {
    std::cout << "v5/v9/v10: skipped because N must be a multiple of 128\n";
    csv << "v5/v9/v10 skipped," << n << ",0,0,0,0\n";
    benchmark_kernel("v6 thread coarsening", launch_v6_thread_coarsening, m, n,
                     k, d_c, c_bytes, h_ref, h_out, csv, cublas_perf);
  }
  csv.close();

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
