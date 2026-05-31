#include "gemm_benchmark.cuh"
#include "sgemm_kernels.cuh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename Launch>
float benchmark_tuned_kernel(const std::string& kernel, const std::string& config,
                             Launch launch, int m, int n, int k, int bm,
                             int bn, int bk, int tm, int tn, int threads,
                             int smem_bytes, float* d_c, size_t c_bytes,
                             const std::vector<float>& ref,
                             std::vector<float>& out, std::ofstream& csv,
                             float cublas_gflops) {
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
  const float perf = gflops(m, n, k, avg_ms);
  const bool ok = compare_result(ref, out);
  const float ratio = perf / cublas_gflops;

  std::cout << kernel << " " << config << ": " << avg_ms << " ms, " << perf
            << " GFLOPS, ratio=" << ratio << ", matched=" << ok << '\n';
  csv << kernel << "," << config << "," << n << "," << bm << "," << bn
      << "," << bk << "," << tm << "," << tn << "," << threads << ","
      << smem_bytes << "," << avg_ms << "," << perf << "," << ratio << ","
      << (ok ? 1 : 0) << '\n';
  return perf;
}

template <int BM, int BN, int BK, int TM, int TN>
void run_thread_tile_config(int m, int n, int k, float alpha, const float* d_a,
                            const float* d_b, float beta, float* d_c,
                            size_t c_bytes, const std::vector<float>& ref,
                            std::vector<float>& out, std::ofstream& csv,
                            float cublas_gflops) {
  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);
  constexpr int threads = (BM / TM) * (BN / TN);
  static_assert(threads <= 1024);
  constexpr int smem_bytes = (BM * BK + BK * BN) * static_cast<int>(sizeof(float));

  dim3 grid(ceil_div(n, BN), ceil_div(m, BM));
  dim3 block(threads);
  const std::string config = "BM" + std::to_string(BM) + "_BN" +
                             std::to_string(BN) + "_BK" + std::to_string(BK) +
                             "_TM" + std::to_string(TM) + "_TN" +
                             std::to_string(TN);
  benchmark_tuned_kernel(
      "thread_tile", config,
      [&]() {
        sgemm_v3_thread_tile<BM, BN, BK, TM, TN>
            <<<grid, block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, BM, BN, BK, TM, TN, threads, smem_bytes, d_c, c_bytes, ref,
      out, csv, cublas_gflops);
}

template <int BM, int BN, int BK, int TM, int TN>
void run_vectorized_config(int m, int n, int k, float alpha, const float* d_a,
                           const float* d_b, float beta, float* d_c,
                           size_t c_bytes, const std::vector<float>& ref,
                           std::vector<float>& out, std::ofstream& csv,
                           float cublas_gflops) {
  if (m % BM != 0 || n % BN != 0 || k % BK != 0) return;
  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);
  constexpr int threads = (BM / TM) * (BN / TN);
  static_assert(threads <= 1024);
  constexpr int smem_bytes = (BK * BM + BK * BN) * static_cast<int>(sizeof(float));

  dim3 grid(n / BN, m / BM);
  dim3 block(threads);
  const std::string config = "BM" + std::to_string(BM) + "_BN" +
                             std::to_string(BN) + "_BK" + std::to_string(BK) +
                             "_TM" + std::to_string(TM) + "_TN" +
                             std::to_string(TN);
  benchmark_tuned_kernel(
      "vectorized", config,
      [&]() {
        sgemm_v4_vectorized<BM, BN, BK, TM, TN>
            <<<grid, block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, BM, BN, BK, TM, TN, threads, smem_bytes, d_c, c_bytes, ref,
      out, csv, cublas_gflops);
}

template <int BM, int BN, int BK, int TM, int TN>
void run_double_buffer_config(int m, int n, int k, float alpha, const float* d_a,
                              const float* d_b, float beta, float* d_c,
                              size_t c_bytes, const std::vector<float>& ref,
                              std::vector<float>& out, std::ofstream& csv,
                              float cublas_gflops) {
  if (m % BM != 0 || n % BN != 0 || k % BK != 0) return;
  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);
  constexpr int threads = (BM / TM) * (BN / TN);
  static_assert(threads <= 1024);
  constexpr int smem_bytes =
      2 * (BK * BM + BK * BN) * static_cast<int>(sizeof(float));

  dim3 grid(n / BN, m / BM);
  dim3 block(threads);
  const std::string config = "BM" + std::to_string(BM) + "_BN" +
                             std::to_string(BN) + "_BK" + std::to_string(BK) +
                             "_TM" + std::to_string(TM) + "_TN" +
                             std::to_string(TN);
  benchmark_tuned_kernel(
      "double_buffer", config,
      [&]() {
        sgemm_v5_double_buffer<BM, BN, BK, TM, TN>
            <<<grid, block>>>(m, n, k, alpha, d_a, d_b, beta, d_c);
        CHECK_CUDA(cudaGetLastError());
      },
      m, n, k, BM, BN, BK, TM, TN, threads, smem_bytes, d_c, c_bytes, ref,
      out, csv, cublas_gflops);
}

int main(int argc, char** argv) {
  int n = 2048;
  if (argc > 1) n = std::atoi(argv[1]);
  if (n <= 0 || n % 128 != 0) {
    std::cerr << "Usage: " << argv[0] << " [square_size_multiple_of_128]\n";
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
  for (int i = 0; i < kRepeat; ++i) launch_cublas();
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float cublas_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const float cublas_avg_ms = cublas_ms / kRepeat;
  const float cublas_perf = gflops(m, n, k, cublas_avg_ms);
  std::cout << "N=" << n << ", cuBLAS=" << cublas_perf << " GFLOPS\n";

  std::ofstream csv("gemm_blocksize_tuning.csv");
  csv << "Kernel,Config,N,BM,BN,BK,TM,TN,Threads,SharedMemoryBytes,TimeMs,"
         "GFLOPS,RatioToCuBLAS,Matched\n";
  csv << "cuBLAS,baseline," << n << ",0,0,0,0,0,0,0," << cublas_avg_ms
      << "," << cublas_perf << ",1,1\n";

  run_thread_tile_config<64, 64, 8, 4, 4>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                          c_bytes, h_ref, h_out, csv,
                                          cublas_perf);
  run_thread_tile_config<64, 128, 8, 4, 8>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                           c_bytes, h_ref, h_out, csv,
                                           cublas_perf);
  run_thread_tile_config<128, 64, 8, 8, 4>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                           c_bytes, h_ref, h_out, csv,
                                           cublas_perf);
  run_thread_tile_config<128, 128, 8, 8, 8>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                            c_bytes, h_ref, h_out, csv,
                                            cublas_perf);
  run_thread_tile_config<128, 128, 16, 8, 8>(m, n, k, alpha, d_a, d_b, beta,
                                             d_c, c_bytes, h_ref, h_out, csv,
                                             cublas_perf);

  run_vectorized_config<64, 128, 8, 4, 8>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                          c_bytes, h_ref, h_out, csv,
                                          cublas_perf);
  run_vectorized_config<128, 64, 8, 8, 4>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                          c_bytes, h_ref, h_out, csv,
                                          cublas_perf);
  run_vectorized_config<128, 128, 8, 8, 8>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                           c_bytes, h_ref, h_out, csv,
                                           cublas_perf);
  run_vectorized_config<128, 128, 16, 8, 8>(m, n, k, alpha, d_a, d_b, beta, d_c,
                                            c_bytes, h_ref, h_out, csv,
                                            cublas_perf);

  run_double_buffer_config<64, 128, 8, 4, 8>(m, n, k, alpha, d_a, d_b, beta,
                                             d_c, c_bytes, h_ref, h_out, csv,
                                             cublas_perf);
  run_double_buffer_config<128, 64, 8, 8, 4>(m, n, k, alpha, d_a, d_b, beta,
                                             d_c, c_bytes, h_ref, h_out, csv,
                                             cublas_perf);
  run_double_buffer_config<128, 128, 8, 8, 8>(m, n, k, alpha, d_a, d_b, beta,
                                              d_c, c_bytes, h_ref, h_out, csv,
                                              cublas_perf);
  run_double_buffer_config<128, 128, 16, 8, 8>(m, n, k, alpha, d_a, d_b, beta,
                                               d_c, c_bytes, h_ref, h_out, csv,
                                               cublas_perf);

  csv.close();

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
