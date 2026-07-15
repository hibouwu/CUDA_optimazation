#include "gemm_benchmark.cuh"
#include "backends/tc0_baseline.cuh"
#include "backends/tc1_tc2_tma.cuh"
#include "backends/tc3_pipeline.cuh"
#include "backends/tc4a_warp_specialized.cuh"
#include "backends/tc4bc_cluster.cuh"
#include "backends/tc5_persistent.cuh"
#include "backends/tc6_nvfp4.cuh"
#include "cutlass_sm110_backends.cuh"
#include "sm110_backend_registry.cuh"
#include "requant/nvfp4_reference.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using gemm_sm110::wants_backend;

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " [square_size] "
            << gemm_sm110::kBackendUsage << '\n'
            << "  " << program << " M N K "
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

size_t count_byte_mismatches(const std::vector<std::uint8_t>& ref,
                             const std::vector<std::uint8_t>& got) {
  if (ref.size() != got.size()) return std::max(ref.size(), got.size());
  size_t errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) ++errors;
  }
  return errors;
}

struct Nvfp4ErrorStats {
  double rmse = 0.0;
  float max_abs_error = 0.0f;
};

Nvfp4ErrorStats compute_nvfp4_error(
    const std::vector<float>& fp32_ref,
    const std::vector<std::uint8_t>& quantized,
    const std::vector<std::uint8_t>& block_scales, float tensor_scale) {
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < fp32_ref.size(); ++i) {
    const std::uint8_t packed = quantized[i / 2];
    const std::uint8_t nibble =
        (i & 1u) == 0u ? static_cast<std::uint8_t>(packed >> 4)
                       : static_cast<std::uint8_t>(packed & 0x0fu);
    const float block_scale = gemm_sm110::requant::decode_positive_e4m3(
        block_scales[i / gemm_sm110::requant::kNvfp4BlockSize]);
    const float reconstructed =
        gemm_sm110::requant::decode_e2m1(nibble) * block_scale *
        tensor_scale;
    const float error = reconstructed - fp32_ref[i];
    squared_error += static_cast<double>(error) * error;
    max_abs_error = std::max(max_abs_error, std::fabs(error));
  }

  Nvfp4ErrorStats stats;
  stats.rmse = std::sqrt(squared_error / fp32_ref.size());
  stats.max_abs_error = max_abs_error;
  return stats;
}

template <typename Launch>
float benchmark_tc6_kernel(
    Launch launch, int m, int n, int k, std::uint8_t* d_quantized,
    std::uint8_t* d_block_scales,
    const gemm_sm110::requant::Nvfp4ReferenceResult& ref,
    const std::vector<float>& fp32_ref, std::ofstream& csv,
    float reference_gflops) {
  const size_t quantized_bytes = ref.quantized.size();
  const size_t scale_bytes = ref.block_scales.size();
  CHECK_CUDA(cudaMemset(d_quantized, 0, quantized_bytes));
  CHECK_CUDA(cudaMemset(d_block_scales, 0, scale_bytes));
  for (int i = 0; i < kWarmup; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaMemset(d_quantized, 0, quantized_bytes));
  CHECK_CUDA(cudaMemset(d_block_scales, 0, scale_bytes));

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

  std::vector<std::uint8_t> quantized(quantized_bytes);
  std::vector<std::uint8_t> block_scales(scale_bytes);
  CHECK_CUDA(cudaMemcpy(quantized.data(), d_quantized, quantized_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(block_scales.data(), d_block_scales, scale_bytes,
                        cudaMemcpyDeviceToHost));

  const size_t value_mismatches =
      count_byte_mismatches(ref.quantized, quantized);
  const size_t scale_mismatches =
      count_byte_mismatches(ref.block_scales, block_scales);
  const Nvfp4ErrorStats ref_error =
      compute_nvfp4_error(fp32_ref, ref.quantized, ref.block_scales,
                          ref.tensor_scale);
  const Nvfp4ErrorStats got_error =
      compute_nvfp4_error(fp32_ref, quantized, block_scales,
                          ref.tensor_scale);
  const bool ok =
      got_error.rmse <= ref_error.rmse * 1.10 + 1.0e-6 &&
      got_error.max_abs_error <= ref_error.max_abs_error * 1.25f + 1.0e-5f;
  const float avg_ms = total_ms / kRepeat;
  const float perf = gflops(m, n, k, avg_ms);
  const float ratio =
      reference_gflops > 0.0f ? perf / reference_gflops : 0.0f;
  std::cout << "tc6 fused NVFP4 TCGen05 epilogue: " << avg_ms
            << " ms, " << perf << " GFLOPS, ratio=" << ratio
            << "x, matched=" << ok
            << ", tensor_scale=" << ref.tensor_scale
            << ", bit_exact="
            << (value_mismatches == 0 && scale_mismatches == 0)
            << ", value_mismatches=" << value_mismatches
            << ", scale_mismatches=" << scale_mismatches
            << ", rmse=" << got_error.rmse
            << " (ref=" << ref_error.rmse << ")"
            << ", max_abs_error=" << got_error.max_abs_error
            << " (ref=" << ref_error.max_abs_error << ")\n";
  csv << "tc6,fused NVFP4 TCGen05 epilogue," << n
      << ",fp16->nvfp4,cuBLAS Tensor Core quantized," << avg_ms << ","
      << perf << "," << ratio << "," << (ok ? 1 : 0) << '\n';
  return avg_ms;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  int m = 1024;
  int n = 1024;
  int k = 1024;
  std::string backend_filter = "all";

  if (argc == 2 || argc == 3) {
    n = std::atoi(argv[1]);
    m = n;
    k = n;
    if (argc == 3) backend_filter = argv[2];
  } else if (argc == 4 || argc == 5) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    if (argc == 5) backend_filter = argv[4];
  } else if (argc > 5) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (m <= 0 || n <= 0 || k <= 0 ||
      !gemm_sm110::is_valid_backend_filter(backend_filter)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const bool run_reference = needs_cublas_reference(backend_filter);
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

  std::cout << "M=" << m << ", N=" << n << ", K=" << k << '\n';
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
	  if (const char* pattern = std::getenv("GEMM_INPUT_PATTERN")) {
	    if (std::string(pattern) == "row_id") {
	      std::fill(h_a.begin(), h_a.end(), 0.0f);
	      std::fill(h_b.begin(), h_b.end(), 0.0f);
	      const int diagonal = std::min(m, k);
	      for (int row = 0; row < diagonal; ++row) {
	        h_a[static_cast<size_t>(row) * k + row] = 1.0f;
	      }
	      for (int k_idx = 0; k_idx < k; ++k_idx) {
	        const float row_value = static_cast<float>(k_idx);
	        for (int n_idx = 0; n_idx < n; ++n_idx) {
	          h_b[static_cast<size_t>(k_idx) * n + n_idx] = row_value;
	        }
	      }
	    }
	  }

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
    } else if (m % 256 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("cutlass"), n, csv,
          "current CUTLASS reference config requires M%256=0, N%128=0, "
          "and K%64=0");
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
    if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc0"), n, csv,
          "requires M,N,K multiples of 16");
    } else {
      using Tc0Runner = gemm_sm110::backends::Tc0Runner;
      Tc0Runner tc0_runner(d_a_half, d_b_half, d_c, m, n, k);
      auto launch_tc0 = [&]() { tc0_runner.launch(); };
      std::vector<float> h_tc0(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc0", "tc0 CUDA WMMA Tensor Core baseline", "fp16->fp32",
          "cuBLAS Tensor Core", launch_tc0, m, n, k, d_c, c_bytes, h_ref_tc,
          h_tc0, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc1a")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc1a"), n, csv,
                                "requires an SM110-family target");
    } else {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc1a"), n, csv,
          "disabled pending linear SMEM descriptor validation");
    }
  }

  if (wants_backend(backend_filter, "tc1b")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc1b"), n, csv,
                                "requires an SM110-family target");
    } else {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc1b"), n, csv,
          "disabled pending linear SMEM descriptor validation");
    }
  }

  if (wants_backend(backend_filter, "tc2a")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc2a"), n, csv,
                                "requires an SM110-family target");
    } else if (m % 128 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc2a"), n, csv,
          "requires M,N multiples of 128 and K a multiple of 64");
    } else {
      gemm_sm110::backends::Tc2aRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc2a", "tc2a 2D TMA SW128-SMEM TCGen05", "fp16->fp32",
          "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
          output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc2b")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc2b"), n, csv,
                                "requires an SM110-family target");
    } else if (m % 128 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc2b"), n, csv,
          "requires M,N multiples of 128 and K a multiple of 64");
    } else {
      gemm_sm110::backends::Tc2bRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc2b", "tc2b 3D TMA SW128-SMEM TCGen05", "fp16->fp32",
          "cuBLAS Tensor Core", launch, m, n, k, d_c, c_bytes, h_ref_tc,
          output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc3")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc3"), n, csv,
                                "requires an SM110-family target");
    } else if (m % 128 != 0 || n % 128 != 0 || k % 64 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc3"), n, csv,
          "requires M,N multiples of 128 and K a multiple of 64");
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
    if (m % 128 != 0 || n % 256 != 0 || k % 128 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc4a"), n, csv,
          "requires M%128=0, N%256=0, and K%128=0");
    } else {
      gemm_sm110::backends::Tc4aRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4a", "tc4a warp-specialized TMA/TCGen05 pipeline",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc4b")) {
    if (m % 256 != 0 || n % 256 != 0 || k % 128 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc4b"), n, csv,
          "requires M,N multiples of 256 and K a multiple of 128");
    } else {
      gemm_sm110::backends::Tc4bcRunner<false> runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4b", "tc4b 2-SM cluster TMA/TCGen05 pipeline",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc4c")) {
    if (m % 256 != 0 || n % 256 != 0 || k % 128 != 0) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc4c"), n, csv,
          "requires M,N multiples of 256 and K a multiple of 128");
    } else {
      gemm_sm110::backends::Tc4bcRunner<true> runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc4c", "tc4c warp-specialized 2-SM cluster pipeline",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5a")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5a"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5aRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5a", "tc5a static persistent 1-SM TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5b")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5b"), n, csv,
                                "requires an SM110-family target");
    } else {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5b"), n, csv,
          "disabled pending dynamic work-queue stability validation");
    }
  }

  if (wants_backend(backend_filter, "tc5c")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5c"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5cRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5c", "tc5c static persistent M128N128K128 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5d")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5d"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5dRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5d", "tc5d static persistent M128N256K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5e")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5e"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5eRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5e", "tc5e static persistent M128N128K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5f")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5f"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5fRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5f", "tc5f static persistent M128N256K128 stage1 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5g")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5g"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5gRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5g", "tc5g static persistent M128N256K64 stage1 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5h")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5h"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5hRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5h", "tc5h overlapped epilogue M128N256K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5i")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5i"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5iRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5i", "tc5i overlapped epilogue M128N128K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5j")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5j"), n, csv,
                                "requires an SM110-family target");
    } else {
      gemm_sm110::backends::Tc5jRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5j", "tc5j overlapped epilogue M128N256K128 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5k")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5k"), n, csv,
                                "requires an SM110-family target");
    } else if (std::getenv("TC5K_RUN_EXPERIMENT") == nullptr) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5k"), n, csv,
          "disabled pending M64 epilogue performance validation");
    } else {
      gemm_sm110::backends::Tc5kRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5k",
          "tc5k experimental overlapped epilogue M64N256K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5l")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5l"), n, csv,
                                "requires an SM110-family target");
    } else if (std::getenv("TC5L_RUN_EXPERIMENT") == nullptr) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5l"), n, csv,
          "disabled pending B-reuse M256N256 validation");
    } else {
      gemm_sm110::backends::Tc5lRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5l", "tc5l experimental B-reuse M256N256K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5m")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5m"), n, csv,
                                "requires an SM110-family target");
    } else if (std::getenv("TC5M_RUN_EXPERIMENT") == nullptr) {
      write_unavailable_backend(
          *gemm_sm110::find_backend("tc5m"), n, csv,
          "disabled pending overlapped B-reuse M256N128 validation");
    } else {
      gemm_sm110::backends::Tc5mRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5m",
          "tc5m experimental overlapped B-reuse M256N128K64 TCGen05 GEMM",
          "fp16->fp32", "cuBLAS Tensor Core", launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc6")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc6"), n, csv,
                                "requires an SM110-family target");
    } else {
      const auto nvfp4_reference =
          gemm_sm110::requant::make_nvfp4_reference(h_ref_tc);
      std::uint8_t* d_tc6_values = nullptr;
      std::uint8_t* d_tc6_scales = nullptr;
      CHECK_CUDA(cudaMalloc(&d_tc6_values,
                            nvfp4_reference.quantized.size()));
      CHECK_CUDA(cudaMalloc(&d_tc6_scales,
                            nvfp4_reference.block_scales.size()));

      gemm_sm110::backends::Tc6Runner runner(
          d_a_half, d_b_half_nk, d_tc6_values, d_tc6_scales, m, n, k,
          1.0f / nvfp4_reference.tensor_scale);
      auto launch = [&]() { runner.launch(); };
      benchmark_tc6_kernel(launch, m, n, k, d_tc6_values, d_tc6_scales,
                           nvfp4_reference, h_ref_tc, csv,
                           cublas_tc_perf);

      CHECK_CUDA(cudaFree(d_tc6_values));
      CHECK_CUDA(cudaFree(d_tc6_scales));
    }
  }

  csv.close();
  if (d_c != nullptr) CHECK_CUDA(cudaFree(d_c));
  if (d_a_half != nullptr) CHECK_CUDA(cudaFree(d_a_half));
  if (d_b_half != nullptr) CHECK_CUDA(cudaFree(d_b_half));
  if (d_b_half_nk != nullptr) CHECK_CUDA(cudaFree(d_b_half_nk));
  return 0;
}
