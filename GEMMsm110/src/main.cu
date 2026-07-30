#include "gemm_benchmark.cuh"
#include "backends/tc0_baseline.cuh"
#include "backends/tc1_tc2_tma.cuh"
#include "backends/tc3_pipeline.cuh"
#include "backends/tc4a_warp_specialized.cuh"
#include "backends/tc4bc_cluster.cuh"
#include "backends/tc5_persistent.cuh"
#include "backends/tc6_nvfp4.cuh"
#include "backends/shapeopt_specialized.cuh"
#include "cublaslt_reference.cuh"
#include "sm110_backend_registry.cuh"
#include "requant/nvfp4_reference.cuh"

#ifndef GEMM_SM110_ENABLE_CUTLASS
#define GEMM_SM110_ENABLE_CUTLASS 1
#endif

#if GEMM_SM110_ENABLE_CUTLASS
#include "cutlass_sm110_backends.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using gemm_sm110::wants_backend;

constexpr const char* kTensorCoreReferenceName =
    "cuBLASLt Matmul heuristic";

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " [square_size] "
            << gemm_sm110::kBackendUsage << " [none|bias|relu|gelu|residual]\n"
            << "  " << program << " M N K "
            << gemm_sm110::kBackendUsage << " [none|bias|relu|gelu|residual]\n";
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
      << ",fp16->fp32," << kTensorCoreReferenceName << ",0,0,0,0\n";
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

template <typename Launch>
float tune_launch_ms(Launch launch, int repeats = 20) {
  for (int i = 0; i < 3; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / repeats;
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
  std::string backend_filter = "core";
  gemm_sm110::references::EpilogueMode epilogue_mode =
      gemm_sm110::references::EpilogueMode::kNone;

  auto is_positive_integer = [](const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    for (const char* ptr = value; *ptr != '\0'; ++ptr) {
      if (*ptr < '0' || *ptr > '9') return false;
    }
    return std::atoi(value) > 0;
  };

  if (argc >= 4 && is_positive_integer(argv[1]) &&
      is_positive_integer(argv[2]) && is_positive_integer(argv[3])) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    if (argc >= 5) backend_filter = argv[4];
    if (argc >= 6 &&
        !gemm_sm110::references::parse_epilogue_mode(argv[5],
                                                     &epilogue_mode)) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (argc > 6) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  } else if (argc >= 2 && is_positive_integer(argv[1])) {
    n = std::atoi(argv[1]);
    m = n;
    k = n;
    if (argc >= 3) backend_filter = argv[2];
    if (argc >= 4 &&
        !gemm_sm110::references::parse_epilogue_mode(argv[3],
                                                     &epilogue_mode)) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (argc > 4) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  } else if (argc > 1) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (m <= 0 || n <= 0 || k <= 0 ||
      !gemm_sm110::is_valid_backend_filter(backend_filter)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const bool run_reference = needs_cublas_reference(backend_filter);

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
  float* d_bias = nullptr;
  float* d_residual = nullptr;
  if (d_c_bytes > 0) {
    CHECK_CUDA(cudaMalloc(&d_c, d_c_bytes));
  }

  const bool device_supports_tc3_sm110 = device_prop.major == 11;

  std::ofstream csv("sgemm_sm110_benchmark.csv");
  csv << "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,"
         "RatioToReference,Matched\n";

  std::cout << "M=" << m << ", N=" << n << ", K=" << k << '\n';
  std::cout << "Backend filter=" << backend_filter << '\n';
  std::cout << "Epilogue="
            << gemm_sm110::references::to_string(epilogue_mode) << '\n';
  std::cout << "Benchmark policy: warmup=" << kWarmup
            << ", timed repeats=" << kRepeat
            << " per backend before moving to the next backend\n";

  const size_t a_half_bytes = static_cast<size_t>(m) * k * sizeof(half);
  const size_t b_half_bytes = static_cast<size_t>(k) * n * sizeof(half);
  const size_t bias_bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> h_a(static_cast<size_t>(m) * k);
	  std::vector<float> h_b(static_cast<size_t>(k) * n);
	  std::vector<float> h_ref_tc(static_cast<size_t>(m) * n);
  std::vector<float> h_bias(static_cast<size_t>(n));
  std::vector<float> h_residual(static_cast<size_t>(m) * n);
	  fill_inputs(h_a, h_b);
  fill_epilogue_inputs(h_bias, h_residual);
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
  if (epilogue_mode == gemm_sm110::references::EpilogueMode::kBias) {
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), bias_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (epilogue_mode == gemm_sm110::references::EpilogueMode::kResidual) {
    CHECK_CUDA(cudaMalloc(&d_residual, d_c_bytes));
    CHECK_CUDA(cudaMemcpy(d_residual, h_residual.data(), d_c_bytes,
                          cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaMemcpy(d_a_half, h_a_half.data(), a_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half, h_b_half.data(), b_half_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_half_nk, h_b_half_nk.data(), b_half_bytes,
                        cudaMemcpyHostToDevice));

  float cublas_tc_perf = 0.0f;
  float cublas_avg_ms = 0.0f;
  std::unique_ptr<gemm_sm110::references::CublasLtMatmulReference>
      cublaslt_reference;
  if (run_reference) {
    cublasHandle_t version_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&version_handle));
    print_gemm_environment(version_handle);
    CHECK_CUBLAS(cublasDestroy(version_handle));

    cublaslt_reference =
        std::make_unique<gemm_sm110::references::CublasLtMatmulReference>(
            d_a_half, d_b_half, d_c, m, n, k, epilogue_mode, d_bias,
            d_residual);
    std::cout << "cuBLASLt heuristic algorithms returned="
              << cublaslt_reference->returned_algorithms()
              << ", selected workspace="
              << cublaslt_reference->selected_workspace_bytes()
              << " bytes, workspace limit="
              << cublaslt_reference->workspace_bytes() << " bytes\n";
    auto launch_cublas_tensor_core = [&]() {
      cublaslt_reference->launch();
    };

    cublas_avg_ms =
        benchmark_reference(launch_cublas_tensor_core, d_c, c_bytes, h_ref_tc);
    cublas_tc_perf = gflops(m, n, k, cublas_avg_ms);

    std::cout << "Tensor Core reference:\n";
    std::cout << kTensorCoreReferenceName << ": " << cublas_avg_ms << " ms, "
              << cublas_tc_perf << " GFLOPS\n";
    csv << "cublas_tc," << kTensorCoreReferenceName << "," << n
        << ",fp16->fp32," << kTensorCoreReferenceName << "," << cublas_avg_ms << ","
        << cublas_tc_perf << ",1,1\n";
  }

  if (wants_backend(backend_filter, "shapeopt")) {
    if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
        device_supports_tc3_sm110 && m == 2048 && n == 2048 && k == 2048) {
      gemm_sm110::backends::Tc5aRunner shapeopt_runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "shapeopt", "ShapeOpt custom square tc5a TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
          d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
          2e-3f);
    } else if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
               device_supports_tc3_sm110 && m == 4096 && n == 64 &&
               k == 4096) {
      int variant = -1;
      if (const char* env = std::getenv("SHAPEOPT_SKINNY_N_VARIANT")) {
        variant = std::atoi(env);
      }
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      if (variant < 0) {
        gemm_sm110::backends::Tc5Runner<64, 128, 2> raw_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        gemm_sm110::backends::Tc5OverlapRunner<128, 64, 64, 4>
            overlap_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        const float raw_ms =
            tune_launch_ms([&]() { raw_runner.launch(); });
        const float overlap_ms =
            tune_launch_ms([&]() { overlap_runner.launch(); });
        const bool use_overlap = overlap_ms < raw_ms;
        std::cout << "ShapeOpt skinny-N autotune: raw=" << raw_ms
                  << " ms, overlap=" << overlap_ms
                  << " ms, chosen="
                  << (use_overlap ? "overlap" : "raw") << '\n';
        auto launch_shapeopt = [&]() {
          if (use_overlap) {
            overlap_runner.launch();
          } else {
            raw_runner.launch();
          }
        };
        benchmark_kernel(
            "shapeopt",
            use_overlap
                ? "ShapeOpt custom skinny-N autotuned tc5a TileN64K64 GEMM"
                : "ShapeOpt custom skinny-N autotuned tc5 TileN64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 1) {
        gemm_sm110::backends::Tc5Runner<64, 64, 2> shapeopt_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-N tc5 TileN64K64 TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 2) {
        gemm_sm110::backends::Tc5Runner<64, 64, 4> shapeopt_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-N tc5 TileN64K64S4 TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 3) {
        gemm_sm110::backends::Tc5Runner<64, 128, 1> shapeopt_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-N tc5 TileN64K128S1 TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 4) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 64, 64, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-N tc5a TileN64K64 TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else {
        gemm_sm110::backends::Tc5Runner<64, 128, 2> shapeopt_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt", "ShapeOpt custom skinny-N tc5 TileN64 TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      }
    } else if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
               device_supports_tc3_sm110 && m == 1024 && n == 1024 &&
               k == 1000) {
      constexpr int kPaddedK = 1024;
      half* d_a_padded = nullptr;
      half* d_b_nk_padded = nullptr;
      CHECK_CUDA(cudaMalloc(&d_a_padded,
                            static_cast<size_t>(m) * kPaddedK *
                                sizeof(half)));
      CHECK_CUDA(cudaMalloc(&d_b_nk_padded,
                            static_cast<size_t>(n) * kPaddedK *
                                sizeof(half)));
      gemm_sm110::backends::shapeopt_detail::launch_pad_k_major_rows(
          d_a_half, d_a_padded, m, k, kPaddedK);
      gemm_sm110::backends::shapeopt_detail::launch_pad_k_major_rows(
          d_b_half_nk, d_b_nk_padded, n, k, kPaddedK);
      CHECK_CUDA(cudaDeviceSynchronize());
      gemm_sm110::backends::Tc5aRunner shapeopt_runner(
          d_a_padded, d_b_nk_padded, d_c, m, n, kPaddedK);
      auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "shapeopt", "ShapeOpt custom tail-K padded tc5 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
          d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
          2e-3f);
      CHECK_CUDA(cudaFree(d_a_padded));
      CHECK_CUDA(cudaFree(d_b_nk_padded));
    } else if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
               device_supports_tc3_sm110 && m == 1152 && n == 768 &&
               k == 1024) {
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      int variant = -1;
      if (const char* env = std::getenv("SHAPEOPT_TAIL_MN_VARIANT")) {
        variant = std::atoi(env);
      }
      if (variant < 0) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc5aRunner tc5a_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3>
            cluster_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 3>
            cluster_k128_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM,
                                n, k);
        gemm_sm110::backends::Tc5TailMnN192Runner<64, 4>
            n192_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        const float tc5a_ms =
            tune_launch_ms([&]() { tc5a_runner.launch(); });
        const float cluster_ms =
            tune_launch_ms([&]() { cluster_runner.launch(); });
        const float cluster_k128_ms =
            tune_launch_ms([&]() { cluster_k128_runner.launch(); });
        const float n192_ms =
            tune_launch_ms([&]() { n192_runner.launch(); });
        int chosen = 0;
        float chosen_ms = tc5a_ms;
        if (cluster_ms < chosen_ms) {
          chosen = 1;
          chosen_ms = cluster_ms;
        }
        if (cluster_k128_ms < chosen_ms) {
          chosen = 3;
          chosen_ms = cluster_k128_ms;
        }
        if (n192_ms < chosen_ms) {
          chosen = 2;
          chosen_ms = n192_ms;
        }
        std::cout << "ShapeOpt tail-MN autotune: tc5a=" << tc5a_ms
                  << " ms, padded_cluster=" << cluster_ms
                  << " ms, padded_cluster_k128=" << cluster_k128_ms
                  << " ms, split_n192=" << n192_ms
                  << " ms, chosen="
                  << (chosen == 3
                          ? "padded_cluster_k128"
                          : (chosen == 2
                                 ? "split_n192"
                                 : (chosen == 1 ? "padded_cluster"
                                                : "tc5a")))
                  << '\n';
        auto launch_shapeopt = [&]() {
          if (chosen == 3) {
            cluster_k128_runner.launch();
          } else if (chosen == 2) {
            n192_runner.launch();
          } else if (chosen == 1) {
            cluster_runner.launch();
          } else {
            tc5a_runner.launch();
          }
        };
        const char* shapeopt_name =
            chosen == 3
                ? "ShapeOpt custom tail-MN autotuned padded cluster tc4c "
                  "M256N256K128S3 GEMM"
            : chosen == 2
                ? "ShapeOpt custom tail-MN autotuned split-N192 tc5 "
                  "M128N192K64S4 GEMM"
            : chosen == 1
                ? "ShapeOpt custom tail-MN autotuned padded cluster tc4c "
                  "M256N256K64S3 GEMM"
                : "ShapeOpt custom tail-MN autotuned tc5a TCGen05 GEMM";
        benchmark_kernel(
            "shapeopt", shapeopt_name,
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 1) {
        gemm_sm110::backends::Tc5TailMnPairNRunner<64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN pair-N tc5 TileN256x2K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 2) {
        gemm_sm110::backends::Tc5TailMnPairNRunner<128, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN pair-N tc5 TileN256x2K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 3) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 256, 64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN tc5a M128N256K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 4) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 256, 64, 3>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN tc5a M128N256K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 5) {
        gemm_sm110::backends::Tc5M64Runner<256, 64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN direct M64 tc5 M64N256K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 6) {
        gemm_sm110::backends::Tc5M64Runner<128, 64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN direct M64 tc5 M64N128K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 7) {
        auto launch_shapeopt = [&]() {
          gemm_sm110::backends::shapeopt_detail::launch_wmma_m64n32_shared(
              d_a_half, d_b_half, d_c, m, n, k);
        };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN shared WMMA M64N32 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 8) {
        gemm_sm110::backends::Tc5TailMnPairMRunner<64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN pair-M tc5 TileM128x2N256K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 9) {
        gemm_sm110::backends::Tc5TailMnPairMRunner<128, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN pair-M tc5 TileM128x2N256K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 10) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<2, 256, 128, 1, 128>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN row-major splitK2 tc5 M128N256K128",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 11) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<4, 256, 128, 1, 128>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN row-major splitK4 tc5 M128N256K128",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 12) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<2, 256, 64, 2, 128>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN row-major splitK2 tc5 M128N256K64",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 13) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<4, 256, 64, 2, 128>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN row-major splitK4 tc5 M128N256K64",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 14) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 256, 64, 4, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap M64 tc5 M64N256K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 15) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 128, 64, 4, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap M64 tc5 M64N128K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 16) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 128, 64, 4, 8>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap M64 tc5 M64N128K64E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 17) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 256, 64, 4, 8>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap M64 tc5 M64N256K64E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 18) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 128, 64, 2, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap tc5 M128N128K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 19) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 128, 64, 3, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap tc5 M128N128K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 20) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 128, 64, 4, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap tc5 M128N128K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 21) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 256, 64, 4, 4, 3, 16, 27>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN fixed 9x3 tc5 M128N256K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 22) {
        gemm_sm110::backends::Tc5OverlapRunner<128, 256, 64, 4, 8>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN overlap tc5 M128N256K64S4E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 23) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 2>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 24) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 2>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 25) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 26) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<192, 64, 3>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N192K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 27) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<192, 128, 2>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N192K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 28) {
        gemm_sm110::backends::Tc5TailMnN192Runner<64, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN split-N192 tc5 M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 29) {
        gemm_sm110::backends::Tc5TailMnN192Runner<128, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN split-N192 tc5 M128N192K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 30) {
        constexpr int kClusterM = 1024;
        constexpr int kTailM = 128;
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3>
            cluster_runner(d_a_half, d_b_half_nk, d_c, kClusterM,
                           kClusterM, n, k);
        gemm_sm110::backends::Tc5aRunner tail_runner(
            d_a_half + static_cast<size_t>(kClusterM) * k,
            d_b_half_nk, d_c + static_cast<size_t>(kClusterM) * n,
            kTailM, n, k);
        auto launch_shapeopt = [&]() {
          cluster_runner.launch();
          tail_runner.launch();
        };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN split-M cluster tc4c + tail tc5a GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 31) {
        gemm_sm110::backends::Tc5TailMnN192ClusterLaunchRunner<64, 4, false>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN cluster-launch N-fast tc5 "
            "M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 32) {
        gemm_sm110::backends::Tc5TailMnN192ClusterLaunchRunner<64, 4, true>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN cluster-launch M-fast tc5 "
            "M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 33) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3, 8>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K64S3E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 34) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3, 4,
                                                          false>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt diagnostic tail-MN padded cluster tc4c no-store "
            "M256N256K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 35) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3, 4,
                                                          true, 64>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c smem-store64 "
            "M256N256K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 36) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 3, 4,
                                                          true, 128>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c smem-store128 "
            "M256N256K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 41) {
        gemm_sm110::backends::Tc5TailMnN192Runner<64, 4, true>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN split-N192 tc5 no-wait "
            "M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 42) {
        gemm_sm110::backends::Tc5TailMnN192ClusterLaunchRunner<64, 4, false,
                                                               true>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN cluster-launch N-fast tc5 no-wait "
            "M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 43) {
        gemm_sm110::backends::Tc5TailMnN192ClusterLaunchRunner<64, 4, true,
                                                               true>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN cluster-launch M-fast tc5 no-wait "
            "M128N192K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 44) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 64, 4>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 45) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 3>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K128S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 46) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 256, 1>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c M256N256K256S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 47) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapSplitN192PaddedRowsRunner<64, 3>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c split-N192 "
            "M256N192K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 49) {
        constexpr int kPaddedM = 1280;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapSplitN192PaddedRowsRunner<128, 2>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN padded cluster tc4c split-N192 "
            "M256N192K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else if (variant == 50) {
        constexpr int kClusterM = 1024;
        constexpr int kTailM = 128;
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 3>
            cluster_runner(d_a_half, d_b_half_nk, d_c, kClusterM,
                           kClusterM, n, k);
        gemm_sm110::backends::Tc5aRunner tail_runner(
            d_a_half + static_cast<size_t>(kClusterM) * k,
            d_b_half_nk, d_c + static_cast<size_t>(kClusterM) * n,
            kTailM, n, k);
        cudaStream_t cluster_stream = nullptr;
        cudaStream_t tail_stream = nullptr;
        CHECK_CUDA(cudaStreamCreate(&cluster_stream));
        CHECK_CUDA(cudaStreamCreate(&tail_stream));
        auto launch_shapeopt = [&]() {
          cluster_runner.launch(cluster_stream);
          tail_runner.launch(tail_stream);
        };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN concurrent split-M tc4c K128S3 + "
            "tail tc5a GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaStreamDestroy(cluster_stream));
        CHECK_CUDA(cudaStreamDestroy(tail_stream));
      } else if (variant == 51) {
        constexpr int kClusterM = 1024;
        constexpr int kTailM = 128;
        constexpr int kPaddedTailM = 256;
        half* d_tail_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_tail_padded,
                              static_cast<size_t>(kPaddedTailM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half + static_cast<size_t>(kClusterM) * k, d_tail_padded,
            kTailM, kPaddedTailM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 3>
            cluster_runner(d_a_half, d_b_half_nk, d_c, kClusterM,
                           kClusterM, n, k);
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<256, 128, 3>
            tail_runner(d_tail_padded, d_b_half_nk,
                        d_c + static_cast<size_t>(kClusterM) * n,
                        kTailM, kPaddedTailM, n, k);
        cudaStream_t cluster_stream = nullptr;
        cudaStream_t tail_stream = nullptr;
        CHECK_CUDA(cudaStreamCreate(&cluster_stream));
        CHECK_CUDA(cudaStreamCreate(&tail_stream));
        auto launch_shapeopt = [&]() {
          cluster_runner.launch(cluster_stream);
          tail_runner.launch(tail_stream);
        };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom tail-MN concurrent split-M tc4c K128S3 + "
            "padded-tail tc4c GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaStreamDestroy(cluster_stream));
        CHECK_CUDA(cudaStreamDestroy(tail_stream));
        CHECK_CUDA(cudaFree(d_tail_padded));
      } else {
        gemm_sm110::backends::Tc5aRunner shapeopt_runner(
            d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt", "ShapeOpt custom tail-MN tc5a TCGen05 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      }
    } else if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
               device_supports_tc3_sm110 && m == 64 && n == 4096 &&
               k == 4096) {
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      int variant = -1;
      if (const char* env = std::getenv("SHAPEOPT_SKINNY_M_VARIANT")) {
        variant = std::atoi(env);
      }
      if (variant < 0) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 128, 2>
            raw_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 64, 4, 4>
            overlap_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 2, 4>
            wide_k_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 2, 8>
            wide_k_e8_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<
            64, 64, 4, 4, true>
            smem_overlap_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        const float raw_ms =
            tune_launch_ms([&]() { raw_runner.launch(); });
        const float overlap_ms =
            tune_launch_ms([&]() { overlap_runner.launch(); });
        const float wide_k_ms =
            tune_launch_ms([&]() { wide_k_runner.launch(); });
        const float wide_k_e8_ms =
            tune_launch_ms([&]() { wide_k_e8_runner.launch(); });
        const float smem_overlap_ms =
            tune_launch_ms([&]() { smem_overlap_runner.launch(); });
        int chosen = 2;
        float chosen_ms = wide_k_ms;
        if (wide_k_e8_ms < 0.85f * chosen_ms) {
          chosen = 4;
          chosen_ms = wide_k_e8_ms;
        }
        if (smem_overlap_ms < 0.90f * chosen_ms) {
          chosen = 3;
          chosen_ms = smem_overlap_ms;
        }
        std::cout << "ShapeOpt skinny-M autotune: raw=" << raw_ms
                  << " ms, overlap=" << overlap_ms
                  << " ms, overlap_k256=" << wide_k_ms
                  << " ms, overlap_k256_e8=" << wide_k_e8_ms
                  << " ms, overlap_smem=" << smem_overlap_ms
                  << " ms, chosen="
                  << (chosen == 4
                          ? "overlap_k256_e8"
                          : (chosen == 3
                          ? "overlap_smem"
                          : (chosen == 2
                                 ? "overlap_k256"
                                 : (chosen == 1 ? "overlap" : "raw"))))
                  << '\n';
        auto launch_shapeopt = [&]() {
          if (chosen == 4) {
            wide_k_e8_runner.launch();
          } else if (chosen == 3) {
            smem_overlap_runner.launch();
          } else if (chosen == 2) {
            wide_k_runner.launch();
          } else if (chosen == 1) {
            overlap_runner.launch();
          } else {
            raw_runner.launch();
          }
        };
        benchmark_kernel(
            "shapeopt",
            chosen == 4
                ? "ShapeOpt custom skinny-M autotuned overlap-transpose tc5 "
                  "TileN64K256S2E8 GEMM"
            : chosen == 2
                ? "ShapeOpt custom skinny-M autotuned overlap-transpose tc5 "
                  "TileN64K256 GEMM"
                : (chosen == 3
                ? "ShapeOpt custom skinny-M autotuned overlap smem-transpose "
                  "tc5 TileN64K64S4 GEMM"
                : (chosen == 1
                ? "ShapeOpt custom skinny-M autotuned overlap-transpose tc5 "
                "TileN64K64 GEMM"
                : "ShapeOpt custom skinny-M autotuned direct-transpose tc5 "
                  "TileN64 GEMM")),
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 1) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 64, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 2) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 128, 1>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64K128S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 3) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 64, 4, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 4) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 128, 2, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 5) {
        gemm_sm110::backends::Tc5M64Runner<256, 64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N256K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 6) {
        gemm_sm110::backends::Tc5M64Runner<256, 128, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N256K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 7) {
        gemm_sm110::backends::Tc5M64Runner<256, 128, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N256K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 8) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 256, 64, 4, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap M64 tc5 M64N256K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 9) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<2, 256, 128, 1, 64>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 splitK2 tc5 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 10) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<4, 256, 128, 1, 64>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 splitK4 tc5 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 11) {
        gemm_sm110::backends::Tc5RowMajorSplitKRunner<8, 256, 128, 1, 64>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 splitK8 tc5 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 12) {
        auto launch_shapeopt = [&]() {
          gemm_sm110::backends::shapeopt_detail::launch_wmma_m64n32_shared(
              d_a_half, d_b_half, d_c, m, n, k);
        };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M shared WMMA M64N32 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 13) {
        gemm_sm110::backends::Tc5M64Runner<64, 128, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N64K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 14) {
        gemm_sm110::backends::Tc5M64Runner<64, 64, 2>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N64K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 15) {
        gemm_sm110::backends::Tc5M64Runner<128, 128, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N128K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 16) {
        gemm_sm110::backends::Tc5TransposedSmemStoreRunner<64, 128, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M smem-transpose tc5 TileN64K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 17) {
        gemm_sm110::backends::Tc5TransposedSmemStoreRunner<64, 64, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M smem-transpose tc5 TileN64K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 18) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<32, 128, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN32K128 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 20) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<32, 64, 4, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN32K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 21) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<16, 64, 4, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN16K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 22) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 256, 1>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64K256S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 23) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 256, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64K256S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 24) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 1, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K256S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 25) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 2, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K256S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 26) {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 512, 1>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64K512S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 27) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 512, 1, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K512S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 28) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 2, 8>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K256S2E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 29) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 1, 8>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K256S1E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 30) {
        gemm_sm110::backends::Tc5M64Runner<256, 256, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N256K256 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 31) {
        gemm_sm110::backends::Tc5M64Runner<128, 256, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N128K256 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 32) {
        gemm_sm110::backends::Tc5M64Runner<64, 256, 1>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct M64 tc5 M64N64K256 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 33) {
        gemm_sm110::backends::Tc5OverlapRunner<64, 256, 256, 1, 4>
            shapeopt_runner(d_a_half, d_b_half_nk, d_c, m, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap M64 tc5 M64N256K256 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant >= 34 && variant <= 43) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 256, 2, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M reserved experimental slot using "
            "overlap-transpose tc5 TileN64K256S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 44) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 64, 3, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 45) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 64, 2, 4>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 46) {
        gemm_sm110::backends::Tc5OverlapTransposedStoreRunner<64, 64, 4, 8>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose tc5 TileN64K64S4E8 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 47) {
        gemm_sm110::backends::
            Tc5OverlapTransposedStoreClusterLaunchRunner<64, 64, 4, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose cluster-launch tc5 "
            "TileN64K64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 48) {
        gemm_sm110::backends::
            Tc5OverlapTransposedStoreClusterLaunchRunner<64, 256, 2, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap-transpose cluster-launch tc5 "
            "TileN64K256S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 49) {
        gemm_sm110::backends::Tc5PairMTransposedStoreRunner<64, 64, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M transpose tc5 M128x2N64K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 50) {
        gemm_sm110::backends::Tc5PairMTransposedStoreRunner<64, 128, 1>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M transpose tc5 M128x2N64K128S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 51) {
        gemm_sm110::backends::Tc5PairMTransposedStoreRunner<64, 64, 3>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M transpose tc5 M128x2N64K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 52) {
        gemm_sm110::backends::Tc5PairMTransposedStoreRunner<64, 256, 1>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M transpose tc5 M128x2N64K256S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 53) {
        gemm_sm110::backends::
            Tc5PairMOverlapTransposedStoreRunner<64, 64, 2, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M overlap-transpose tc5 "
            "M128x2N64K64S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 54) {
        gemm_sm110::backends::
            Tc5PairMOverlapTransposedStoreRunner<64, 64, 3, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M overlap-transpose tc5 "
            "M128x2N64K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 55) {
        gemm_sm110::backends::
            Tc5PairMOverlapTransposedStoreRunner<64, 128, 1, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M overlap-transpose tc5 "
            "M128x2N64K128S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 56) {
        gemm_sm110::backends::
            Tc5PairMOverlapTransposedStoreRunner<64, 256, 1, 4>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M pair-M overlap-transpose tc5 "
            "M128x2N64K256S1 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 57) {
        gemm_sm110::backends::
            Tc5OverlapTransposedStoreRunner<64, 64, 4, 4, true>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap smem-transpose tc5 "
            "TileN64K64S4 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 58) {
        gemm_sm110::backends::
            Tc5OverlapTransposedStoreRunner<64, 64, 3, 4, true>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap smem-transpose tc5 "
            "TileN64K64S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 59) {
        gemm_sm110::backends::
            Tc5OverlapTransposedStoreRunner<64, 128, 2, 4, true>
                shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M overlap smem-transpose tc5 "
            "TileN64K128S2 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      } else if (variant == 60) {
        constexpr int kPaddedM = 256;
        half* d_a_padded = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a_padded,
                              static_cast<size_t>(kPaddedM) * k *
                                  sizeof(half)));
        gemm_sm110::backends::shapeopt_detail::launch_pad_rows(
            d_a_half, d_a_padded, m, kPaddedM, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        gemm_sm110::backends::Tc4cOverlapPaddedRowsRunner<64, 128, 3>
            shapeopt_runner(d_a_padded, d_b_half_nk, d_c, m, kPaddedM, n, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt diagnostic skinny-M direct padded cluster tc4c "
            "M256N64K128S3 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
        CHECK_CUDA(cudaFree(d_a_padded));
      } else {
        gemm_sm110::backends::Tc5TransposedStoreRunner<64, 128, 2>
            shapeopt_runner(d_b_half_nk, d_a_half, d_c, n, m, k);
        auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
        benchmark_kernel(
            "shapeopt",
            "ShapeOpt custom skinny-M direct-transpose tc5 TileN64 GEMM",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      }
    } else if (epilogue_mode == gemm_sm110::references::EpilogueMode::kNone &&
               m == 384 && n == 520 && k == 300) {
      constexpr int kPaddedN = 528;
      constexpr int kPaddedK = 304;
      half* d_a_padded = nullptr;
      half* d_b_padded = nullptr;
      CHECK_CUDA(cudaMalloc(&d_a_padded,
                            static_cast<size_t>(m) * kPaddedK *
                                sizeof(half)));
      CHECK_CUDA(cudaMalloc(&d_b_padded,
                            static_cast<size_t>(kPaddedK) * kPaddedN *
                                sizeof(half)));
      gemm_sm110::backends::shapeopt_detail::launch_pad_k_major_rows(
          d_a_half, d_a_padded, m, k, kPaddedK);
      gemm_sm110::backends::shapeopt_detail::launch_pad_2d_rows_cols(
          d_b_half, d_b_padded, k, n, kPaddedK, kPaddedN);
      CHECK_CUDA(cudaDeviceSynchronize());
      auto launch_shapeopt = [&]() {
        gemm_sm110::backends::shapeopt_detail::launch_ragged_padded_wmma(
            d_a_padded, d_b_padded, d_c, m, n, kPaddedN, kPaddedK);
      };
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "shapeopt", "ShapeOpt custom ragged padded WMMA GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
          d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
          2e-3f);
      CHECK_CUDA(cudaFree(d_a_padded));
      CHECK_CUDA(cudaFree(d_b_padded));
    } else if (gemm_sm110::backends::ShapeOptSpecializedRunner::supports(
            m, n, k, epilogue_mode, d_bias, d_residual)) {
      gemm_sm110::backends::ShapeOptSpecializedRunner shapeopt_runner(
          d_a_half, d_b_half_nk, d_c, d_b_half, m, n, k, epilogue_mode,
          d_bias, d_residual);
      auto launch_shapeopt = [&]() { shapeopt_runner.launch(); };
      std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "shapeopt", shapeopt_runner.label(), "fp16->fp32",
          kTensorCoreReferenceName, launch_shapeopt, m, n, k, d_c, c_bytes,
          h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    } else {
      const bool allow_cublaslt_fallback =
          std::getenv("SHAPEOPT_ALLOW_CUBLASLT_FALLBACK") != nullptr &&
          std::string(std::getenv("SHAPEOPT_ALLOW_CUBLASLT_FALLBACK")) == "1";
      if (!allow_cublaslt_fallback) {
        write_unavailable_backend(
            *gemm_sm110::find_backend("shapeopt"), n, csv,
            "no specialized ShapeOpt kernel for this shape/epilogue yet");
      } else {
      std::unique_ptr<gemm_sm110::references::CublasLtMatmulReference>
          owned_shapeopt_reference;
      auto* shapeopt_reference = cublaslt_reference.get();
      if (shapeopt_reference == nullptr) {
        owned_shapeopt_reference =
            std::make_unique<gemm_sm110::references::CublasLtMatmulReference>(
                d_a_half, d_b_half, d_c, m, n, k, epilogue_mode, d_bias,
                d_residual);
        shapeopt_reference = owned_shapeopt_reference.get();
      }
      auto launch_shapeopt = [&]() { shapeopt_reference->launch(); };
      if (shapeopt_reference == cublaslt_reference.get()) {
        std::cout << "ShapeOpt cuBLASLt heuristic fallback router: "
                  << cublas_avg_ms << " ms, " << cublas_tc_perf
                  << " GFLOPS, ratio=1x, matched=1"
                  << " (same selected cuBLASLt heuristic as reference)\n";
        csv << "shapeopt,ShapeOpt cuBLASLt heuristic fallback router," << n
            << ",fp16->fp32," << kTensorCoreReferenceName << ","
            << cublas_avg_ms << "," << cublas_tc_perf << ",1,1\n";
      } else {
        std::vector<float> h_shapeopt(static_cast<size_t>(m) * n);
        benchmark_kernel(
            "shapeopt", "ShapeOpt cuBLASLt heuristic fallback router",
            "fp16->fp32", kTensorCoreReferenceName, launch_shapeopt, m, n, k,
            d_c, c_bytes, h_ref_tc, h_shapeopt, csv, cublas_tc_perf, 2e-2f,
            2e-3f);
      }
      }
    }
  }

  if (wants_backend(backend_filter, "cutlass")) {
#if GEMM_SM110_ENABLE_CUTLASS
    if (!device_supports_tc3_sm110) {
      std::cout << "CUTLASS official Blackwell auto-schedule GEMM: skipped "
                   "because runtime GPU is not an sm110 family target\n";
      csv << "cutlass,CUTLASS official Blackwell auto-schedule GEMM skipped,"
          << n << ",fp16->fp32," << kTensorCoreReferenceName
          << ",0,0,0,0\n";
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
          "fp16->fp32", kTensorCoreReferenceName, launch_cutlass, m, n, k, d_c,
          c_bytes, h_ref_tc, h_cutlass, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
#else
    write_unavailable_backend(*gemm_sm110::find_backend("cutlass"), n, csv,
                              "disabled at build time");
#endif
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
          kTensorCoreReferenceName, launch_tc0, m, n, k, d_c, c_bytes, h_ref_tc,
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
          kTensorCoreReferenceName, launch, m, n, k, d_c, c_bytes, h_ref_tc,
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
          kTensorCoreReferenceName, launch, m, n, k, d_c, c_bytes, h_ref_tc,
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
          "fp16->fp32", kTensorCoreReferenceName, launch_tc3, m, n, k, d_c,
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
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5a", "tc5a overlapped epilogue M128N256K64 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    }
  }

  if (wants_backend(backend_filter, "tc5b")) {
    if (!device_supports_tc3_sm110) {
      write_unavailable_backend(*gemm_sm110::find_backend("tc5b"), n, csv,
                                "requires an SM110-family target");
    } else if (m == 1024 && n == 1024 && k == 1024) {
      gemm_sm110::backends::Tc4bcRunner<true> runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5b",
          "tc5b hybrid 2-SM overlapped M256N256K128 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
    } else {
      gemm_sm110::backends::Tc5aRunner runner(
          d_a_half, d_b_half_nk, d_c, m, n, k);
      auto launch = [&]() { runner.launch(); };
      std::vector<float> output(static_cast<size_t>(m) * n);
      benchmark_kernel(
          "tc5b",
          "tc5b hybrid fallback tc5a M128N256K64 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
          c_bytes, h_ref_tc, output, csv, cublas_tc_perf, 2e-2f, 2e-3f);
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
          "tc5c", "tc5c static persistent 1-SM TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5d", "tc5d static persistent M128N128K128 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5e", "tc5e static persistent M128N256K64 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5f", "tc5f static persistent M128N128K64 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5g", "tc5g static persistent M128N256K128 stage1 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "tc5h", "tc5h static persistent M128N256K64 stage1 TCGen05 GEMM",
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
          "fp16->fp32", kTensorCoreReferenceName, launch, m, n, k, d_c,
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
  if (d_bias != nullptr) CHECK_CUDA(cudaFree(d_bias));
  if (d_residual != nullptr) CHECK_CUDA(cudaFree(d_residual));
  return 0;
}
