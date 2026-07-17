#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                << " code=" << static_cast<int>(err__)                       \
                << " name=" << cudaGetErrorName(err__)                       \
                << " msg=" << cudaGetErrorString(err__) << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                << " status=" << static_cast<int>(status__) << std::endl;    \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

namespace {

constexpr int kWarmup = 3;
constexpr int kRepeat = 20;

float input_a(size_t i) {
  return static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
}

float input_b(size_t i) {
  return static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
}

std::vector<std::uint8_t> pack_fp4_e2m1(size_t elements, bool is_a) {
  std::vector<std::uint8_t> packed((elements + 1) / 2, 0);
  for (size_t i = 0; i < elements; i += 2) {
    const float lo_value = is_a ? input_a(i) : input_b(i);
    const float hi_value = i + 1 < elements
                               ? (is_a ? input_a(i + 1) : input_b(i + 1))
                               : 0.0f;
    __nv_fp4_e2m1 lo(lo_value);
    __nv_fp4_e2m1 hi(hi_value);
    packed[i / 2] = static_cast<std::uint8_t>((lo.__x & 0x0f) |
                                              ((hi.__x & 0x0f) << 4));
  }
  return packed;
}

float decode_fp4_e2m1(const std::vector<std::uint8_t>& packed, size_t index) {
  const std::uint8_t byte = packed[index / 2];
  const std::uint8_t nibble =
      (index & 1u) == 0u ? static_cast<std::uint8_t>(byte & 0x0f)
                         : static_cast<std::uint8_t>(byte >> 4);
  __nv_fp4_e2m1 value;
  value.__x = nibble;
  return static_cast<float>(value);
}

float gflops(int n, float ms) {
  return 2.0f * static_cast<float>(n) * n * n / (ms * 1.0e6f);
}

cublasLtMatrixLayout_t make_layout(cudaDataType_t type, uint64_t rows,
                                   uint64_t cols, int64_t ld,
                                   cublasLtOrder_t order) {
  cublasLtMatrixLayout_t layout = nullptr;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  return layout;
}

bool compare_samples(int n, const std::vector<std::uint8_t>& a,
                     const std::vector<std::uint8_t>& b,
                     const std::vector<half>& got) {
  int errors = 0;
  for (int s = 0; s < 64; ++s) {
    const int row = (s * 131) % n;
    const int col = (s * 197) % n;
    float ref = 0.0f;
    for (int kk = 0; kk < n; ++kk) {
      ref += decode_fp4_e2m1(a, static_cast<size_t>(row) * n + kk) *
             decode_fp4_e2m1(b, static_cast<size_t>(kk) * n + col);
    }
    const float got_value =
        __half2float(got[static_cast<size_t>(row) * n + col]);
    const float diff = std::fabs(ref - got_value);
    const float tol = 1.0f + 0.10f * std::fabs(ref);
    if (diff > tol && ++errors <= 5) {
      std::cerr << "FP4 sample mismatch at (" << row << ", " << col
                << "): ref=" << ref << ", got=" << got_value
                << ", diff=" << diff << ", tol=" << tol << '\n';
    }
  }
  return errors == 0;
}

bool compare_samples(int n, const std::vector<std::uint8_t>& a,
                     const std::vector<std::uint8_t>& b,
                     const std::vector<float>& got) {
  int errors = 0;
  for (int s = 0; s < 64; ++s) {
    const int row = (s * 131) % n;
    const int col = (s * 197) % n;
    float ref = 0.0f;
    for (int kk = 0; kk < n; ++kk) {
      ref += decode_fp4_e2m1(a, static_cast<size_t>(row) * n + kk) *
             decode_fp4_e2m1(b, static_cast<size_t>(kk) * n + col);
    }
    const float got_value = got[static_cast<size_t>(row) * n + col];
    const float diff = std::fabs(ref - got_value);
    const float tol = 1.0f + 0.10f * std::fabs(ref);
    if (diff > tol && ++errors <= 5) {
      std::cerr << "FP4 sample mismatch at (" << row << ", " << col
                << "): ref=" << ref << ", got=" << got_value
                << ", diff=" << diff << ", tol=" << tol << '\n';
    }
  }
  return errors == 0;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 1024;
  if (argc >= 2) {
    n = std::atoi(argv[1]);
  }
  if (n <= 0 || (n % 2) != 0) {
    std::cerr << "Usage: " << argv[0] << " [positive_even_N]\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  std::cout << "N=" << n
            << ", type=CUDA_R_4F_E2M1, descriptor sweep over scale modes,"
               " output types, orders, and compute types\n";

  const size_t elements = static_cast<size_t>(n) * n;
  const std::vector<std::uint8_t> h_a = pack_fp4_e2m1(elements, true);
  const std::vector<std::uint8_t> h_b = pack_fp4_e2m1(elements, false);
  std::vector<half> h_c16(elements);
  std::vector<float> h_c32(elements);
  __nv_fp8_e8m0 scale_one_e8m0(1.0f);
  __nv_fp8_e4m3 scale_one_e4m3(1.0f);
  std::vector<std::uint8_t> h_scales32((elements + 31) / 32,
                                       scale_one_e8m0.__x);
  std::vector<std::uint8_t> h_scales16((elements + 15) / 16,
                                       scale_one_e4m3.__x);

  std::uint8_t* d_a = nullptr;
  std::uint8_t* d_b = nullptr;
  half* d_c16 = nullptr;
  float* d_c32 = nullptr;
  std::uint8_t* d_a_scales32 = nullptr;
  std::uint8_t* d_b_scales32 = nullptr;
  std::uint8_t* d_a_scales16 = nullptr;
  std::uint8_t* d_b_scales16 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, h_a.size()));
  CHECK_CUDA(cudaMalloc(&d_b, h_b.size()));
  CHECK_CUDA(cudaMalloc(&d_c16, h_c16.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_c32, h_c32.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_a_scales32, h_scales32.size()));
  CHECK_CUDA(cudaMalloc(&d_b_scales32, h_scales32.size()));
  CHECK_CUDA(cudaMalloc(&d_a_scales16, h_scales16.size()));
  CHECK_CUDA(cudaMalloc(&d_b_scales16, h_scales16.size()));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), h_b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_a_scales32, h_scales32.data(), h_scales32.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_scales32, h_scales32.data(), h_scales32.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_a_scales16, h_scales16.data(), h_scales16.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_scales16, h_scales16.data(), h_scales16.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c16, 0, h_c16.size() * sizeof(half)));
  CHECK_CUDA(cudaMemset(d_c32, 0, h_c32.size() * sizeof(float)));

  constexpr uint64_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;

  cublasLtHandle_t lt = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&lt));

  struct ProbeConfig {
    const char* name;
    cublasLtOrder_t order;
    bool swap_operands;
    cublasComputeType_t compute_type;
    cublasLtMatmulMatrixScale_t scale_mode;
    bool use_vec16_scales;
    cudaDataType_t output_type;
  };
  const ProbeConfig configs[] = {
      {"row_major_nn_fast16", CUBLASLT_ORDER_ROW, false,
       CUBLAS_COMPUTE_32F_FAST_16F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"row_major_nn_32f", CUBLASLT_ORDER_ROW, false, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"col_major_swap_fast16", CUBLASLT_ORDER_COL, true,
       CUBLAS_COMPUTE_32F_FAST_16F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"col_major_swap_32f", CUBLASLT_ORDER_COL, true, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"col_major_nn_fast16", CUBLASLT_ORDER_COL, false,
       CUBLAS_COMPUTE_32F_FAST_16F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"col_major_nn_32f", CUBLASLT_ORDER_COL, false, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_16F},
      {"col_swap_vec16_16f", CUBLASLT_ORDER_COL, true,
       CUBLAS_COMPUTE_32F_FAST_16F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, true, CUDA_R_16F},
      {"col_swap_vec16_32f", CUBLASLT_ORDER_COL, true, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, true, CUDA_R_32F},
      {"row_vec16_16f", CUBLASLT_ORDER_ROW, false,
       CUBLAS_COMPUTE_32F_FAST_16F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, true, CUDA_R_16F},
      {"row_vec16_32f", CUBLASLT_ORDER_ROW, false, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, true, CUDA_R_32F},
      {"col_swap_vec32_out32", CUBLASLT_ORDER_COL, true, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_32F},
      {"row_vec32_out32", CUBLASLT_ORDER_ROW, false, CUBLAS_COMPUTE_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, false, CUDA_R_32F},
  };

  bool any_heuristic = false;
  bool matched = false;
  for (const ProbeConfig& config : configs) {
    void* d_c = config.output_type == CUDA_R_16F
                    ? static_cast<void*>(d_c16)
                    : static_cast<void*>(d_c32);
    const size_t output_bytes = config.output_type == CUDA_R_16F
                                    ? h_c16.size() * sizeof(half)
                                    : h_c32.size() * sizeof(float);
    CHECK_CUDA(cudaMemset(d_c, 0, output_bytes));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void* workspace = nullptr;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, config.compute_type,
                                          CUDA_R_32F));
    const cublasLtMatmulMatrixScale_t scale_mode = config.scale_mode;
    std::uint8_t* d_a_scales =
        config.use_vec16_scales ? d_a_scales16 : d_a_scales32;
    std::uint8_t* d_b_scales =
        config.use_vec16_scales ? d_b_scales16 : d_b_scales32;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        config.swap_operands ? &d_b_scales : &d_a_scales,
        sizeof(d_a_scales)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        config.swap_operands ? &d_a_scales : &d_b_scales,
        sizeof(d_b_scales)));

    a_desc = make_layout(CUDA_R_4F_E2M1, n, n, n, config.order);
    b_desc = make_layout(CUDA_R_4F_E2M1, n, n, n, config.order);
    c_desc = make_layout(config.output_type, n, n, n, config.order);
    d_desc = make_layout(config.output_type, n, n, n, config.order);

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
        sizeof(kWorkspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        lt, op_desc, a_desc, b_desc, c_desc, d_desc, pref, 1, &heuristic,
        &returned);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned == 0 ||
        heuristic.state != CUBLAS_STATUS_SUCCESS) {
      std::cout << "probe " << config.name << ": heuristic failed status="
                << static_cast<int>(heuristic_status)
                << ", returned=" << returned
                << ", state=" << static_cast<int>(heuristic.state) << '\n';
      CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
      CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
      continue;
    }
    any_heuristic = true;

    CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* matmul_a = config.swap_operands ? d_b : d_a;
    const void* matmul_b = config.swap_operands ? d_a : d_b;
    auto launch = [&]() {
      CHECK_CUBLAS(cublasLtMatmul(
          lt, op_desc, &alpha, matmul_a, a_desc, matmul_b, b_desc, &beta, d_c,
          c_desc, d_c, d_desc, &heuristic.algo, workspace, kWorkspaceBytes,
          0));
    };

    for (int i = 0; i < kWarmup; ++i) launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kRepeat; ++i) launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const float avg_ms = total_ms / kRepeat;
    if (config.output_type == CUDA_R_16F) {
      CHECK_CUDA(cudaMemcpy(h_c16.data(), d_c, output_bytes,
                            cudaMemcpyDeviceToHost));
      matched = compare_samples(n, h_a, h_b, h_c16);
    } else {
      CHECK_CUDA(cudaMemcpy(h_c32.data(), d_c, output_bytes,
                            cudaMemcpyDeviceToHost));
      matched = compare_samples(n, h_a, h_b, h_c32);
    }
    std::cout << "probe " << config.name
              << ": cuBLASLt FP4 E2M1 scale_mode="
              << static_cast<int>(config.scale_mode)
              << ", output_type=" << static_cast<int>(config.output_type)
              << ' ' << avg_ms << " ms, "
              << gflops(n, avg_ms) << " GFLOP/s, matched=" << matched
              << '\n';

    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
    if (matched) {
      break;
    }
  }

  if (!any_heuristic) {
    std::cerr << "cuBLASLt FP4 heuristic failed for every probed descriptor\n";
  } else if (!matched) {
    std::cerr << "cuBLASLt FP4 had a runnable descriptor, but sample "
                 "correctness did not match row-major A*B semantics\n";
  }

  CHECK_CUBLAS(cublasLtDestroy(lt));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c16));
  CHECK_CUDA(cudaFree(d_c32));
  CHECK_CUDA(cudaFree(d_a_scales32));
  CHECK_CUDA(cudaFree(d_b_scales32));
  CHECK_CUDA(cudaFree(d_a_scales16));
  CHECK_CUDA(cudaFree(d_b_scales16));
  return matched ? EXIT_SUCCESS : EXIT_FAILURE;
}
