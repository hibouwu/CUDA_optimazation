#include "r00_common.hpp"
#include <cublasLt.h>
#include <sstream>

inline void lt_check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": cublas status " + std::to_string(status));
}
#define LT_CHECK(expr) lt_check((expr), #expr)
struct LtGemm {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
  cublasLtMatmulPreference_t preference{};
  cublasLtMatmulAlgo_t algorithm{};
  DeviceBuffer<unsigned char> workspace{64 * 1024 * 1024};
  DeviceBuffer<float> scale{1};
  void* a;
  void* b;
  void* d;
  size_t selected_workspace = 0;
  LtGemm(const Options& o, void* input_a, void* input_b, void* output, cudaDataType_t input_type,
         cudaDataType_t output_type)
      : a(input_b), b(input_a), d(output) {
    LT_CHECK(cublasLtCreate(&handle));
    LT_CHECK(cublasLtMatmulDescCreate(
        &desc, o.dtype == "fp32" ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op_a = CUBLAS_OP_N, op_b = CUBLAS_OP_N;
    if (o.dtype == "fp8") {
      // FP8 TN: A row-major is already the storage of its column-major transpose;
      // B is explicitly generated in column-major and D is column-major.
      a = input_a;
      b = input_b;
      op_a = CUBLAS_OP_T;
      LT_CHECK(cublasLtMatrixLayoutCreate(&la, input_type, o.k, o.m, o.k));
      LT_CHECK(cublasLtMatrixLayoutCreate(&lb, input_type, o.k, o.n, o.k));
      LT_CHECK(cublasLtMatrixLayoutCreate(&lc, output_type, o.m, o.n, o.m));
      LT_CHECK(cublasLtMatrixLayoutCreate(&ld, output_type, o.m, o.n, o.m));
      float one = 1;
      CUDA_CHECK(cudaMemcpy(scale.pointer, &one, 4, cudaMemcpyHostToDevice));
      LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &scale.pointer, sizeof(scale.pointer)));
      LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &scale.pointer, sizeof(scale.pointer)));
      int8_t fast_accum = 0;
      LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum,
                                              sizeof(fast_accum)));
    } else {
      // Column-major D^T = B^T A^T, with original row-major allocations; no copy.
      LT_CHECK(cublasLtMatrixLayoutCreate(&la, input_type, o.n, o.k, o.n));
      LT_CHECK(cublasLtMatrixLayoutCreate(&lb, input_type, o.k, o.m, o.k));
      LT_CHECK(cublasLtMatrixLayoutCreate(&lc, output_type, o.n, o.m, o.n));
      LT_CHECK(cublasLtMatrixLayoutCreate(&ld, output_type, o.n, o.m, o.n));
    }
    LT_CHECK(
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    LT_CHECK(
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t limit = workspace.count;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int count = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, preference, 8, candidates,
                                            &count));
    bool found = false;
    for (int i = 0; i < count; ++i)
      if (candidates[i].state == CUBLAS_STATUS_SUCCESS) {
        algorithm = candidates[i].algo;
        selected_workspace = candidates[i].workspaceSize;
        found = true;
        break;
      }
    if (!found)
      throw std::runtime_error(
          "no legal cuBLASLt heuristic candidate for specified layout/precision");
  }
  ~LtGemm() {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
  }
  void launch() {
    float alpha = 1, beta = 0;
    LT_CHECK(cublasLtMatmul(handle, desc, &alpha, a, la, b, lb, &beta, d, lc, d, ld, &algorithm,
                            workspace.pointer, workspace.count, 0));
  }
  std::string metadata(const Options& o) {
    std::ostringstream s;
    s << "{\"library_version\":";
    s << cublasLtGetVersion() << ",\"workspace_limit_bytes\":" << workspace.count
      << ",\"selected_workspace_bytes\":" << selected_workspace
      << ",\"layout_adapted\":" << (o.dtype == "fp8" ? "true" : "false") << ",\"output_layout\":\""
      << (o.dtype == "fp8" ? "column_major" : "row_major")
      << "\",\"a_scale\":1,\"b_scale\":1,\"fast_accum\":false";
    const char* names[] = {"algorithm_id", "tile_id", "stages_id", "split_k", "cluster_shape_id"};
    cublasLtMatmulAlgoConfigAttributes_t attributes[] = {
        CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
    for (int i = 0; i < 5; ++i) {
      int value = 0;
      size_t written = 0;
      auto status = cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attributes[i], &value,
                                                         sizeof(value), &written);
      s << ",\"" << names[i] << "\":";
      if (status == CUBLAS_STATUS_SUCCESS)
        s << value;
      else
        s << "null";
    }
    s << '}';
    return s.str();
  }
};
template <class Input, class Output>
int run(const Options& o, cudaDataType_t input_type, cudaDataType_t output_type) {
  DeviceBuffer<Input> a(size_t(o.m) * o.k), b(size_t(o.k) * o.n);
  DeviceBuffer<Output> d(size_t(o.m) * o.n);
  LtGemm gemm(o, a.pointer, b.pointer, d.pointer, input_type, output_type);
  return measure_gemm(
      o, a.pointer, b.pointer, d.pointer, o.dtype == "fp8", o.dtype == "fp8",
      [&]() { gemm.launch(); }, gemm.metadata(o));
}
int main(int argc, char** argv) {
  try {
    Options o(argc, argv);
    if (o.backend != "cublaslt")
      throw std::runtime_error("this executable implements cublaslt only");
    if (o.dtype == "fp16")
      return run<__half, float>(o, CUDA_R_16F, CUDA_R_32F);
    if (o.dtype == "bf16")
      return run<__nv_bfloat16, float>(o, CUDA_R_16BF, CUDA_R_32F);
    if (o.dtype == "fp8")
      return run<__nv_fp8_e4m3, __nv_bfloat16>(o, CUDA_R_8F_E4M3, CUDA_R_16BF);
    if (o.dtype == "fp32")
      return run<float, float>(o, CUDA_R_32F, CUDA_R_32F);
    throw std::runtime_error("unknown dtype");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
