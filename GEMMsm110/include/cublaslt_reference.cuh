#pragma once

#include "gemm_common.cuh"

#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace gemm_sm110::references {

enum class EpilogueMode {
  kNone,
  kBias,
  kRelu,
  kGelu,
  kResidual,
};

inline const char* to_string(EpilogueMode mode) {
  switch (mode) {
    case EpilogueMode::kNone:
      return "none";
    case EpilogueMode::kBias:
      return "bias";
    case EpilogueMode::kRelu:
      return "relu";
    case EpilogueMode::kGelu:
      return "gelu";
    case EpilogueMode::kResidual:
      return "residual";
  }
  return "unknown";
}

inline bool parse_epilogue_mode(std::string_view value, EpilogueMode* mode) {
  if (value == "none" || value == "default") {
    *mode = EpilogueMode::kNone;
  } else if (value == "bias") {
    *mode = EpilogueMode::kBias;
  } else if (value == "relu") {
    *mode = EpilogueMode::kRelu;
  } else if (value == "gelu") {
    *mode = EpilogueMode::kGelu;
  } else if (value == "residual") {
    *mode = EpilogueMode::kResidual;
  } else {
    return false;
  }
  return true;
}

inline size_t cublaslt_workspace_limit_bytes() {
  constexpr size_t kDefaultWorkspaceBytes = 64ull * 1024ull * 1024ull;
  const char* env = std::getenv("CUBLASLT_WORKSPACE_BYTES");
  if (env == nullptr || env[0] == '\0') {
    return kDefaultWorkspaceBytes;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0ull) {
    return kDefaultWorkspaceBytes;
  }
  return static_cast<size_t>(parsed);
}

class CublasLtMatmulReference {
 public:
  CublasLtMatmulReference(const half* a_row_major, const half* b_row_major,
                          float* d_row_major, int m, int n, int k,
                          EpilogueMode epilogue_mode = EpilogueMode::kNone,
                          const float* bias = nullptr,
                          const float* residual = nullptr)
      : a_(a_row_major),
        b_(b_row_major),
        d_(d_row_major),
        bias_(bias),
        residual_(residual),
        m_(m),
        n_(n),
        k_(k),
        epilogue_mode_(epilogue_mode) {
    CHECK_CUBLAS(cublasLtCreate(&handle_));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(
        &matmul_desc_, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));

    const cublasOperation_t op = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));

    configure_epilogue();

    // cuBLASLt is column-major by default.  Reuse the old cuBLAS trick:
    // row-major C[M,N] is viewed as column-major C'[N,M] = B'[N,K] * A'[K,M].
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lt_a_layout_, CUDA_R_16F, n_, k_, n_));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lt_b_layout_, CUDA_R_16F, k_, m_, k_));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lt_c_layout_, CUDA_R_32F, n_, m_, n_));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lt_d_layout_, CUDA_R_32F, n_, m_, n_));

    workspace_bytes_ = cublaslt_workspace_limit_bytes();
    if (workspace_bytes_ > 0) {
      CHECK_CUDA(cudaMalloc(&workspace_, workspace_bytes_));
    }

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference_));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes_, sizeof(workspace_bytes_)));

    choose_algorithm();
  }

  ~CublasLtMatmulReference() {
    if (workspace_ != nullptr) cudaFree(workspace_);
    if (preference_ != nullptr) cublasLtMatmulPreferenceDestroy(preference_);
    if (lt_d_layout_ != nullptr) cublasLtMatrixLayoutDestroy(lt_d_layout_);
    if (lt_c_layout_ != nullptr) cublasLtMatrixLayoutDestroy(lt_c_layout_);
    if (lt_b_layout_ != nullptr) cublasLtMatrixLayoutDestroy(lt_b_layout_);
    if (lt_a_layout_ != nullptr) cublasLtMatrixLayoutDestroy(lt_a_layout_);
    if (matmul_desc_ != nullptr) cublasLtMatmulDescDestroy(matmul_desc_);
    if (handle_ != nullptr) cublasLtDestroy(handle_);
  }

  CublasLtMatmulReference(const CublasLtMatmulReference&) = delete;
  CublasLtMatmulReference& operator=(const CublasLtMatmulReference&) = delete;

  void launch(cudaStream_t stream = 0) {
    const float* c_input = residual_ != nullptr ? residual_ : d_;
    CHECK_CUBLAS(cublasLtMatmul(
        handle_, matmul_desc_, &alpha_,
        b_, lt_a_layout_,
        a_, lt_b_layout_,
        &beta_,
        c_input, lt_c_layout_,
        d_, lt_d_layout_,
        &algo_,
        workspace_, workspace_bytes_, stream));
  }

  size_t workspace_bytes() const { return workspace_bytes_; }
  size_t selected_workspace_bytes() const { return selected_workspace_bytes_; }
  int returned_algorithms() const { return returned_algorithms_; }

 private:
  void configure_epilogue() {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    switch (epilogue_mode_) {
      case EpilogueMode::kNone:
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        break;
      case EpilogueMode::kBias:
        epilogue = CUBLASLT_EPILOGUE_BIAS;
        break;
      case EpilogueMode::kRelu:
        epilogue = CUBLASLT_EPILOGUE_RELU;
        break;
      case EpilogueMode::kGelu:
        epilogue = CUBLASLT_EPILOGUE_GELU;
        break;
      case EpilogueMode::kResidual:
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        beta_ = 1.0f;
        break;
    }

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue)));

    if (epilogue_mode_ == EpilogueMode::kBias) {
      if (bias_ == nullptr) {
        std::fprintf(stderr, "bias epilogue requires a bias pointer\n");
        std::abort();
      }
      const void* bias_ptr = bias_;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
          sizeof(bias_ptr)));
      const cudaDataType_t bias_type = CUDA_R_32F;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type)));
    }

    if (epilogue_mode_ == EpilogueMode::kResidual && residual_ == nullptr) {
      std::fprintf(stderr, "residual epilogue requires a residual pointer\n");
      std::abort();
    }
  }

  void choose_algorithm() {
    std::array<cublasLtMatmulHeuristicResult_t, 16> results{};
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle_, matmul_desc_, lt_a_layout_, lt_b_layout_,
        lt_c_layout_, lt_d_layout_, preference_,
        static_cast<int>(results.size()), results.data(),
        &returned_algorithms_));

    for (int i = 0; i < returned_algorithms_; ++i) {
      if (results[i].state == CUBLAS_STATUS_SUCCESS &&
          results[i].workspaceSize <= workspace_bytes_) {
        algo_ = results[i].algo;
        selected_workspace_bytes_ = results[i].workspaceSize;
        algorithm_ready_ = true;
        dump_algorithm_config_if_requested(i);
        return;
      }
    }

    std::fprintf(stderr,
                 "cuBLASLt heuristic returned no usable algorithm for "
                 "M=%d N=%d K=%d with workspace=%zu bytes "
                 "(returned=%d)\n",
                 m_, n_, k_, workspace_bytes_, returned_algorithms_);
    std::abort();
  }

  void dump_algorithm_config_if_requested(int heuristic_index) {
    const char* dump = std::getenv("CUBLASLT_DUMP_ALGO");
    if (dump == nullptr || dump[0] == '\0' || dump[0] == '0') return;

    auto get_i32 = [&](cublasLtMatmulAlgoConfigAttributes_t attr) {
      int32_t value = 0;
      size_t written = 0;
      CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
          &algo_, attr, &value, sizeof(value), &written));
      return value;
    };
    auto get_u32 = [&](cublasLtMatmulAlgoConfigAttributes_t attr) {
      uint32_t value = 0;
      size_t written = 0;
      CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
          &algo_, attr, &value, sizeof(value), &written));
      return value;
    };
    auto get_u16 = [&](cublasLtMatmulAlgoConfigAttributes_t attr) {
      uint16_t value = 0;
      size_t written = 0;
      CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
          &algo_, attr, &value, sizeof(value), &written));
      return value;
    };

    std::printf(
        "cuBLASLt selected algo: heuristic_index=%d id=%d tile=%u "
        "splitK=%d reduction=%u swizzle=%u custom=%u stages=%u "
        "inner=%u cluster=%u\n",
        heuristic_index, get_i32(CUBLASLT_ALGO_CONFIG_ID),
        get_u32(CUBLASLT_ALGO_CONFIG_TILE_ID),
        get_i32(CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
        get_u32(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
        get_u32(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
        get_u32(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
        get_u32(CUBLASLT_ALGO_CONFIG_STAGES_ID),
        static_cast<unsigned>(get_u16(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)),
        static_cast<unsigned>(
            get_u16(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)));
  }

  const half* a_ = nullptr;
  const half* b_ = nullptr;
  float* d_ = nullptr;
  const float* bias_ = nullptr;
  const float* residual_ = nullptr;
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  EpilogueMode epilogue_mode_ = EpilogueMode::kNone;
  const float alpha_ = 1.0f;
  float beta_ = 0.0f;

  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t matmul_desc_ = nullptr;
  cublasLtMatrixLayout_t lt_a_layout_ = nullptr;
  cublasLtMatrixLayout_t lt_b_layout_ = nullptr;
  cublasLtMatrixLayout_t lt_c_layout_ = nullptr;
  cublasLtMatrixLayout_t lt_d_layout_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulAlgo_t algo_{};
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
  size_t selected_workspace_bytes_ = 0;
  int returned_algorithms_ = 0;
  bool algorithm_ready_ = false;
};

}  // namespace gemm_sm110::references
