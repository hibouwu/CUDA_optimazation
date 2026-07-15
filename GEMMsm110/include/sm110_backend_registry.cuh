#pragma once

#include <array>
#include <string>
#include <string_view>

namespace gemm_sm110 {

struct BackendDescriptor {
  std::string_view id;
  std::string_view label;
  int stage;
  bool implemented;
};

inline constexpr std::array<BackendDescriptor, 26> kBackendDescriptors{{
    {"cublas_tc", "cuBLAS Tensor Core reference", -1, true},
    {"cutlass", "CUTLASS official Blackwell auto schedule", -1, true},
    {"tc0", "CUDA WMMA Tensor Core baseline", 0, true},
    {"tc1a", "2D TMA linear-SMEM TCGen05 minimal", 1, false},
    {"tc1b", "3D TMA linear-SMEM TCGen05 minimal", 1, false},
    {"tc2a", "2D TMA SW128 TCGen05", 2, true},
    {"tc2b", "3D TMA SW128 TCGen05", 2, true},
    {"tc3", "multi-stage 2D TMA SW128 TCGen05 pipeline", 3, true},
    {"tc4a", "warp-specialized TMA/TCGen05 pipeline", 4, true},
    {"tc4b", "2-SM cluster TMA/TCGen05 pipeline", 4, true},
    {"tc4c", "warp-specialized 2-SM cluster pipeline", 4, true},
    {"tc5a", "static persistent 1-SM TCGen05 scheduler", 5, true},
    {"tc5b", "software dynamic persistent 1-SM TCGen05 scheduler", 5,
     false},
    {"tc5c", "static persistent 1-SM TCGen05 M128N128K128 scheduler", 5,
     true},
    {"tc5d", "static persistent 1-SM TCGen05 M128N256K64 scheduler", 5,
     true},
    {"tc5e", "static persistent 1-SM TCGen05 M128N128K64 scheduler", 5,
     true},
    {"tc5f", "static persistent 1-SM TCGen05 M128N256K128 stage1 scheduler",
     5, true},
    {"tc5g", "static persistent 1-SM TCGen05 M128N256K64 stage1 scheduler",
     5, true},
    {"tc5h", "overlapped epilogue 1-SM TCGen05 M128N256K64 scheduler", 5,
     true},
    {"tc5i", "overlapped epilogue 1-SM TCGen05 M128N128K64 scheduler", 5,
     true},
    {"tc5j", "overlapped epilogue 1-SM TCGen05 M128N256K128 scheduler", 5,
     true},
    {"tc5k", "overlapped epilogue 1-SM TCGen05 M64N256K64 scheduler", 5,
     false},
    {"tc5l", "B-reuse 1-SM TCGen05 M256N256K64 scheduler", 5, false},
    {"tc5m", "overlapped B-reuse 1-SM TCGen05 M256N128K64 scheduler", 5,
     false},
    {"tc5n", "hybrid 2-SM overlap for 1024 plus tc5h fallback", 5, true},
    {"tc6", "fused NVFP4 TCGen05 epilogue", 6, true},
}};

inline bool is_backend_group(const std::string& filter) {
  return filter == "all" || filter == "references" ||
         filter == "stage0" || filter == "stage1" || filter == "stage2" ||
         filter == "stage3" || filter == "stage4" || filter == "stage5" ||
         filter == "stage6";
}

inline bool is_valid_backend_filter(const std::string& filter) {
  if (is_backend_group(filter)) {
    return true;
  }
  for (const auto& backend : kBackendDescriptors) {
    if (backend.id == filter) {
      return true;
    }
  }
  return false;
}

inline bool wants_backend(const std::string& filter,
                          std::string_view backend_id) {
  if (filter == "all") {
    return true;
  }
  if (filter == backend_id) {
    return true;
  }

  for (const auto& backend : kBackendDescriptors) {
    if (backend.id != backend_id) {
      continue;
    }
    if (filter == "references") {
      return backend.stage == -1;
    }
    if (filter.size() == 6 && filter.rfind("stage", 0) == 0) {
      return backend.stage == filter[5] - '0';
    }
  }
  return false;
}

inline const BackendDescriptor* find_backend(std::string_view id) {
  for (const auto& backend : kBackendDescriptors) {
    if (backend.id == id) {
      return &backend;
    }
  }
  return nullptr;
}

inline constexpr std::string_view kBackendUsage =
    "[all|references|stage0..stage6|cublas_tc|cutlass|"
    "tc0|tc1a|tc1b|tc2a|tc2b|tc3|tc4a|tc4b|tc4c|"
    "tc5a|tc5b|tc5c|tc5d|tc5e|tc5f|tc5g|tc5h|tc5i|tc5j|tc5k|tc5l|tc5m|tc5n|tc6]";

}  // namespace gemm_sm110
