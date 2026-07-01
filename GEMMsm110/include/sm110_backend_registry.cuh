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

inline constexpr std::array<BackendDescriptor, 13> kBackendDescriptors{{
    {"cublas_tc", "cuBLAS Tensor Core reference", -1, true},
    {"cutlass", "CUTLASS official Blackwell auto schedule", -1, true},
    {"tc0", "CUDA WMMA Tensor Core baseline", 0, true},
    {"tc1a", "2D TMA linear-SMEM TCGen05 minimal", 1, true},
    {"tc1b", "3D TMA linear-SMEM TCGen05 minimal", 1, true},
    {"tc2a", "2D TMA SW128 TCGen05", 2, true},
    {"tc2b", "3D TMA SW128 TCGen05", 2, true},
    {"tc3", "multi-stage 2D TMA SW128 TCGen05 pipeline", 3, true},
    {"tc4a", "warp-specialized TMA/TCGen05 pipeline", 4, true},
    {"tc4b", "2-SM cluster TMA/TCGen05 pipeline", 4, true},
    {"tc4c", "warp-specialized 2-SM cluster pipeline", 4, true},
    {"tc5a", "static persistent TCGen05 scheduler", 5, true},
    {"tc5b", "hardware CLC persistent TCGen05 scheduler", 5, true},
}};

inline bool is_backend_group(const std::string& filter) {
  return filter == "all" || filter == "references" ||
         filter == "stage0" || filter == "stage1" || filter == "stage2" ||
         filter == "stage3" || filter == "stage4" || filter == "stage5";
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
    "[all|references|stage0..stage5|cublas_tc|cutlass|"
    "tc0|tc1a|tc1b|tc2a|tc2b|tc3|tc4a|tc4b|tc4c|tc5a|tc5b]";

}  // namespace gemm_sm110
