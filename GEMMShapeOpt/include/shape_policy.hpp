#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace gemm_shape_opt {

enum class ShapeClass {
  kSquareLike,
  kRaggedTile,
  kSkinnyM,
  kSkinnyN,
  kGemvLikeM,
  kGemvLikeN,
  kSmall,
};

struct Shape {
  int m = 0;
  int n = 0;
  int k = 0;
};

struct ShapeFeatures {
  ShapeClass shape_class = ShapeClass::kSquareLike;
  bool tile_m_aligned = false;
  bool tile_n_aligned = false;
  bool tile_k_aligned = false;
  int m_tail = 0;
  int n_tail = 0;
  int k_tail = 0;
  int64_t flops = 0;
};

inline const char* to_string(ShapeClass shape_class) {
  switch (shape_class) {
    case ShapeClass::kSquareLike:
      return "square_like";
    case ShapeClass::kRaggedTile:
      return "ragged_tile";
    case ShapeClass::kSkinnyM:
      return "skinny_m";
    case ShapeClass::kSkinnyN:
      return "skinny_n";
    case ShapeClass::kGemvLikeM:
      return "gemv_like_m";
    case ShapeClass::kGemvLikeN:
      return "gemv_like_n";
    case ShapeClass::kSmall:
      return "small";
  }
  return "unknown";
}

inline bool is_aligned(int value, int alignment) {
  return alignment > 0 && value % alignment == 0;
}

inline ShapeFeatures classify_shape(Shape shape) {
  ShapeFeatures features{};
  features.tile_m_aligned = is_aligned(shape.m, 128);
  features.tile_n_aligned = is_aligned(shape.n, 256);
  features.tile_k_aligned = is_aligned(shape.k, 64);
  features.m_tail = shape.m % 128;
  features.n_tail = shape.n % 256;
  features.k_tail = shape.k % 64;
  features.flops = 2ll * shape.m * shape.n * shape.k;

  const int min_mn = std::min(shape.m, shape.n);
  const int max_mn = std::max(shape.m, shape.n);
  if (shape.m <= 1) {
    features.shape_class = ShapeClass::kGemvLikeM;
  } else if (shape.n <= 1) {
    features.shape_class = ShapeClass::kGemvLikeN;
  } else if (shape.m <= 128 && shape.n >= 1024) {
    features.shape_class = ShapeClass::kSkinnyM;
  } else if (shape.n <= 128 && shape.m >= 1024) {
    features.shape_class = ShapeClass::kSkinnyN;
  } else if (features.flops < (1ll << 30)) {
    features.shape_class = ShapeClass::kSmall;
  } else if (!features.tile_m_aligned || !features.tile_n_aligned ||
             !features.tile_k_aligned) {
    features.shape_class = ShapeClass::kRaggedTile;
  } else if (max_mn <= 2 * min_mn) {
    features.shape_class = ShapeClass::kSquareLike;
  } else {
    features.shape_class = ShapeClass::kRaggedTile;
  }
  return features;
}

inline std::string_view default_backend_suite(ShapeClass shape_class) {
  switch (shape_class) {
    case ShapeClass::kSquareLike:
      return "core";
    case ShapeClass::kRaggedTile:
      return "cublas_tc cutlass tc5a tc5b";
    case ShapeClass::kSkinnyM:
    case ShapeClass::kSkinnyN:
    case ShapeClass::kGemvLikeM:
    case ShapeClass::kGemvLikeN:
      return "cublas_tc cutlass";
    case ShapeClass::kSmall:
      return "cublas_tc cutlass tc2a tc2b tc5a tc5b";
  }
  return "cublas_tc";
}

}  // namespace gemm_shape_opt
