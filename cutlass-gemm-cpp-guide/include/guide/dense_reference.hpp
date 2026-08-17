// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace guide {

inline float deterministic_value(std::size_t row, std::size_t col, std::uint64_t seed) {
  std::uint64_t x = seed ^ (row * 0x9e3779b97f4a7c15ULL) ^ (col * 0xbf58476d1ce4e5b9ULL);
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  int bucket = static_cast<int>((x ^ (x >> 31)) % 17ULL) - 8;
  return static_cast<float>(bucket) * 0.125f;
}

inline std::vector<float> dense_reference_mnk(std::vector<float> const& a_mk,
                                              std::vector<float> const& b_nk,
                                              int m, int n, int k,
                                              float alpha = 1.0f,
                                              float beta = 0.0f,
                                              std::vector<float> const* c_mn = nullptr) {
  std::vector<float> d(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        sum += static_cast<double>(a_mk[static_cast<std::size_t>(row) * k + kk]) *
               static_cast<double>(b_nk[static_cast<std::size_t>(col) * k + kk]);
      }
      double c = c_mn ? (*c_mn)[static_cast<std::size_t>(row) * n + col] : 0.0;
      d[static_cast<std::size_t>(row) * n + col] = static_cast<float>(alpha * sum + beta * c);
    }
  }
  return d;
}

struct ErrorMetrics {
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t max_abs_index = 0;
  bool finite = true;
};

template <class Observed>
inline ErrorMetrics compare_full(std::vector<float> const& expected,
                                 std::vector<Observed> const& observed) {
  ErrorMetrics metrics;
  if (expected.size() != observed.size()) {
    metrics.max_abs = std::numeric_limits<double>::infinity();
    metrics.max_rel = std::numeric_limits<double>::infinity();
    metrics.finite = false;
    return metrics;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    double actual = static_cast<double>(observed[i]);
    double wanted = static_cast<double>(expected[i]);
    if (!std::isfinite(actual)) metrics.finite = false;
    double abs_error = std::abs(actual - wanted);
    double rel_error = abs_error / std::max(std::abs(wanted), 1.0e-12);
    if (abs_error > metrics.max_abs) {
      metrics.max_abs = abs_error;
      metrics.max_abs_index = i;
    }
    metrics.max_rel = std::max(metrics.max_rel, rel_error);
  }
  return metrics;
}

}  // namespace guide
