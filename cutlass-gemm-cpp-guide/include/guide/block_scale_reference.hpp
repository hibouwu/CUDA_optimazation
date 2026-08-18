// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace guide {

inline std::vector<float> block_scaled_reference_mnk(
    std::vector<float> const& a_mk, std::vector<float> const& b_nk,
    std::vector<float> const& sfa, std::vector<float> const& sfb,
    int m, int n, int k, int scale_vector_size) {
  int sfk = (k + scale_vector_size - 1) / scale_vector_size;
  if (a_mk.size() != static_cast<std::size_t>(m) * k ||
      b_nk.size() != static_cast<std::size_t>(n) * k ||
      sfa.size() != static_cast<std::size_t>(m) * sfk ||
      sfb.size() != static_cast<std::size_t>(n) * sfk) {
    throw std::invalid_argument("block-scaled logical tensor sizes do not match M/N/K/SV");
  }
  std::vector<float> d(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        int sf = kk / scale_vector_size;
        double av = static_cast<double>(a_mk[static_cast<std::size_t>(row) * k + kk]);
        double bv = static_cast<double>(b_nk[static_cast<std::size_t>(col) * k + kk]);
        double as = static_cast<double>(sfa[static_cast<std::size_t>(row) * sfk + sf]);
        double bs = static_cast<double>(sfb[static_cast<std::size_t>(col) * sfk + sf]);
        sum += (av * as) * (bv * bs);
      }
      d[static_cast<std::size_t>(row) * n + col] = static_cast<float>(sum);
    }
  }
  return d;
}

}  // namespace guide
