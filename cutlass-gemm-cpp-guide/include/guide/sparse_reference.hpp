// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace guide {

struct Sparse24Group {
  std::array<float, 2> values{};
  std::array<unsigned char, 2> positions{};
};

inline Sparse24Group compress_sparse24(std::array<float, 4> const& group) {
  Sparse24Group compressed;
  int count = 0;
  for (int i = 0; i < 4; ++i) {
    if (group[i] != 0.0f) {
      if (count == 2) throw std::invalid_argument("2:4 group contains more than two nonzeros");
      compressed.values[count] = group[i];
      compressed.positions[count] = static_cast<unsigned char>(i);
      ++count;
    }
  }
  if (count != 2) throw std::invalid_argument("2:4 group must contain exactly two nonzeros");
  return compressed;
}

inline std::array<float, 4> decompress_sparse24(Sparse24Group const& group) {
  if (group.positions[0] >= 4 || group.positions[1] >= 4 ||
      group.positions[0] == group.positions[1]) {
    throw std::invalid_argument("invalid 2:4 metadata");
  }
  std::array<float, 4> dense{};
  dense[group.positions[0]] = group.values[0];
  dense[group.positions[1]] = group.values[1];
  return dense;
}

}  // namespace guide
