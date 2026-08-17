// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <stdexcept>

namespace guide {

// Independent closed form for CUTLASS Sm1xx K-major 128x4 scale chunks.
// The K element coordinate is converted to a logical scale coordinate by SV.
inline std::size_t sm1xx_scale_offset(int mn, int k_element, int mn_extent,
                                      int k_extent, int scale_vector_size) {
  if (mn < 0 || mn >= mn_extent || k_element < 0 || k_element >= k_extent) {
    throw std::out_of_range("logical scale coordinate is outside the tensor");
  }
  int sf = k_element / scale_vector_size;
  int block_mn = mn / 128;
  int block_sf = sf / 4;
  int blocks_mn = (mn_extent + 127) / 128;
  int within_mn = mn % 128;
  int lane = within_mn % 32;
  int warp = within_mn / 32;
  int within_sf = sf % 4;
  std::size_t block = static_cast<std::size_t>(block_sf) * blocks_mn + block_mn;
  return block * 512 + static_cast<std::size_t>(lane) * 16 + warp * 4 + within_sf;
}

inline std::size_t sm1xx_scale_storage_size(int mn_extent, int k_extent,
                                            int scale_vector_size) {
  int blocks_mn = (mn_extent + 127) / 128;
  int sf_extent = (k_extent + scale_vector_size - 1) / scale_vector_size;
  int blocks_sf = (sf_extent + 3) / 4;
  return static_cast<std::size_t>(blocks_mn) * blocks_sf * 512;
}

}  // namespace guide
