#pragma once
// Host-only reference: affine composition, independent of the device MAD loop.
#include <cstdint>

namespace synchronization_reference {
inline std::uint32_t initial(unsigned seed, unsigned thread) {
  return std::uint32_t(seed) + std::uint32_t(17) * thread;
}
inline std::uint32_t advance(std::uint32_t value, std::uint64_t steps) {
  std::uint32_t multiplier = 1664525u, addend = 1013904223u;
  std::uint32_t total_multiplier = 1, total_addend = 0;
  while (steps) {
    if (steps & 1) {
      total_addend = multiplier * total_addend + addend;
      total_multiplier = multiplier * total_multiplier;
    }
    addend = multiplier * addend + addend;
    multiplier *= multiplier;
    steps >>= 1;
  }
  return total_multiplier * value + total_addend;
}
inline bool selected(int mode, unsigned thread, unsigned threads) {
  return mode == 0 ? thread % 32 >= 16 : thread / 32 == threads / 32 - 1;
}
inline std::uint32_t expected(int mode, bool skew, unsigned thread,
                              unsigned threads, unsigned seed, std::uint64_t phases) {
  auto steps = skew && selected(mode, thread, threads) ? phases * 256 : 0;
  return advance(initial(seed, thread), steps);
}
}
