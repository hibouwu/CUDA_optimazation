#pragma once
// Host-only oracle. Sum the modular address progression instead of reproducing
// the device pipeline; all totals here use 64-bit arithmetic before uint32 wrap.
#include <cstdint>
#include <stdexcept>

namespace async_copy_reference {
using u64 = std::uint64_t;
constexpr u64 array_words = 8388608 / 4;
constexpr unsigned threads = 128;

inline u64 floor_sum(u64 n, u64 modulus, u64 stride, u64 first) {
  u64 total = 0;
  for (;;) {
    if (stride >= modulus) {
      total += n * (n - 1) / 2 * (stride / modulus);
      stride %= modulus;
    }
    if (first >= modulus) {
      total += n * (first / modulus);
      first %= modulus;
    }
    const u64 end = stride * n + first;
    if (end < modulus) return total;
    n = end / modulus;
    first = end % modulus;
    const u64 previous_modulus = modulus;
    modulus = stride;
    stride = previous_modulus;
  }
}

inline void validate(unsigned bytes, unsigned blocks, unsigned block,
                     unsigned thread, u64 steps) {
  if ((bytes != 4 && bytes != 8 && bytes != 16) || blocks == 0 ||
      blocks > 65536 || block >= blocks || thread >= threads ||
      steps == 0 || steps > 65536)
    throw std::invalid_argument("S12 reference coordinates outside bounded domain");
}

inline u64 word_index(unsigned bytes, unsigned blocks, unsigned block,
                      unsigned producer, u64 step, unsigned word) {
  const u64 words = bytes / 4;
  return ((u64(block) * threads + producer + step * blocks * threads) * words + word) % array_words;
}

inline std::uint32_t payload_word(unsigned bytes, unsigned blocks, unsigned block,
                                  unsigned producer, u64 step, unsigned word,
                                  unsigned seed) {
  return std::uint32_t(17 * word_index(bytes, blocks, block, producer, step, word) + seed);
}

inline std::uint32_t checksum(unsigned bytes, unsigned blocks, unsigned block,
                               unsigned consumer, u64 steps, unsigned seed) {
  validate(bytes, blocks, block, consumer, steps);
  const unsigned producer = (consumer + 1) % threads;
  const u64 words = bytes / 4, stride = u64(blocks) * threads * words;
  u64 result = 0;
  for (unsigned word = 0; word < words; ++word) {
    const u64 first = (u64(block) * threads + producer) * words + word;
    const u64 unwrapped = steps * first + stride * steps * (steps - 1) / 2;
    const u64 indices = unwrapped - array_words * floor_sum(steps, array_words, stride, first);
    result += 17 * indices + steps * seed;
  }
  return std::uint32_t(result);
}

inline u64 final_step(unsigned slot, unsigned stages, u64 steps) {
  if ((stages != 1 && stages != 2 && stages != 4) || slot >= stages || steps <= slot)
    throw std::invalid_argument("S12 final slot was not filled");
  return slot + ((steps - 1 - slot) / stages) * stages;
}
}
