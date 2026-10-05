#pragma once
// Host-only S14 reference: logical payload offsets exclude the leading guard.
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tma_bulk_reference {
using u64 = std::uint64_t;
using u32 = std::uint32_t;
constexpr u64 slots = 32;

inline void payload_domain(u64 bytes) {
  if (bytes != 1024 && bytes != 4096 && bytes != 8192 && bytes != 16384 &&
      bytes != 32768 && bytes != 65536)
    throw std::invalid_argument("S14 payload outside finite matrix");
}
inline u64 multiply(u64 left, u64 right) {
  if (right && left > std::numeric_limits<u64>::max() / right)
    throw std::overflow_error("S14 uint64 product");
  return left * right;
}
inline u64 allocation_bytes(u64 blocks, u64 payload) {
  payload_domain(payload);
  if (!blocks) throw std::invalid_argument("S14 zero CTA count");
  const u64 data = multiply(multiply(blocks, slots), payload);
  if (data > std::numeric_limits<u64>::max() - 32)
    throw std::overflow_error("S14 allocation plus two16-byte guards");
  return data + 32;
}
inline u64 slot_offset(u64 block, u64 iteration, u64 blocks, u64 payload) {
  allocation_bytes(blocks, payload);
  if (block >= blocks) throw std::invalid_argument("S14 CTA coordinate");
  return (block * slots + iteration % slots) * payload;
}
inline u64 completed_payload(u64 blocks, u64 iterations, u64 payload) {
  allocation_bytes(blocks, payload);
  if (!iterations || iterations > 65536)
    throw std::invalid_argument("S14 iteration domain");
  return multiply(multiply(blocks, iterations), payload);
}
inline u32 global_input_word(u64 block, u64 iteration, u64 word, u64 blocks,
                             u64 payload, u32 seed) {
  const u64 offset = slot_offset(block, iteration, blocks, payload);
  if (word >= payload/4) throw std::invalid_argument("S14 tile word coordinate");
  // Modular word arithmetic is deliberately separate from64-bit addresses.
  return u32(17) * u32(offset/4 + word) + seed;
}
inline u32 shared_input_word(u64 block, u64 word, u64 blocks, u64 payload, u32 seed) {
  allocation_bytes(blocks, payload);
  if (block >= blocks || word >= payload/4)
    throw std::invalid_argument("S14 shared word coordinate");
  return u32(29) * u32(block * (payload/4) + word) + seed;
}
inline bool destination_slot_written(u64 slot, u64 iterations) {
  if (slot >= slots || !iterations || iterations > 65536)
    throw std::invalid_argument("S14 destination coverage coordinate");
  return iterations >= slots || slot < iterations;
}
inline u32 output_poison(u32 expected) { return ~expected; }
inline u32 destination_word(u64 block, u64 slot, u64 word, u64 blocks,
                            u64 payload, u64 iterations, u32 seed) {
  const u32 value = shared_input_word(block, word, blocks, payload, seed);
  return destination_slot_written(slot, iterations) ? value : output_poison(value);
}
} // namespace tma_bulk_reference
