#pragma once
// Host-only arithmetic oracle for the S13 contract. No device pipeline is copied.
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace global_duplex_reference {
using u64=std::uint64_t;
using u32=std::uint32_t;

inline u64 aligned_array_bytes(u64 requested,u64 total_threads) {
  if(!requested||!total_threads||total_threads>std::numeric_limits<u64>::max()/16)
    throw std::invalid_argument("S13 allocation domain");
  const u64 alignment=total_threads*16;
  const u64 groups=requested/alignment+(requested%alignment!=0);
  if(groups>std::numeric_limits<u64>::max()/alignment)
    throw std::overflow_error("S13 aligned allocation overflow");
  return groups*alignment;
}

inline u64 groups_per_sweep(u64 array_bytes,u64 total_threads) {
  if(!array_bytes||array_bytes%16||!total_threads||array_bytes/16%total_threads)
    throw std::invalid_argument("S13 whole vector groups required");
  return array_bytes/16/total_threads;
}

inline u64 vector_index(u64 group,unsigned request,unsigned ratio,u64 thread,
                         u64 array_bytes,u64 total_threads) {
  const u64 groups=groups_per_sweep(array_bytes,total_threads);
  if(!ratio||ratio>4||request>=ratio||thread>=total_threads||group>=groups)
    throw std::invalid_argument("S13 request coordinate");
  // The largest intermediate is below 4*(array_bytes/16), hence fits uint64.
  return ((group*ratio+request)*total_threads+thread)%(array_bytes/16);
}

inline u32 word_value(u64 word_index,unsigned multiplier,u32 seed) {
  return u32(multiplier)*u32(word_index)+seed;
}

inline u32 read_checksum(u64 array_bytes,u64 total_threads,u64 thread,
                          unsigned reads,unsigned iterations,u32 seed) {
  const u64 groups=groups_per_sweep(array_bytes,total_threads);
  if(thread>=total_threads||reads>4||!iterations||iterations>16)
    throw std::invalid_argument("S13 checksum coordinate");
  // Sum the four lane indices across all vectors owned by this thread.
  // Reduce every polynomial in Z/(2^32); no large signed intermediate or loop.
  const u32 g=u32(groups),t=u32(thread),T=u32(total_threads);
  const u32 indices=u32(8)*T*g*(g-u32(1))+u32(16)*t*g+u32(6)*g;
  const u32 sweep=u32(17)*indices+u32(4)*g*seed;
  return sweep*u32(reads)*u32(iterations);
}

inline u64 requested_work(u64 array_bytes,unsigned reads,unsigned writes,unsigned iterations) {
  const bool allowed=(reads==1&&writes==0)||(reads==0&&writes==1)||
    (reads==1&&(writes==1||writes==2||writes==4))||
    (writes==1&&(reads==2||reads==4));
  if(!allowed||!iterations||iterations>16)
    throw std::invalid_argument("S13 finite ratio/iteration matrix");
  const u64 multiplier=u64(iterations)*(reads+writes);
  if(!array_bytes||array_bytes>std::numeric_limits<u64>::max()/multiplier)
    throw std::overflow_error("S13 uint64 requested work overflow");
  return array_bytes*multiplier;
}
}
