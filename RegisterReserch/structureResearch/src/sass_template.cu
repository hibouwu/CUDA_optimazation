#include <cstdint>

extern "C" __global__ __launch_bounds__(32, 1) void sass_register_probe(
    const std::uint32_t* sources, std::uint64_t* elapsed_cycles,
    std::uint32_t* sinks, int iterations) {
  const int lane = threadIdx.x;
  const std::uint32_t s0 = sources[0 * 32 + lane];
  const std::uint32_t s1 = sources[1 * 32 + lane];
  const std::uint32_t s2 = sources[2 * 32 + lane];
  const std::uint32_t s3 = sources[3 * 32 + lane];
  const std::uint32_t s4 = sources[4 * 32 + lane];
  const std::uint32_t s5 = sources[5 * 32 + lane];
  const std::uint32_t s6 = sources[6 * 32 + lane];
  const std::uint32_t s7 = sources[7 * 32 + lane];
  const std::uint32_t s8 = sources[8 * 32 + lane];
  const std::uint32_t s9 = sources[9 * 32 + lane];
  const std::uint32_t s10 = sources[10 * 32 + lane];
  const std::uint32_t s11 = sources[11 * 32 + lane];
  const std::uint32_t s12 = sources[12 * 32 + lane];
  const std::uint32_t s13 = sources[13 * 32 + lane];
  const std::uint32_t s14 = sources[14 * 32 + lane];
  const std::uint32_t s15 = sources[15 * 32 + lane];

  std::uint32_t acc0 = 0x1234567u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc1 = 0x2345678u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc2 = 0x3456789u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc3 = 0x456789au + static_cast<std::uint32_t>(lane);

  __syncwarp();
  std::uint64_t start;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) : : "memory");

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int unroll = 0; unroll < 16; ++unroll) {
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc0)
                   : "r"(s0), "r"(s1));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc1)
                   : "r"(s2), "r"(s3));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc2)
                   : "r"(s4), "r"(s5));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc3)
                   : "r"(s6), "r"(s7));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc0)
                   : "r"(s8), "r"(s9));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc1)
                   : "r"(s10), "r"(s11));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc2)
                   : "r"(s12), "r"(s13));
      asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                   : "+r"(acc3)
                   : "r"(s14), "r"(s15));
    }
  }

  std::uint64_t stop;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) : : "memory");
  __syncwarp();

  elapsed_cycles[lane] = stop - start;
  sinks[lane] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^
                s6 ^ s7 ^ s8 ^ s9 ^ s10 ^ s11 ^ s12 ^ s13 ^ s14 ^ s15;
}
