#include <cstdint>

#define LOAD_SOURCE(index)                                                     \
  const std::uint32_t s##index = sources[(index) * 32 + lane]

#define IMAD(destination, source0, source1)                                     \
  asm volatile("mad.lo.u32 %0, %1, %2, %0;"                                   \
               : "+r"(destination)                                             \
               : "r"(source0), "r"(source1))

extern "C" __global__ __launch_bounds__(32, 1) void sass_register_probe(
    const std::uint32_t* sources, std::uint64_t* elapsed_cycles,
    std::uint32_t* sinks, int iterations) {
  const int lane = threadIdx.x;
  LOAD_SOURCE(0);
  LOAD_SOURCE(1);
  LOAD_SOURCE(2);
  LOAD_SOURCE(3);
  LOAD_SOURCE(4);
  LOAD_SOURCE(5);
  LOAD_SOURCE(6);
  LOAD_SOURCE(7);
  LOAD_SOURCE(8);
  LOAD_SOURCE(9);
  LOAD_SOURCE(10);
  LOAD_SOURCE(11);
  LOAD_SOURCE(12);
  LOAD_SOURCE(13);
  LOAD_SOURCE(14);
  LOAD_SOURCE(15);
  LOAD_SOURCE(16);
  LOAD_SOURCE(17);
  LOAD_SOURCE(18);
  LOAD_SOURCE(19);
  LOAD_SOURCE(20);
  LOAD_SOURCE(21);
  LOAD_SOURCE(22);
  LOAD_SOURCE(23);
  LOAD_SOURCE(24);
  LOAD_SOURCE(25);
  LOAD_SOURCE(26);
  LOAD_SOURCE(27);
  LOAD_SOURCE(28);
  LOAD_SOURCE(29);
  LOAD_SOURCE(30);
  LOAD_SOURCE(31);
  LOAD_SOURCE(32);
  LOAD_SOURCE(33);
  LOAD_SOURCE(34);
  LOAD_SOURCE(35);
  LOAD_SOURCE(36);
  LOAD_SOURCE(37);
  LOAD_SOURCE(38);
  LOAD_SOURCE(39);
  LOAD_SOURCE(40);
  LOAD_SOURCE(41);
  LOAD_SOURCE(42);
  LOAD_SOURCE(43);
  LOAD_SOURCE(44);
  LOAD_SOURCE(45);
  LOAD_SOURCE(46);
  LOAD_SOURCE(47);
  LOAD_SOURCE(48);
  LOAD_SOURCE(49);
  LOAD_SOURCE(50);
  LOAD_SOURCE(51);
  LOAD_SOURCE(52);
  LOAD_SOURCE(53);
  LOAD_SOURCE(54);
  LOAD_SOURCE(55);
  LOAD_SOURCE(56);
  LOAD_SOURCE(57);
  LOAD_SOURCE(58);
  LOAD_SOURCE(59);
  LOAD_SOURCE(60);
  LOAD_SOURCE(61);
  LOAD_SOURCE(62);
  LOAD_SOURCE(63);

  std::uint32_t out0 = 0x1234567u + static_cast<std::uint32_t>(lane);
  std::uint32_t out1 = 0x2345678u + static_cast<std::uint32_t>(lane);
  std::uint32_t out2 = 0x3456789u + static_cast<std::uint32_t>(lane);
  std::uint32_t out3 = 0x456789au + static_cast<std::uint32_t>(lane);

  __syncwarp();
  std::uint64_t start;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) : : "memory");

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int unroll = 0; unroll < 4; ++unroll) {
      IMAD(out0, s0, s1);
      IMAD(out1, s2, s3);
      IMAD(out2, s4, s5);
      IMAD(out3, s6, s7);
      IMAD(out0, s8, s9);
      IMAD(out1, s10, s11);
      IMAD(out2, s12, s13);
      IMAD(out3, s14, s15);
      IMAD(out0, s16, s17);
      IMAD(out1, s18, s19);
      IMAD(out2, s20, s21);
      IMAD(out3, s22, s23);
      IMAD(out0, s24, s25);
      IMAD(out1, s26, s27);
      IMAD(out2, s28, s29);
      IMAD(out3, s30, s31);
      IMAD(out0, s32, s33);
      IMAD(out1, s34, s35);
      IMAD(out2, s36, s37);
      IMAD(out3, s38, s39);
      IMAD(out0, s40, s41);
      IMAD(out1, s42, s43);
      IMAD(out2, s44, s45);
      IMAD(out3, s46, s47);
      IMAD(out0, s48, s49);
      IMAD(out1, s50, s51);
      IMAD(out2, s52, s53);
      IMAD(out3, s54, s55);
      IMAD(out0, s56, s57);
      IMAD(out1, s58, s59);
      IMAD(out2, s60, s61);
      IMAD(out3, s62, s63);
    }
  }

  std::uint64_t stop;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) : : "memory");
  __syncwarp();

  elapsed_cycles[lane] = stop - start;
  sinks[lane] = out0 ^ out1 ^ out2 ^ out3;
}

#undef IMAD
#undef LOAD_SOURCE
