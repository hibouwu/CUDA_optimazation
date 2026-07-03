#include <cstdint>

#define LOAD_SOURCE(index)                                                     \
  const float s##index = reinterpret_cast<const float*>(sources)[(index) * 32 + lane]

#define FFMA(destination, source0, source1)                                    \
  asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(destination)              \
               : "f"(source0), "f"(source1))

extern "C" __global__ __launch_bounds__(32, 1) void sass_register_probe(
    const float* sources, std::uint64_t* elapsed_cycles,
    float* sinks, int iterations) {
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
  float out0 = 1.23f + static_cast<float>(lane);
  float out1 = 2.34f + static_cast<float>(lane);
  float out2 = 3.45f + static_cast<float>(lane);
  float out3 = 4.56f + static_cast<float>(lane);

  __syncwarp();
  std::uint64_t start;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) : : "memory");

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int unroll = 0; unroll < 8; ++unroll) {
      FFMA(out0, s0, s1);
      FFMA(out1, s2, s3);
      FFMA(out2, s4, s5);
      FFMA(out3, s6, s7);
      FFMA(out0, s8, s9);
      FFMA(out1, s10, s11);
      FFMA(out2, s12, s13);
      FFMA(out3, s14, s15);
      FFMA(out0, s16, s17);
      FFMA(out1, s18, s19);
      FFMA(out2, s20, s21);
      FFMA(out3, s22, s23);
      FFMA(out0, s24, s25);
      FFMA(out1, s26, s27);
      FFMA(out2, s28, s29);
      FFMA(out3, s30, s31);
    }
  }

  std::uint64_t stop;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) : : "memory");
  __syncwarp();

  elapsed_cycles[lane] = stop - start;
  sinks[lane] = out0 + out1 + out2 + out3;
}

#undef FFMA
#undef LOAD_SOURCE
