// Compiler-only dialect probe. Do not execute: no TMEM/descriptors are initialized.
// The asm spelling is reduced from snapshot/include/cute/arch/copy_sm100.hpp:378-380.
#if defined(PROBE_WITH_COMMA)
#define PROBE_SEPARATOR ,
#else
#define PROBE_SEPARATOR
#endif

using uint32_t = unsigned int;
using uint64_t = unsigned long long;

__device__ __forceinline__ void device_wrapper(uint64_t const& src_addr,
                                             uint32_t const& dst_addr) {
  asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                :
                : "r"(dst_addr) PROBE_SEPARATOR "l"(src_addr));
}

extern "C" __global__ void asm01_emit(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  device_wrapper(src_addr, dst_addr);
#endif
}
