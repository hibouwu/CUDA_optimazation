/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Extracted compiler probe; never execute this kernel.
#include <cute/config.hpp>
#include <cute/arch/config.hpp>
#if defined(__CUDA_ARCH__)
#if !defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
#error "Expected original CUTLASS configuration to enable the actual TMEM branch"
#endif
#if __CUDA_ARCH__ != PROBE_EXPECT_ARCH
#error "Unexpected actual device architecture"
#endif
#endif
namespace cute {
#line 366 "snapshot/include/cute/arch/copy_sm100.hpp"
namespace SM100::TMEM::UTCCP {

// 128 data path lanes, 256-bit pattern, 1cta mode
struct SM100_UTCCP_128dp256bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 128 data path lanes, 256-bit pattern, 2cta mode
struct SM100_UTCCP_128dp256bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.128x256b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

struct SM100_UTCCP_128dp128bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x128b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

struct SM100_UTCCP_128dp128bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.128x128b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};


// 4 data path lanes, 256-bit pattern, 1cta mode
struct SM100_UTCCP_4dp256bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.4x256b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4 data path lanes, 256-bit pattern, 2cta mode
struct SM100_UTCCP_4dp256bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        asm volatile ("tcgen05.cp.cta_group::2.4x256b [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4x32 data path lanes (broadcast), 128-bit pattern, 1cta mode
struct SM100_UTCCP_4x32dp128bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4x32 data path lanes (broadcast), 128-bit pattern, 2cta mode
struct SM100_UTCCP_4x32dp128bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast like 4x32dp), 128-bit pattern, 1cta mode
struct SM100_UTCCP_2x64dp128bitlw0213_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13  [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast like 4x32dp), 128-bit pattern, 2cta mode
struct SM100_UTCCP_2x64dp128bitlw0213_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13  [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast seperately in upper and lower 64dp), 128-bit pattern, 1cta mode
// data_row[0:31] -> DP[0:63]
// data_row[32:63] -> DP[64:127]
struct SM100_UTCCP_2x64dp128bitlw0123_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast seperately in upper and lower 64dp), 128-bit pattern, 2cta mode
// data_row[0:31] -> DP[0:63]
// data_row[32:63] -> DP[64:127]
struct SM100_UTCCP_2x64dp128bitlw0123_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
    :
    : "r"(dst_addr), "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

} // end namespace SM100::TMEM::UTCCP

} // namespace cute
#line 1 "asm01_probe_entry.cu"
extern "C" __global__ void asm01_emit_00(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_01(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_2cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_02(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp128bit_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_03(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp128bit_2cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_04(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_4dp256bit_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_05(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_4dp256bit_2cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_06(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_07(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_08(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_09(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0213_2cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_10(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0123_1cta::copy(src_addr, dst_addr);
#endif
}
extern "C" __global__ void asm01_emit_11(uint64_t src_addr, uint32_t dst_addr) {
#if defined(PROBE_EMIT)
  cute::SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0123_2cta::copy(src_addr, dst_addr);
#endif
}
