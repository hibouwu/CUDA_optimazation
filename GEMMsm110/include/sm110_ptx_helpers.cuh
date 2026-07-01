#pragma once

// Thin SM110 instruction wrappers used by the handwritten kernels.
//
// The implementation style follows learn-cuda/02e_matmul_sm100: CUDA owns
// tensor-map creation and the kernel owns all scheduling, address arithmetic,
// barriers, TMA, TCGen05, and TMEM operations.  CUTLASS/CuTe types are
// intentionally not used here.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace gemm_sm110::ptx {

constexpr int kWarpSize = 32;

__device__ __forceinline__ uint32_t elect_one() {
  uint32_t elected = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "elect.sync _|p, %1;\n\t"
      "@p mov.u32 %0, 1;\n\t"
      "}"
      : "+r"(elected)
      : "r"(0xffffffff));
  return elected;
}

__device__ __forceinline__ uint32_t smem_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t barrier,
                                               uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(barrier), "r"(arrivals));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t barrier,
                                               uint32_t phase) {
  constexpr uint32_t kSuspendHint = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred ready;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
      "ready, [%0], %1, %2;\n\t"
      "@!ready bra.uni WAIT_%=;\n\t"
      "}"
      :
      : "r"(barrier), "r"(phase), "r"(kSuspendHint)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint32_t barrier, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
      "_, [%0], %1;"
      :
      : "r"(barrier), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(
    uint32_t dst, const CUtensorMap* tensor_map, int x, int y, int z,
    uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global."
      "mbarrier::complete_tx::bytes.cta_group::1 "
      "[%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(dst), "l"(tensor_map), "r"(x), "r"(y), "r"(z),
        "r"(barrier)
      : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    uint32_t dst, const CUtensorMap* tensor_map, int x, int y,
    uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(dst), "l"(tensor_map), "r"(x), "r"(y), "r"(barrier)
      : "memory");
}

template <int CtaGroup = 1>
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem,
                                            uint32_t columns) {
  asm volatile(
      "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 "
      "[%0], %1;"
      :
      : "r"(dst_smem), "r"(columns), "n"(CtaGroup));
}

template <int CtaGroup = 1>
__device__ __forceinline__ void tmem_dealloc(uint32_t base,
                                              uint32_t columns) {
  asm volatile(
      "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
      :
      : "r"(base), "r"(columns), "n"(CtaGroup));
}

template <int CtaGroup = 1>
__device__ __forceinline__ void mma_f16(uint32_t accumulator,
                                        uint64_t descriptor_a,
                                        uint64_t descriptor_b,
                                        uint32_t instruction_descriptor,
                                        bool accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred use_d;\n\t"
      "setp.ne.b32 use_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::%5.kind::f16 "
      "[%0], %1, %2, %3, use_d;\n\t"
      "}"
      :
      : "r"(accumulator), "l"(descriptor_a), "l"(descriptor_b),
        "r"(instruction_descriptor), "r"(static_cast<int>(accumulate)),
        "n"(CtaGroup));
}

template <int CtaGroup = 1>
__device__ __forceinline__ void mma_commit(uint32_t barrier) {
  asm volatile(
      "tcgen05.commit.cta_group::%1."
      "mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(barrier), "n"(CtaGroup)
      : "memory");
}

__device__ __forceinline__ void tmem_load_32x32b_x8(
    uint32_t address, float (&values)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__host__ __device__ constexpr uint64_t encode_smem(uint64_t value) {
  return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t sw128_k_major_descriptor(
    uint32_t smem) {
  constexpr uint32_t kStrideByteOffset = 8 * 128;
  return encode_smem(smem) |
         (encode_smem(kStrideByteOffset) << 32ULL) |
         (1ULL << 46ULL) | (2ULL << 61ULL);
}

inline void check_driver(CUresult status, const char* where) {
  if (status == CUDA_SUCCESS) return;
  const char* message = "unknown CUDA driver error";
  cuGetErrorString(status, &message);
  std::fprintf(stderr, "CUDA driver failure in %s: %s\n", where, message);
  std::abort();
}

inline void encode_tiled_3d_sw128(CUtensorMap* tensor_map,
                                  const half* base,
                                  uint64_t global_height,
                                  uint64_t global_width,
                                  uint32_t tile_height,
                                  uint32_t tile_width) {
  // [height,width] -> [width/64,height,64], with the contiguous 64-wide
  // mode first in CUtensorMap coordinate order.
  constexpr uint32_t kRank = 3;
  uint64_t global_dim[kRank] = {64, global_height, global_width / 64};
  uint64_t global_stride[kRank - 1] = {
      global_width * sizeof(half), 64 * sizeof(half)};
  uint32_t box_dim[kRank] = {64, tile_height, tile_width / 64};
  uint32_t element_stride[kRank] = {1, 1, 1};

  check_driver(
      cuTensorMapEncodeTiled(
          tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, kRank,
          const_cast<half*>(base), global_dim, global_stride, box_dim,
          element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(3D SW128)");
}

inline void encode_tiled_2d_sw128(CUtensorMap* tensor_map,
                                  const half* base,
                                  uint64_t global_height,
                                  uint64_t global_width,
                                  uint32_t tile_height) {
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {global_width, global_height};
  uint64_t global_stride[kRank - 1] = {
      global_width * sizeof(half)};
  uint32_t box_dim[kRank] = {64, tile_height};
  uint32_t element_stride[kRank] = {1, 1};

  check_driver(
      cuTensorMapEncodeTiled(
          tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, kRank,
          const_cast<half*>(base), global_dim, global_stride, box_dim,
          element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(2D SW128)");
}

}  // namespace gemm_sm110::ptx
