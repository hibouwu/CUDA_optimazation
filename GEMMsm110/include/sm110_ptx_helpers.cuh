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

__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint32_t block_rank_in_cluster() {
  uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
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

__device__ __forceinline__ void mbarrier_arrive(uint32_t barrier) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :
               : "r"(barrier)
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
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  asm volatile(
      "tcgen05.relinquish_alloc_permit.cta_group::%0.sync.aligned;"
      :
      : "n"(CtaGroup));
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

__device__ __forceinline__ void mma_f16_cta_group2(
    uint32_t accumulator, uint64_t descriptor_a, uint64_t descriptor_b,
    uint32_t instruction_descriptor, bool accumulate) {
  uint32_t mask[8] = {};
  asm volatile(
      "{\n\t"
      ".reg .pred use_d;\n\t"
      "setp.ne.b32 use_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 "
      "[%0], %1, %2, %3, "
      "{%5, %6, %7, %8, %9, %10, %11, %12}, use_d;\n\t"
      "}"
      :
      : "r"(accumulator), "l"(descriptor_a), "l"(descriptor_b),
        "r"(instruction_descriptor), "r"(static_cast<int>(accumulate)),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
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

template <int CtaGroup = 1>
__device__ __forceinline__ void mma_commit_multicast(uint32_t barrier,
                                                      uint16_t cta_mask) {
  asm volatile(
      "tcgen05.commit.cta_group::%2."
      "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
      "[%0], %1;"
      :
      : "r"(barrier), "h"(cta_mask), "n"(CtaGroup)
      : "memory");
}

__device__ __forceinline__ void tmem_load_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tmem_load_32x32b_x8_no_wait(
    uint32_t address, float (&values)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7])
      : "r"(address));
}

__device__ __forceinline__ void tmem_load_32x32b_x8(
    uint32_t address, float (&values)[8]) {
  tmem_load_32x32b_x8_no_wait(address, values);
  tmem_load_wait();
}

__device__ __forceinline__ void tmem_load_32x32b_x16(
    uint32_t address, float (&values)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7]), "=f"(values[8]),
        "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
        "=f"(values[12]), "=f"(values[13]), "=f"(values[14]),
        "=f"(values[15])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tmem_load_16x256b_x8(
    uint32_t address, float (&values)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7]), "=f"(values[8]),
        "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
        "=f"(values[12]), "=f"(values[13]), "=f"(values[14]),
        "=f"(values[15]), "=f"(values[16]), "=f"(values[17]),
        "=f"(values[18]), "=f"(values[19]), "=f"(values[20]),
        "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
        "=f"(values[24]), "=f"(values[25]), "=f"(values[26]),
        "=f"(values[27]), "=f"(values[28]), "=f"(values[29]),
        "=f"(values[30]), "=f"(values[31])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tmem_load_16x256b_x16(
    uint32_t address, float (&values)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x16.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
      : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
        "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
        "=f"(values[6]), "=f"(values[7]), "=f"(values[8]),
        "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
        "=f"(values[12]), "=f"(values[13]), "=f"(values[14]),
        "=f"(values[15]), "=f"(values[16]), "=f"(values[17]),
        "=f"(values[18]), "=f"(values[19]), "=f"(values[20]),
        "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
        "=f"(values[24]), "=f"(values[25]), "=f"(values[26]),
        "=f"(values[27]), "=f"(values[28]), "=f"(values[29]),
        "=f"(values[30]), "=f"(values[31]), "=f"(values[32]),
        "=f"(values[33]), "=f"(values[34]), "=f"(values[35]),
        "=f"(values[36]), "=f"(values[37]), "=f"(values[38]),
        "=f"(values[39]), "=f"(values[40]), "=f"(values[41]),
        "=f"(values[42]), "=f"(values[43]), "=f"(values[44]),
        "=f"(values[45]), "=f"(values[46]), "=f"(values[47]),
        "=f"(values[48]), "=f"(values[49]), "=f"(values[50]),
        "=f"(values[51]), "=f"(values[52]), "=f"(values[53]),
        "=f"(values[54]), "=f"(values[55]), "=f"(values[56]),
        "=f"(values[57]), "=f"(values[58]), "=f"(values[59]),
        "=f"(values[60]), "=f"(values[61]), "=f"(values[62]),
        "=f"(values[63])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void store_global_l1_no_allocate_v8_f32_values(
    float* dst, float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7) {
  asm volatile(
      "st.global.L1::no_allocate.L2::evict_first.v8.f32 "
      "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "l"(dst), "f"(value0), "f"(value1), "f"(value2), "f"(value3),
        "f"(value4), "f"(value5), "f"(value6), "f"(value7)
      : "memory");
}

__device__ __forceinline__ void store_global_l1_no_allocate_v8_f32(
    float* dst, const float (&values)[8]) {
  store_global_l1_no_allocate_v8_f32_values(
      dst, values[0], values[1], values[2], values[3], values[4],
      values[5], values[6], values[7]);
}

__device__ __forceinline__ void store_global_l1_no_allocate_v2_f32(
    float* dst, float value0, float value1) {
  asm volatile(
      "st.global.L1::no_allocate.v2.f32 [%0], {%1, %2};"
      :
      : "l"(dst), "f"(value0), "f"(value1)
      : "memory");
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

__device__ __forceinline__ uint64_t inter_k_major_descriptor(
    uint32_t smem) {
  constexpr uint32_t kStrideByteOffset = 8 * 16;
  return encode_smem(smem) |
         (encode_smem(kStrideByteOffset) << 32ULL) |
         (1ULL << 46ULL);
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

inline void encode_tiled_3d_inter(CUtensorMap* tensor_map,
                                  const half* base,
                                  uint64_t global_height,
                                  uint64_t global_width,
                                  uint32_t tile_height,
                                  uint32_t tile_width) {
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
          CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(3D INTER)");
}

inline void encode_tiled_2d_sw128_strided(CUtensorMap* tensor_map,
                                          const half* base,
                                          uint64_t global_height,
                                          uint64_t global_width,
                                          uint64_t row_stride,
                                          uint32_t tile_height) {
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {global_width, global_height};
  uint64_t global_stride[kRank - 1] = {
      row_stride * sizeof(half)};
  uint32_t box_dim[kRank] = {64, tile_height};
  uint32_t element_stride[kRank] = {1, 1};

  check_driver(
      cuTensorMapEncodeTiled(
          tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, kRank,
          const_cast<half*>(base), global_dim, global_stride, box_dim,
          element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(2D SW128 strided)");
}

inline void encode_tiled_2d_sw128(CUtensorMap* tensor_map,
                                  const half* base,
                                  uint64_t global_height,
                                  uint64_t global_width,
                                  uint32_t tile_height) {
  encode_tiled_2d_sw128_strided(tensor_map, base, global_height,
                                global_width, global_width, tile_height);
}

inline void encode_tiled_2d_inter(CUtensorMap* tensor_map,
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
          CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(2D INTER)");
}

}  // namespace gemm_sm110::ptx
