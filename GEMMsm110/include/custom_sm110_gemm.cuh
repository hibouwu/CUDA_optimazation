#pragma once

// Low-level SM110 TCGen05 GEMM kernels owned by this project.
//
// This does not instantiate CUTLASS CollectiveBuilder, GemmUniversal, or a
// CUTLASS device adapter. It uses CuTe only as the low-level encoding layer for
// TCGen05 MMA/TMEM instructions and layouts.
//
// The TCGen05/TMEM sequence is adapted from NVIDIA CUTLASS tutorial
// examples/cute/tutorial/blackwell/01_mma_sm100.cu (BSD-3-Clause).
// Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.

#include "gemm_common.cuh"

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>

namespace gemm_sm110::custom_backend {

using namespace cute;

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc3SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> a;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> b;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a() {
    return make_tensor(make_smem_ptr(a.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), BSmemLayout{});
  }
};

template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler, class TiledMma>
__global__ void tc3_tcgen05_gemm_kernel(ATensor matrix_a, BTensor matrix_b,
                                        CTensor matrix_c, DTensor matrix_d,
                                        MmaTiler mma_tiler,
                                        TiledMma tiled_mma) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  auto tile_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor g_a =
      local_tile(matrix_a, mma_tiler, tile_coord, Step<_1, X, _1>{});
  Tensor g_b =
      local_tile(matrix_b, mma_tiler, tile_coord, Step<X, _1, _1>{});
  Tensor g_c =
      local_tile(matrix_c, mma_tiler, tile_coord, Step<_1, _1, X>{});
  Tensor g_d =
      local_tile(matrix_d, mma_tiler, tile_coord, Step<_1, _1, X>{});

  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  ThrMMA cta_mma = tiled_mma.get_slice(_0{});
  Tensor t_cg_a = cta_mma.partition_A(g_a);
  Tensor t_cg_b = cta_mma.partition_B(g_b);
  Tensor t_cg_c = cta_mma.partition_C(g_c);
  Tensor t_cg_d = cta_mma.partition_C(g_d);
  Tensor t_cr_a = cta_mma.make_fragment_A(s_a);
  Tensor t_cr_b = cta_mma.make_fragment_B(s_b);
  Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool mma_warp = threadIdx.x < kWarpSize;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (mma_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }
  __syncthreads();
  t_ct_acc.data() = storage.tmem_base_ptr;

  if (mma_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  int mma_phase = 0;
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int k_tile = 0; k_tile < size<3>(t_cg_a); ++k_tile) {
    cooperative_copy<128>(threadIdx.x, t_cg_a(_, _, _, k_tile), s_a);
    cooperative_copy<128>(threadIdx.x, t_cg_b(_, _, _, k_tile), s_b);
    __syncthreads();

    if (mma_warp) {
      for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
        gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
             t_ct_acc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier);
    }
    cute::wait_barrier(storage.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, t_ct_acc);
  ThrCopy thread_copy = tmem_to_register.get_slice(threadIdx.x);
  Tensor t_dt_acc = thread_copy.partition_S(t_ct_acc);
  Tensor t_dg_d = thread_copy.partition_D(t_cg_d);
  using Accumulator = typename decltype(t_ct_acc)::value_type;
  Tensor t_dr_acc = make_tensor<Accumulator>(shape(t_dg_d));

  copy(tmem_to_register, t_dt_acc, t_dr_acc);
  copy(t_dr_acc, t_dg_d);

  __syncthreads();
  if (mma_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
#else
  (void)matrix_a;
  (void)matrix_b;
  (void)matrix_c;
  (void)matrix_d;
  (void)mma_tiler;
  (void)tiled_mma;
#endif
}

class Tc3Runner {
 public:
  Tc3Runner(const half* a, const half* b, float* d, int m, int n, int k)
      : a_(reinterpret_cast<const cutlass::half_t*>(a)),
        b_(reinterpret_cast<const cutlass::half_t*>(b)),
        d_(d),
        m_(m),
        n_(n),
        k_(k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc3 custom TCGen05 requires M,N multiples of 128 and K "
                   "a multiple of 64\n");
      std::abort();
    }
  }

  void launch() {
    auto layout_a =
        make_layout(make_shape(m_, k_), make_stride(k_, Int<1>{}));
    // The benchmark stores B as row-major KxN. Its logical MMA view is NxK.
    auto layout_b =
        make_layout(make_shape(n_, k_), make_stride(Int<1>{}, n_));
    auto layout_d =
        make_layout(make_shape(m_, n_), make_stride(n_, Int<1>{}));

    auto matrix_a = make_tensor(make_gmem_ptr(a_), layout_a);
    auto matrix_b = make_tensor(make_gmem_ptr(b_), layout_b);
    auto matrix_c = make_tensor(make_gmem_ptr(d_), layout_d);
    auto matrix_d = make_tensor(make_gmem_ptr(d_), layout_d);

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float, kTileM,
                             kTileN, UMMA::Major::K, UMMA::Major::MN>{});
    auto mma_tiler =
        make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto mma_shape_a = partition_shape_A(
        tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
    auto mma_shape_b = partition_shape_B(
        tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
    [[maybe_unused]] auto smem_layout_a = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<cutlass::half_t>{}, mma_shape_a);
    [[maybe_unused]] auto smem_layout_b = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<cutlass::half_t>{}, mma_shape_b);
    using Storage = Tc3SharedStorage<
        cutlass::half_t, cutlass::half_t, decltype(smem_layout_a),
        decltype(smem_layout_b)>;

    auto* kernel =
        &tc3_tcgen05_gemm_kernel<Storage, decltype(matrix_a),
                                 decltype(matrix_b), decltype(matrix_c),
                                 decltype(matrix_d), decltype(mma_tiler),
                                 decltype(tiled_mma)>;
    static bool configured = false;
    if (!configured) {
      check_cuda(cudaFuncSetAttribute(
                     kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(sizeof(Storage))),
                 "cudaFuncSetAttribute(tc3)");
      configured = true;
    }

    const dim3 grid(m_ / kTileM, n_ / kTileN);
    kernel<<<grid, 128, sizeof(Storage)>>>(matrix_a, matrix_b, matrix_c,
                                          matrix_d, mma_tiler, tiled_mma);
    check_cuda(cudaGetLastError(), "tc3_tcgen05_gemm_kernel launch");
  }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  const cutlass::half_t* a_;
  const cutlass::half_t* b_;
  float* d_;
  int m_;
  int n_;
  int k_;
};

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc4SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> a;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> b;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a() {
    return make_tensor(make_smem_ptr(a.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), BSmemLayout{});
  }
};

template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler, class TiledMma, class TmaAtomA,
          class TmaAtomB>
__global__ void tc4_tma_tcgen05_gemm_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, TiledMma tiled_mma,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  auto tile_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor g_a =
      local_tile(matrix_a, mma_tiler, tile_coord, Step<_1, X, _1>{});
  Tensor g_b =
      local_tile(matrix_b, mma_tiler, tile_coord, Step<X, _1, _1>{});
  Tensor g_c =
      local_tile(matrix_c, mma_tiler, tile_coord, Step<_1, _1, X>{});
  Tensor g_d =
      local_tile(matrix_d, mma_tiler, tile_coord, Step<_1, _1, X>{});

  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  ThrMMA cta_mma = tiled_mma.get_slice(_0{});
  Tensor t_cg_a = cta_mma.partition_A(g_a);
  Tensor t_cg_b = cta_mma.partition_B(g_b);
  Tensor t_cg_c = cta_mma.partition_C(g_c);
  Tensor t_cg_d = cta_mma.partition_C(g_d);
  Tensor t_cr_a = cta_mma.make_fragment_A(s_a);
  Tensor t_cr_b = cta_mma.make_fragment_B(s_b);
  Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool mma_warp = threadIdx.x < kWarpSize;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (mma_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }
  __syncthreads();
  t_ct_acc.data() = storage.tmem_base_ptr;

  auto [t_a_g_a, t_a_s_a] =
      tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                    group_modes<0, 3>(s_a), group_modes<0, 3>(t_cg_a));
  auto [t_b_g_b, t_b_s_b] =
      tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                    group_modes<0, 3>(s_b), group_modes<0, 3>(t_cg_b));
  const int transaction_bytes =
      sizeof(make_tensor_like(t_a_s_a)) + sizeof(make_tensor_like(t_b_s_b));

  if (mma_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    cute::initialize_barrier(storage.tma_barrier, 1);
  }
  int mma_phase = 0;
  int tma_phase = 0;
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int k_tile = 0; k_tile < size<3>(t_cg_a); ++k_tile) {
    if (mma_warp && elected_thread) {
      cute::set_barrier_transaction_bytes(storage.tma_barrier,
                                          transaction_bytes);
      copy(tma_atom_a.with(storage.tma_barrier), t_a_g_a(_, k_tile),
           t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier), t_b_g_b(_, k_tile),
           t_b_s_b);
    }

    cute::wait_barrier(storage.tma_barrier, tma_phase);
    tma_phase ^= 1;

    if (mma_warp) {
      for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
        gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
             t_ct_acc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier);
    }
    cute::wait_barrier(storage.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, t_ct_acc);
  ThrCopy thread_copy = tmem_to_register.get_slice(threadIdx.x);
  Tensor t_dt_acc = thread_copy.partition_S(t_ct_acc);
  Tensor t_dg_d = thread_copy.partition_D(t_cg_d);
  using Accumulator = typename decltype(t_ct_acc)::value_type;
  Tensor t_dr_acc = make_tensor<Accumulator>(shape(t_dg_d));
  copy(tmem_to_register, t_dt_acc, t_dr_acc);
  copy(t_dr_acc, t_dg_d);

  __syncthreads();
  if (mma_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
#else
  (void)matrix_a;
  (void)matrix_b;
  (void)matrix_c;
  (void)matrix_d;
  (void)mma_tiler;
  (void)tiled_mma;
  (void)tma_atom_a;
  (void)tma_atom_b;
#endif
}

class Tc4Runner {
 public:
  Tc4Runner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc4 custom TMA TCGen05 requires M,N multiples of 128 "
                   "and K a multiple of 64\n");
      std::abort();
    }

    const auto* typed_a = reinterpret_cast<const cutlass::half_t*>(a);
    const auto* typed_b = reinterpret_cast<const cutlass::half_t*>(b);
    auto layout_a =
        make_layout(make_shape(m, k), make_stride(k, Int<1>{}));
    auto layout_b =
        make_layout(make_shape(n, k), make_stride(Int<1>{}, n));
    auto layout_d =
        make_layout(make_shape(m, n), make_stride(n, Int<1>{}));
    auto matrix_a = make_tensor(make_gmem_ptr(typed_a), layout_a);
    auto matrix_b = make_tensor(make_gmem_ptr(typed_b), layout_b);
    auto matrix_c = make_tensor(make_gmem_ptr(d), layout_d);
    auto matrix_d = make_tensor(make_gmem_ptr(d), layout_d);

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float, kTileM,
                             kTileN, UMMA::Major::K, UMMA::Major::MN>{});
    auto mma_tiler =
        make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto mma_shape_a = partition_shape_A(
        tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
    auto mma_shape_b = partition_shape_B(
        tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
    auto smem_layout_a = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<cutlass::half_t>{}, mma_shape_a);
    auto smem_layout_b = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<cutlass::half_t>{}, mma_shape_b);
    using Storage = Tc4SharedStorage<
        cutlass::half_t, cutlass::half_t, decltype(smem_layout_a),
        decltype(smem_layout_b)>;

    Copy_Atom tma_atom_a =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_a, smem_layout_a,
                      select<0, 2>(mma_tiler));
    Copy_Atom tma_atom_b =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_b, smem_layout_b,
                      select<1, 2>(mma_tiler));
    auto matrix_a_tma = tma_atom_a.get_tma_tensor(shape(matrix_a));
    auto matrix_b_tma = tma_atom_b.get_tma_tensor(shape(matrix_b));

    auto* kernel =
        &tc4_tma_tcgen05_gemm_kernel<
            Storage, decltype(matrix_a_tma), decltype(matrix_b_tma),
            decltype(matrix_c), decltype(matrix_d), decltype(mma_tiler),
            decltype(tiled_mma), decltype(tma_atom_a), decltype(tma_atom_b)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc4)");

    const dim3 grid(m / kTileM, n / kTileN);
    launch_ = [=]() mutable {
      kernel<<<grid, 128, sizeof(Storage)>>>(
          matrix_a_tma, matrix_b_tma, matrix_c, matrix_d, mma_tiler,
          tiled_mma, tma_atom_a, tma_atom_b);
      check_cuda(cudaGetLastError(), "tc4_tma_tcgen05_gemm_kernel launch");
    };
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

struct alignas(16) Tc5ClcResponse {
  uint32_t data[4];
};

struct Tc5WorkTile {
  int m;
  int n;
  int valid;
};

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc5SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> a;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> b;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint64_t clc_barrier;
  alignas(16) Tc5ClcResponse clc_response;
  alignas(16) cute::uint32_t tmem_base_ptr;
  Tc5WorkTile work_tile;

  CUTE_DEVICE constexpr auto tensor_a() {
    return make_tensor(make_smem_ptr(a.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), BSmemLayout{});
  }
};

template <class Storage>
CUTE_DEVICE void tc5_issue_clc_query(Storage& storage) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  const uint32_t response_address =
      cute::cast_smem_ptr_to_uint(&storage.clc_response);
  const uint32_t barrier_address =
      cute::cast_smem_ptr_to_uint(&storage.clc_barrier);
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.multicast::cluster::all.b128 "
      "[%0], [%1];"
      :
      : "r"(response_address), "r"(barrier_address)
      : "memory");
#else
  (void)storage;
#endif
}

template <class Storage>
CUTE_DEVICE Tc5WorkTile tc5_read_clc_response(Storage& storage) {
  Tc5WorkTile tile{-1, -1, 0};
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  const uint32_t response_address =
      cute::cast_smem_ptr_to_uint(&storage.clc_response);
  uint32_t valid = 0;
  asm volatile(
      "{\n"
      ".reg .pred canceled;\n"
      ".reg .b128 response;\n"
      "ld.shared.b128 response, [%3];\n"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
      "canceled, response;\n"
      "selp.u32 %2, 1, 0, canceled;\n"
      "@canceled "
      "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%0, %1, _, _}, response;\n"
      "}\n"
      : "=r"(tile.m), "=r"(tile.n), "=r"(valid)
      : "r"(response_address)
      : "memory");
  tile.valid = static_cast<int>(valid);
  cutlass::arch::fence_view_async_shared();
#endif
  return tile;
}

template <bool UseClc, class SharedStorage, class ATensor, class BTensor,
          class CTensor, class DTensor, class MmaTiler, class TiledMma,
          class TmaAtomA, class TmaAtomB>
__global__ void tc5_persistent_tma_tcgen05_gemm_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, TiledMma tiled_mma,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b, int tiles_m, int tiles_n) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool mma_warp = threadIdx.x < kWarpSize;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (mma_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }

  if (mma_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    cute::initialize_barrier(storage.tma_barrier, 1);
    if constexpr (UseClc) {
      cute::initialize_barrier(storage.clc_barrier, 1);
    }
  }
  if (threadIdx.x == 0) {
    if constexpr (UseClc) {
      storage.work_tile = {static_cast<int>(blockIdx.x),
                           static_cast<int>(blockIdx.y), 1};
    } else {
      const int tile_id = static_cast<int>(blockIdx.x);
      storage.work_tile = {tile_id / tiles_n, tile_id % tiles_n,
                           tile_id < tiles_m * tiles_n ? 1 : 0};
    }
  }
  __syncthreads();

  int mma_phase = 0;
  int tma_phase = 0;
  int clc_phase = 0;
  int static_tile_id = static_cast<int>(blockIdx.x);

  while (storage.work_tile.valid) {
    if constexpr (UseClc) {
      if (threadIdx.x == 0) {
        cute::set_barrier_transaction_bytes(storage.clc_barrier,
                                            sizeof(Tc5ClcResponse));
        tc5_issue_clc_query(storage);
      }
    }

    const auto tile_coord =
        make_coord(storage.work_tile.m, storage.work_tile.n, _);
    Tensor g_a =
        local_tile(matrix_a, mma_tiler, tile_coord, Step<_1, X, _1>{});
    Tensor g_b =
        local_tile(matrix_b, mma_tiler, tile_coord, Step<X, _1, _1>{});
    Tensor g_c =
        local_tile(matrix_c, mma_tiler, tile_coord, Step<_1, _1, X>{});
    Tensor g_d =
        local_tile(matrix_d, mma_tiler, tile_coord, Step<_1, _1, X>{});

    ThrMMA cta_mma = tiled_mma.get_slice(_0{});
    Tensor t_cg_a = cta_mma.partition_A(g_a);
    Tensor t_cg_b = cta_mma.partition_B(g_b);
    Tensor t_cg_c = cta_mma.partition_C(g_c);
    Tensor t_cg_d = cta_mma.partition_C(g_d);
    Tensor t_cr_a = cta_mma.make_fragment_A(s_a);
    Tensor t_cr_b = cta_mma.make_fragment_B(s_b);
    Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);
    t_ct_acc.data() = storage.tmem_base_ptr;

    auto [t_a_g_a, t_a_s_a] =
        tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(s_a), group_modes<0, 3>(t_cg_a));
    auto [t_b_g_b, t_b_s_b] =
        tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(s_b), group_modes<0, 3>(t_cg_b));
    const int transaction_bytes =
        sizeof(make_tensor_like(t_a_s_a)) + sizeof(make_tensor_like(t_b_s_b));

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    for (int k_tile = 0; k_tile < size<3>(t_cg_a); ++k_tile) {
      if (mma_warp && elected_thread) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier,
                                            transaction_bytes);
        copy(tma_atom_a.with(storage.tma_barrier), t_a_g_a(_, k_tile),
             t_a_s_a);
        copy(tma_atom_b.with(storage.tma_barrier), t_b_g_b(_, k_tile),
             t_b_s_b);
      }
      cute::wait_barrier(storage.tma_barrier, tma_phase);
      tma_phase ^= 1;

      if (mma_warp) {
        for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
          gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
               t_ct_acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&storage.mma_barrier);
      }
      cute::wait_barrier(storage.mma_barrier, mma_phase);
      mma_phase ^= 1;
    }

    TiledCopy tmem_to_register =
        make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, t_ct_acc);
    ThrCopy thread_copy = tmem_to_register.get_slice(threadIdx.x);
    Tensor t_dt_acc = thread_copy.partition_S(t_ct_acc);
    Tensor t_dg_d = thread_copy.partition_D(t_cg_d);
    using Accumulator = typename decltype(t_ct_acc)::value_type;
    Tensor t_dr_acc = make_tensor<Accumulator>(shape(t_dg_d));
    copy(tmem_to_register, t_dt_acc, t_dr_acc);
    copy(t_dr_acc, t_dg_d);
    __syncthreads();

    if constexpr (UseClc) {
      cute::wait_barrier(storage.clc_barrier, clc_phase);
      clc_phase ^= 1;
      if (threadIdx.x == 0) {
        storage.work_tile = tc5_read_clc_response(storage);
      }
    } else {
      if (threadIdx.x == 0) {
        static_tile_id += static_cast<int>(gridDim.x);
        storage.work_tile = {
            static_tile_id / tiles_n, static_tile_id % tiles_n,
            static_tile_id < tiles_m * tiles_n ? 1 : 0};
      }
    }
    __syncthreads();
  }

  if (mma_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
#else
  (void)matrix_a;
  (void)matrix_b;
  (void)matrix_c;
  (void)matrix_d;
  (void)mma_tiler;
  (void)tiled_mma;
  (void)tma_atom_a;
  (void)tma_atom_b;
  (void)tiles_m;
  (void)tiles_n;
#endif
}

template <bool UseClc>
class Tc5Runner {
 public:
  Tc5Runner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc5 custom persistent TCGen05 requires M,N multiples of "
                   "128 and K a multiple of 64\n");
      std::abort();
    }

    const auto* typed_a = reinterpret_cast<const cutlass::half_t*>(a);
    const auto* typed_b = reinterpret_cast<const cutlass::half_t*>(b);
    auto layout_a =
        make_layout(make_shape(m, k), make_stride(k, Int<1>{}));
    auto layout_b =
        make_layout(make_shape(n, k), make_stride(Int<1>{}, n));
    auto layout_d =
        make_layout(make_shape(m, n), make_stride(n, Int<1>{}));
    auto matrix_a = make_tensor(make_gmem_ptr(typed_a), layout_a);
    auto matrix_b = make_tensor(make_gmem_ptr(typed_b), layout_b);
    auto matrix_c = make_tensor(make_gmem_ptr(d), layout_d);
    auto matrix_d = make_tensor(make_gmem_ptr(d), layout_d);

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float, kTileM,
                             kTileN, UMMA::Major::K, UMMA::Major::MN>{});
    auto mma_tiler =
        make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto mma_shape_a = partition_shape_A(
        tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
    auto mma_shape_b = partition_shape_B(
        tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
    auto smem_layout_a = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<cutlass::half_t>{}, mma_shape_a);
    auto smem_layout_b = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<cutlass::half_t>{}, mma_shape_b);
    using Storage = Tc5SharedStorage<
        cutlass::half_t, cutlass::half_t, decltype(smem_layout_a),
        decltype(smem_layout_b)>;

    Copy_Atom tma_atom_a =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_a, smem_layout_a,
                      select<0, 2>(mma_tiler));
    Copy_Atom tma_atom_b =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_b, smem_layout_b,
                      select<1, 2>(mma_tiler));
    auto matrix_a_tma = tma_atom_a.get_tma_tensor(shape(matrix_a));
    auto matrix_b_tma = tma_atom_b.get_tma_tensor(shape(matrix_b));

    auto* kernel =
        &tc5_persistent_tma_tcgen05_gemm_kernel<
            UseClc, Storage, decltype(matrix_a_tma),
            decltype(matrix_b_tma), decltype(matrix_c), decltype(matrix_d),
            decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_a),
            decltype(tma_atom_b)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc5)");

    const int tiles_m = m / kTileM;
    const int tiles_n = n / kTileN;
    dim3 grid;
    if constexpr (UseClc) {
      grid = dim3(tiles_m, tiles_n);
    } else {
      int device = 0;
      cudaDeviceProp properties{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5)");
      check_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties(tc5)");
      const int workers =
          min(tiles_m * tiles_n, properties.multiProcessorCount);
      grid = dim3(workers);
    }

    launch_ = [=]() mutable {
      kernel<<<grid, 128, sizeof(Storage)>>>(
          matrix_a_tma, matrix_b_tma, matrix_c, matrix_d, mma_tiler,
          tiled_mma, tma_atom_a, tma_atom_b, tiles_m, tiles_n);
      check_cuda(cudaGetLastError(),
                 "tc5_persistent_tma_tcgen05_gemm_kernel launch");
    };
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

using Tc5StaticRunner = Tc5Runner<false>;
using Tc5ClcRunner = Tc5Runner<true>;

}  // namespace gemm_sm110::custom_backend
