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
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <memory>

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
  static constexpr int kStages = 2;
  static constexpr int kAStageElements = cute::cosize_v<ASmemLayout>;
  static constexpr int kBStageElements = cute::cosize_v<BSmemLayout>;

  alignas(128)
      cute::ArrayEngine<TypeA, kStages * kAStageElements> a;
  alignas(128)
      cute::ArrayEngine<TypeB, kStages * kBStageElements> b;
  alignas(16) cute::uint64_t mma_barrier[kStages];
  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a(int stage) {
    return make_tensor(
        make_smem_ptr(a.begin() + stage * kAStageElements), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b(int stage) {
    return make_tensor(
        make_smem_ptr(b.begin() + stage * kBStageElements), BSmemLayout{});
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

  ThrMMA cta_mma = tiled_mma.get_slice(_0{});
  Tensor t_cg_a = cta_mma.partition_A(g_a);
  Tensor t_cg_b = cta_mma.partition_B(g_b);
  Tensor t_cg_c = cta_mma.partition_C(g_c);
  Tensor t_cg_d = cta_mma.partition_C(g_d);
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
                    group_modes<0, 3>(storage.tensor_a(0)),
                    group_modes<0, 3>(t_cg_a));
  auto [t_b_g_b, t_b_s_b] =
      tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                    group_modes<0, 3>(storage.tensor_b(0)),
                    group_modes<0, 3>(t_cg_b));
  const int transaction_bytes =
      sizeof(make_tensor_like(t_a_s_a)) + sizeof(make_tensor_like(t_b_s_b));

  if (mma_warp && elected_thread) {
    for (int stage = 0; stage < SharedStorage::kStages; ++stage) {
      cute::initialize_barrier(storage.mma_barrier[stage], 1);
      cute::initialize_barrier(storage.tma_barrier[stage], 1);
    }
  }
  int mma_phase[SharedStorage::kStages] = {0, 0};
  int tma_phase[SharedStorage::kStages] = {0, 0};
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  const int k_tile_count = size<3>(t_cg_a);

  for (int load_tile = 0;
       load_tile < min(k_tile_count, SharedStorage::kStages); ++load_tile) {
    const int load_stage = load_tile % SharedStorage::kStages;
    Tensor load_s_a = storage.tensor_a(load_stage);
    Tensor load_s_b = storage.tensor_b(load_stage);
    auto [load_t_a_g_a, load_t_a_s_a] =
        tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(load_s_a),
                      group_modes<0, 3>(t_cg_a));
    auto [load_t_b_g_b, load_t_b_s_b] =
        tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(load_s_b),
                      group_modes<0, 3>(t_cg_b));
    if (mma_warp && elected_thread) {
      cute::set_barrier_transaction_bytes(storage.tma_barrier[load_stage],
                                          transaction_bytes);
      copy(tma_atom_a.with(storage.tma_barrier[load_stage]),
           load_t_a_g_a(_, load_tile), load_t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier[load_stage]),
           load_t_b_g_b(_, load_tile), load_t_b_s_b);
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    const int stage = k_tile % SharedStorage::kStages;

    cute::wait_barrier(storage.tma_barrier[stage], tma_phase[stage]);
    tma_phase[stage] ^= 1;

    Tensor current_s_a = storage.tensor_a(stage);
    Tensor current_s_b = storage.tensor_b(stage);
    Tensor t_cr_a = cta_mma.make_fragment_A(current_s_a);
    Tensor t_cr_b = cta_mma.make_fragment_B(current_s_b);

    if (mma_warp) {
      for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
        gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
             t_ct_acc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier[stage]);
    }

    const int reuse_tile = k_tile + SharedStorage::kStages;
    if (reuse_tile < k_tile_count) {
      cute::wait_barrier(storage.mma_barrier[stage], mma_phase[stage]);
      mma_phase[stage] ^= 1;

      auto [reuse_t_a_g_a, reuse_t_a_s_a] =
          tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                        group_modes<0, 3>(current_s_a),
                        group_modes<0, 3>(t_cg_a));
      auto [reuse_t_b_g_b, reuse_t_b_s_b] =
          tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                        group_modes<0, 3>(current_s_b),
                        group_modes<0, 3>(t_cg_b));
      if (mma_warp && elected_thread) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier[stage],
                                            transaction_bytes);
        copy(tma_atom_a.with(storage.tma_barrier[stage]),
             reuse_t_a_g_a(_, reuse_tile), reuse_t_a_s_a);
        copy(tma_atom_b.with(storage.tma_barrier[stage]),
             reuse_t_b_g_b(_, reuse_tile), reuse_t_b_s_b);
      }
    }
  }

  for (int stage = 0;
       stage < min(k_tile_count, SharedStorage::kStages); ++stage) {
    cute::wait_barrier(storage.mma_barrier[stage], mma_phase[stage]);
    mma_phase[stage] ^= 1;
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

template <int TileN>
class Tc4RunnerImpl {
 public:
  Tc4RunnerImpl(const half* a, const half* b, float* d, int m, int n,
                int k) {
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
  static constexpr int kTileN = TileN;
  static constexpr int kTileK = 128;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

class Tc4OneSmRunner {
 public:
  Tc4OneSmRunner(const half* a, const half* b, float* d, int m, int n,
                 int k) {
    if (n >= 1024 && n % 256 == 0) {
      auto implementation =
          std::make_shared<Tc4RunnerImpl<256>>(a, b, d, m, n, k);
      launch_ = [implementation]() { implementation->launch(); };
    } else {
      auto implementation =
          std::make_shared<Tc4RunnerImpl<128>>(a, b, d, m, n, k);
      launch_ = [implementation]() { implementation->launch(); };
    }
  }

  void launch() { launch_(); }

 private:
  std::function<void()> launch_;
};

template <class TypeA, class TypeB, class TypeD, class ASmemLayout,
          class BSmemLayout, class DSmemLayout>
struct Tc4TwoSmSharedStorage {
  static constexpr int kStages = 3;
  static constexpr int kAStageElements = cute::cosize_v<ASmemLayout>;
  static constexpr int kBStageElements = cute::cosize_v<BSmemLayout>;

  union alignas(128) {
    struct alignas(128) {
      alignas(128)
          cute::ArrayEngine<TypeA, kStages * kAStageElements> a;
      alignas(128)
          cute::ArrayEngine<TypeB, kStages * kBStageElements> b;
    } mainloop;
    alignas(128)
        cute::ArrayEngine<TypeD, cute::cosize_v<DSmemLayout>> d;
  } tensors;
  alignas(16) cute::uint64_t mma_barrier[kStages];
  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a(int stage) {
    return make_tensor(
        make_smem_ptr(tensors.mainloop.a.begin() +
                      stage * kAStageElements),
        ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b(int stage) {
    return make_tensor(
        make_smem_ptr(tensors.mainloop.b.begin() +
                      stage * kBStageElements),
        BSmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_d() {
    return make_tensor(make_smem_ptr(tensors.d.begin()), DSmemLayout{});
  }
};

template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler, class TiledMma, class ClusterShape,
          class EpiTiler, class TmaAtomA, class TmaAtomB, class TmaAtomD>
__global__ void tc4_2sm_tma_tcgen05_gemm_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, EpiTiler epi_tiler, TiledMma tiled_mma,
    ClusterShape cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b,
    CUTE_GRID_CONSTANT TmaAtomD const tma_atom_d) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename TiledMma::AtomThrID{}));
  auto mma_coord_vmnk =
      make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                 blockIdx.x / size<0>(cluster_layout_vmnk), blockIdx.y, _);
  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);

  Tensor g_a =
      local_tile(matrix_a, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor g_b =
      local_tile(matrix_b, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor g_c =
      local_tile(matrix_c, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor g_d =
      local_tile(matrix_d, mma_tiler, mma_coord, Step<_1, _1, X>{});

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor t_cg_a = cta_mma.partition_A(g_a);
  Tensor t_cg_b = cta_mma.partition_B(g_b);
  Tensor t_cg_c = cta_mma.partition_C(g_c);
  Tensor t_cg_d = cta_mma.partition_C(g_d);
  Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool mma_warp = threadIdx.x < kWarpSize;
  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (mma_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }
  __syncthreads();
  t_ct_acc.data() = storage.tmem_base_ptr;

  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
      static_cast<int>(cute::block_rank_in_cluster()));
  const bool leader_cta =
      get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  auto [t_a_g_a, t_a_s_a] = tma_partition(
      tma_atom_a, get<2>(cta_in_cluster_coord_vmnk),
      make_layout(size<2>(cluster_layout_vmnk)),
      group_modes<0, 3>(storage.tensor_a(0)),
      group_modes<0, 3>(t_cg_a));
  auto [t_b_g_b, t_b_s_b] = tma_partition(
      tma_atom_b, get<1>(cta_in_cluster_coord_vmnk),
      make_layout(size<1>(cluster_layout_vmnk)),
      group_modes<0, 3>(storage.tensor_b(0)),
      group_modes<0, 3>(t_cg_b));

  const uint16_t tma_mask_a = create_tma_multicast_mask<2>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  const uint16_t tma_mask_b = create_tma_multicast_mask<1>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  const uint16_t mma_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk);
  const int transaction_bytes =
      size<0>(cluster_layout_vmnk) *
      (sizeof(make_tensor_like(t_a_s_a)) +
       sizeof(make_tensor_like(t_b_s_b)));

  if (mma_warp && elected_thread) {
    const int participants = size<1>(cluster_layout_vmnk) +
                             size<2>(cluster_layout_vmnk) - 1;
    for (int stage = 0; stage < SharedStorage::kStages; ++stage) {
      cute::initialize_barrier(storage.mma_barrier[stage], participants);
      cute::initialize_barrier(storage.tma_barrier[stage], 1);
    }
  }
  int mma_phase[SharedStorage::kStages] = {};
  int tma_phase[SharedStorage::kStages] = {};
  cute::cluster_sync();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  const int k_tile_count = size<3>(t_cg_a);

  for (int load_tile = 0;
       load_tile < min(k_tile_count, SharedStorage::kStages); ++load_tile) {
    const int load_stage = load_tile % SharedStorage::kStages;
    Tensor load_s_a = storage.tensor_a(load_stage);
    Tensor load_s_b = storage.tensor_b(load_stage);
    auto [load_t_a_g_a, load_t_a_s_a] = tma_partition(
        tma_atom_a, get<2>(cta_in_cluster_coord_vmnk),
        make_layout(size<2>(cluster_layout_vmnk)),
        group_modes<0, 3>(load_s_a), group_modes<0, 3>(t_cg_a));
    auto [load_t_b_g_b, load_t_b_s_b] = tma_partition(
        tma_atom_b, get<1>(cta_in_cluster_coord_vmnk),
        make_layout(size<1>(cluster_layout_vmnk)),
        group_modes<0, 3>(load_s_b), group_modes<0, 3>(t_cg_b));
    if (mma_warp && elected_thread) {
      if (leader_cta) {
        cute::set_barrier_transaction_bytes(
            storage.tma_barrier[load_stage], transaction_bytes);
      }
      copy(tma_atom_a.with(storage.tma_barrier[load_stage], tma_mask_a),
           load_t_a_g_a(_, load_tile), load_t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier[load_stage], tma_mask_b),
           load_t_b_g_b(_, load_tile), load_t_b_s_b);
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    const int stage = k_tile % SharedStorage::kStages;
    Tensor current_s_a = storage.tensor_a(stage);
    Tensor current_s_b = storage.tensor_b(stage);
    Tensor t_cr_a = cta_mma.make_fragment_A(current_s_a);
    Tensor t_cr_b = cta_mma.make_fragment_B(current_s_b);

    if (leader_cta) {
      cute::wait_barrier(storage.tma_barrier[stage], tma_phase[stage]);
      tma_phase[stage] ^= 1;
      if (mma_warp) {
        for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
          gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
               t_ct_acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(
            &storage.mma_barrier[stage], mma_mask_c);
      }
    }

    const int reuse_tile = k_tile + SharedStorage::kStages;
    if (reuse_tile < k_tile_count) {
      cute::wait_barrier(storage.mma_barrier[stage], mma_phase[stage]);
      mma_phase[stage] ^= 1;

      auto [reuse_t_a_g_a, reuse_t_a_s_a] = tma_partition(
          tma_atom_a, get<2>(cta_in_cluster_coord_vmnk),
          make_layout(size<2>(cluster_layout_vmnk)),
          group_modes<0, 3>(current_s_a), group_modes<0, 3>(t_cg_a));
      auto [reuse_t_b_g_b, reuse_t_b_s_b] = tma_partition(
          tma_atom_b, get<1>(cta_in_cluster_coord_vmnk),
          make_layout(size<1>(cluster_layout_vmnk)),
          group_modes<0, 3>(current_s_b), group_modes<0, 3>(t_cg_b));
      if (mma_warp && elected_thread) {
        if (leader_cta) {
          cute::set_barrier_transaction_bytes(
              storage.tma_barrier[stage], transaction_bytes);
        }
        copy(tma_atom_a.with(storage.tma_barrier[stage], tma_mask_a),
             reuse_t_a_g_a(_, reuse_tile), reuse_t_a_s_a);
        copy(tma_atom_b.with(storage.tma_barrier[stage], tma_mask_b),
             reuse_t_b_g_b(_, reuse_tile), reuse_t_b_s_b);
      }
    }
  }

  for (int stage = 0;
       stage < min(k_tile_count, SharedStorage::kStages); ++stage) {
    cute::wait_barrier(storage.mma_barrier[stage], mma_phase[stage]);
    mma_phase[stage] ^= 1;
  }

  auto epi_tile = make_tile(epi_tiler);
  Tensor t_acc_epi = zipped_divide(t_ct_acc, epi_tile);
  Tensor g_d_epi = zipped_divide(t_cg_d, epi_tile);
  Tensor s_d_epi = storage.tensor_d();
  auto [t_sg_g_d, t_sg_s_d] =
      tma_partition(tma_atom_d, s_d_epi, g_d_epi);

  TiledCopy tmem_to_register = make_tmem_copy(
      SM100_TMEM_LOAD_32dp32b1x{}, t_acc_epi(_, _0{}));
  ThrCopy thread_copy = tmem_to_register.get_slice(threadIdx.x);
  Tensor t_rt_acc = thread_copy.partition_S(t_acc_epi);
  Tensor t_rs_d = thread_copy.partition_D(s_d_epi);
  Tensor t_rr_d = make_fragment_like(t_rs_d);

  CUTE_UNROLL
  for (int epi_tile_idx = 0; epi_tile_idx < size<2>(t_rt_acc);
       ++epi_tile_idx) {
    copy(tmem_to_register, t_rt_acc(_, _, epi_tile_idx), t_rr_d);
    copy_aligned(t_rr_d, t_rs_d);
    tma_store_fence();
    __syncthreads();
    if (mma_warp && elected_thread) {
      copy(tma_atom_d, t_sg_s_d, t_sg_g_d(_, epi_tile_idx));
      tma_store_arrive();
      tma_store_wait<0>();
    }
    __syncthreads();
  }

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
  (void)epi_tiler;
  (void)tiled_mma;
  (void)cluster_shape;
  (void)tma_atom_a;
  (void)tma_atom_b;
  (void)tma_atom_d;
#endif
}

template <int ClusterN>
class Tc4TwoSmRunner {
 public:
  Tc4TwoSmRunner(const half* a, const half* b, float* d, int m, int n,
                 int k) {
    if (m % kTileM != 0 || n % kClusterTileN != 0 ||
        k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc4 custom 2-SM TMA TCGen05 requires M a multiple of "
                   "256, N a multiple of %d, and K a multiple of 128\n",
                   kClusterTileN);
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
        SM100_MMA_F16BF16_2x1SM_SS<
            cutlass::half_t, cutlass::half_t, float, kTileM, kTileN,
            UMMA::Major::K, UMMA::Major::MN>{});
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
    auto mma_shape_d = partition_shape_C(
        tiled_mma, make_shape(Int<kTileM>{}, Int<kTileN>{}));
    auto epi_tiler =
        make_tile(size<0, 0>(mma_shape_d), size<0, 1>(mma_shape_d) / Int<2>{});
    auto smem_layout_d_mn = tile_to_shape(
        UMMA::Layout_K_SW128_Atom<float>{},
        make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
    auto smem_layout_d = group<0, 2>(smem_layout_d_mn);
    using Storage = Tc4TwoSmSharedStorage<
        cutlass::half_t, cutlass::half_t, float, decltype(smem_layout_a),
        decltype(smem_layout_b), decltype(smem_layout_d)>;

    auto cluster_shape = make_shape(Int<2>{}, Int<ClusterN>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(
        make_layout(cluster_shape),
        make_tile(typename decltype(tiled_mma)::AtomThrID{}));
    Copy_Atom tma_atom_a = make_tma_atom_A_sm100(
        SM100_TMA_2SM_LOAD_MULTICAST{}, matrix_a, smem_layout_a, mma_tiler,
        tiled_mma, cluster_layout_vmnk);
    Copy_Atom tma_atom_b = make_tma_atom_B_sm100(
        SM100_TMA_2SM_LOAD_MULTICAST{}, matrix_b, smem_layout_b, mma_tiler,
        tiled_mma, cluster_layout_vmnk);
    auto matrix_a_tma = tma_atom_a.get_tma_tensor(shape(matrix_a));
    auto matrix_b_tma = tma_atom_b.get_tma_tensor(shape(matrix_b));
    Copy_Atom tma_atom_d = make_tma_atom(
        SM90_TMA_STORE{}, matrix_d, smem_layout_d, epi_tiler);
    auto matrix_d_tma = tma_atom_d.get_tma_tensor(shape(matrix_d));

    auto* kernel = &tc4_2sm_tma_tcgen05_gemm_kernel<
        Storage, decltype(matrix_a_tma), decltype(matrix_b_tma),
        decltype(matrix_c), decltype(matrix_d_tma), decltype(mma_tiler),
        decltype(tiled_mma), decltype(cluster_shape), decltype(epi_tiler),
        decltype(tma_atom_a), decltype(tma_atom_b), decltype(tma_atom_d)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc4 2-SM)");

    const dim3 cluster(size<0>(cluster_shape), size<1>(cluster_shape),
                       size<2>(cluster_shape));
    const dim3 grid(
        size(ceil_div(m, kTileM * size<1>(cluster_layout_vmnk))) *
            cluster.x,
        size(ceil_div(n, kTileN * size<2>(cluster_layout_vmnk))) *
            cluster.y);
    launch_ = [=]() mutable {
      cutlass::ClusterLaunchParams params = {
          grid, dim3(128), cluster, static_cast<int>(sizeof(Storage))};
      const cutlass::Status status = cutlass::launch_kernel_on_cluster(
          params, reinterpret_cast<void const*>(kernel), matrix_a_tma,
          matrix_b_tma, matrix_c, matrix_d_tma, mma_tiler, epi_tiler,
          tiled_mma, cluster_shape, tma_atom_a, tma_atom_b, tma_atom_d);
      if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "tc4 custom 2-SM launch failed: %s\n",
                     cutlassGetStatusString(status));
        std::abort();
      }
      check_cuda(cudaGetLastError(),
                 "tc4_2sm_tma_tcgen05_gemm_kernel launch");
    };
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 256;
  static constexpr int kTileN = 256;
  static constexpr int kTileK = 128;
  static constexpr int kClusterTileN = kTileN * ClusterN;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

class Tc4Runner {
 public:
  Tc4Runner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m >= 1024 && n >= 1024 && m % 256 == 0 && n % 512 == 0 &&
        k % 64 == 0) {
      if (n >= 4096) {
        auto implementation =
            std::make_shared<Tc4TwoSmRunner<2>>(a, b, d, m, n, k);
        launch_ = [implementation]() { implementation->launch(); };
      } else {
        auto implementation =
            std::make_shared<Tc4TwoSmRunner<1>>(a, b, d, m, n, k);
        launch_ = [implementation]() { implementation->launch(); };
      }
    } else {
      auto implementation =
          std::make_shared<Tc4OneSmRunner>(a, b, d, m, n, k);
      launch_ = [implementation]() { implementation->launch(); };
    }
  }

  void launch() { launch_(); }

 private:
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

template <class TypeA, class TypeB, class TypeD, class ASmemLayout,
          class BSmemLayout, class DSmemLayout>
struct Tc5SharedStorage {
  static constexpr int kStages = 2;
  static constexpr int kAStageElements = cute::cosize_v<ASmemLayout>;
  static constexpr int kBStageElements = cute::cosize_v<BSmemLayout>;

  union alignas(128) {
    struct alignas(128) {
      alignas(128)
          cute::ArrayEngine<TypeA, kStages * kAStageElements> a;
      alignas(128)
          cute::ArrayEngine<TypeB, kStages * kBStageElements> b;
    } mainloop;
    alignas(128)
        cute::ArrayEngine<TypeD, cute::cosize_v<DSmemLayout>> d;
  } tensors;
  alignas(16) cute::uint64_t mma_barrier[kStages];
  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint64_t clc_barrier;
  alignas(16) Tc5ClcResponse clc_response;
  alignas(16) cute::uint32_t tmem_base_ptr;
  Tc5WorkTile work_tile;

  CUTE_DEVICE constexpr auto tensor_a(int stage) {
    return make_tensor(
        make_smem_ptr(tensors.mainloop.a.begin() +
                      stage * kAStageElements),
        ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b(int stage) {
    return make_tensor(
        make_smem_ptr(tensors.mainloop.b.begin() +
                      stage * kBStageElements),
        BSmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_d() {
    return make_tensor(make_smem_ptr(tensors.d.begin()), DSmemLayout{});
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
          class EpiTiler, class TmaAtomA, class TmaAtomB, class TmaAtomD>
__global__ void tc5_persistent_tma_tcgen05_gemm_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, EpiTiler epi_tiler, TiledMma tiled_mma,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b,
    CUTE_GRID_CONSTANT TmaAtomD const tma_atom_d, int tiles_m, int tiles_n) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool mma_warp = threadIdx.x < kWarpSize;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (mma_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }

  if (mma_warp && elected_thread) {
    for (int stage = 0; stage < SharedStorage::kStages; ++stage) {
      cute::initialize_barrier(storage.mma_barrier[stage], 1);
      cute::initialize_barrier(storage.tma_barrier[stage], 1);
    }
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

  int mma_phase[SharedStorage::kStages] = {0, 0};
  int tma_phase[SharedStorage::kStages] = {0, 0};
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
    Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);
    t_ct_acc.data() = storage.tmem_base_ptr;

    auto [t_a_g_a, t_a_s_a] =
        tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(storage.tensor_a(0)),
                      group_modes<0, 3>(t_cg_a));
    auto [t_b_g_b, t_b_s_b] =
        tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(storage.tensor_b(0)),
                      group_modes<0, 3>(t_cg_b));
    const int transaction_bytes =
        sizeof(make_tensor_like(t_a_s_a)) + sizeof(make_tensor_like(t_b_s_b));

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    const int k_tile_count = size<3>(t_cg_a);

    if (mma_warp && elected_thread) {
      cute::set_barrier_transaction_bytes(storage.tma_barrier[0],
                                          transaction_bytes);
      copy(tma_atom_a.with(storage.tma_barrier[0]), t_a_g_a(_, 0),
           t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier[0]), t_b_g_b(_, 0),
           t_b_s_b);
    }

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      const int stage = k_tile % SharedStorage::kStages;

      cute::wait_barrier(storage.tma_barrier[stage], tma_phase[stage]);
      tma_phase[stage] ^= 1;

      Tensor current_s_a = storage.tensor_a(stage);
      Tensor current_s_b = storage.tensor_b(stage);
      Tensor t_cr_a = cta_mma.make_fragment_A(current_s_a);
      Tensor t_cr_b = cta_mma.make_fragment_B(current_s_b);

      const int next_tile = k_tile + 1;
      if (next_tile < k_tile_count) {
        const int next_stage = next_tile % SharedStorage::kStages;
        Tensor next_s_a = storage.tensor_a(next_stage);
        Tensor next_s_b = storage.tensor_b(next_stage);
        auto [next_t_a_g_a, next_t_a_s_a] =
            tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                          group_modes<0, 3>(next_s_a),
                          group_modes<0, 3>(t_cg_a));
        auto [next_t_b_g_b, next_t_b_s_b] =
            tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                          group_modes<0, 3>(next_s_b),
                          group_modes<0, 3>(t_cg_b));
        if (mma_warp && elected_thread) {
          cute::set_barrier_transaction_bytes(storage.tma_barrier[next_stage],
                                              transaction_bytes);
          copy(tma_atom_a.with(storage.tma_barrier[next_stage]),
               next_t_a_g_a(_, next_tile), next_t_a_s_a);
          copy(tma_atom_b.with(storage.tma_barrier[next_stage]),
               next_t_b_g_b(_, next_tile), next_t_b_s_b);
        }
      }

      if (mma_warp) {
        for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
          gemm(tiled_mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block),
               t_ct_acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&storage.mma_barrier[stage]);
      }
      cute::wait_barrier(storage.mma_barrier[stage], mma_phase[stage]);
      mma_phase[stage] ^= 1;
    }

    auto epi_tile = make_tile(epi_tiler);
    Tensor t_acc_epi = zipped_divide(t_ct_acc, epi_tile);
    Tensor g_d_epi = zipped_divide(t_cg_d, epi_tile);
    Tensor s_d_epi = storage.tensor_d();
    auto [t_sg_g_d, t_sg_s_d] =
        tma_partition(tma_atom_d, s_d_epi, g_d_epi);

    TiledCopy tmem_to_register = make_tmem_copy(
        SM100_TMEM_LOAD_32dp32b1x{}, t_acc_epi(_, _0{}));
    ThrCopy thread_copy = tmem_to_register.get_slice(threadIdx.x);
    Tensor t_rt_acc = thread_copy.partition_S(t_acc_epi);
    Tensor t_rs_d = thread_copy.partition_D(s_d_epi);
    Tensor t_rr_d = make_fragment_like(t_rs_d);

    CUTE_UNROLL
    for (int epi_tile_idx = 0; epi_tile_idx < size<2>(t_rt_acc);
         ++epi_tile_idx) {
      copy(tmem_to_register, t_rt_acc(_, _, epi_tile_idx), t_rr_d);
      copy_aligned(t_rr_d, t_rs_d);
      tma_store_fence();
      __syncthreads();
      if (mma_warp && elected_thread) {
        copy(tma_atom_d, t_sg_s_d, t_sg_g_d(_, epi_tile_idx));
        tma_store_arrive();
        tma_store_wait<0>();
      }
      __syncthreads();
    }

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
  (void)epi_tiler;
  (void)tiled_mma;
  (void)tma_atom_a;
  (void)tma_atom_b;
  (void)tma_atom_d;
  (void)tiles_m;
  (void)tiles_n;
#endif
}

template <bool UseClc, int TileN>
class Tc5RunnerImpl {
 public:
  Tc5RunnerImpl(const half* a, const half* b, float* d, int m, int n,
                int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc5 custom persistent TCGen05 requires M,N multiples of "
                   "128 and K a multiple of 128\n");
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
    auto mma_shape_d = partition_shape_C(
        tiled_mma, make_shape(Int<kTileM>{}, Int<kTileN>{}));
    auto epi_tiler =
        make_tile(size<0, 0>(mma_shape_d), size<0, 1>(mma_shape_d) / Int<2>{});
    auto smem_layout_d_mn = tile_to_shape(
        UMMA::Layout_K_SW128_Atom<float>{},
        make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
    auto smem_layout_d = group<0, 2>(smem_layout_d_mn);
    using Storage = Tc5SharedStorage<
        cutlass::half_t, cutlass::half_t, float, decltype(smem_layout_a),
        decltype(smem_layout_b), decltype(smem_layout_d)>;

    Copy_Atom tma_atom_a =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_a, smem_layout_a,
                      select<0, 2>(mma_tiler));
    Copy_Atom tma_atom_b =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_b, smem_layout_b,
                      select<1, 2>(mma_tiler));
    auto matrix_a_tma = tma_atom_a.get_tma_tensor(shape(matrix_a));
    auto matrix_b_tma = tma_atom_b.get_tma_tensor(shape(matrix_b));
    Copy_Atom tma_atom_d = make_tma_atom(
        SM90_TMA_STORE{}, matrix_d, smem_layout_d, epi_tiler);
    auto matrix_d_tma = tma_atom_d.get_tma_tensor(shape(matrix_d));

    auto* kernel =
        &tc5_persistent_tma_tcgen05_gemm_kernel<
            UseClc, Storage, decltype(matrix_a_tma),
            decltype(matrix_b_tma), decltype(matrix_c),
            decltype(matrix_d_tma), decltype(mma_tiler),
            decltype(tiled_mma), decltype(epi_tiler), decltype(tma_atom_a),
            decltype(tma_atom_b), decltype(tma_atom_d)>;
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
          matrix_a_tma, matrix_b_tma, matrix_c, matrix_d_tma, mma_tiler,
          epi_tiler, tiled_mma, tma_atom_a, tma_atom_b, tma_atom_d,
          tiles_m, tiles_n);
      check_cuda(cudaGetLastError(),
                 "tc5_persistent_tma_tcgen05_gemm_kernel launch");
    };
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = TileN;
  static constexpr int kTileK = 128;

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

template <bool UseClc>
class Tc5Runner {
 public:
  Tc5Runner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (n >= 1024 && n % 256 == 0) {
      auto implementation =
          std::make_shared<Tc5RunnerImpl<UseClc, 256>>(a, b, d, m, n, k);
      launch_ = [implementation]() { implementation->launch(); };
    } else {
      auto implementation =
          std::make_shared<Tc5RunnerImpl<UseClc, 128>>(a, b, d, m, n, k);
      launch_ = [implementation]() { implementation->launch(); };
    }
  }

  void launch() { launch_(); }

 private:
  std::function<void()> launch_;
};

using Tc5StaticRunner = Tc5Runner<false>;
using Tc5ClcRunner = Tc5Runner<true>;

}  // namespace gemm_sm110::custom_backend
