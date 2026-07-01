#pragma once

// Stage 4a: tc3 plus warp-level producer/consumer specialization.
//
// warp 0: TMEM allocation and TCGen05 MMA consumer
// warp 1: TMA producer
// all warps: epilogue/readback
//
// Tile shape, two-stage buffers, SW128 layouts, and direct epilogue are kept
// identical to tc3.

#include "../custom_sm110_gemm.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace gemm_sm110::backends {

using namespace cute;

template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler, class TiledMma, class TmaAtomA,
          class TmaAtomB>
__global__ void tc4a_warp_specialized_kernel(
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

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const bool consumer_warp = warp == 0;
  const bool producer_warp = warp == 1;
  const uint32_t elected_thread = cute::elect_one_sync();

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (consumer_warp) {
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

  if (consumer_warp && elected_thread) {
    for (int stage = 0; stage < SharedStorage::kStages; ++stage) {
      cute::initialize_barrier(storage.mma_barrier[stage], 1);
      cute::initialize_barrier(storage.tma_barrier[stage], 1);
    }
  }
  int mma_phase[SharedStorage::kStages] = {};
  int tma_phase[SharedStorage::kStages] = {};
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  const int k_tile_count = size<3>(t_cg_a);

  for (int load_tile = 0;
       load_tile < min(k_tile_count, SharedStorage::kStages); ++load_tile) {
    const int stage = load_tile % SharedStorage::kStages;
    Tensor load_s_a = storage.tensor_a(stage);
    Tensor load_s_b = storage.tensor_b(stage);
    auto [load_t_a_g_a, load_t_a_s_a] =
        tma_partition(tma_atom_a, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(load_s_a),
                      group_modes<0, 3>(t_cg_a));
    auto [load_t_b_g_b, load_t_b_s_b] =
        tma_partition(tma_atom_b, Int<0>{}, Layout<_1>{},
                      group_modes<0, 3>(load_s_b),
                      group_modes<0, 3>(t_cg_b));
    if (producer_warp && elected_thread) {
      cute::set_barrier_transaction_bytes(storage.tma_barrier[stage],
                                          transaction_bytes);
      copy(tma_atom_a.with(storage.tma_barrier[stage]),
           load_t_a_g_a(_, load_tile), load_t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier[stage]),
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

    if (consumer_warp) {
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
      if (producer_warp && elected_thread) {
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
  if (consumer_warp) {
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

class Tc4aRunner {
 public:
  Tc4aRunner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc4a requires M,N multiples of 128 and K a multiple "
                   "of 64\n");
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
        SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float,
                             kTileM, kTileN, UMMA::Major::K,
                             UMMA::Major::MN>{});
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
    using Storage = custom_backend::Tc4SharedStorage<
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

    auto* kernel = &tc4a_warp_specialized_kernel<
        Storage, decltype(matrix_a_tma), decltype(matrix_b_tma),
        decltype(matrix_c), decltype(matrix_d), decltype(mma_tiler),
        decltype(tiled_mma), decltype(tma_atom_a), decltype(tma_atom_b)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc4a)");

    const dim3 grid(m / kTileM, n / kTileN);
    launch_ = [=]() mutable {
      kernel<<<grid, 128, sizeof(Storage)>>>(
          matrix_a_tma, matrix_b_tma, matrix_c, matrix_d, mma_tiler,
          tiled_mma, tma_atom_a, tma_atom_b);
      check_cuda(cudaGetLastError(), "tc4a_warp_specialized_kernel launch");
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

}  // namespace gemm_sm110::backends
