#pragma once

// Stage 4b/4c paired 2-SM experiment.
//
// tc4b: 2-SM cluster MMA, TMA and MMA issued by warp 0.
// tc4c: identical kernel, except warp 1 is the dedicated TMA producer.
//
// Both use a fixed 256x128x64 MMA tile, two SW128 stages, a 2x1 cluster, and
// the same direct TMEM->register->GMEM epilogue.

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace gemm_sm110::backends {

using namespace cute;

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc4bcSharedStorage {
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

template <bool WarpSpecialized, class SharedStorage, class ATensor,
          class BTensor, class CTensor, class DTensor, class MmaTiler,
          class TiledMma, class ClusterShape, class TmaAtomA, class TmaAtomB>
__global__ void tc4bc_2sm_cluster_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, TiledMma tiled_mma, ClusterShape cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b) {
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

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const bool consumer_warp = warp == 0;
  const bool producer_warp = WarpSpecialized ? warp == 1 : warp == 0;
  const uint32_t elected_thread = cute::elect_one_sync();

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (consumer_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }
  __syncthreads();
  t_ct_acc.data() = storage.tmem_base_ptr;

  auto cta_coord = cluster_layout_vmnk.get_flat_coord(
      static_cast<int>(cute::block_rank_in_cluster()));
  const bool leader_cta = get<0>(cta_coord) == Int<0>{};
  const uint16_t tma_mask_a =
      create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord);
  const uint16_t tma_mask_b =
      create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord);
  const uint16_t mma_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk, cta_coord) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk, cta_coord);

  auto [t_a_g_a, t_a_s_a] = tma_partition(
      tma_atom_a, get<2>(cta_coord),
      make_layout(size<2>(cluster_layout_vmnk)),
      group_modes<0, 3>(storage.tensor_a(0)),
      group_modes<0, 3>(t_cg_a));
  auto [t_b_g_b, t_b_s_b] = tma_partition(
      tma_atom_b, get<1>(cta_coord),
      make_layout(size<1>(cluster_layout_vmnk)),
      group_modes<0, 3>(storage.tensor_b(0)),
      group_modes<0, 3>(t_cg_b));
  const int transaction_bytes =
      size<0>(cluster_layout_vmnk) *
      (sizeof(make_tensor_like(t_a_s_a)) +
       sizeof(make_tensor_like(t_b_s_b)));

  if (consumer_warp && elected_thread) {
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
    const int stage = load_tile % SharedStorage::kStages;
    Tensor load_s_a = storage.tensor_a(stage);
    Tensor load_s_b = storage.tensor_b(stage);
    auto [load_t_a_g_a, load_t_a_s_a] = tma_partition(
        tma_atom_a, get<2>(cta_coord),
        make_layout(size<2>(cluster_layout_vmnk)),
        group_modes<0, 3>(load_s_a), group_modes<0, 3>(t_cg_a));
    auto [load_t_b_g_b, load_t_b_s_b] = tma_partition(
        tma_atom_b, get<1>(cta_coord),
        make_layout(size<1>(cluster_layout_vmnk)),
        group_modes<0, 3>(load_s_b), group_modes<0, 3>(t_cg_b));
    if (producer_warp && elected_thread) {
      if (leader_cta) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier[stage],
                                            transaction_bytes);
      }
      copy(tma_atom_a.with(storage.tma_barrier[stage], tma_mask_a),
           load_t_a_g_a(_, load_tile), load_t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier[stage], tma_mask_b),
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
      if (consumer_warp) {
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
          tma_atom_a, get<2>(cta_coord),
          make_layout(size<2>(cluster_layout_vmnk)),
          group_modes<0, 3>(current_s_a), group_modes<0, 3>(t_cg_a));
      auto [reuse_t_b_g_b, reuse_t_b_s_b] = tma_partition(
          tma_atom_b, get<1>(cta_coord),
          make_layout(size<1>(cluster_layout_vmnk)),
          group_modes<0, 3>(current_s_b), group_modes<0, 3>(t_cg_b));
      if (producer_warp && elected_thread) {
        if (leader_cta) {
          cute::set_barrier_transaction_bytes(storage.tma_barrier[stage],
                                              transaction_bytes);
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
  (void)cluster_shape;
  (void)tma_atom_a;
  (void)tma_atom_b;
#endif
}

template <bool WarpSpecialized>
class Tc4bcRunner {
 public:
  Tc4bcRunner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc4b/tc4c require M a multiple of 256, N a multiple "
                   "of 128, and K a multiple of 64\n");
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
    using Storage = Tc4bcSharedStorage<
        cutlass::half_t, cutlass::half_t, decltype(smem_layout_a),
        decltype(smem_layout_b)>;

    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
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

    auto* kernel = &tc4bc_2sm_cluster_kernel<
        WarpSpecialized, Storage, decltype(matrix_a_tma),
        decltype(matrix_b_tma), decltype(matrix_c), decltype(matrix_d),
        decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
        decltype(tma_atom_a), decltype(tma_atom_b)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc4b/tc4c)");

    const dim3 cluster(2, 1, 1);
    const dim3 grid((m / kTileM) * cluster.x, n / kTileN);
    launch_ = [=]() mutable {
      cutlass::ClusterLaunchParams params = {
          grid, dim3(128), cluster, static_cast<int>(sizeof(Storage))};
      const cutlass::Status status = cutlass::launch_kernel_on_cluster(
          params, reinterpret_cast<void const*>(kernel), matrix_a_tma,
          matrix_b_tma, matrix_c, matrix_d, mma_tiler, tiled_mma,
          cluster_shape, tma_atom_a, tma_atom_b);
      if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "tc4b/tc4c cluster launch failed: %s\n",
                     cutlassGetStatusString(status));
        std::abort();
      }
      check_cuda(cudaGetLastError(), "tc4bc_2sm_cluster_kernel launch");
    };
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 256;
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

using Tc4bRunner = Tc4bcRunner<false>;
using Tc4cRunner = Tc4bcRunner<true>;

}  // namespace gemm_sm110::backends
