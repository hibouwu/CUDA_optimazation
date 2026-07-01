#pragma once

// Stage 5 persistent schedulers built on the tc4c computation path.
//
// Shared computation path:
//   - fixed 256x128x64 2-SM MMA and 2x1 cluster
//   - two SW128 TMA stages
//   - warp 0 MMA consumer, warp 1 TMA producer
//   - two alternating TMEM accumulator slots
//   - warpgroup 1 (warps 4..7) exclusively performs epilogue/readback
//
// Scheduler variable:
//   UseClc=false -> tc5a finite persistent workers + static grid stride
//   UseClc=true  -> tc5b full launch grid + hardware CLC cancellation

#include "tc4bc_cluster.cuh"

#include <cutlass/detail/sm100_tmem_helper.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace gemm_sm110::backends {

struct alignas(16) Tc5ClcResponse {
  uint32_t data[4];
};

struct Tc5WorkTile {
  int m;
  int n;
  int valid;
};

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc5PersistentSharedStorage {
  static constexpr int kStages = 2;
  static constexpr int kAStageElements = cute::cosize_v<ASmemLayout>;
  static constexpr int kBStageElements = cute::cosize_v<BSmemLayout>;

  alignas(128)
      cute::ArrayEngine<TypeA, kStages * kAStageElements> a;
  alignas(128)
      cute::ArrayEngine<TypeB, kStages * kBStageElements> b;
  alignas(16) cute::uint64_t mma_barrier[kStages];
  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint64_t clc_barrier;
  alignas(16) Tc5ClcResponse clc_response;
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
CUTE_DEVICE Tc5WorkTile tc5_read_clc_response(Storage& storage,
                                               int cluster_rank_x) {
  Tc5WorkTile tile{-1, -1, 0};
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  const uint32_t response_address =
      cute::cast_smem_ptr_to_uint(&storage.clc_response);
  uint32_t first_x = 0;
  uint32_t first_y = 0;
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
      : "=r"(first_x), "=r"(first_y), "=r"(valid)
      : "r"(response_address)
      : "memory");
  if (valid != 0) {
    const int virtual_x = static_cast<int>(first_x) + cluster_rank_x;
    tile.m = virtual_x / 2;
    tile.n = static_cast<int>(first_y);
    tile.valid = 1;
  }
  cutlass::arch::fence_view_async_shared();
#else
  (void)cluster_rank_x;
#endif
  return tile;
}

template <bool UseClc, class SharedStorage, class ATensor, class BTensor,
          class CTensor, class DTensor, class MmaTiler, class TiledMma,
          class ClusterShape, class TmaAtomA, class TmaAtomB>
__global__ void tc5_persistent_cluster_kernel(
    ATensor matrix_a, BTensor matrix_b, CTensor matrix_c, DTensor matrix_d,
    MmaTiler mma_tiler, TiledMma tiled_mma, ClusterShape cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b, int tiles_m, int tiles_n) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  extern __shared__ char shared_memory[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename TiledMma::AtomThrID{}));
  auto cta_coord = cluster_layout_vmnk.get_flat_coord(
      static_cast<int>(cute::block_rank_in_cluster()));
  const int cluster_rank_x = static_cast<int>(cute::block_rank_in_cluster());
  const bool leader_cta = get<0>(cta_coord) == Int<0>{};
  const uint16_t tma_mask_a =
      create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord);
  const uint16_t tma_mask_b =
      create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord);
  const uint16_t mma_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk, cta_coord) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk, cta_coord);

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const bool consumer_warp = warp == 0;
  const bool producer_warp = warp == 1;
  // TMEM load addresses encode a data-path lane selected by warp_id % 4.
  // Give the epilogue a complete, dedicated warpgroup so its physical
  // warps 4..7 map exactly to logical TMEM-copy warps 0..3.  Warps 2/3
  // intentionally remain idle: folding the epilogue into only two warps
  // would either leave half the accumulator uncovered or violate the TMEM
  // data-path ownership rule.
  const bool epilogue_warp = warp >= 4;
  const uint32_t elected_thread = cute::elect_one_sync();

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (consumer_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &storage.tmem_base_ptr);
  }

  if (consumer_warp && elected_thread) {
    const int participants = size<1>(cluster_layout_vmnk) +
                             size<2>(cluster_layout_vmnk) - 1;
    for (int stage = 0; stage < SharedStorage::kStages; ++stage) {
      cute::initialize_barrier(storage.mma_barrier[stage], participants);
      cute::initialize_barrier(storage.tma_barrier[stage], 1);
    }
    if constexpr (UseClc) {
      cute::initialize_barrier(storage.clc_barrier, 1);
    }
  }
  int mma_phase[SharedStorage::kStages] = {};
  int tma_phase[SharedStorage::kStages] = {};
  int clc_phase = 0;
  cute::cluster_sync();

  const int worker_cluster = static_cast<int>(blockIdx.x) / 2;
  const int worker_clusters = static_cast<int>(gridDim.x) / 2;
  int static_work_id = worker_cluster;
  Tc5WorkTile work_tile;
  if constexpr (UseClc) {
    work_tile = {static_cast<int>(blockIdx.x) / 2,
                 static_cast<int>(blockIdx.y), 1};
  } else {
    work_tile = {static_work_id / tiles_n, static_work_id % tiles_n,
                 static_work_id < tiles_m * tiles_n ? 1 : 0};
  }
  int accumulator_slot = 0;

  while (work_tile.valid) {
    if constexpr (UseClc) {
      // The CLC response and completion are multicast to every CTA in the
      // cluster, so every local full barrier must expect the 16-byte
      // transaction before the leader submits the query.
      if (producer_warp && elected_thread) {
        cute::set_barrier_transaction_bytes(storage.clc_barrier,
                                            sizeof(Tc5ClcResponse));
      }
      cute::cluster_sync();
      if (leader_cta && producer_warp && elected_thread) {
        tc5_issue_clc_query(storage);
      }
    }

    auto tile_coord = make_coord(work_tile.m, work_tile.n, _);
    Tensor g_a =
        local_tile(matrix_a, mma_tiler, tile_coord, Step<_1, X, _1>{});
    Tensor g_b =
        local_tile(matrix_b, mma_tiler, tile_coord, Step<X, _1, _1>{});
    Tensor g_c =
        local_tile(matrix_c, mma_tiler, tile_coord, Step<_1, _1, X>{});
    Tensor g_d =
        local_tile(matrix_d, mma_tiler, tile_coord, Step<_1, _1, X>{});

    ThrMMA cta_mma = tiled_mma.get_slice(get<0>(cta_coord));
    Tensor t_cg_a = cta_mma.partition_A(g_a);
    Tensor t_cg_b = cta_mma.partition_B(g_b);
    Tensor t_cg_c = cta_mma.partition_C(g_c);
    Tensor t_cg_d = cta_mma.partition_C(g_d);
    Tensor t_ct_acc = cta_mma.make_fragment_C(t_cg_c);
    const uint32_t accumulator_columns =
        cutlass::detail::find_tmem_tensor_col_offset(t_ct_acc);
    t_ct_acc.data() =
        storage.tmem_base_ptr + accumulator_slot * accumulator_columns;

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

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    const int k_tile_count = size<3>(t_cg_a);
    if (consumer_warp || producer_warp) {
      for (int load_tile = 0;
           load_tile < min(k_tile_count, SharedStorage::kStages);
           ++load_tile) {
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

        if (leader_cta && consumer_warp) {
          cute::wait_barrier(storage.tma_barrier[stage],
                             tma_phase[stage]);
          tma_phase[stage] ^= 1;
          for (int k_block = 0; k_block < size<2>(t_cr_a); ++k_block) {
            gemm(tiled_mma, t_cr_a(_, _, k_block),
                 t_cr_b(_, _, k_block), t_ct_acc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive_multicast_2x1SM(
              &storage.mma_barrier[stage], mma_mask_c);
        }

        const int reuse_tile = k_tile + SharedStorage::kStages;
        if (reuse_tile < k_tile_count) {
          cute::wait_barrier(storage.mma_barrier[stage],
                             mma_phase[stage]);
          mma_phase[stage] ^= 1;
          auto [reuse_t_a_g_a, reuse_t_a_s_a] = tma_partition(
              tma_atom_a, get<2>(cta_coord),
              make_layout(size<2>(cluster_layout_vmnk)),
              group_modes<0, 3>(current_s_a),
              group_modes<0, 3>(t_cg_a));
          auto [reuse_t_b_g_b, reuse_t_b_s_b] = tma_partition(
              tma_atom_b, get<1>(cta_coord),
              make_layout(size<1>(cluster_layout_vmnk)),
              group_modes<0, 3>(current_s_b),
              group_modes<0, 3>(t_cg_b));
          if (producer_warp && elected_thread) {
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
    }

    __syncthreads();
    if (epilogue_warp) {
      TiledCopy tmem_to_register =
          make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, t_ct_acc);
      const int epilogue_thread = static_cast<int>(threadIdx.x) - 128;
      ThrCopy thread_copy = tmem_to_register.get_slice(epilogue_thread);
      Tensor t_dt_acc = thread_copy.partition_S(t_ct_acc);
      Tensor t_dg_d = thread_copy.partition_D(t_cg_d);
      using Accumulator = typename decltype(t_ct_acc)::value_type;
      Tensor t_dr_acc = make_tensor<Accumulator>(shape(t_dg_d));
      copy(tmem_to_register, t_dt_acc, t_dr_acc);
      copy(t_dr_acc, t_dg_d);
    }
    __syncthreads();

    accumulator_slot ^= 1;
    if constexpr (UseClc) {
      cute::wait_barrier(storage.clc_barrier, clc_phase);
      clc_phase ^= 1;
      work_tile = tc5_read_clc_response(storage, cluster_rank_x);
    } else {
      static_work_id += worker_clusters;
      work_tile = {static_work_id / tiles_n, static_work_id % tiles_n,
                   static_work_id < tiles_m * tiles_n ? 1 : 0};
    }
  }

  cute::cluster_sync();
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
                   "tc5a/tc5b require M a multiple of 256, N a multiple "
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
    using Storage = Tc5PersistentSharedStorage<
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

    auto* kernel = &tc5_persistent_cluster_kernel<
        UseClc, Storage, decltype(matrix_a_tma), decltype(matrix_b_tma),
        decltype(matrix_c), decltype(matrix_d), decltype(mma_tiler),
        decltype(tiled_mma), decltype(cluster_shape), decltype(tma_atom_a),
        decltype(tma_atom_b)>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc5)");

    const int tiles_m = m / kTileM;
    const int tiles_n = n / kTileN;
    dim3 grid;
    if constexpr (UseClc) {
      grid = dim3(tiles_m * 2, tiles_n);
    } else {
      int device = 0;
      cudaDeviceProp properties{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice(tc5)");
      check_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties(tc5)");
      const int resident_clusters =
          max(1, properties.multiProcessorCount / 2);
      const int worker_clusters =
          min(tiles_m * tiles_n, resident_clusters);
      grid = dim3(worker_clusters * 2, 1);
    }

    const dim3 cluster(2, 1, 1);
    launch_ = [=]() mutable {
      cutlass::ClusterLaunchParams params = {
          grid, dim3(256), cluster, static_cast<int>(sizeof(Storage))};
      const cutlass::Status status = cutlass::launch_kernel_on_cluster(
          params, reinterpret_cast<void const*>(kernel), matrix_a_tma,
          matrix_b_tma, matrix_c, matrix_d, mma_tiler, tiled_mma,
          cluster_shape, tma_atom_a, tma_atom_b, tiles_m, tiles_n);
      if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "tc5 cluster launch failed: %s\n",
                     cutlassGetStatusString(status));
        std::abort();
      }
      check_cuda(cudaGetLastError(),
                 "tc5_persistent_cluster_kernel launch");
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

using Tc5aRunner = Tc5Runner<false>;
using Tc5bRunner = Tc5Runner<true>;

}  // namespace gemm_sm110::backends
