#pragma once

// Stage 1/2 paired experiment:
//   Rank3=false, Sw128=false -> tc1a (2D TMA, linear/INTER SMEM)
//   Rank3=true,  Sw128=false -> tc1b (3D TMA, linear/INTER SMEM)
//   Rank3=false, Sw128=true  -> tc2a (2D TMA, SW128 SMEM)
//   Rank3=true,  Sw128=true  -> tc2b (3D TMA, SW128 SMEM)
//
// All four variants share the same tile, CTA, barrier order, TCGen05 MMA, and
// epilogue. Rank3 and Sw128 are the only compile-time experimental variables.

#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace gemm_sm110::backends {

using namespace cute;

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct Tc12SharedStorage {
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
__global__ void tc12_minimal_tma_tcgen05_kernel(
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
  const bool mma_warp = threadIdx.x < 32;
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

template <bool Rank3, bool Sw128>
class Tc12Runner {
 public:
  Tc12Runner(const half* a, const half* b, float* d, int m, int n, int k) {
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
      std::fprintf(stderr,
                   "tc1/tc2 require M,N multiples of 128 and K a multiple "
                   "of 64\n");
      std::abort();
    }

    const auto* typed_a = reinterpret_cast<const cutlass::half_t*>(a);
    const auto* typed_b = reinterpret_cast<const cutlass::half_t*>(b);
    auto layout_d =
        make_layout(make_shape(m, n), make_stride(n, Int<1>{}));
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

    auto smem_layout_a = [&]() {
      if constexpr (Sw128) {
        return UMMA::tile_to_mma_shape(
            UMMA::Layout_K_SW128_Atom<cutlass::half_t>{}, mma_shape_a);
      } else {
        return UMMA::tile_to_mma_shape(
            UMMA::Layout_K_INTER_Atom<cutlass::half_t>{}, mma_shape_a);
      }
    }();
    auto smem_layout_b = [&]() {
      if constexpr (Sw128) {
        return UMMA::tile_to_mma_shape(
            UMMA::Layout_MN_SW128_Atom<cutlass::half_t>{}, mma_shape_b);
      } else {
        return UMMA::tile_to_mma_shape(
            UMMA::Layout_MN_INTER_Atom<cutlass::half_t>{}, mma_shape_b);
      }
    }();

    using Storage = Tc12SharedStorage<
        cutlass::half_t, cutlass::half_t, decltype(smem_layout_a),
        decltype(smem_layout_b)>;

    if constexpr (Rank3) {
      auto layout_a = make_layout(
          make_shape(m, k, Int<1>{}),
          make_stride(k, Int<1>{}, static_cast<int64_t>(m) * k));
      auto layout_b = make_layout(
          make_shape(n, k, Int<1>{}),
          make_stride(Int<1>{}, n, static_cast<int64_t>(n) * k));
      auto matrix_a_3d = make_tensor(make_gmem_ptr(typed_a), layout_a);
      auto matrix_b_3d = make_tensor(make_gmem_ptr(typed_b), layout_b);
      initialize<Storage>(matrix_a_3d, matrix_b_3d, matrix_c, matrix_d,
                          mma_tiler, tiled_mma, smem_layout_a,
                          smem_layout_b, m, n);
    } else {
      auto layout_a =
          make_layout(make_shape(m, k), make_stride(k, Int<1>{}));
      auto layout_b =
          make_layout(make_shape(n, k), make_stride(Int<1>{}, n));
      auto matrix_a_2d = make_tensor(make_gmem_ptr(typed_a), layout_a);
      auto matrix_b_2d = make_tensor(make_gmem_ptr(typed_b), layout_b);
      initialize<Storage>(matrix_a_2d, matrix_b_2d, matrix_c, matrix_d,
                          mma_tiler, tiled_mma, smem_layout_a,
                          smem_layout_b, m, n);
    }
  }

  void launch() { launch_(); }

 private:
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 64;

  template <class Storage, class MatrixA, class MatrixB, class MatrixC,
            class MatrixD, class MmaTiler, class TiledMma,
            class SmemLayoutA, class SmemLayoutB>
  void initialize(MatrixA matrix_a, MatrixB matrix_b, MatrixC matrix_c,
                  MatrixD matrix_d, MmaTiler mma_tiler, TiledMma tiled_mma,
                  SmemLayoutA smem_layout_a, SmemLayoutB smem_layout_b,
                  int m, int n) {
    Copy_Atom tma_atom_a =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_a, smem_layout_a,
                      select<0, 2>(mma_tiler));
    Copy_Atom tma_atom_b =
        make_tma_atom(SM90_TMA_LOAD{}, matrix_b, smem_layout_b,
                      select<1, 2>(mma_tiler));
    auto matrix_a_tma_full = tma_atom_a.get_tma_tensor(shape(matrix_a));
    auto matrix_b_tma_full = tma_atom_b.get_tma_tensor(shape(matrix_b));

    if constexpr (Rank3) {
      auto matrix_a_tma = matrix_a_tma_full(_, _, 0);
      auto matrix_b_tma = matrix_b_tma_full(_, _, 0);
      configure_launch<Storage>(
          matrix_a_tma, matrix_b_tma, matrix_c, matrix_d, mma_tiler,
          tiled_mma, tma_atom_a, tma_atom_b, m, n);
    } else {
      configure_launch<Storage>(
          matrix_a_tma_full, matrix_b_tma_full, matrix_c, matrix_d,
          mma_tiler, tiled_mma, tma_atom_a, tma_atom_b, m, n);
    }
  }

  template <class Storage, class MatrixA, class MatrixB, class MatrixC,
            class MatrixD, class MmaTiler, class TiledMma, class TmaAtomA,
            class TmaAtomB>
  void configure_launch(MatrixA matrix_a, MatrixB matrix_b, MatrixC matrix_c,
                        MatrixD matrix_d, MmaTiler mma_tiler,
                        TiledMma tiled_mma, TmaAtomA tma_atom_a,
                        TmaAtomB tma_atom_b, int m, int n) {
    auto* kernel = &tc12_minimal_tma_tcgen05_kernel<
        Storage, MatrixA, MatrixB, MatrixC, MatrixD, MmaTiler, TiledMma,
        TmaAtomA, TmaAtomB>;
    check_cuda(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(sizeof(Storage))),
               "cudaFuncSetAttribute(tc1/tc2)");
    const dim3 grid(m / kTileM, n / kTileN);
    launch_ = [=]() mutable {
      kernel<<<grid, 128, sizeof(Storage)>>>(
          matrix_a, matrix_b, matrix_c, matrix_d, mma_tiler, tiled_mma,
          tma_atom_a, tma_atom_b);
      check_cuda(cudaGetLastError(),
                 "tc12_minimal_tma_tcgen05_kernel launch");
    };
  }

  static void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA failure in %s: %s\n", where,
                   cudaGetErrorString(status));
      std::abort();
    }
  }

  std::function<void()> launch_;
};

using Tc1aRunner = Tc12Runner<false, false>;
using Tc1bRunner = Tc12Runner<true, false>;
using Tc2aRunner = Tc12Runner<false, true>;
using Tc2bRunner = Tc12Runner<true, true>;

}  // namespace gemm_sm110::backends
