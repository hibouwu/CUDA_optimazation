#include <cutlass/half.h>

#include <cuda_runtime.h>

#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <int Tid, class TmemCopy, class AccTensor, class GmemTensor>
void print_copy_thread(const char* label, TmemCopy const& tmem_copy,
                       AccTensor const& tCtAcc, GmemTensor const& tCgC) {
  auto thr_copy = tmem_copy.get_slice(Int<Tid>{});
  Tensor tD = thr_copy.partition_D(tCgC);
  print(label);
  print(" tid ");
  print(Tid);
  print(" tD:\t");
  print(tD);
  print("\n");
}

template <int TileM, int TileN>
void print_accumulator_layout(const char* label) {
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;

  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, TileM, TileN,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto cta_mma = tiled_mma.get_slice(_0{});

  Layout layout_c = make_layout(make_shape(Int<TileM>{}, Int<TileN>{}),
                                make_stride(Int<TileN>{}, Int<1>{}));
  Tensor gC = make_tensor(make_gmem_ptr(static_cast<TypeC*>(nullptr)),
                          layout_c);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  print(label);
  print(" tiled_mma:\n");
  print(tiled_mma);
  print(label);
  print(" tCgC:\t");
  print(tCgC);
  print("\n");
  print(label);
  print(" tCtAcc:\t");
  print(tCtAcc);
  print("\n");

  auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_16dp256b16x{}, tCtAcc);
  print(label);
  print(" tmem_copy 16dp256b16x:\n");
  print(tmem_copy);
  print("\n");
  print_copy_thread<0>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<1>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<4>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<8>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<16>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<31>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<32>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<64>(label, tmem_copy, tCtAcc, tCgC);
  print_copy_thread<96>(label, tmem_copy, tCtAcc, tCgC);
}

template <int TileM, int TileN>
void print_accumulator_layout_2sm(const char* label) {
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;

  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, TileM, TileN,
                                 UMMA::Major::K, UMMA::Major::K>{});
  auto cta_mma = tiled_mma.get_slice(_0{});

  Layout layout_c = make_layout(make_shape(Int<TileM>{}, Int<TileN>{}),
                                make_stride(Int<TileN>{}, Int<1>{}));
  Tensor gC = make_tensor(make_gmem_ptr(static_cast<TypeC*>(nullptr)),
                          layout_c);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  print(label);
  print(" tiled_mma:\n");
  print(tiled_mma);
  print(label);
  print(" tCgC:\t");
  print(tCgC);
  print("\n");
  print(label);
  print(" tCtAcc:\t");
  print(tCtAcc);
  print("\n");

  auto tmem_copy_32 = make_tmem_copy(SM100_TMEM_LOAD_32dp32b16x{}, tCtAcc);
  print(label);
  print(" tmem_copy 32dp32b16x:\n");
  print(tmem_copy_32);
  print("\n");
  print_copy_thread<0>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<1>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<31>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<32>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<63>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<64>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<95>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<96>(label, tmem_copy_32, tCtAcc, tCgC);
  print_copy_thread<127>(label, tmem_copy_32, tCtAcc, tCgC);
}

template <int TileM, int TileN>
__global__ void print_source_partition_kernel() {
  if (threadIdx.x != 0) return;

  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;

  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, TileM, TileN,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto cta_mma = tiled_mma.get_slice(_0{});
  Layout layout_c = make_layout(make_shape(Int<TileM>{}, Int<TileN>{}),
                                make_stride(Int<TileN>{}, Int<1>{}));
  Tensor gC = make_tensor(make_gmem_ptr(static_cast<TypeC*>(nullptr)),
                          layout_c);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
  auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_16dp256b16x{}, tCtAcc);

  auto thr0 = tmem_copy.get_slice(Int<0>{});
  auto thr32 = tmem_copy.get_slice(Int<32>{});
  auto thr64 = tmem_copy.get_slice(Int<64>{});
  auto thr96 = tmem_copy.get_slice(Int<96>{});
  Tensor tS0 = thr0.partition_S(tCtAcc);
  Tensor tS32 = thr32.partition_S(tCtAcc);
  Tensor tS64 = thr64.partition_S(tCtAcc);
  Tensor tS96 = thr96.partition_S(tCtAcc);
  print("device tS M");
  print(TileM);
  print(" tid0:\t");
  print(tS0);
  print("\n");
  print("device tS M");
  print(TileM);
  print(" tid32:\t");
  print(tS32);
  print("\n");
  print("device tS M");
  print(TileM);
  print(" tid64:\t");
  print(tS64);
  print("\n");
  print("device tS M");
  print(TileM);
  print(" tid96:\t");
  print(tS96);
  print("\n");
}

int main() {
  print_accumulator_layout<64, 256>("M64N256");
  print_accumulator_layout<128, 256>("M128N256");
  print_accumulator_layout_2sm<256, 128>("M256N128_2SM");
  print_accumulator_layout_2sm<256, 256>("M256N256_2SM");
  print_source_partition_kernel<64, 256><<<1, 32>>>();
  print_source_partition_kernel<128, 256><<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
