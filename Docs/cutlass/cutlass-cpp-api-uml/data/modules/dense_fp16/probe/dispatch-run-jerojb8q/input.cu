// Shape/type evidence only. The optional kernel is compiled, never launched.
// Reuse the immutable recipe source that produced run-334xibhm.
#define PROBE_HOST_TYPES
#define main prior_dense_type_main_not_called
#include "run-334xibhm/input.cu"
#undef main
#include <utility>

using namespace cute;
using SmemTensorA = decltype(make_tensor(make_smem_ptr(static_cast<CollectiveMainloop::SmemAllocTypeA*>(nullptr)),
                                      CollectiveMainloop::SmemLayoutA{}));
using SmemTensorB = decltype(make_tensor(make_smem_ptr(static_cast<CollectiveMainloop::SmemAllocTypeB*>(nullptr)),
                                      CollectiveMainloop::SmemLayoutB{}));
using FragmentA = decltype(TiledMma::make_fragment_A(std::declval<SmemTensorA>()));
using FragmentB = decltype(TiledMma::make_fragment_B(std::declval<SmemTensorB>()));
using AccShape = decltype(partition_shape_C(TiledMma{}, take<0,2>(CollectiveMainloop::TileShape{})));
using BulkAccumulator = decltype(cutlass::detail::make_sm100_accumulator<
    GemmKernel::AccumulatorPipelineStageCount, GemmKernel::IsOverlappingAccum>(
    TiledMma{}, AccShape{}, GemmKernel::EpilogueTile{}));
using Accumulator = decltype(std::declval<BulkAccumulator&>()(_,_,_,int{}));
using ASlice = decltype(std::declval<FragmentA&>()(_,_,int{},int{}));
using BSlice = decltype(std::declval<FragmentB&>()(_,_,int{},int{}));
using AVector = decltype(std::declval<ASlice const&>()(_,int{}));
using BVector = decltype(std::declval<BSlice const&>()(_,int{}));
using DVector = decltype(std::declval<Accumulator&>()(_,int{},int{}));
using CVector = decltype(std::declval<Accumulator const&>()(_,int{},int{}));
template<class T> constexpr int rank_of = T::layout_type::rank;
template<class T> constexpr int value_bytes = decltype(size<0>(typename T::layout_type{}))::value * sizeof(typename T::engine_type::value_type);
static_assert(rank_of<FragmentA> == 4 && rank_of<FragmentB> == 4);
static_assert(rank_of<ASlice> == 2 && rank_of<BSlice> == 2 && rank_of<Accumulator> == 3);
static_assert(is_rmem<typename ASlice::engine_type>::value && is_rmem<typename BSlice::engine_type>::value);
static_assert(is_rmem<typename Accumulator::engine_type>::value && is_tmem<typename Accumulator::engine_type>::value);
static_assert(value_bytes<ASlice> == 8 && value_bytes<BSlice> == 8);
static_assert(rank_of<AVector> == 1 && rank_of<BVector> == 1 && rank_of<DVector> == 1 && rank_of<CVector> == 1);
static_assert(!std::is_reference_v<DVector>, "D(_,m,ns) is a temporary, requiring gemm's D&& forwarding overload");
static_assert(is_same_v<DVector, CVector>);

#if defined(COMPILE_DISPATCH_KERNEL)
// Valid template instantiation evidence, not a valid runtime memory/launch setup.
__global__ void dense_gemm_dispatch_probe(FragmentA a, FragmentB b, Accumulator c, int k, int stage) {
  TiledMma tiled_mma;
  cute::gemm(tiled_mma, a(_,_,k,stage), b(_,_,k,stage), c);
}
#else
template<class T> void tensor_evidence(char const* name) {
  show_type<T>(name);
  std::printf("%s.shape=", name); print(shape(typename T::layout_type{})); std::printf("\n");
  std::printf("%s.rank=%d\n", name, rank_of<T>);
  std::printf("%s.value_bytes=%zu\n", name, sizeof(typename T::engine_type::value_type));
  std::printf("%s.V_elements=%d\n", name, int(size<0>(typename T::layout_type{})));
  std::printf("%s.is_rmem=%d\n", name, int(is_rmem<typename T::engine_type>::value));
  std::printf("%s.is_tmem=%d\n", name, int(is_tmem<typename T::engine_type>::value));
}
int main() {
  // Only type/shape operations execute. No GPU API or instruction wrapper is called.
  tensor_evidence<FragmentA>("FragmentA");
  tensor_evidence<FragmentB>("FragmentB");
  tensor_evidence<BulkAccumulator>("BulkAccumulator");
  tensor_evidence<ASlice>("ASlice");
  tensor_evidence<BSlice>("BSlice");
  tensor_evidence<Accumulator>("Accumulator");
  tensor_evidence<AVector>("AVector");
  tensor_evidence<BVector>("BVector");
  tensor_evidence<DVector>("DVector");
  tensor_evidence<CVector>("CVector");
  std::printf("Kernel.IsOverlappingAccum=%d\n", int(GemmKernel::IsOverlappingAccum));
  std::printf("Dispatch4.A_V_bytes=%d\n", value_bytes<ASlice>);
  std::printf("Dispatch4.B_V_bytes=%d\n", value_bytes<BSlice>);
  std::printf("Dispatch4.M=%d\n", int(size<1>(typename ASlice::layout_type{})));
  std::printf("Dispatch4.N=%d\n", int(size<1>(typename BSlice::layout_type{})));
  std::printf("Mainloop.K_blocks=%d\n", int(size<2>(typename FragmentA::layout_type{})));
  std::printf("DVector.is_reference=%d\n", int(std::is_reference_v<DVector>));
}
#endif
