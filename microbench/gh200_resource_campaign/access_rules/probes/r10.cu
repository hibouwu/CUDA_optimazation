// R10: fixed 2600x3000x2000 GEMM, explicit physical B row stride.
#include "gaps_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#ifdef R10_TRACE
#include "r10_trace.hpp"
#endif

using namespace cute;
#ifndef R10_CFG
#error "compile with -DR10_CFG=0|1|2"
#endif
#if R10_CFG == 0
constexpr const char* kConfigName = "cfg_a";
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_2, _1, _1>;
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define R10_STAGES_AUTO 1
#elif R10_CFG == 1
constexpr const char* kConfigName = "cfg_b";
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_64, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#define R10_STAGES_AUTO 1
#elif R10_CFG == 2
constexpr const char* kConfigName = "cfg_c";
using KernelTile = Shape<_256, _128, _64>;
using KernelCluster = Shape<_1, _2, _1>;
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define R10_STAGES_AUTO 0
#else
#error "unknown R10_CFG"
#endif
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, KernelTile, KernelCluster, EpilogueTile,
    float, float, void, cutlass::layout::RowMajor, 4, float, cutlass::layout::RowMajor, 4,
    EpilogueSchedule>::CollectiveOp;
#if R10_STAGES_AUTO
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Epilogue::SharedStorage))>;
#else
using StageCount = cutlass::gemm::collective::StageCount<4>;
#endif
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t, cutlass::layout::RowMajor,
    8, cutlass::half_t, cutlass::layout::RowMajor, 8, float, KernelTile, KernelCluster, StageCount,
    MainloopSchedule>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;


#ifdef R10_TRACE
#include "r10_trace.hpp"
#endif

void cutlass_check(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
}

struct Args {
  int m = 2600, n = 3000, k = 2000;
  int64_t lda = 2000, ldb = 3000, ldd = 3000;
  Args(int argc, char** argv) {
    if (argc % 2 == 0) throw std::runtime_error("--key value pairs required");
    for (int i = 1; i < argc; i += 2) {
      std::string key = argv[i];
      int64_t value = std::stoll(argv[i + 1]);
      if (key == "--m") m = int(value);
      else if (key == "--n") n = int(value);
      else if (key == "--k") k = int(value);
      else if (key == "--lda") lda = value;
      else if (key == "--ldb") ldb = value;
      else if (key == "--ldd") ldd = value;
      else throw std::runtime_error("unknown argument " + key);
    }
    // This probe implements the nine planned physical layouts only.
    if (m != 2600 || n != 3000 || k != 2000 || lda != k || ldd != n ||
        (ldb != 3000 && ldb != 3008 && ldb != 3072))
      throw std::runtime_error("outside R10 matrix");
    gaps::validate_layout(m, k, lda);
    gaps::validate_layout(k, n, ldb);
    gaps::validate_layout(m, n, ldd);
  }
};

int main(int argc, char** argv) try {
  Args o(argc, argv);
  DeviceBuffer<__half> a(size_t(o.m) * o.lda), b(size_t(o.k) * o.ldb);
  DeviceBuffer<float> d(size_t(o.m) * o.ldd);
  // CUTLASS views B as (N,K,L): physical row-major KxN is stride (1,ldb,0).
  typename Kernel::StrideA sa{};
  typename Kernel::StrideB sb{};
  typename Kernel::StrideD sd{};
  get<0>(sa) = o.lda;
  get<1>(sb) = o.ldb;
  get<0>(sd) = o.ldd;
  get<2>(sa) = get<2>(sb) = get<2>(sd) = int64_t(0);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {o.m, o.n, o.k, 1},
      {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
       reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
      {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
  Gemm gemm;
  cutlass_check(gemm.can_implement(arguments), "can_implement");
  DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
  cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
  dim3 grid = Gemm::get_grid_shape(gemm.params());
  int ctas = int(grid.x * grid.y * grid.z);
  constexpr int max_ctas = 1024;
  if (ctas > max_ctas) throw std::runtime_error("trace allocation too small");

  gaps::fill_input_strided<<<256, 256>>>(a.pointer, o.m, o.k, o.lda, 17, true);
  gaps::fill_input_strided<<<256, 256>>>(b.pointer, o.k, o.n, o.ldb, 17, false);
  gaps::fill_output_sentinel<<<256, 256>>>(d.pointer, d.count);
  CUDA_CHECK(cudaGetLastError());
  // Identical scratch allocation and preparation in plain and trace variants.
  // Plain never passes this pointer to a kernel and does not write instrumentation.
  DeviceBuffer<uint64_t> trace(size_t(max_ctas) * 8);
  CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#ifdef R10_TRACE
  CUDA_CHECK(cudaMemcpyToSymbol(r10_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
#endif
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  auto timed = [&]() {
    CUDA_CHECK(cudaEventRecord(begin));
    cutlass_check(gemm.run(), "run");
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    return ms * 1000;
  };
  std::vector<float> warmup;
  bool converged = false;
  for (int i = 0; i < 30; ++i) {
    warmup.push_back(timed());
    if (i >= 7) {
      std::vector<float> last(warmup.end() - 5, warmup.end());
      if (coefficient_of_variation(last) <= 0.02) {
        converged = true;
        break;
      }
    }
  }
  // Same GPU preparation after warmup, immediately before both sampled calls.
  CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
  CUDA_CHECK(cudaDeviceSynchronize());
  float elapsed = timed();
  std::vector<uint64_t> host_trace;
#ifdef R10_TRACE
  host_trace.resize(size_t(ctas) * 8);
  CUDA_CHECK(cudaMemcpy(host_trace.data(), trace.pointer, host_trace.size() * 8,
                        cudaMemcpyDeviceToHost));
#endif
  Errors errors = gaps::check_gemm_output(d.pointer, o.m, o.n, o.k, o.ldd);
  size_t bad_b = gaps::padding_errors(b.pointer, o.k, o.n, o.ldb, __float2half(65504.f));
  size_t bad_a = gaps::padding_errors(a.pointer, o.m, o.k, o.lda, __float2half(65504.f));
  size_t bad_d = gaps::padding_errors(d.pointer, o.m, o.n, o.ldd, gaps::output_sentinel());
  bool valid = errors.nonfinite == 0 && errors.max_storage_error == 0 &&
               bad_a + bad_b + bad_d == 0;
  std::cout << std::setprecision(17)
            << "{\"event\":\"measurement\",\"config\":\"" << kConfigName
            << "\",\"m\":" << o.m << ",\"n\":" << o.n << ",\"k\":" << o.k
            << ",\"lda\":" << o.lda << ",\"ldb\":" << o.ldb << ",\"ldd\":" << o.ldd
            << ",\"elapsed_us\":" << elapsed << ",\"grid_ctas\":" << ctas
            << ",\"stages\":" << int(Mainloop::DispatchPolicy::Stages)
            << ",\"threads\":" << Kernel::MaxThreadsPerBlock
            << ",\"smem_bytes\":" << sizeof(typename Kernel::SharedStorage)
            << ",\"max_active_ctas_per_sm\":" << Gemm::maximum_active_blocks()
            << ",\"warmup_converged\":" << (converged ? "true" : "false")
            << ",\"scratch_bytes\":" << trace.count * 8
            << ",\"preparation_protocol\":\"symmetric_scratch_memset_sync_v3\""
            << ",\"work_flop\":" << uint64_t(2) * o.m * o.n * o.k
            << ",\"warmup_us\":[";
  for (size_t i = 0; i < warmup.size(); ++i) std::cout << (i ? "," : "") << warmup[i];
  std::cout << "],\"trace\":[";
  for (size_t i = 0; i < host_trace.size(); ++i)
    std::cout << (i ? "," : "") << host_trace[i];
#if R10_CFG == 1
  std::cout << "],\"trace_schema\":4,\"trace_cta_limit\":4,"
            << "\"trace_consumer_groups\":[0],\"trace_tile\":\"last\"";
#else
  std::cout << "],\"trace_schema\":2,\"trace_consumer_groups\":[0,1],"
            << "\"trace_tile\":\"last\"";
#endif
  std::cout << ",\"padding\":{\"errors\":" << bad_a + bad_b + bad_d
            << ",\"checked_elements\":" << uint64_t(o.k) * (o.ldb - o.n)
            << ",\"b_sentinel_fp16_bits\":31743}}\n";
  gaps::print_check(errors, bad_a + bad_b + bad_d);
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return valid ? 0 : 2;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
