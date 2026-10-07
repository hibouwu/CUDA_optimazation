// R14: 4-CTA persistent GEMM, per-role/per-tile tracing. See run_r14.py.
#include "gaps_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#ifdef R14_TRACE
#include "r14_trace.hpp"
#ifndef R14_EPILOGUE_OVERLAY_ACTIVE
#error "R14 trace must compile the epilogue overlay, not the original sibling header"
#endif
#endif

using namespace cute;
#ifndef R14_CFG
#error "compile with -DR14_CFG=0|1|2"
#endif
#if R14_CFG == 0
constexpr const char* kConfigName = "cfg_a";
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_2, _1, _1>;
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define R14_STAGES_AUTO 1
#elif R14_CFG == 1
constexpr const char* kConfigName = "cfg_b";
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_64, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#define R14_STAGES_AUTO 1
#elif R14_CFG == 2
constexpr const char* kConfigName = "baseline";
using KernelTile = Shape<_128, _256, _64>;
using KernelCluster = Shape<_2, _1, _1>;
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define R14_STAGES_AUTO 0
#else
#error "unknown R14_CFG"
#endif
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, KernelTile, KernelCluster, EpilogueTile,
    float, float, void, cutlass::layout::RowMajor, 4, float, cutlass::layout::RowMajor, 4,
    EpilogueSchedule>::CollectiveOp;
#if R14_STAGES_AUTO
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
#if R14_CFG != 1
static_assert(Kernel::NumMMAThreads == 256, "recheck cooperative TMA issuing warp mapping");
#endif

constexpr int kTraceWords = 4 + 3 * 8 * 24;
constexpr int kTraceCtas = 4;
constexpr const char* kTracePair = R14_PAIR == 0 ? "main" : R14_PAIR == 1 ? "output"
                                         : R14_PAIR == 2 ? "supply" : "critical";
constexpr unsigned kTraceMask = R14_PAIR == 0 ? 0x06 : R14_PAIR == 1 ? 0x2c
                                           : R14_PAIR == 2 ? 0x03 : 0x21;
constexpr const char* kTraceClock = R14_PAIR == 3 ? "sm_cycle_and_globaltimer" : "sm_cycle_only";

static void cutlass_check(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + what + ": " +
                             cutlassGetStatusString(status));
}

struct Args {
  std::string mode = "warm";
  int m = 256, n = 2048, k = 128;
  Args(int argc, char** argv) {
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--mode") mode = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else throw std::runtime_error("unknown option " + key);
    }
  }
};

static void print_u64(const char* name, const std::vector<uint64_t>& v) {
  std::cout << ",\"" << name << "\":[";
  for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << v[i];
  std::cout << ']';
}

static void print_f(const char* name, const std::vector<double>& v) {
  std::cout << ",\"" << name << "\":[";
  for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << v[i];
  std::cout << ']';
}

int main(int argc, char** argv) {
  try {
    Args o(argc, argv);
    std::cout << std::setprecision(10);
    if (o.mode != "warm" && o.mode != "setup")
      throw std::runtime_error("unknown mode " + o.mode);

    Options check(1, argv);  // Deterministic dyadic witness, seed 17.
    check.m = o.m, check.n = o.n, check.k = o.k;
    int64_t lda = o.k, ldb = o.n, ldd = o.n;
    int tn = int(size<1>(KernelTile{}));
    if (o.m != 256 || (o.n != 2 * tn && o.n != 16 * tn) ||
        (o.k != 128 && o.k != 512 && o.k != 4096))
      throw std::runtime_error("shape outside R14 matrix");
    gaps::validate_layout(o.m, o.k, lda);
    gaps::validate_layout(o.k, o.n, ldb);
    gaps::validate_layout(o.m, o.n, ldd);
    DeviceBuffer<__half> a(size_t(o.m) * lda), b(size_t(o.k) * ldb);
    DeviceBuffer<float> d(size_t(o.m) * ldd);
    typename Kernel::StrideA sa = make_stride(lda, Int<1>{}, int64_t(o.m) * lda);
    typename Kernel::StrideB sb = make_stride(Int<1>{}, ldb, int64_t(o.k) * ldb);
    typename Kernel::StrideD sd = make_stride(ldd, Int<1>{}, int64_t(o.m) * ldd);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {o.m, o.n, o.k, 1},
                                       {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
                                        reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
                                       {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    arguments.hw_info.device_id = 0;
    arguments.hw_info.sm_count = 4;
    arguments.scheduler.max_swizzle_size = 1;
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    dim3 grid = Gemm::get_grid_shape(gemm.params());
    int ctas = int(grid.x * grid.y * grid.z);
    if (ctas != kTraceCtas)
      throw std::runtime_error("R14 requires exactly 4 CTAs");

    // Same preparation order as R00 measure_gemm (fill A, B; D = NaN pattern).
    gaps::fill_input_strided<<<256, 256>>>(a.pointer, o.m, o.k, lda, check.seed, true);
    gaps::fill_input_strided<<<256, 256>>>(b.pointer, o.k, o.n, ldb, check.seed, false);
    CUDA_CHECK(cudaGetLastError());
    gaps::fill_output_sentinel<<<256, 256>>>(d.pointer, d.count);
    DeviceBuffer<uint64_t> trace(size_t(kTraceCtas) * kTraceWords);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#ifdef R14_TRACE
    CUDA_CHECK(cudaMemcpyToSymbol(r14_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
    const bool traced = true;
#else
    const bool traced = false;
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    auto timed = [&]() {
      CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(begin));
      cutlass_check(gemm.run(), "run");
      CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaEventSynchronize(end));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      return ms;
    };
    // Read every role/tile record after the timed call.
    auto report = [&](const char* label, float ms, const std::vector<float>& warm) {
      std::vector<uint64_t> host;
#ifdef R14_TRACE
      host.resize(size_t(ctas) * kTraceWords);
      CUDA_CHECK(cudaMemcpy(host.data(), trace.pointer, host.size() * 8, cudaMemcpyDeviceToHost));
#endif
      std::vector<double> warm_us;
      for (float w : warm) warm_us.push_back(w * 1e3);
      std::cout << "{\"event\":\"call\",\"label\":\"" << label << "\",\"elapsed_us\":" << ms * 1e3
                << ",\"warmup_calls\":" << warm.size();
      print_f("warmup_us", warm_us);
      print_u64("trace", host);
      std::cout << "}\n";
    };
    auto warm_sequence = [&]() {
      std::vector<float> warm;
      for (int i = 0; i < 30; ++i) {
        warm.push_back(timed());
        if (i >= 7 && coefficient_of_variation({warm.end() - 5, warm.end()}) <= 0.02)
          break;
      }
      float ms = timed();
      report("warm", ms, warm);
      if (warm.size() < 8 || coefficient_of_variation({warm.end() - 5, warm.end()}) > 0.02)
        throw std::runtime_error("last five warmups CV exceeds 2% after 30 calls");
    };

    std::cout << "{\"event\":\"setup\",\"config\":\"" << kConfigName
              << "\",\"traced\":" << (traced ? "true" : "false") << ",\"mode\":\"" << o.mode
              << "\",\"m\":" << o.m << ",\"n\":" << o.n << ",\"k\":" << o.k << ",\"grid\":["
              << grid.x << ',' << grid.y << ',' << grid.z << "],\"tile\":["
              << int(size<0>(KernelTile{})) << ',' << int(size<1>(KernelTile{})) << ','
              << int(size<2>(KernelTile{})) << "],\"cluster\":[" << int(size<0>(KernelCluster{}))
              << ',' << int(size<1>(KernelCluster{})) << "],\"epilogue_tile\":["
              << int(size<0>(EpilogueTile{})) << ',' << int(size<1>(EpilogueTile{}))
              << "],\"stages\":" << int(Mainloop::DispatchPolicy::Stages)
              << ",\"threads\":" << Kernel::MaxThreadsPerBlock
              << ",\"smem\":" << sizeof(typename Kernel::SharedStorage)
              << ",\"smem_epilogue\":" << sizeof(typename Epilogue::SharedStorage)
              << ",\"max_active_ctas_per_sm\":" << Gemm::maximum_active_blocks()
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"lda\":" << lda << ",\"ldb\":" << ldb << ",\"ldd\":" << ldd
              << ",\"tiles_per_cta\":" << o.n / (2 * tn)
              << ",\"trace_version\":6,\"trace_selected_cta\":" << (R14_PAIR == 3 ? -1 : 0)
              << ",\"trace_pair\":\"" << kTracePair << "\",\"trace_mask\":" << kTraceMask
              << ",\"trace_clock\":\"" << kTraceClock << "\""
              << ",\"source_release_role\":" << (R14_CFG == 1 ? -1 : 2)
              << ",\"scratch_bytes\":" << trace.count * sizeof(uint64_t)
              << ",\"trace_words\":" << kTraceWords << "}\n";
    if (o.mode == "setup")
      return 0;
    warm_sequence();

    Errors e = gaps::check_gemm_output(d.pointer, o.m, o.n, o.k, ldd, check.seed,
                                       o.m * o.n);
    size_t padding_bad = gaps::padding_errors(a.pointer, o.m, o.k, lda, __half(65504.f))
        + gaps::padding_errors(b.pointer, o.k, o.n, ldb, __half(65504.f))
        + gaps::padding_errors(d.pointer, o.m, o.n, ldd, gaps::output_sentinel());
    bool ok = e.nonfinite == 0 && e.max_storage_error == 0 && padding_bad == 0;
    gaps::print_check(e, padding_bad);
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
