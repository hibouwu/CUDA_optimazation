// V02: predict-then-measure run of the fixed CUTLASS kernel (NDEBUG build), copied from
// r09_probe.cu. Kernel types, warm protocol and sampled FP64 check are unchanged.
//   v02_plain: original CUTLASS headers; its event time is the measured value.
//   v02_trace: -DV02_TRACE and an include overlay (v02_run.py): per CTA, clock64/globaltimer at
//              fixed sites (v02_trace.hpp) plus a count of tiles processed by the CTA.
// Change from r09_probe.cu: in the warm sequence the trace record is zeroed (and the device
// synchronized, in both builds) before the final timed call, so the tile counter and stamps
// belong to that call only.
//
// Modes:
//   warm:      R00 measure_gemm sequence: synchronized event-timed calls until the last 5 have
//              CV<=2% (i>=7, at most 30), then one timed call. Trace of that call is read.
//   idle_warm: one untimed call (module load), idle --idle-s, one timed cold call (trace read),
//              then the warm sequence above (trace read again).
//   timer_res: globaltimer update step seen by one thread spinning on it.
// A 20 us clock64/globaltimer probe kernel runs after each traced call (after the event window).
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <chrono>
#include <map>
#include <thread>
#ifdef V02_TRACE
#include "v02_trace.hpp"
#endif

using namespace cute;
using KernelTile = Shape<_128, _256, _64>;
using KernelCluster = Shape<_2, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, KernelTile, KernelCluster,
    Shape<_128, _32>, float, float, void, cutlass::layout::RowMajor, 4, float,
    cutlass::layout::RowMajor, 4, cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t, cutlass::layout::RowMajor,
    8, cutlass::half_t, cutlass::layout::RowMajor, 8, float, KernelTile, KernelCluster,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

constexpr int kTraceWords = 32;    // per CTA: 12 slots x (clock64, globaltimer) + smid
constexpr int kTraceCtas = 1024;

static void cutlass_check(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + what + ": " +
                             cutlassGetStatusString(status));
}

// One block per SM, thread 0 spins `window_ns` of globaltimer (same probe as R07).
__global__ void clock_probe(uint64_t window_ns, uint64_t* out) {
  if (threadIdx.x != 0)
    return;
  uint64_t t0, t1, c0, c1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  c0 = clock64();
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  } while (t1 - t0 < window_ns);
  c1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  out[2 * blockIdx.x] = c1 - c0;
  out[2 * blockIdx.x + 1] = t1 - t0;
}

// Records each change of globaltimer seen by one thread: (globaltimer, clock64) pairs.
__global__ void timer_resolution(int changes, uint64_t* out) {
  uint64_t last, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(last));
  for (int i = 0; i < changes;) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now != last) {
      out[2 * i] = now;
      out[2 * i + 1] = clock64();
      last = now;
      ++i;
    }
  }
}

struct Args {
  std::string mode = "warm";
  int m = 2048, n = 2048, k = 2048;
  double idle_s = 1.5;
  Args(int argc, char** argv) {
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--mode") mode = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else if (key == "--idle-s") idle_s = std::stod(v);
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

static int run_timer_resolution() {
  const int changes = 4096;
  DeviceBuffer<uint64_t> out(2 * changes);
  timer_resolution<<<1, 1>>>(changes, out.pointer);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> raw(2 * changes);
  CUDA_CHECK(cudaMemcpy(raw.data(), out.pointer, raw.size() * 8, cudaMemcpyDeviceToHost));
  std::map<uint64_t, int> histogram;
  for (int i = 1; i < changes; ++i) ++histogram[raw[2 * i] - raw[2 * (i - 1)]];
  double cycles = double(raw[2 * changes - 1] - raw[1]);
  double nanos = double(raw[2 * changes - 2] - raw[0]);
  std::cout << "{\"event\":\"timer_res\",\"changes\":" << changes << ",\"span_ns\":" << nanos
            << ",\"span_cycles\":" << cycles << ",\"ghz\":" << cycles / nanos
            << ",\"step_histogram\":{";
  bool first = true;
  for (auto& [step, count] : histogram) {
    std::cout << (first ? "" : ",") << '"' << step << "\":" << count;
    first = false;
  }
  std::cout << "}}\n";
  return 0;
}

int main(int argc, char** argv) {
  try {
    Args o(argc, argv);
    std::cout << std::setprecision(10);
    if (o.mode == "timer_res")
      return run_timer_resolution();
    if (o.mode != "warm" && o.mode != "idle_warm")
      throw std::runtime_error("unknown mode " + o.mode);

    Options check(1, argv);  // R00 sampled FP64 check: default 4096 samples, seed 17
    check.m = o.m, check.n = o.n, check.k = o.k;
    DeviceBuffer<__half> a(size_t(o.m) * o.k), b(size_t(o.k) * o.n);
    DeviceBuffer<float> d(size_t(o.m) * o.n);
    auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(o.m, o.k, 1));
    auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(o.n, o.k, 1));
    auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(o.m, o.n, 1));
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {o.m, o.n, o.k, 1},
                                       {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
                                        reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
                                       {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    dim3 grid = Gemm::get_grid_shape(gemm.params());
    int ctas = int(grid.x * grid.y * grid.z);
    if (ctas > kTraceCtas)
      throw std::runtime_error("grid larger than trace buffer");

    // Same preparation order as R00 measure_gemm (fill A, B; D = NaN pattern).
    fill_input<<<256, 256>>>(a.pointer, o.m, o.k, check.seed, true, false);
    fill_input<<<256, 256>>>(b.pointer, o.k, o.n, check.seed, false, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d.pointer, 0xff, d.count * sizeof(float)));
    DeviceBuffer<uint64_t> probe(2 * size_t(properties.multiProcessorCount));
#ifdef V02_TRACE
    DeviceBuffer<uint64_t> trace(size_t(kTraceCtas) * kTraceWords);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
    CUDA_CHECK(cudaMemcpyToSymbol(v02_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
    const bool traced = true;
#else
    const bool traced = false;
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
      return ms;
    };
    // Reads the trace of the call just completed, then runs the post-call clock probe.
    auto report = [&](const char* label, float ms, const std::vector<float>& warm) {
      std::vector<uint64_t> host;
#ifdef V02_TRACE
      host.resize(size_t(ctas) * kTraceWords);
      CUDA_CHECK(cudaMemcpy(host.data(), trace.pointer, host.size() * 8, cudaMemcpyDeviceToHost));
#endif
      clock_probe<<<properties.multiProcessorCount, 32>>>(20000, probe.pointer);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint64_t> raw(probe.count);
      CUDA_CHECK(cudaMemcpy(raw.data(), probe.pointer, raw.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<double> ghz;
      for (size_t i = 0; i < raw.size(); i += 2) ghz.push_back(double(raw[i]) / double(raw[i + 1]));
      std::sort(ghz.begin(), ghz.end());
      std::vector<double> warm_us;
      for (float w : warm) warm_us.push_back(w * 1e3);
      std::cout << "{\"event\":\"call\",\"label\":\"" << label << "\",\"elapsed_us\":" << ms * 1e3
                << ",\"probe_ghz_median\":" << ghz[ghz.size() / 2]
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
#ifdef V02_TRACE
      CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#endif
      CUDA_CHECK(cudaDeviceSynchronize());
      float ms = timed();
      report("warm", ms, warm);
    };

    std::cout << "{\"event\":\"setup\",\"traced\":" << (traced ? "true" : "false")
              << ",\"mode\":\"" << o.mode << "\",\"m\":" << o.m << ",\"n\":" << o.n
              << ",\"k\":" << o.k << ",\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z
              << "],\"threads\":" << Kernel::MaxThreadsPerBlock
              << ",\"smem\":" << sizeof(typename Kernel::SharedStorage)
              << ",\"trace_words\":" << kTraceWords << "}\n";
    if (o.mode == "idle_warm") {
      cutlass_check(gemm.run(), "run");  // module load, not timed
      CUDA_CHECK(cudaDeviceSynchronize());
      std::this_thread::sleep_for(std::chrono::duration<double>(o.idle_s));
      float ms = timed();
      report("cold", ms, {});
    }
    warm_sequence();

    Errors e = check_output<__half, float>(check, d.pointer, false);
    bool ok = e.nonfinite == 0 && e.max_storage_error <= 1e-5;
    std::cout << "{\"event\":\"check\",\"samples\":" << e.indices.size()
              << ",\"max_storage_reference_error\":" << e.max_storage_error
              << ",\"nonfinite\":" << e.nonfinite << ",\"status\":\""
              << (ok ? "ok" : "numeric_error") << "\"}\n";
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
