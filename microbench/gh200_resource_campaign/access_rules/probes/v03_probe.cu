// V03: in-call SM clock calibration and predict-then-measure run of the fixed CUTLASS kernel
// (NDEBUG build), copied from v02_probe.cu. Kernel types, warm protocol and sampled FP64 check
// are unchanged.
//   v03_plain: original CUTLASS headers; its event time is the measured value.
//   v03_trace: -DV03_TRACE and an include overlay (v03_run.py): per CTA, clock64/globaltimer at
//              fixed sites (v03_trace.hpp), a tile counter, and one (clock64, globaltimer) stamp
//              per tile after mma_tail (time-resolved clock).
// Changes from v02_probe.cu:
//   --swizzle S   scheduler max_swizzle_size (default 1, as V02)
//   --sm-count G  KernelHardwareInfo::sm_count (persistent grid size; 0 = all SMs, as V02)
//   --nvml-us P   poll NVML (GPU and module instant power, SM clock, clock-event reasons) at most
//                 every P us while each warm-sequence and timed call is in flight (main thread
//                 polls cudaEventQuery instead of blocking in cudaEventSynchronize)
//   --samples N   FP64 check samples (default 4096, as V02)
//   --sustain-ms T  diagnostic only: after the warm rule converges, keep issuing synchronized
//                 calls until T ms after the first warm-up call, then the timed call (NVML needs
//                 tens of ms of load to show power; this is not the warm protocol)
//   check failures are reported, not thrown.
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <chrono>
#include <dlfcn.h>
#include <map>
#include <thread>
#ifdef V03_TRACE
#include "v03_trace.hpp"
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

// Per CTA: 12 slots x (clock64, globaltimer), smid (24), tile count (25), then from word 32 one
// (clock64, globaltimer) pair per tile. Must match v03_run.py TRACE_HEADER.
constexpr int kTileStamps = 128;
constexpr int kTraceWords = 32 + 2 * kTileStamps;
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

// Minimal NVML access through dlopen (no header or link dependency). Field ids from nvml.h:
// NVML_FI_DEV_POWER_INSTANT = 186; scope 0 = GPU, 1 = module (GH200 module power).
struct Nvml {
  struct FieldValue {  // layout of nvmlFieldValue_t
    unsigned field_id, scope_id;
    long long timestamp, latency_us;
    int value_type, status;
    union { double d; unsigned u; unsigned long ul; unsigned long long ull; long long sll; } value;
  };
  using Device = void*;
  int (*init)() = nullptr;
  int (*by_uuid)(const char*, Device*) = nullptr;
  int (*fields)(Device, int, FieldValue*) = nullptr;
  int (*clock)(Device, int, unsigned*) = nullptr;
  int (*reasons)(Device, unsigned long long*) = nullptr;
  int (*power)(Device, unsigned*) = nullptr;
  Device device = nullptr;
  bool ok = false;

  explicit Nvml(const cudaDeviceProp& properties) {
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return;
    init = (int (*)())dlsym(lib, "nvmlInit_v2");
    by_uuid = (int (*)(const char*, Device*))dlsym(lib, "nvmlDeviceGetHandleByUUID");
    fields = (int (*)(Device, int, FieldValue*))dlsym(lib, "nvmlDeviceGetFieldValues");
    clock = (int (*)(Device, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
    reasons = (int (*)(Device, unsigned long long*))dlsym(
        lib, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!reasons)
      reasons = (int (*)(Device, unsigned long long*))dlsym(
          lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
    power = (int (*)(Device, unsigned*))dlsym(lib, "nvmlDeviceGetPowerUsage");
    if (!init || !by_uuid || !fields || !clock || !reasons || !power || init() != 0) return;
    char uuid[64];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(properties.uuid.bytes);
    std::snprintf(uuid, sizeof(uuid),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                  b[13], b[14], b[15]);
    ok = by_uuid(uuid, &device) == 0;
  }
};

// NVML poller driven from the main thread: while a timed call is in flight the host polls
// cudaEventQuery and takes one NVML row every period (no extra thread). One row per poll, host
// steady_clock ns, plus the NVML query latency.
struct NvmlPoller {
  struct Row {
    int64_t t_ns, latency_ns;
    double gpu_w, module_w, avg_w;
    unsigned sm_mhz;
    unsigned long long reasons;
  };
  Nvml& nvml;
  int period_us;
  int64_t last_ns = 0;
  std::vector<Row> rows;
  NvmlPoller(Nvml& nvml, int period_us) : nvml(nvml), period_us(period_us) {
    rows.reserve(1 << 16);
  }
  static int64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void poll() {
    int64_t t = now();
    if (t - last_ns < int64_t(period_us) * 1000)
      return;
    last_ns = t;
    Nvml::FieldValue f[2]{};
    f[0].field_id = 186, f[0].scope_id = 0;
    f[1].field_id = 186, f[1].scope_id = 1;
    Row r{t, 0, -1, -1, -1, 0, 0};
    if (nvml.fields(nvml.device, 2, f) == 0) {
      if (f[0].status == 0) r.gpu_w = f[0].value.u / 1e3;
      if (f[1].status == 0) r.module_w = f[1].value.u / 1e3;
    }
    unsigned mw = 0;
    if (nvml.power(nvml.device, &mw) == 0) r.avg_w = mw / 1e3;
    nvml.clock(nvml.device, 1 /* NVML_CLOCK_SM */, &r.sm_mhz);
    nvml.reasons(nvml.device, &r.reasons);
    r.latency_ns = now() - t;
    rows.push_back(r);
  }
};

struct Args {
  std::string mode = "warm";
  int m = 2048, n = 2048, k = 2048, swizzle = 1, sm_count = 0, nvml_us = 0, samples = 4096;
  int sustain_ms = 0;
  Args(int argc, char** argv) {
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--mode") mode = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else if (key == "--swizzle") swizzle = std::stoi(v);
      else if (key == "--sm-count") sm_count = std::stoi(v);
      else if (key == "--nvml-us") nvml_us = std::stoi(v);
      else if (key == "--samples") samples = std::stoi(v);
      else if (key == "--sustain-ms") sustain_ms = std::stoi(v);
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
    if (o.mode != "warm")
      throw std::runtime_error("unknown mode " + o.mode);

    Options check(1, argv);  // R00 sampled FP64 check, seed 17
    check.m = o.m, check.n = o.n, check.k = o.k, check.samples = o.samples;
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
    arguments.scheduler.max_swizzle_size = o.swizzle;
    if (o.sm_count > 0)
      arguments.hw_info.sm_count = o.sm_count;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    const auto& sp = gemm.params().scheduler;
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
#ifdef V03_TRACE
    DeviceBuffer<uint64_t> trace(size_t(kTraceCtas) * kTraceWords);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
    CUDA_CHECK(cudaMemcpyToSymbol(v03_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
    const bool traced = true;
#else
    const bool traced = false;
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "{\"event\":\"setup\",\"traced\":" << (traced ? "true" : "false")
              << ",\"mode\":\"" << o.mode << "\",\"m\":" << o.m << ",\"n\":" << o.n
              << ",\"k\":" << o.k << ",\"swizzle\":" << o.swizzle << ",\"sm_count\":" << o.sm_count
              << ",\"raster_actual\":\""
              << (sp.raster_order_ == cutlass::gemm::kernel::detail::RasterOrder::AlongN ? "N" : "M")
              << "\",\"log_swizzle\":" << sp.log_swizzle_size_
              << ",\"blocks_per_problem\":" << sp.blocks_per_problem_
              << ",\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z
              << "],\"threads\":" << Kernel::MaxThreadsPerBlock
              << ",\"smem\":" << sizeof(typename Kernel::SharedStorage)
              << ",\"trace_words\":" << kTraceWords << ",\"tile_stamps\":" << kTileStamps << "}\n";

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    std::unique_ptr<Nvml> nvml;
    std::unique_ptr<NvmlPoller> poller;
    if (o.nvml_us > 0) {
      nvml = std::make_unique<Nvml>(properties);
      if (nvml->ok)
        poller = std::make_unique<NvmlPoller>(*nvml, o.nvml_us);
    }
    auto timed = [&]() {
      CUDA_CHECK(cudaEventRecord(begin));
      cutlass_check(gemm.run(), "run");
      CUDA_CHECK(cudaEventRecord(end));
      if (poller) {
        poller->last_ns = 0;  // one row right after launch
        while (cudaEventQuery(end) == cudaErrorNotReady) poller->poll();
      }
      CUDA_CHECK(cudaEventSynchronize(end));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      return ms;
    };

    // Warm sequence: R00 measure_gemm rule, then one timed call whose trace is read.
    int64_t warm_start_ns = NvmlPoller::now();
    std::vector<float> warm;
    for (int i = 0; i < 30; ++i) {
      warm.push_back(timed());
      if (i >= 7 && coefficient_of_variation({warm.end() - 5, warm.end()}) <= 0.02)
        break;
    }
    while (NvmlPoller::now() - warm_start_ns < int64_t(o.sustain_ms) * 1000000)
      warm.push_back(timed());
#ifdef V03_TRACE
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    int64_t call_start_ns = NvmlPoller::now();
    float ms = timed();
    int64_t call_end_ns = NvmlPoller::now();

    std::vector<uint64_t> host;
#ifdef V03_TRACE
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
    std::cout << "{\"event\":\"call\",\"label\":\"warm\",\"elapsed_us\":" << ms * 1e3
              << ",\"probe_ghz_median\":" << ghz[ghz.size() / 2]
              << ",\"warmup_calls\":" << warm.size();
    print_f("warmup_us", warm_us);
    print_u64("trace", host);
    std::cout << "}\n";
    if (o.nvml_us > 0) {
      std::cout << "{\"event\":\"nvml\",\"ok\":" << (poller ? "true" : "false")
                << ",\"period_us\":" << o.nvml_us << ",\"warm_start_ns\":0"
                << ",\"call_start_ns\":" << call_start_ns - warm_start_ns
                << ",\"call_end_ns\":" << call_end_ns - warm_start_ns << ",\"rows\":[";
      if (poller) {
        for (size_t i = 0; i < poller->rows.size(); ++i) {
          const auto& r = poller->rows[i];
          std::cout << (i ? "," : "") << '[' << r.t_ns - warm_start_ns << ',' << r.gpu_w << ','
                    << r.module_w << ',' << r.avg_w << ',' << r.sm_mhz << ',' << r.reasons << ','
                    << r.latency_ns << ']';
        }
      }
      std::cout << "]}\n";
    }

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
