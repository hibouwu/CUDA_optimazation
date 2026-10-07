// R07: launch-overhead split and clock-under-load probe.
//
// Backends: the fixed R00 CUTLASS kernel (same type definitions as r00_cutlass.cu),
// cuBLASLt FP16->FP32 (first legal heuristic, 64 MiB workspace, as r00_gemm.cu),
// and an empty kernel launched either <<<1,32>>> or with the CUTLASS launch shape.
//
// Modes:
//   timing: per-call time as (a) isolated event pair around one call, host synchronized
//           between calls, (b) back-to-back stream launches, (c) one CUDA graph of N calls.
//   r00seq: R00 warm-up/timed-call sequence after idle, with clock probes before and after.
//   loop:   warm call -> idle -> cold clock probe -> one cold call -> back-to-back batches
//           each batch followed by a short clock64/globaltimer probe; host realtime stamps.
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <cublasLt.h>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <time.h>

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

#define LT_CHECK(expr)                                                       \
  do {                                                                       \
    cublasStatus_t s_ = (expr);                                              \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                         \
      throw std::runtime_error(std::string(#expr) + ": " + std::to_string(s_)); \
  } while (0)

static void cutlass_check(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + what + ": " +
                             cutlassGetStatusString(status));
}

static int64_t realtime_ns() {
  timespec t{};
  clock_gettime(CLOCK_REALTIME, &t);
  return int64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}

__global__ void empty_kernel() {}

// One block per SM: spin `window_ns` of globaltimer and record clock64 cycles.
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

struct ClockProbe {
  int blocks;
  DeviceBuffer<uint64_t> buffer;
  explicit ClockProbe(int sms) : blocks(sms), buffer(2 * size_t(sms)) {}
  void enqueue(cudaStream_t stream, uint64_t window_ns) {
    clock_probe<<<blocks, 32, 0, stream>>>(window_ns, buffer.pointer);
    CUDA_CHECK(cudaGetLastError());
  }
  // Returns {median, min, max} GHz over SMs. Call after the stream is synchronized.
  std::vector<double> read() {
    std::vector<uint64_t> raw(2 * blocks);
    CUDA_CHECK(cudaMemcpy(raw.data(), buffer.pointer, raw.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> ghz;
    for (int i = 0; i < blocks; ++i)
      ghz.push_back(double(raw[2 * i]) / double(raw[2 * i + 1]));
    std::sort(ghz.begin(), ghz.end());
    return {ghz[ghz.size() / 2], ghz.front(), ghz.back()};
  }
};

struct Args {
  std::string backend = "cutlass", mode = "timing";
  int m = 2048, n = 2048, k = 2048, isolated = 50, burst = 200;
  double seconds = 8, batch_ms = 50, idle_s = 3, tail_s = 2;
  Args(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--backend") backend = v;
      else if (key == "--mode") mode = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else if (key == "--isolated") isolated = std::stoi(v);
      else if (key == "--burst") burst = std::stoi(v);
      else if (key == "--seconds") seconds = std::stod(v);
      else if (key == "--batch-ms") batch_ms = std::stod(v);
      else if (key == "--idle-s") idle_s = std::stod(v);
      else if (key == "--tail-s") tail_s = std::stod(v);
      else throw std::runtime_error("unknown option " + key);
    }
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
  }
};

// cuBLASLt FP16 row-major via column-major D^T = B^T A^T, as in r00_gemm.cu.
struct LtGemm {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t la{}, lb{}, ld{};
  cublasLtMatmulPreference_t pref{};
  cublasLtMatmulAlgo_t algo{};
  DeviceBuffer<unsigned char> workspace{64 * 1024 * 1024};
  const void *a, *b;
  void* d;
  LtGemm(const Args& o, const void* A, const void* B, void* D) : a(B), b(A), d(D) {
    LT_CHECK(cublasLtCreate(&handle));
    LT_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, o.n, o.k, o.n));
    LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, o.k, o.m, o.k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, o.n, o.m, o.n));
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t limit = workspace.count;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &limit, sizeof(limit)));
    cublasLtMatmulHeuristicResult_t found[8]{};
    int count = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld, ld, pref, 8, found, &count));
    int i = 0;
    while (i < count && found[i].state != CUBLAS_STATUS_SUCCESS) ++i;
    if (i == count) throw std::runtime_error("no legal cuBLASLt candidate");
    algo = found[i].algo;
  }
  void launch(cudaStream_t s) {
    float alpha = 1, beta = 0;
    LT_CHECK(cublasLtMatmul(handle, desc, &alpha, a, la, b, lb, &beta, d, ld, d, ld, &algo,
                            workspace.pointer, workspace.count, s));
  }
};

struct Times {
  std::vector<double> isolated_us, host_call_us;
  double burst_us = 0, graph_us = 0;
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

static Times measure(const Args& o, cudaStream_t s, const std::function<void(cudaStream_t)>& call) {
  Times t;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float ms = 0;
  for (int i = 0; i < 20; ++i) call(s);  // warm-up
  CUDA_CHECK(cudaStreamSynchronize(s));
  // (a) isolated: idle stream, event pair around one call; includes host launch path.
  for (int i = 0; i < o.isolated; ++i) {
    CUDA_CHECK(cudaEventRecord(e0, s));
    auto h0 = std::chrono::steady_clock::now();
    call(s);
    auto h1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    t.isolated_us.push_back(ms * 1e3);
    t.host_call_us.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
  }
  // (b) back-to-back stream launches.
  CUDA_CHECK(cudaEventRecord(e0, s));
  for (int i = 0; i < o.burst; ++i) call(s);
  CUDA_CHECK(cudaEventRecord(e1, s));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  t.burst_us = ms * 1e3 / o.burst;
  // (c) one graph of `burst` calls; second launch is timed.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
  for (int i = 0; i < o.burst; ++i) call(s);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaEventRecord(e0, s));
  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaEventRecord(e1, s));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  t.graph_us = ms * 1e3 / o.burst;
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  return t;
}

static void print_list(const char* name, const std::vector<double>& v) {
  std::cout << ",\"" << name << "\":[";
  for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << v[i];
  std::cout << ']';
}

// R00 measure_gemm sequence: synchronized warm-up calls until the last 5 have CV<=2% (i>=7,
// at most 30), then one timed call; clock probe right before the sequence and after it.
static void run_r00_sequence(const Args& o, cudaStream_t s, ClockProbe& probe,
                             const std::function<void(cudaStream_t)>& call) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  call(s);  // module load, outside the sequence
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::this_thread::sleep_for(std::chrono::duration<double>(o.idle_s));
  probe.enqueue(s, 20000);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto before = probe.read();
  auto timed = [&]() {
    float ms = 0;
    CUDA_CHECK(cudaEventRecord(e0, s));
    call(s);
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
  };
  std::vector<float> warm;
  for (int i = 0; i < 30; ++i) {
    warm.push_back(timed());
    if (i >= 7 && coefficient_of_variation({warm.end() - 5, warm.end()}) <= 0.02) break;
  }
  float last = timed();
  probe.enqueue(s, 20000);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto after = probe.read();
  std::vector<double> warm_us;
  for (float w : warm) warm_us.push_back(w * 1e3);
  std::cout << "{\"event\":\"r00seq\",\"timed_us\":" << last * 1e3
            << ",\"ghz_before\":" << before[0] << ",\"ghz_after\":" << after[0]
            << ",\"warmup_calls\":" << warm.size();
  print_list("warmup_us", warm_us);
  std::cout << "}\n";
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
}

static void run_loop(const Args& o, cudaStream_t s, ClockProbe& probe,
                     const std::function<void(cudaStream_t)>& call) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float ms = 0;
  call(s);  // loads the module/library kernel; excluded from the cold call below
  CUDA_CHECK(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::duration<double>(o.idle_s));
  // Cold: clock probe right after idle, then one call, then another probe.
  probe.enqueue(s, 20000);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto cold_before = probe.read();
  int64_t cold_ns = realtime_ns();
  CUDA_CHECK(cudaEventRecord(e0, s));
  call(s);
  CUDA_CHECK(cudaEventRecord(e1, s));
  probe.enqueue(s, 20000);
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  auto cold_after = probe.read();
  std::cout << std::setprecision(9) << "{\"event\":\"cold\",\"t_ns\":" << cold_ns
            << ",\"call_us\":" << ms * 1e3 << ",\"ghz_before\":" << cold_before[0]
            << ",\"ghz_after\":" << cold_after[0] << "}\n";
  // Calibrate batch length with 5 back-to-back calls.
  CUDA_CHECK(cudaEventRecord(e0, s));
  for (int i = 0; i < 5; ++i) call(s);
  CUDA_CHECK(cudaEventRecord(e1, s));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  int batch = std::max(1, int(o.batch_ms / (ms / 5)));
  int64_t start = realtime_ns();
  std::cout << "{\"event\":\"loop_start\",\"t_ns\":" << start << ",\"batch\":" << batch << "}\n";
  for (int index = 0; realtime_ns() - start < int64_t(o.seconds * 1e9); ++index) {
    CUDA_CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < batch; ++i) call(s);
    CUDA_CHECK(cudaEventRecord(e1, s));
    probe.enqueue(s, 20000);  // 20 us after a batch of ~50 ms: <0.1% duty
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    auto g = probe.read();
    std::cout << "{\"event\":\"batch\",\"index\":" << index << ",\"t_ns\":" << realtime_ns()
              << ",\"call_us\":" << ms * 1e3 / batch << ",\"ghz\":" << g[0]
              << ",\"ghz_min\":" << g[1] << ",\"ghz_max\":" << g[2] << "}\n";
  }
  std::cout << "{\"event\":\"loop_end\",\"t_ns\":" << realtime_ns() << "}\n";
  std::this_thread::sleep_for(std::chrono::duration<double>(o.tail_s));
  probe.enqueue(s, 20000);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::cout << "{\"event\":\"tail\",\"t_ns\":" << realtime_ns()
            << ",\"ghz\":" << probe.read()[0] << "}\n";
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  try {
    Args o(argc, argv);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ClockProbe probe(prop.multiProcessorCount);
    std::function<void(cudaStream_t)> call;

    // Inputs exist for every backend so that the GEMM paths can be checked.
    bool gemm = o.backend != "empty1";  // empty_cutlass_shape needs CUTLASS params for its grid
    size_t ma = gemm ? size_t(o.m) * o.k : 1, mb = gemm ? size_t(o.k) * o.n : 1;
    size_t md = gemm ? size_t(o.m) * o.n : 1;
    DeviceBuffer<__half> a(ma), b(mb);
    DeviceBuffer<float> d(md);
    Options check(1, argv);  // reuse R00 sampled FP64 check
    check.m = o.m, check.n = o.n, check.k = o.k, check.samples = 256;
    if (gemm) {
      fill_input<<<256, 256>>>(a.pointer, o.m, o.k, check.seed, true, false);
      fill_input<<<256, 256>>>(b.pointer, o.k, o.n, check.seed, false, false);
      CUDA_CHECK(cudaMemset(d.pointer, 0xff, md * sizeof(float)));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    Gemm cutlass_gemm;
    std::unique_ptr<DeviceBuffer<unsigned char>> workspace;
    std::unique_ptr<LtGemm> lt;
    const int smem = int(sizeof(typename Kernel::SharedStorage));
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[1];
    dim3 cutlass_grid;
    if (o.backend == "cutlass" || o.backend == "empty_cutlass_shape") {
      auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},
                                                 make_shape(o.m, o.k, 1));
      auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},
                                                 make_shape(o.n, o.k, 1));
      auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{},
                                                 make_shape(o.m, o.n, 1));
      typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    {o.m, o.n, o.k, 1},
                                    {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
                                     reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
                                    {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
      cutlass_check(cutlass_gemm.can_implement(args), "can_implement");
      workspace.reset(new DeviceBuffer<unsigned char>(
          std::max<size_t>(1, Gemm::get_workspace_size(args))));
      cutlass_check(cutlass_gemm.initialize(args, workspace->pointer), "initialize");
      cutlass_grid = Gemm::get_grid_shape(cutlass_gemm.params());
      call = [&](cudaStream_t st) { cutlass_check(cutlass_gemm.run(st), "run"); };
    }
    if (o.backend == "empty_cutlass_shape") {
      // Same grid/cluster/threads/dynamic SMEM as the CUTLASS launch at this M, N; no work.
      CUDA_CHECK(cudaFuncSetAttribute(empty_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem));
      cfg.gridDim = cutlass_grid;
      cfg.blockDim = dim3(Kernel::MaxThreadsPerBlock, 1, 1);
      cfg.dynamicSmemBytes = smem;
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = 2, attr[0].val.clusterDim.y = 1, attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr, cfg.numAttrs = 1;
      call = [&](cudaStream_t st) {
        cfg.stream = st;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, empty_kernel));
      };
      gemm = false;  // nothing written to D
    } else if (o.backend == "cublaslt") {
      lt.reset(new LtGemm(o, a.pointer, b.pointer, d.pointer));
      call = [&](cudaStream_t st) { lt->launch(st); };
    } else if (o.backend == "empty1") {
      call = [&](cudaStream_t st) {
        empty_kernel<<<1, 32, 0, st>>>();
        CUDA_CHECK(cudaGetLastError());
      };
    } else if (o.backend != "cutlass") {
      throw std::runtime_error("unknown backend " + o.backend);
    }

    std::cout << std::setprecision(9);
    if (o.mode == "loop") {
      run_loop(o, s, probe, call);
    } else if (o.mode == "r00seq") {
      run_r00_sequence(o, s, probe, call);
    } else if (o.mode == "timing") {
      Times t = measure(o, s, call);
      std::cout << "{\"event\":\"timing\",\"backend\":\"" << o.backend << "\",\"m\":" << o.m
                << ",\"n\":" << o.n << ",\"k\":" << o.k << ",\"smem\":" << smem
                << ",\"cutlass_grid\":[" << cutlass_grid.x << ',' << cutlass_grid.y << ','
                << cutlass_grid.z << ']'
                << ",\"isolated_median_us\":" << median(t.isolated_us)
                << ",\"host_call_median_us\":" << median(t.host_call_us)
                << ",\"burst_us\":" << t.burst_us << ",\"graph_us\":" << t.graph_us
                << ",\"burst_n\":" << o.burst;
      print_list("isolated_us", t.isolated_us);
      std::cout << "}\n";
    } else {
      throw std::runtime_error("unknown mode " + o.mode);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (gemm) {
      Errors e = check_output<__half, float>(check, d.pointer, false);
      bool ok = e.nonfinite == 0 && e.max_storage_error <= 1e-5;
      std::cout << "{\"event\":\"check\",\"samples\":" << e.indices.size()
                << ",\"max_storage_reference_error\":" << e.max_storage_error
                << ",\"status\":\"" << (ok ? "ok" : "numeric_error") << "\"}\n";
      return ok ? 0 : 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
