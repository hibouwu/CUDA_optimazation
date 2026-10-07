// R08: wave/tail scaling and tile traversal (raster, swizzle, cluster) of the fixed CUTLASS GEMM.
//
// Kernel types are identical to r00_cutlass.cu / r07_probe.cu (FP16 -> FP32, 128x256x64, 4 stage,
// TMA warp-specialized cooperative, ElementC=void, epilogue 128x32). The cluster M extent is
// R08_CLUSTER_M (default 2); a 1x1x1 build is a separate binary.
//
// Modes:
//   time:  module-load calls, idle, then `--calls` isolated calls (event pair around gemm.run()),
//          each followed by a 20 us clock64/globaltimer probe; optional CUDA graph of `--graph`
//          calls (device time without per-call host submission).
//   stamp: the same Kernel::operator() launched through a wrapper __global__ that records, per CTA,
//          %smid, globaltimer at entry (thread 0) and the latest globaltimer after operator()
//          returns (lane 0 of every warp, atomicMax). The mainloop code is not modified.
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <chrono>
#include <thread>

#ifndef R08_CLUSTER_M
#define R08_CLUSTER_M 2
#endif

using namespace cute;
using KernelTile = Shape<_128, _256, _64>;
using KernelCluster = Shape<Int<R08_CLUSTER_M>, _1, _1>;
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
using RasterOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

static void cutlass_check(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + what + ": " +
                             cutlassGetStatusString(status));
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Per CTA: [0] entry globaltimer, [1] latest exit globaltimer, [2] smid, [3] linear CTA id.
template <class Operator>
__global__ __launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void r08_stamped_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params,
                        unsigned long long* stamps) {
  extern __shared__ char smem[];
  unsigned cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    stamps[4 * cta] = globaltimer();
    stamps[4 * cta + 2] = smid;
    stamps[4 * cta + 3] = cta;
  }
  Operator op;
  op(params, smem);
  if ((threadIdx.x & 31) == 0)
    atomicMax(&stamps[4 * cta + 1], (unsigned long long)globaltimer());
}

__global__ void clock_probe(uint64_t window_ns, uint64_t* out) {
  if (threadIdx.x != 0)
    return;
  uint64_t t0 = globaltimer(), t1;
  uint64_t c0 = clock64();
  do {
    t1 = globaltimer();
  } while (t1 - t0 < window_ns);
  uint64_t c1 = clock64();
  t1 = globaltimer();
  out[2 * blockIdx.x] = c1 - c0;
  out[2 * blockIdx.x + 1] = t1 - t0;
}

struct ClockProbe {
  int blocks;
  DeviceBuffer<uint64_t> buffer;
  explicit ClockProbe(int sms) : blocks(sms), buffer(2 * size_t(sms)) {}
  void enqueue(cudaStream_t stream) {
    clock_probe<<<blocks, 32, 0, stream>>>(20000, buffer.pointer);
    CUDA_CHECK(cudaGetLastError());
  }
  double median_ghz() {
    std::vector<uint64_t> raw(2 * blocks);
    CUDA_CHECK(cudaMemcpy(raw.data(), buffer.pointer, raw.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> ghz;
    for (int i = 0; i < blocks; ++i)
      ghz.push_back(double(raw[2 * i]) / double(raw[2 * i + 1]));
    std::sort(ghz.begin(), ghz.end());
    return ghz[ghz.size() / 2];
  }
};

struct Args {
  std::string mode = "time", raster = "h";
  int m = 2048, n = 2048, k = 4096, swizzle = 1, calls = 5, graph = 0, launches = 3;
  double idle_s = 0.5;
  Args(int argc, char** argv) {
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--mode") mode = v;
      else if (key == "--raster") raster = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else if (key == "--swizzle") swizzle = std::stoi(v);
      else if (key == "--calls") calls = std::stoi(v);
      else if (key == "--graph") graph = std::stoi(v);
      else if (key == "--launches") launches = std::stoi(v);
      else if (key == "--idle-s") idle_s = std::stod(v);
      else throw std::runtime_error("unknown option " + key);
    }
    if (raster != "h" && raster != "m" && raster != "n")
      throw std::runtime_error("raster must be h, m or n");
  }
};

template <class T>
static void print_list(const char* name, const std::vector<T>& v) {
  std::cout << ",\"" << name << "\":[";
  for (size_t i = 0; i < v.size(); ++i)
    std::cout << (i ? "," : "") << v[i];
  std::cout << ']';
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
  try {
    Args o(argc, argv);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ClockProbe probe(prop.multiProcessorCount);

    DeviceBuffer<__half> a(size_t(o.m) * o.k), b(size_t(o.k) * o.n);
    DeviceBuffer<float> d(size_t(o.m) * o.n);
    Options check(1, argv);  // R00 sampled FP64 check on the same dyadic witness
    check.m = o.m, check.n = o.n, check.k = o.k, check.samples = 256;
    fill_input<<<256, 256>>>(a.pointer, o.m, o.k, check.seed, true, false);
    fill_input<<<256, 256>>>(b.pointer, o.k, o.n, check.seed, false, false);
    CUDA_CHECK(cudaMemset(d.pointer, 0xff, size_t(o.m) * o.n * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(o.m, o.k, 1));
    auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(o.n, o.k, 1));
    auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(o.m, o.n, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {o.m, o.n, o.k, 1},
                                  {reinterpret_cast<cutlass::half_t*>(a.pointer), sa,
                                   reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
                                  {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
    args.scheduler.max_swizzle_size = o.swizzle;
    args.scheduler.raster_order = o.raster == "m"   ? RasterOptions::AlongM
                                  : o.raster == "n" ? RasterOptions::AlongN
                                                    : RasterOptions::Heuristic;
    Gemm gemm;
    cutlass_check(gemm.can_implement(args), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(args)));
    cutlass_check(gemm.initialize(args, workspace.pointer, s), "initialize");
    const auto& params = gemm.params();
    const auto& sp = params.scheduler;
    dim3 grid = Gemm::get_grid_shape(params);
    const int smem = int(sizeof(typename Kernel::SharedStorage));
    const int threads = int(Kernel::MaxThreadsPerBlock);

    // Co-resident cluster count reported by the runtime for this launch shape.
    int max_active_clusters = -1;
    {
      cudaLaunchConfig_t cfg{};
      cudaLaunchAttribute attr[1];
      cfg.gridDim = grid;
      cfg.blockDim = dim3(threads, 1, 1);
      cfg.dynamicSmemBytes = smem;
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = R08_CLUSTER_M, attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr, cfg.numAttrs = 1;
      auto status = cudaOccupancyMaxActiveClusters(
          &max_active_clusters, cutlass::device_kernel<Kernel>, &cfg);
      if (status != cudaSuccess) {
        cudaGetLastError();
        max_active_clusters = -1;
      }
    }

    std::cout << std::setprecision(9) << "{\"event\":\"config\",\"m\":" << o.m << ",\"n\":" << o.n
              << ",\"k\":" << o.k << ",\"cluster_m\":" << R08_CLUSTER_M << ",\"raster_option\":\""
              << o.raster << "\",\"max_swizzle\":" << o.swizzle
              << ",\"raster_actual\":\""
              << (sp.raster_order_ == cutlass::gemm::kernel::detail::RasterOrder::AlongN ? "N" : "M")
              << "\",\"log_swizzle\":" << sp.log_swizzle_size_
              << ",\"blocks_per_problem\":" << sp.blocks_per_problem_
              << ",\"sched_tiles_m\":" << sp.problem_tiles_m_ * sp.cluster_shape_m_
              << ",\"sched_tiles_n\":" << sp.problem_tiles_n_ * sp.cluster_shape_n_
              << ",\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z << ']'
              << ",\"smem\":" << smem << ",\"threads\":" << threads
              << ",\"max_active_clusters\":" << max_active_clusters
              << ",\"sm_count\":" << prop.multiProcessorCount << "}\n";

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float ms = 0;
    auto call = [&]() { cutlass_check(gemm.run(s), "run"); };
    for (int i = 0; i < 2; ++i)
      call();  // module load and first touch; not timed
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (o.mode == "time") {
      std::this_thread::sleep_for(std::chrono::duration<double>(o.idle_s));
      probe.enqueue(s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      double ghz_before = probe.median_ghz();
      std::vector<double> call_us, ghz_after;
      for (int i = 0; i < o.calls; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        call();
        CUDA_CHECK(cudaEventRecord(e1, s));
        probe.enqueue(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        call_us.push_back(ms * 1e3);
        ghz_after.push_back(probe.median_ghz());
      }
      double graph_us = 0, graph_ghz_after = 0;
      if (o.graph > 0) {
        cudaGraph_t g;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        for (int i = 0; i < o.graph; ++i)
          call();
        CUDA_CHECK(cudaStreamEndCapture(s, &g));
        CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
        CUDA_CHECK(cudaGraphUpload(exec, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(exec, s));
        CUDA_CHECK(cudaEventRecord(e1, s));
        probe.enqueue(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        graph_us = ms * 1e3 / o.graph;
        graph_ghz_after = probe.median_ghz();
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(g));
      }
      std::cout << "{\"event\":\"time\",\"call_median_us\":" << median(call_us)
                << ",\"ghz_before\":" << ghz_before << ",\"ghz_after_median\":"
                << median(ghz_after) << ",\"graph_n\":" << o.graph << ",\"graph_us\":" << graph_us
                << ",\"graph_ghz_after\":" << graph_ghz_after;
      print_list("call_us", call_us);
      print_list("ghz_after", ghz_after);
      std::cout << "}\n";
    } else if (o.mode == "stamp") {
      auto kernel = r08_stamped_kernel<Kernel>;
      CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      int ctas = int(grid.x * grid.y * grid.z);
      DeviceBuffer<unsigned long long> stamps(4 * size_t(ctas));
      cudaLaunchConfig_t cfg{};
      cudaLaunchAttribute attr[1];
      cfg.gridDim = grid;
      cfg.blockDim = dim3(threads, 1, 1);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = s;
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = R08_CLUSTER_M, attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr, cfg.numAttrs = 1;
      CUDA_CHECK(cudaMemset(d.pointer, 0xff, size_t(o.m) * o.n * sizeof(float)));
      for (int launch = 0; launch < o.launches; ++launch) {
        std::this_thread::sleep_for(std::chrono::duration<double>(o.idle_s));
        CUDA_CHECK(cudaMemset(stamps.pointer, 0, stamps.count * 8));
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, params, stamps.pointer));
        CUDA_CHECK(cudaEventRecord(e1, s));
        probe.enqueue(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<unsigned long long> raw(stamps.count);
        CUDA_CHECK(cudaMemcpy(raw.data(), stamps.pointer, raw.size() * 8,
                              cudaMemcpyDeviceToHost));
        // Tiles per CTA from the scheduler's linear index (see StaticPersistentTileScheduler):
        // AlongN: blockIdx.x + blockIdx.y * gridDim.x; AlongM: blockIdx.x * gridDim.y + blockIdx.y.
        bool along_n = sp.raster_order_ == cutlass::gemm::kernel::detail::RasterOrder::AlongN;
        std::vector<unsigned long long> start, end, smid;
        std::vector<int> tiles;
        for (int c = 0; c < ctas; ++c) {
          int bx = c % grid.x, by = (c / grid.x) % grid.y;
          uint64_t linear = along_n ? uint64_t(bx) + uint64_t(by) * grid.x
                                    : uint64_t(bx) * grid.y + by;
          int count = 0;
          for (uint64_t q = linear; q < sp.blocks_per_problem_; q += uint64_t(ctas))
            ++count;
          start.push_back(raw[4 * c]);
          end.push_back(raw[4 * c + 1]);
          smid.push_back(raw[4 * c + 2]);
          tiles.push_back(count);
        }
        std::cout << "{\"event\":\"stamp\",\"launch\":" << launch << ",\"event_us\":" << ms * 1e3
                  << ",\"ghz_after\":" << probe.median_ghz();
        print_list("start_ns", start);
        print_list("end_ns", end);
        print_list("smid", smid);
        print_list("tiles", tiles);
        std::cout << "}\n";
      }
    } else {
      throw std::runtime_error("unknown mode " + o.mode);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    Errors e = check_output<__half, float>(check, d.pointer, false);
    bool ok = e.nonfinite == 0 && e.max_storage_error <= 1e-5;
    std::cout << "{\"event\":\"check\",\"samples\":" << e.indices.size()
              << ",\"max_storage_reference_error\":" << e.max_storage_error
              << ",\"status\":\"" << (ok ? "ok" : "numeric_error") << "\"}\n";
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
