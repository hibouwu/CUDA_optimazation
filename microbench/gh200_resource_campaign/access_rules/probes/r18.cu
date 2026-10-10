// R18/R19: physical-grid matched boundaries, explicit zeros, and recorded work coordinates.
// Copied from the earlier probe: explicit lda/ldb/ldd, warm protocol, dyadic exact sampled check.
#include "gaps_common.hpp"
#include <algorithm>
#include <memory>
#ifdef R10_WORK_SWAP
// Experimental initial-work permutation; every later advance keeps the same stride.
__device__ __constant__ int r10_work_swap55_91;
#endif
#ifdef R13_FIRST_SUPPLY
__device__ __constant__ unsigned r13_first_supply_events;
#endif
#ifdef R10_PRODUCER_DELAY
__device__ __constant__ unsigned r10_producer_delay_ns;
__device__ __constant__ uint64_t* r10_producer_delay_trace;
__device__ __forceinline__ void r10_delay_first_producer() {
  if (blockIdx.x!=55 || blockIdx.y!=0 || blockIdx.z!=0) return;
  if (threadIdx.x==0) {
    uint64_t begin_ns, now;
#ifdef R18_DUAL_CLOCK
    uint64_t begin_c, end_c, end_ns;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin_c) :: "memory");
#endif
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin_ns) :: "memory");
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory"); }
    while (now-begin_ns < r10_producer_delay_ns);
#ifdef R18_DUAL_CLOCK
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_c) :: "memory");
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_ns) :: "memory");
    r10_producer_delay_trace[0]=begin_c; r10_producer_delay_trace[1]=begin_ns;
    r10_producer_delay_trace[2]=end_c; r10_producer_delay_trace[3]=end_ns;
#endif
  }
  __syncwarp();
}
#endif
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#ifdef V06_TRACE
#include "v06_trace.hpp"
#ifndef V06_KERNEL_OVERLAY_ACTIVE
#error "V06 trace must compile the kernel overlay headers"
#endif
#endif

using namespace cute;
#ifndef V06_CFG
#error "compile with -DV06_CFG=0|1|2"
#endif
#if V06_CFG == 0 || V06_CFG == 3
#if V06_CFG == 3
constexpr const char* kConfigName = "cfg_a1";
#else
constexpr const char* kConfigName = "cfg_a";
#endif
using KernelTile = Shape<_128, _128, _64>;
#if V06_CFG == 3
using KernelCluster = Shape<_1, _1, _1>;
#else
using KernelCluster = Shape<_2, _1, _1>;
#endif
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define V06_STAGES_AUTO 1
#elif V06_CFG == 1
constexpr const char* kConfigName = "cfg_b";
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_64, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#define V06_STAGES_AUTO 1
#elif V06_CFG == 2 || V06_CFG == 4
#if V06_CFG == 4
constexpr const char* kConfigName = "cfg_c1";
#else
constexpr const char* kConfigName = "cfg_c";
#endif
using KernelTile = Shape<_256, _128, _64>;
#if V06_CFG == 4
using KernelCluster = Shape<_1, _1, _1>;
#else
using KernelCluster = Shape<_1, _2, _1>;
#endif
using EpilogueTile = Shape<_128, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#define V06_STAGES_AUTO 0
#else
#error "unknown V06_CFG"
#endif
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, KernelTile, KernelCluster, EpilogueTile,
    float, float, void, cutlass::layout::RowMajor, 4, float, cutlass::layout::RowMajor, 4,
    EpilogueSchedule>::CollectiveOp;
#if defined(R18_STAGE_COUNT)
using StageCount = cutlass::gemm::collective::StageCount<R18_STAGE_COUNT>;
#elif V06_STAGES_AUTO
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
#if V06_CFG != 1
static_assert(Kernel::NumMMAThreads == 256, "recheck cooperative TMA issuing warp mapping");
#endif

#if defined(R13_STAGE_PROGRESS)
constexpr const char* kTraceVersion="r13-stage-progress-v1";
#elif defined(R13_STAGE_SPARSE)
constexpr const char* kTraceVersion="r13-stage-sparse-v1";
#elif defined(R18_DUAL_CLOCK)
constexpr const char* kTraceVersion="r18-dual-clock-events";
#elif defined(R18_MATCH_DUAL_LAYOUT)
constexpr const char* kTraceVersion="r18-wide-clock-events";
#elif defined(R15_OUTPUT_NS)
constexpr const char* kTraceVersion="r15-first-output-ns";
#elif defined(V08_ENDS)
constexpr const char* kTraceVersion="v08-ends";
#elif defined(R18_LIGHT)
constexpr const char* kTraceVersion="r18-light-events";
#else
constexpr const char* kTraceVersion="r18-work-coordinates";
#endif
#if defined(R18_DUAL_CLOCK) || defined(R18_MATCH_DUAL_LAYOUT)
constexpr int kTraceTileWords = 10;
#else
constexpr int kTraceTileWords = 6;
#endif
constexpr int kTraceWords = 16 + 2 * 64 * kTraceTileWords;  // = V06CtaWords


static void cutlass_check(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS ") + what + ": " +
                             cutlassGetStatusString(status));
}

struct Args {
  std::string mode = "warm";
  int m = 1280, n = 1536, k = 1536;
  int zero_m=-1,zero_n=-1,storage_m=0,storage_n=0,swizzle=1,evict=0;
  int input_map_m=0,input_map_n=0;
  int sm_count = 0;
  std::string input_mode = "dyadic";
  std::string warmup_policy = "adaptive";
  std::string history_mode = "off";
  std::string history_source = "original";
  int mirror_target_slot = 0;
  std::string raster_order = "heuristic";
  int seed = 17;
  int work_swap55_91 = 0;
  unsigned producer_delay_ns = 0;
  unsigned first_supply_events = 0;
  int64_t lda = 0, ldb = 0, ldd = 0;
  int64_t alloc_lda = 0, alloc_ldb = 0;
  int b_offset_bytes = 0, b_allocation_extra_bytes = 0;
  Args(int argc, char** argv) {
    if (argc % 2 == 0)
      throw std::runtime_error("options are --key value pairs");
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string key = argv[i], v = argv[i + 1];
      if (key == "--mode") mode = v;
      else if (key == "--m") m = std::stoi(v);
      else if (key == "--n") n = std::stoi(v);
      else if (key == "--k") k = std::stoi(v);
      else if (key == "--lda") lda = std::stoll(v);
      else if (key == "--ldb") ldb = std::stoll(v);
      else if (key == "--ldd") ldd = std::stoll(v);
      else if (key == "--alloc-lda") alloc_lda = std::stoll(v);
      else if (key == "--alloc-ldb") alloc_ldb = std::stoll(v);
      else if (key == "--b-offset-bytes") b_offset_bytes = std::stoi(v);
      else if (key == "--b-allocation-extra-bytes") b_allocation_extra_bytes = std::stoi(v);
      else if(key=="--zero-m")zero_m=std::stoi(v);
      else if(key=="--zero-n")zero_n=std::stoi(v);
      else if(key=="--storage-m")storage_m=std::stoi(v);
      else if(key=="--storage-n")storage_n=std::stoi(v);
      else if(key=="--input-map-m")input_map_m=std::stoi(v);
      else if(key=="--input-map-n")input_map_n=std::stoi(v);
      else if(key=="--swizzle")swizzle=std::stoi(v);
      else if(key=="--evict")evict=std::stoi(v);
      else if(key=="--sm-count")sm_count=std::stoi(v);
      else if(key=="--input-mode")input_mode=v;
      else if(key=="--warmup-policy")warmup_policy=v;
      else if(key=="--history-mode")history_mode=v;
      else if(key=="--history-source")history_source=v;
      else if(key=="--mirror-target-slot")mirror_target_slot=std::stoi(v);
      else if(key=="--raster-order")raster_order=v;
      else if(key=="--seed")seed=std::stoi(v);
      else if(key=="--work-swap55-91")work_swap55_91=std::stoi(v);
      else if(key=="--producer-delay-ns")producer_delay_ns=std::stoul(v);
      else if(key=="--first-supply-events")first_supply_events=std::stoul(v);
      else throw std::runtime_error("unknown option " + key);
    }
    if (first_supply_events > 1) throw std::runtime_error("first-supply-events must be 0 or 1");
#ifndef R13_FIRST_SUPPLY
    if (first_supply_events) throw std::runtime_error("first supply events require experiment build");
#endif
    if (producer_delay_ns != 0 && producer_delay_ns != 512)
      throw std::runtime_error("producer-delay-ns is the declared 0/512 ns pair");
#ifndef R10_PRODUCER_DELAY
    if (producer_delay_ns) throw std::runtime_error("producer delay requires the experiment build");
#endif
    if (work_swap55_91 != 0 && work_swap55_91 != 1)
      throw std::runtime_error("work-swap55-91 must be 0 or 1");
#ifndef R10_WORK_SWAP
    if (work_swap55_91) throw std::runtime_error("work swap requires the experiment build");
#endif
    if (b_offset_bytes < 0 || b_allocation_extra_bytes < b_offset_bytes ||
        b_offset_bytes % 16 || b_allocation_extra_bytes % 16)
      throw std::runtime_error("B base offset must be 16-byte aligned and fit extra allocation");
    if (input_mode != "dyadic" && input_mode != "zero" && input_mode != "random")
      throw std::runtime_error("input-mode must be dyadic, zero or random");
    if (seed < 0)
      throw std::runtime_error("seed must be nonnegative");
    if (warmup_policy != "adaptive" && warmup_policy != "common-sw1-16")
      throw std::runtime_error("warmup-policy must be adaptive or common-sw1-16");
    if (history_mode != "off" && history_mode != "single" && history_mode != "series")
      throw std::runtime_error("history-mode must be off, single or series");
    if (history_mode != "off" && warmup_policy != "common-sw1-16")
      throw std::runtime_error("history observation requires common-sw1-16 prefix");
    if (history_source != "original" && history_source != "same" && history_source != "mirror")
      throw std::runtime_error("history-source must be original, same or mirror");
    if ((mirror_target_slot != 0 && mirror_target_slot != 1) ||
        (history_source == "original" && mirror_target_slot != 0))
      throw std::runtime_error("mirror target slot requires paired source allocations");
    if (history_source != "original" && (history_mode != "series" || swizzle != 1))
      throw std::runtime_error("mirror source comparison uses the sw1 history series");
    if (raster_order != "heuristic" && raster_order != "along_m" && raster_order != "along_n")
      throw std::runtime_error("raster-order must be heuristic, along_m or along_n");
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

__global__ void zero_panel(__half* data,int rows,int cols,int64_t stride,int zero_row,int zero_col){
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(rows)*cols;i+=size_t(gridDim.x)*blockDim.x){
    int row=i/cols,col=i%cols;
    if((zero_row>=0&&row>=zero_row)||(zero_col>=0&&col>=zero_col))data[size_t(row)*stride+col]=__half(0.f);
  }
}

static size_t nonzero_input_rectangle(const __half* data,int rows,int columns,int64_t stride){
  std::vector<uint16_t> bits(size_t(rows)*columns);
  CUDA_CHECK(cudaMemcpy2D(bits.data(),size_t(columns)*2,data,size_t(stride)*2,
                         size_t(columns)*2,rows,cudaMemcpyDeviceToHost));
  return std::count_if(bits.begin(),bits.end(),[](uint16_t x){return x!=0;});
}

inline Errors check_boundary_output(const float* data, int m, int n, int k, int64_t ldd,
                                int seed, int requested, int zero_m, int zero_n,
                                int input_mode, const __half* a, const __half* b,
                                int64_t lda, int64_t ldb, double& max_error_ratio) {
  gaps::validate_layout(m, n, ldd);
  if (k <= 0 || seed < 0)
    throw std::runtime_error("positive K and nonnegative witness seed required");
  Errors result;
  result.indices = gaps::checked_indices(m, n, requested, unsigned(seed));
  DeviceBuffer<uint64_t> indices(result.indices.size());
  DeviceBuffer<float> values(result.indices.size());
  CUDA_CHECK(cudaMemcpy(indices.pointer, result.indices.data(), indices.count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  gaps::gather_strided<<<(indices.count + 255) / 256, 256>>>(
      data, indices.pointer, values.pointer, int(indices.count), n, ldd);
  CUDA_CHECK(cudaGetLastError());
  result.values.resize(indices.count);
  CUDA_CHECK(cudaMemcpy(result.values.data(), values.pointer, values.count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<__half> host_a, host_b;
  if (input_mode == 2) {
    host_a.resize(size_t(m) * k);
    host_b.resize(size_t(k) * n);
    CUDA_CHECK(cudaMemcpy2D(host_a.data(), size_t(k) * 2, a, size_t(lda) * 2,
                           size_t(k) * 2, m, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(host_b.data(), size_t(n) * 2, b, size_t(ldb) * 2,
                           size_t(n) * 2, k, cudaMemcpyDeviceToHost));
  }
  double squared_error = 0, squared_reference = 0;
  for (size_t q = 0; q < result.indices.size(); ++q) {
    uint64_t index = result.indices[q];
    int row = int(index / n), col = int(index % n);
    bool zero = input_mode == 1 || (zero_m >= 0 && row >= zero_m) ||
                (zero_n >= 0 && col >= zero_n);
    double reference = 0, sum_abs_products = 0;
    if (!zero && input_mode == 2) {
      for (int t = 0; t < k; ++t) {
        double product = double(float(host_a[size_t(row) * k + t])) *
                         double(float(host_b[size_t(t) * n + col]));
        reference += product;
        sum_abs_products += std::abs(product);
      }
    } else if (!zero) {
      reference = gaps::dyadic_reference(row, col, k, seed);
    }
    float value = result.values[q];
    if (!std::isfinite(value)) {
      ++result.nonfinite;
      continue;
    }
    double error = std::abs(double(value) - reference);
    if (input_mode == 2)
      max_error_ratio = std::max(max_error_ratio, error / (0x1p-20 + 0x1p-21 * sum_abs_products));
    result.max_absolute = std::max(result.max_absolute, error);
    result.max_storage_error = std::max(result.max_storage_error,
                                       std::abs(double(value) - double(float(reference))));
    if (reference != 0)
      result.max_relative_nonzero = std::max(result.max_relative_nonzero,
                                            error / std::abs(reference));
    squared_error += error * error;
    squared_reference += reference * reference;
  }
  result.sample_relative_l2 = std::sqrt(squared_reference ? squared_error / squared_reference
                                                         : squared_error);
  return result;
}


int main(int argc, char** argv) {
  try {
    Args o(argc, argv);
    std::cout << std::setprecision(10);
    if (o.mode != "warm" && o.mode != "setup" && o.mode != "check-work-swap" && o.mode != "check-producer-delay" && o.mode != "check-first-supply")
      throw std::runtime_error("unknown mode " + o.mode);

    Options check(1, argv);
    check.m = o.m, check.n = o.n, check.k = o.k;
    check.seed = o.seed;
    int input_mode = o.input_mode == "random" ? 2 : o.input_mode == "zero" ? 1 : 0;
    int64_t lda = o.lda ? o.lda : o.k;
    int64_t ldb = o.ldb ? o.ldb : o.n, ldd = o.ldd ? o.ldd : o.n;
    int tn = int(size<1>(KernelTile{}));
    gaps::validate_layout(o.m, o.k, lda);
    gaps::validate_layout(o.k, o.n, ldb);
    gaps::validate_layout(o.m, o.n, ldd);
    int storage_m=o.storage_m?o.storage_m:o.m,storage_n=o.storage_n?o.storage_n:o.n;
    if(storage_m<o.m||storage_n<o.n||ldb<storage_n||ldd<storage_n || (o.swizzle!=1&&o.swizzle!=8))
      throw std::runtime_error("invalid physical storage/swizzle");
    const bool separate_input_map=o.input_map_m||o.input_map_n;
    int input_map_m=o.input_map_m?o.input_map_m:o.m;
    int input_map_n=o.input_map_n?o.input_map_n:o.n;
    if(input_map_m<o.m||input_map_m>storage_m||input_map_n<o.n||input_map_n>storage_n)
      throw std::runtime_error("input tensor map must cover logical input and fit allocated storage");
    int64_t alloc_lda=o.alloc_lda?o.alloc_lda:lda,alloc_ldb=o.alloc_ldb?o.alloc_ldb:ldb;
    if(alloc_lda<lda||alloc_ldb<ldb)throw std::runtime_error("allocation pitch is smaller than physical pitch");
    const size_t b_payload_elements=size_t(o.k)*alloc_ldb;
    DeviceBuffer<__half> a(size_t(storage_m)*alloc_lda), b(b_payload_elements+o.b_allocation_extra_bytes/sizeof(__half));
    const bool mirror_sources = o.history_source != "original";
    std::unique_ptr<DeviceBuffer<__half>> mirror_a, mirror_b;
    if (mirror_sources) {
      mirror_a = std::make_unique<DeviceBuffer<__half>>(a.count);
      mirror_b = std::make_unique<DeviceBuffer<__half>>(b.count);
    }
    __half* target_a = o.mirror_target_slot ? mirror_a->pointer : a.pointer;
    __half* b_data=b.pointer+o.b_offset_bytes/sizeof(__half);
    __half* mirror_b_data=mirror_sources ? mirror_b->pointer+o.b_offset_bytes/sizeof(__half) : nullptr;
    __half* target_b = o.mirror_target_slot ? mirror_b_data : b_data;
    __half* prefix_a = o.history_source == "mirror" ? (o.mirror_target_slot ? a.pointer : mirror_a->pointer) : target_a;
    __half* prefix_b = o.history_source == "mirror" ? (o.mirror_target_slot ? b_data : mirror_b_data) : target_b;
    DeviceBuffer<float> d(size_t(storage_m)*ldd);
    typename Kernel::StrideA sa = make_stride(lda, Int<1>{}, int64_t(o.m) * lda);
    typename Kernel::StrideB sb = make_stride(Int<1>{}, ldb, int64_t(o.k) * ldb);
    typename Kernel::StrideD sd = make_stride(ldd, Int<1>{}, int64_t(o.m) * ldd);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {o.m, o.n, o.k, 1},
                                       {reinterpret_cast<cutlass::half_t*>(target_a), sa,
                                        reinterpret_cast<cutlass::half_t*>(target_b), sb},
                                       {{1.f, 0.f}, nullptr, sd, d.pointer, sd}};
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    char gpu_uuid[41] = "GPU-";
    int uuid_pos = 4;
    for (int i = 0; i < 16; ++i) {
      if (i == 4 || i == 6 || i == 8 || i == 10) gpu_uuid[uuid_pos++] = '-';
      std::snprintf(gpu_uuid + uuid_pos, 3, "%02x", unsigned(uint8_t(properties.uuid.bytes[i])));
      uuid_pos += 2;
    }
    if (o.sm_count < 0 || o.sm_count > properties.multiProcessorCount)
      throw std::runtime_error("sm-count must be zero (all SMs) or within the device SM count");
    int scheduler_sms = o.sm_count ? o.sm_count : properties.multiProcessorCount;
    if (scheduler_sms < int(size(KernelCluster{})))
      throw std::runtime_error("sm-count is smaller than one cluster");
    DeviceBuffer<unsigned> eviction(std::max<size_t>(1,size_t(properties.l2CacheSize)*2/4));
    unsigned eviction_seed=17;
    if(o.evict<0||o.evict>1)throw std::runtime_error("invalid eviction mode");
    arguments.hw_info.device_id = 0;
    arguments.hw_info.sm_count = scheduler_sms;
    arguments.scheduler.max_swizzle_size = o.swizzle;
    using RasterOption = decltype(arguments.scheduler.raster_order);
    arguments.scheduler.raster_order = o.raster_order == "along_m" ? RasterOption::AlongM :
                                      o.raster_order == "along_n" ? RasterOption::AlongN : RasterOption::Heuristic;
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    auto launch_params=gemm.params();
    if(separate_input_map){
      auto input_shape=make_shape(input_map_m,input_map_n,o.k,1);
      if(!Mainloop::can_implement(input_shape,arguments.mainloop))
        throw std::runtime_error("input tensor-map shape is not implementable");
      // Only source descriptors change. Kernel problem_shape, output descriptor,
      // scheduler and work count remain those constructed for the logical GEMM.
      launch_params.mainloop=Mainloop::to_underlying_arguments(input_shape,arguments.mainloop,nullptr);
    }
    dim3 grid = Gemm::get_grid_shape(launch_params);
    const char* resolved_raster = launch_params.scheduler.raster_order_ ==
        decltype(launch_params.scheduler.raster_order_)::AlongN ? "along_n" : "along_m";
    int ctas = int(grid.x * grid.y * grid.z);
    if (ctas <= 0 || ctas > properties.multiProcessorCount)
      throw std::runtime_error("unexpected persistent grid");

#ifdef R10_WORK_SWAP
    if ((o.work_swap55_91 || o.mode == "check-work-swap") &&
        (V06_CFG != 1 || o.m != 2304 || o.n != 3072 || o.k != 1024 ||
         ctas != 132 || o.swizzle != 1 || std::string(resolved_raster) != "along_m"))
      throw std::runtime_error("work swap is only the declared cfg_b persistent pair");
    CUDA_CHECK(cudaMemcpyToSymbol(r10_work_swap55_91, &o.work_swap55_91, sizeof(int)));
#else
    if (o.mode == "check-work-swap") throw std::runtime_error("work swap check requires experiment build");
#endif

#ifdef R10_PRODUCER_DELAY
    if ((o.producer_delay_ns || o.first_supply_events || o.mode == "check-producer-delay" || o.mode == "check-first-supply") &&
        (V06_CFG != 1 || o.m != 2304 || o.n != 3072 || o.k != 1024 ||
         ctas != 132 || o.swizzle != 1 || o.work_swap55_91 || std::string(resolved_raster) != "along_m"))
      throw std::runtime_error("producer delay is only the declared cfg_b no-swap pair");
    CUDA_CHECK(cudaMemcpyToSymbol(r10_producer_delay_ns, &o.producer_delay_ns, sizeof(unsigned)));
#else
    if (o.mode == "check-producer-delay") throw std::runtime_error("producer delay check requires experiment build");
#endif

#ifndef R13_FIRST_SUPPLY
    if (o.mode == "check-first-supply") throw std::runtime_error("first supply check requires experiment build");
#endif
    // Both parameter sets exist before input/cache preparation. The common
    // history uses the same kernel; its source identity is an explicit condition.
    const bool common_history = o.warmup_policy == "common-sw1-16";
    const bool history_series = o.history_mode == "series";
    const bool history_enabled = o.history_mode != "off";
    const char* history_protocol = history_series ? "prefix16-target19-ends-v1"
        : history_enabled ? "single-common16-ends-v1" : "none";
    if (history_enabled && o.mode != "setup") {
#ifndef V08_ENDS
      throw std::runtime_error("history observation requires the ends binary");
#endif
    }
    const int warmup_swizzle = common_history ? 1 : o.swizzle;
    auto warmup_launch_params = launch_params;
    if (common_history) {
      auto warmup_arguments = arguments;
      warmup_arguments.scheduler.max_swizzle_size = 1;
      Gemm warmup_gemm;
      cutlass_check(warmup_gemm.can_implement(warmup_arguments), "warmup can_implement");
      if (Gemm::get_workspace_size(warmup_arguments) > workspace.count)
        throw std::runtime_error("common history needs a different workspace");
      cutlass_check(warmup_gemm.initialize(warmup_arguments, workspace.pointer), "warmup initialize");
      warmup_launch_params = warmup_gemm.params();
      warmup_launch_params.mainloop = launch_params.mainloop;
      if (o.history_source == "mirror") {
        warmup_arguments.mainloop.ptr_A = reinterpret_cast<cutlass::half_t*>(prefix_a);
        warmup_arguments.mainloop.ptr_B = reinterpret_cast<cutlass::half_t*>(prefix_b);
        warmup_launch_params.mainloop = Mainloop::to_underlying_arguments(
            make_shape(input_map_m,input_map_n,o.k,1),warmup_arguments.mainloop,nullptr);
      }
      dim3 warmup_grid = Gemm::get_grid_shape(warmup_launch_params);
      if (warmup_grid.x != grid.x || warmup_grid.y != grid.y || warmup_grid.z != grid.z)
        throw std::runtime_error("common history changes persistent grid");
    }

    // Both single and series allocate the same compact capacity, before fills.
    // Only series records D2D heads; no kernel trace layout or pointer changes.
    constexpr int history_prefix_calls = 16, history_target_calls = 19;
    constexpr int history_calls = history_prefix_calls + history_target_calls;
    std::unique_ptr<DeviceBuffer<uint64_t>> history;
    if (history_enabled) history = std::make_unique<DeviceBuffer<uint64_t>>(size_t(history_calls)*ctas*16);
    std::vector<uint64_t> saved_history;

    // Same preparation order as R00 measure_gemm (fill A, B; D = NaN pattern).
    int input_rows=separate_input_map?storage_m:o.m,input_columns=separate_input_map?storage_n:o.n;
    gaps::fill_input_strided<<<256, 256>>>(a.pointer, input_rows, o.k, lda, check.seed, true, input_mode,o.alloc_lda?a.count:0);
    gaps::fill_input_strided<<<256, 256>>>(b_data, o.k, input_columns, ldb, check.seed, false, input_mode,o.alloc_ldb?b_payload_elements:0);
    CUDA_CHECK(cudaGetLastError());
    if(o.zero_m>=0) zero_panel<<<256,256>>>(a.pointer,input_rows,o.k,lda,o.zero_m,-1);
    if(o.zero_n>=0) zero_panel<<<256,256>>>(b_data,o.k,input_columns,ldb,-1,o.zero_n);
    CUDA_CHECK(cudaGetLastError());
    if (mirror_sources) {
      gaps::fill_input_strided<<<256,256>>>(mirror_a->pointer,input_rows,o.k,lda,check.seed,true,input_mode,o.alloc_lda?a.count:0);
      gaps::fill_input_strided<<<256,256>>>(mirror_b_data,o.k,input_columns,ldb,check.seed,false,input_mode,o.alloc_ldb?b_payload_elements:0);
      if(o.zero_m>=0) zero_panel<<<256,256>>>(mirror_a->pointer,input_rows,o.k,lda,o.zero_m,-1);
      if(o.zero_n>=0) zero_panel<<<256,256>>>(mirror_b_data,o.k,input_columns,ldb,-1,o.zero_n);
      CUDA_CHECK(cudaGetLastError());
    }
    gaps::fill_output_sentinel<<<256, 256>>>(d.pointer, d.count);
    DeviceBuffer<uint64_t> trace(size_t(ctas) * kTraceWords);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#ifdef V06_TRACE
    CUDA_CHECK(cudaMemcpyToSymbol(v06_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
    const bool traced = true;
#else
    const bool traced = false;
#endif
#ifdef R10_PRODUCER_DELAY
#ifdef R13_FIRST_SUPPLY
    DeviceBuffer<uint64_t> delay_trace(8);
    CUDA_CHECK(cudaMemcpyToSymbol(r13_first_supply_events,&o.first_supply_events,sizeof(unsigned)));
#else
    DeviceBuffer<uint64_t> delay_trace(4);
#endif
    CUDA_CHECK(cudaMemset(delay_trace.pointer,0,delay_trace.count*8));
    CUDA_CHECK(cudaMemcpyToSymbol(r10_producer_delay_trace, &delay_trace.pointer, sizeof(delay_trace.pointer)));
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    std::vector<cudaEvent_t> history_begin, history_end;
    if (history_enabled) {
      history_begin.resize(history_calls); history_end.resize(history_calls);
      for (int j=0; j<history_calls; ++j) {
        CUDA_CHECK(cudaEventCreate(&history_begin[j]));
        CUDA_CHECK(cudaEventCreate(&history_end[j]));
      }
    }
    int history_index = 0;
    auto timed = [&](typename Gemm::Params& params) {
      if(o.evict)evict_buffer<<<256,256>>>(eviction.pointer,eviction.count,++eviction_seed);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t call_begin = history_enabled ? history_begin[history_index] : begin;
      cudaEvent_t call_end = history_enabled ? history_end[history_index] : end;
      CUDA_CHECK(cudaEventRecord(call_begin));
      cutlass_check(Gemm::run(params), "run");
      CUDA_CHECK(cudaEventRecord(call_end));
      CUDA_CHECK(cudaEventSynchronize(call_end));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, call_begin, call_end));
      if (history_series) {
        CUDA_CHECK(cudaMemcpy2DAsync(history->pointer+size_t(history_index)*ctas*16,16*sizeof(uint64_t),
            trace.pointer,kTraceWords*sizeof(uint64_t),16*sizeof(uint64_t),ctas,cudaMemcpyDeviceToDevice));
      }
      if (history_enabled) ++history_index;
      return ms;
    };
    // Read every role/tile record after the timed call.
    auto report = [&](const char* label, float ms, const std::vector<float>& warm) {
      std::vector<uint64_t> host;
#ifdef V06_TRACE
      static_assert(kTraceWords == V06CtaWords, "trace layout mismatch");
      host.resize(size_t(ctas) * kTraceWords);
      if (history_series) {
        for (int cta=0; cta<ctas; ++cta)
          std::copy_n(saved_history.data()+(size_t(history_calls-1)*ctas+cta)*16,16,
                      host.data()+size_t(cta)*kTraceWords);
      } else {
        CUDA_CHECK(cudaMemcpy(host.data(), trace.pointer, host.size() * 8, cudaMemcpyDeviceToHost));
      }
      for (int cta = 0; cta < ctas; ++cta)
        if (host[size_t(cta) * kTraceWords + 10])
          throw std::runtime_error("V06 trace tile capacity exceeded");
#endif
#if defined(R10_PRODUCER_DELAY) && defined(R18_DUAL_CLOCK)
      std::vector<uint64_t> delay_events(4);
      CUDA_CHECK(cudaMemcpy(delay_events.data(),delay_trace.pointer,32,cudaMemcpyDeviceToHost));
#ifdef R13_FIRST_SUPPLY
      std::vector<uint64_t> first_events(4);
      CUDA_CHECK(cudaMemcpy(first_events.data(),delay_trace.pointer+4,32,cudaMemcpyDeviceToHost));
#endif
#endif
      std::vector<double> warm_us;
      for (float w : warm) warm_us.push_back(w * 1e3);
      std::cout << "{\"event\":\"call\",\"label\":\"" << label << "\",\"elapsed_us\":" << ms * 1e3
                << ",\"warmup_calls\":" << warm.size()
                << ",\"warmup_policy\":\"" << o.warmup_policy << "\""
                << ",\"warmup_swizzle\":" << warmup_swizzle
                << ",\"warmup_fixed_calls\":" << (common_history ? 16 : 0)
                << ",\"target_swizzle\":" << o.swizzle
                << ",\"history_protocol\":\"" << history_protocol << "\""
                << ",\"history_source\":\"" << o.history_source << "\""
                << ",\"mirror_target_slot\":" << o.mirror_target_slot
                << ",\"source_copies\":" << (mirror_sources ? 2 : 1)
                << ",\"target_a_ptr\":" << reinterpret_cast<uintptr_t>(target_a)
                << ",\"target_b_ptr\":" << reinterpret_cast<uintptr_t>(target_b)
                << ",\"b_buffer_ptr\":" << reinterpret_cast<uintptr_t>(o.mirror_target_slot ? mirror_b->pointer : b.pointer)
                << ",\"first_supply_events\":" << o.first_supply_events
#ifdef R13_FIRST_SUPPLY
                << ",\"first_supply_protocol\":\"r13-first-supply-v1\""
                << ",\"first_supply_scratch_bytes\":64"
#endif
                << ",\"producer_delay_ns\":" << o.producer_delay_ns
#ifdef R10_PRODUCER_DELAY
                << ",\"producer_delay_protocol\":\"cta55-pre-first-load-v1\""
#endif
                << ",\"work_swap55_91\":" << o.work_swap55_91
                << ",\"b_offset_bytes\":" << o.b_offset_bytes
                << ",\"b_allocation_extra_bytes\":" << o.b_allocation_extra_bytes
                << ",\"b_payload_elements\":" << b_payload_elements
                << ",\"prefix_a_ptr\":" << reinterpret_cast<uintptr_t>(prefix_a)
                << ",\"prefix_b_ptr\":" << reinterpret_cast<uintptr_t>(prefix_b)
                << ",\"target_calls\":" << (history_series ? history_target_calls : 1)
                << ",\"preceding_target_calls\":" << (history_series ? history_target_calls-1 : 0)
                << ",\"warmup_last5_cv\":" << coefficient_of_variation({warm.end()-5,warm.end()});
#if defined(R10_PRODUCER_DELAY) && defined(R18_DUAL_CLOCK)
      print_u64("producer_delay_trace",delay_events);
#ifdef R13_FIRST_SUPPLY
      print_u64("first_supply_trace",first_events);
#endif
#endif
      if (history_enabled) {
        float prefix_ms=0,observed_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&prefix_ms,history_begin[0],history_end[history_prefix_calls-1]));
        CUDA_CHECK(cudaEventElapsedTime(&observed_ms,history_begin[0],history_end[history_index-1]));
        std::cout << ",\"history_recorded_calls\":" << history_index
                  << ",\"history_prefix_event_us\":" << prefix_ms*1e3
                  << ",\"history_event_us\":" << observed_ms*1e3;
        std::vector<double> gaps_us;
        for (int j=0; j<history_index-1; ++j) {
          float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,history_end[j],history_begin[j+1]));
          gaps_us.push_back(ms*1e3);
        }
        print_f("history_inter_call_event_us",gaps_us);
      }
      print_f("warmup_us", warm_us);
      print_u64("trace", host);
      std::cout << "}\n";
    };
    auto warm_sequence = [&]() {
      std::vector<float> warm;
      if (history_series) {
        std::vector<double> call_us, gaps_us;
        for (int j=0; j<history_prefix_calls; ++j) {
          float ms=timed(warmup_launch_params); warm.push_back(ms); call_us.push_back(ms*1e3);
        }
        for (int j=0; j<history_target_calls; ++j) call_us.push_back(timed(launch_params)*1e3);
        CUDA_CHECK(cudaDeviceSynchronize());
        saved_history.resize(history->count);
        CUDA_CHECK(cudaMemcpy(saved_history.data(),history->pointer,history->count*sizeof(uint64_t),cudaMemcpyDeviceToHost));
        for (int j=0; j<history_calls-1; ++j) {
          float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,history_end[j],history_begin[j+1]));
          gaps_us.push_back(ms*1e3);
        }
        float prefix_ms=0,series_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&prefix_ms,history_begin[0],history_end[history_prefix_calls-1]));
        CUDA_CHECK(cudaEventElapsedTime(&series_ms,history_begin[0],history_end[history_calls-1]));
        std::cout << "{\"event\":\"history_series\",\"protocol\":\"" << history_protocol << "\""
                  << ",\"prefix_calls\":" << history_prefix_calls << ",\"target_calls\":" << history_target_calls
                  << ",\"prefix_swizzle\":1,\"target_swizzle\":" << o.swizzle
                  << ",\"ctas\":" << ctas << ",\"head_words\":16,\"prefix_elapsed_us\":" << prefix_ms*1e3
                  << ",\"series_elapsed_us\":" << series_ms*1e3;
        print_f("call_elapsed_us",call_us); print_f("inter_call_event_us",gaps_us);
        print_u64("heads",saved_history); std::cout << "}\n";
        report("history-last-target",call_us.back()/1e3,warm);
        return;
      }
      if (common_history) {
        for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));
        float ms = timed(launch_params);
        report("warm", ms, warm);
        return;  // Fixed history: CV is a diagnostic, never a retry condition.
      }
      for (int i = 0; i < 30; ++i) {
        warm.push_back(timed(launch_params));
        if (i >= 7 && coefficient_of_variation({warm.end() - 5, warm.end()}) <= 0.02)
          break;
      }
      float ms = timed(launch_params);
      report("warm", ms, warm);
      if (warm.size() < 8 || coefficient_of_variation({warm.end() - 5, warm.end()}) > 0.02)
        throw std::runtime_error("last five warmups CV exceeds 2% after 30 calls");
    };

    std::cout << "{\"event\":\"setup\",\"config\":\"" << kConfigName
              << "\",\"gpu_uuid\":\"" << gpu_uuid
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
              << ",\"requested_sm_count\":" << o.sm_count
              << ",\"scheduler_sm_count\":" << scheduler_sms
              << ",\"input_mode\":\"" << o.input_mode << "\",\"seed\":" << o.seed
              << ",\"warmup_policy\":\"" << o.warmup_policy << "\""
              << ",\"warmup_swizzle\":" << warmup_swizzle
              << ",\"warmup_fixed_calls\":" << (common_history ? 16 : 0)
              << ",\"history_mode\":\"" << o.history_mode << "\""
              << ",\"history_protocol\":\"" << history_protocol << "\""
                << ",\"history_source\":\"" << o.history_source << "\""
                << ",\"mirror_target_slot\":" << o.mirror_target_slot
                << ",\"source_copies\":" << (mirror_sources ? 2 : 1)
                << ",\"target_a_ptr\":" << reinterpret_cast<uintptr_t>(target_a)
                << ",\"target_b_ptr\":" << reinterpret_cast<uintptr_t>(target_b)
                << ",\"b_buffer_ptr\":" << reinterpret_cast<uintptr_t>(o.mirror_target_slot ? mirror_b->pointer : b.pointer)
                << ",\"first_supply_events\":" << o.first_supply_events
#ifdef R13_FIRST_SUPPLY
                << ",\"first_supply_protocol\":\"r13-first-supply-v1\""
                << ",\"first_supply_scratch_bytes\":64"
#endif
                << ",\"producer_delay_ns\":" << o.producer_delay_ns
#ifdef R10_PRODUCER_DELAY
                << ",\"producer_delay_protocol\":\"cta55-pre-first-load-v1\""
#endif
                << ",\"work_swap55_91\":" << o.work_swap55_91
                << ",\"b_offset_bytes\":" << o.b_offset_bytes
                << ",\"b_allocation_extra_bytes\":" << o.b_allocation_extra_bytes
                << ",\"b_payload_elements\":" << b_payload_elements
                << ",\"prefix_a_ptr\":" << reinterpret_cast<uintptr_t>(prefix_a)
                << ",\"prefix_b_ptr\":" << reinterpret_cast<uintptr_t>(prefix_b)
              << ",\"history_capacity_bytes\":" << (history_enabled ? history->count*sizeof(uint64_t) : 0)
              << ",\"raster_order\":\"" << o.raster_order << "\""
              << ",\"resolved_raster_order\":\"" << resolved_raster << "\""
              << std::setprecision(17)
              << ",\"check_atol\":" << (input_mode == 2 ? 0x1p-20 : 0.0)
              << ",\"check_sum_abs_rtol\":" << (input_mode == 2 ? 0x1p-21 : 0.0)
              << std::setprecision(10)
              << ",\"lda\":" << lda << ",\"ldb\":" << ldb << ",\"ldd\":" << ldd
              << ",\"alloc_lda\":" << alloc_lda << ",\"alloc_ldb\":" << alloc_ldb
              << ",\"allocated_a_elements\":" << a.count << ",\"allocated_b_elements\":" << b.count
              << ",\"evict\":" << o.evict << ",\"eviction_bytes\":" << eviction.count*4
              << ",\"zero_m\":" << o.zero_m << ",\"zero_n\":" << o.zero_n
              << ",\"storage_m\":" << storage_m << ",\"storage_n\":" << storage_n << ",\"swizzle\":" << o.swizzle
              << ",\"input_map_m\":" << input_map_m << ",\"input_map_n\":" << input_map_n
              << ",\"initialized_input_rows\":" << input_rows << ",\"initialized_input_columns\":" << input_columns
              << ",\"trace_tile_capacity\":64,\"trace_version\":\"" << kTraceVersion << "\""
              << ",\"trace_tile_words\":" << kTraceTileWords
              << ",\"scratch_bytes\":" << trace.count * sizeof(uint64_t)
              << ",\"trace_words\":" << kTraceWords << "}\n";
    if (o.mode == "setup")
      return 0;
#if defined(R10_WORK_SWAP) || defined(R10_PRODUCER_DELAY)
    if (o.mode == "check-work-swap" || o.mode == "check-producer-delay" || o.mode == "check-first-supply") {
      // Outside timing: compare all output bits in one allocation, not a sampled hash.
      std::vector<float> baseline(size_t(o.m)*ldd), swapped(baseline.size());
      unsigned flag=0;
      auto set_check_flag = [&](unsigned value) {
#ifdef R13_FIRST_SUPPLY
        if (o.mode == "check-first-supply") {
          CUDA_CHECK(cudaMemcpyToSymbol(r13_first_supply_events, &value, sizeof(value)));
          return;
        }
#endif
#ifdef R10_PRODUCER_DELAY
        if (o.mode == "check-producer-delay") {
          CUDA_CHECK(cudaMemcpyToSymbol(r10_producer_delay_ns, &value, sizeof(value)));
          return;
        }
#endif
#ifdef R10_WORK_SWAP
        CUDA_CHECK(cudaMemcpyToSymbol(r10_work_swap55_91, &value, sizeof(value)));
#endif
      };
      set_check_flag(flag);
      cutlass_check(Gemm::run(launch_params), "baseline check run");
      CUDA_CHECK(cudaMemcpy(baseline.data(), d.pointer, baseline.size()*sizeof(float), cudaMemcpyDeviceToHost));
      gaps::fill_output_sentinel<<<256,256>>>(d.pointer, d.count);
      flag=o.mode == "check-producer-delay" ? 512 : 1;
      set_check_flag(flag);
      cutlass_check(Gemm::run(launch_params), "swap check run");
      CUDA_CHECK(cudaMemcpy(swapped.data(), d.pointer, swapped.size()*sizeof(float), cudaMemcpyDeviceToHost));
      size_t different=0, nonfinite=0;
      for (int i=0; i<o.m; ++i) for (int j=0; j<o.n; ++j) {
        size_t pos=size_t(i)*ldd+j;
        different += std::memcmp(&baseline[pos], &swapped[pos], sizeof(float)) != 0;
        nonfinite += !std::isfinite(baseline[pos]) || !std::isfinite(swapped[pos]);
      }
      std::cout << "{\"event\":\"" << (o.mode == "check-first-supply" ? "first_supply_check" : o.mode == "check-producer-delay" ? "producer_delay_check" : "work_swap_check") << "\",\"elements\":" << size_t(o.m)*o.n
                << ",\"different\":" << different << ",\"nonfinite\":" << nonfinite << "}\n";
      return different == 0 && nonfinite == 0 ? 0 : 2;
    }
#endif
    warm_sequence();

    double max_error_ratio = 0;
    Errors e = check_boundary_output(d.pointer, o.m, o.n, o.k, ldd, check.seed,
                                       4096,o.zero_m,o.zero_n,input_mode,target_a,target_b,
                                       lda,ldb,max_error_ratio);
    size_t padding_bad = gaps::padding_errors(a.pointer, input_rows, o.k, lda, __half(65504.f))
        + gaps::padding_errors(b_data, o.k, input_columns, ldb, __half(65504.f))
        + gaps::padding_errors(d.pointer, o.m, o.n, ldd, gaps::output_sentinel());
    if (mirror_sources) {
      padding_bad += gaps::padding_errors(mirror_a->pointer,input_rows,o.k,lda,__half(65504.f))
          + gaps::padding_errors(mirror_b_data,o.k,input_columns,ldb,__half(65504.f));
    }
    size_t zero_storage_bad=0,zero_storage_checked=0;
    if(separate_input_map&&o.zero_m==o.m&&storage_m>o.m){
      zero_storage_checked+=size_t(storage_m-o.m)*o.k;
      zero_storage_bad+=nonzero_input_rectangle(a.pointer+size_t(o.m)*lda,storage_m-o.m,o.k,lda);
    }
    if(separate_input_map&&o.zero_n==o.n&&storage_n>o.n){
      zero_storage_checked+=size_t(o.k)*(storage_n-o.n);
      zero_storage_bad+=nonzero_input_rectangle(b_data+o.n,o.k,storage_n-o.n,ldb);
    }
    if(separate_input_map)
      std::cout << "{\"event\":\"input_zero_storage\",\"checked\":" << zero_storage_checked
                << ",\"nonzero\":" << zero_storage_bad << "}\n";
    padding_bad+=zero_storage_bad;
    // The input map may cover extra rows; the output must still leave them untouched.
    if(separate_input_map&&storage_m>o.m){
      std::vector<uint32_t> extra_rows(size_t(storage_m-o.m)*ldd);
      CUDA_CHECK(cudaMemcpy(extra_rows.data(),d.pointer+size_t(o.m)*ldd,
                           extra_rows.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
      for(uint32_t bits:extra_rows)padding_bad+=bits!=0x7fc12345u;
    }
    bool numerical_ok = input_mode == 2 ? max_error_ratio <= 1 : e.max_storage_error == 0;
    bool ok = e.nonfinite == 0 && numerical_ok && padding_bad == 0;
    if (input_mode == 2)
      std::cout << "{\"event\":\"tolerance\",\"max_error_ratio\":" << max_error_ratio << "}\n";
    gaps::print_check(e, padding_bad, numerical_ok);
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    for (auto event:history_begin) CUDA_CHECK(cudaEventDestroy(event));
    for (auto event:history_end) CUDA_CHECK(cudaEventDestroy(event));
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
