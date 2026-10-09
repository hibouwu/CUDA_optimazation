// R18/R19: physical-grid matched boundaries, explicit zeros, and recorded work coordinates.
// Copied from the earlier probe: explicit lda/ldb/ldd, warm protocol, dyadic exact sampled check.
#include "gaps_common.hpp"
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
#if V06_STAGES_AUTO
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

#if defined(R18_DUAL_CLOCK)
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
  int sm_count = 0;
  std::string input_mode = "dyadic";
  int seed = 17;
  int64_t lda = 0, ldb = 0, ldd = 0;
  int64_t alloc_lda = 0, alloc_ldb = 0;
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
      else if(key=="--zero-m")zero_m=std::stoi(v);
      else if(key=="--zero-n")zero_n=std::stoi(v);
      else if(key=="--storage-m")storage_m=std::stoi(v);
      else if(key=="--storage-n")storage_n=std::stoi(v);
      else if(key=="--swizzle")swizzle=std::stoi(v);
      else if(key=="--evict")evict=std::stoi(v);
      else if(key=="--sm-count")sm_count=std::stoi(v);
      else if(key=="--input-mode")input_mode=v;
      else if(key=="--seed")seed=std::stoi(v);
      else throw std::runtime_error("unknown option " + key);
    }
    if (input_mode != "dyadic" && input_mode != "zero" && input_mode != "random")
      throw std::runtime_error("input-mode must be dyadic, zero or random");
    if (seed < 0)
      throw std::runtime_error("seed must be nonnegative");
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
    if (o.mode != "warm" && o.mode != "setup")
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
    int64_t alloc_lda=o.alloc_lda?o.alloc_lda:lda,alloc_ldb=o.alloc_ldb?o.alloc_ldb:ldb;
    if(alloc_lda<lda||alloc_ldb<ldb)throw std::runtime_error("allocation pitch is smaller than physical pitch");
    DeviceBuffer<__half> a(size_t(storage_m)*alloc_lda), b(size_t(o.k)*alloc_ldb);
    DeviceBuffer<float> d(size_t(storage_m)*ldd);
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
    Gemm gemm;
    cutlass_check(gemm.can_implement(arguments), "can_implement");
    DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
    cutlass_check(gemm.initialize(arguments, workspace.pointer), "initialize");
    dim3 grid = Gemm::get_grid_shape(gemm.params());
    int ctas = int(grid.x * grid.y * grid.z);
    if (ctas <= 0 || ctas > properties.multiProcessorCount)
      throw std::runtime_error("unexpected persistent grid");

    // Same preparation order as R00 measure_gemm (fill A, B; D = NaN pattern).
    gaps::fill_input_strided<<<256, 256>>>(a.pointer, o.m, o.k, lda, check.seed, true, input_mode,o.alloc_lda?a.count:0);
    gaps::fill_input_strided<<<256, 256>>>(b.pointer, o.k, o.n, ldb, check.seed, false, input_mode,o.alloc_ldb?b.count:0);
    CUDA_CHECK(cudaGetLastError());
    if(o.zero_m>=0) zero_panel<<<256,256>>>(a.pointer,o.m,o.k,lda,o.zero_m,-1);
    if(o.zero_n>=0) zero_panel<<<256,256>>>(b.pointer,o.k,o.n,ldb,-1,o.zero_n);
    CUDA_CHECK(cudaGetLastError());
    gaps::fill_output_sentinel<<<256, 256>>>(d.pointer, d.count);
    DeviceBuffer<uint64_t> trace(size_t(ctas) * kTraceWords);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * 8));
#ifdef V06_TRACE
    CUDA_CHECK(cudaMemcpyToSymbol(v06_trace_ptr, &trace.pointer, sizeof(trace.pointer)));
    const bool traced = true;
#else
    const bool traced = false;
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    auto timed = [&]() {
      if(o.evict)evict_buffer<<<256,256>>>(eviction.pointer,eviction.count,++eviction_seed);
      CUDA_CHECK(cudaGetLastError());
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
#ifdef V06_TRACE
      static_assert(kTraceWords == V06CtaWords, "trace layout mismatch");
      host.resize(size_t(ctas) * kTraceWords);
      CUDA_CHECK(cudaMemcpy(host.data(), trace.pointer, host.size() * 8, cudaMemcpyDeviceToHost));
      for (int cta = 0; cta < ctas; ++cta)
        if (host[size_t(cta) * kTraceWords + 10])
          throw std::runtime_error("V06 trace tile capacity exceeded");
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
              << ",\"trace_tile_capacity\":64,\"trace_version\":\"" << kTraceVersion << "\""
              << ",\"trace_tile_words\":" << kTraceTileWords
              << ",\"scratch_bytes\":" << trace.count * sizeof(uint64_t)
              << ",\"trace_words\":" << kTraceWords << "}\n";
    if (o.mode == "setup")
      return 0;
    warm_sequence();

    double max_error_ratio = 0;
    Errors e = check_boundary_output(d.pointer, o.m, o.n, o.k, ldd, check.seed,
                                       4096,o.zero_m,o.zero_n,input_mode,a.pointer,b.pointer,
                                       lda,ldb,max_error_ratio);
    size_t padding_bad = gaps::padding_errors(a.pointer, o.m, o.k, lda, __half(65504.f))
        + gaps::padding_errors(b.pointer, o.k, o.n, ldb, __half(65504.f))
        + gaps::padding_errors(d.pointer, o.m, o.n, ldd, gaps::output_sentinel());
    bool numerical_ok = input_mode == 2 ? max_error_ratio <= 1 : e.max_storage_error == 0;
    bool ok = e.nonfinite == 0 && numerical_ok && padding_bad == 0;
    if (input_mode == 2)
      std::cout << "{\"event\":\"tolerance\",\"max_error_ratio\":" << max_error_ratio << "}\n";
    gaps::print_check(e, padding_bad, numerical_ok);
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
