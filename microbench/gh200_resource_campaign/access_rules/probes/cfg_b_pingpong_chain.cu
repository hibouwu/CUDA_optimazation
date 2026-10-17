// E: the actual S6 CUTLASS cfg_b pingpong kernel, three persistent output tiles.
// Plain uses the unmodified factory. Trace uses the frozen V06 overlay plus
// four explicit call-boundary stamps; its global trace stores are observed cost.
#include "r00_common.hpp"
#include "cfg_b_pingpong_trace.hpp"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#if CFG_B_PINGPONG_TRACE && !defined(V06_KERNEL_OVERLAY_ACTIVE)
#error "Trace requires the declared frozen kernel and collective overlay"
#endif

using namespace cute;
constexpr int M = 384, N = 128, K = 832, GuardElements = 16;
constexpr int OutputElements = M * N;
constexpr uint32_t GuardBits = 0x7fc12345u;
using KernelTile = Shape<_128, _128, _64>;
using KernelCluster = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_64, _32>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    KernelTile, KernelCluster, EpilogueTile, float, float,
    void, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8, float,
    KernelTile, KernelCluster, cutlass::gemm::collective::StageCount<6>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
static_assert(Kernel::MaxThreadsPerBlock == 384 && Kernel::NumMMAThreads == 128);
static_assert(Mainloop::DispatchPolicy::Stages == 6);
static_assert(sizeof(Mainloop::TensorStorage) == 196608);
static_assert(Kernel::SharedStorageSize == 215040);
static_assert(Kernel::MathWarpGroupOrderBarrier::SequenceDepth == 2);
static_assert(Kernel::MathWarpGroupOrderBarrier::SequenceLength == 2);

struct EOptions {
  int seed = 3, warmup_limit = 60;
  bool inspect = false;
  EOptions(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--inspect-program") { inspect = true; continue; }
      if (i + 1 == argc) throw std::runtime_error("missing CLI value");
      std::string value = argv[++i]; size_t end = 0;
      int parsed = std::stoi(value, &end);
      if (end != value.size()) throw std::runtime_error("noninteger CLI value");
      if (key == "--seed") seed = parsed;
      else if (key == "--warmup-limit") warmup_limit = parsed;
      else throw std::runtime_error("unknown CLI option: " + key);
    }
    if (seed != 3 && seed != 22 && seed != 41) throw std::runtime_error("seed must be 3, 22 or 41");
    if (warmup_limit < 8 || warmup_limit > 60) throw std::runtime_error("warmup-limit must be 8..60");
  }
};
void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
}
std::string json_string(const std::string& input) {
  std::ostringstream out; out << '"';
  for (unsigned char c : input) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << c;
  }
  out << '"'; return out.str();
}
std::string uuid_string(const cudaDeviceProp& properties) {
  std::ostringstream out; out << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
    out << std::setw(2) << unsigned(uint8_t(properties.uuid.bytes[i]));
  }
  return out.str();
}
void inspect_program() {
  std::cout << "{\"schema\":\"gh200-cfg-b-pingpong-program/1\",\"trace\":"
      << (CFG_B_PINGPONG_TRACE ? "true" : "false")
      << ",\"cuda_api_calls\":0,\"descriptor\":null,\"factory\":{\"NumThreads\":"
      << Kernel::MaxThreadsPerBlock << ",\"NumMMAThreads\":" << Kernel::NumMMAThreads
      << ",\"Stages\":" << Mainloop::DispatchPolicy::Stages
      << ",\"SharedStorageSize\":" << Kernel::SharedStorageSize
      << ",\"mainloop_tensor_storage_bytes\":" << sizeof(Mainloop::TensorStorage)
      << ",\"epilogue_tensor_storage_bytes\":" << sizeof(Epilogue::TensorStorage)
      << ",\"StagesC\":" << Epilogue::DispatchPolicy::StagesC << ",\"StagesD\":" << Epilogue::DispatchPolicy::StagesD
      << ",\"ReuseSmemC\":" << (Epilogue::DispatchPolicy::ReuseSmemC ? "true" : "false")
      << ",\"DelayTmaStore\":" << (Epilogue::DispatchPolicy::DelayTmaStore ? "true" : "false")
      << ",\"software_tile\":[128,128,64],\"cluster\":[1,1,1],\"epilogue_tile\":[64,32]"
      << ",\"instruction_shape\":[" << Gemm::InstructionShape::kM << ','
      << Gemm::InstructionShape::kN << ',' << Gemm::InstructionShape::kK
      << "],\"ordered_sequence\":[" << Kernel::MathWarpGroupOrderBarrier::SequenceDepth
      << ',' << Kernel::MathWarpGroupOrderBarrier::SequenceLength << "]}}\n";
}
std::vector<float> reference(int seed) {
  // Independent integer oracle. FP16 inputs are exact /32 dyadics; every
  // partial integer sum is bounded by 832*64 and is exactly representable.
  std::vector<float> result(OutputElements);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int64_t numerator = 0;
      for (int inner = 0; inner < K; ++inner) {
        int a = (7 * row + 13 * inner + 3 * seed) % 17 - 8;
        int b = (5 * inner + 11 * col + 5 * seed) % 17 - 8;
        numerator += int64_t(a) * b;
      }
      result[size_t(row) * N + col] = float(numerator) / 1024.0f;
    }
  }
  return result;
}
struct Trial {
  float event_ms = 0;
  uint64_t errors = 0, guard_errors = 0;
  double max_abs_error = 0;
  std::vector<float> output;
  std::array<uint32_t, 2 * GuardElements> guards{};
  EPingpongTrace trace{};
};
void emit_clock(const EClockPair& stamp) {
  std::cout << "{\"cycle\":" << stamp.cycle << ",\"ns\":" << stamp.ns << '}';
}
void emit_trace(const EPingpongTrace& trace) {
  const char* names[EEventCount] = {"first_mma", "main_end", "epilogue_permit", "store_tail_return",
      "mma_return", "math_arrive_after", "store_return", "epilogue_arrive_after"};
  std::cout << "{\"entry\":{\"cycle\":" << trace.entry.cycle << ",\"ns\":" << trace.entry.ns
      << ",\"smid\":" << trace.entry_smid << "},\"producer_first_work\":";
  emit_clock(trace.producer_first_work);
  std::cout << ",\"overflow\":" << trace.overflow << ",\"outputs\":[";
  for (int output = 0; output < 3; ++output) {
    int wg = output % 2, local_seq = output / 2;
    const auto& record = trace.outputs[wg][local_seq];
    std::cout << (output ? "," : "") << "{\"output\":" << output << ",\"consumer_wg\":" << wg
        << ",\"local_seq\":" << record.local_seq << ",\"coord\":[" << record.m_idx << ',' << record.n_idx
        << "],\"l_idx\":" << record.l_idx << ",\"valid\":" << record.valid;
    for (int event = 0; event < EEventCount; ++event) {
      std::cout << ',' << json_string(names[event]) << ':';
      emit_clock(record.events[event]);
    }
    std::cout << '}';
  }
  std::cout << "],\"finals\":[";
  for (int wg = 0; wg < 2; ++wg) {
    std::cout << (wg ? "," : "") << "{\"consumer_wg\":" << wg << ",\"cycle\":"
        << trace.finals[wg].time.cycle << ",\"ns\":" << trace.finals[wg].time.ns
        << ",\"tiles\":" << trace.finals[wg].tiles << ",\"smid\":" << trace.finals[wg].smid << '}';
  }
  std::cout << "],\"qualification\":\"consumer leader timestamps with in-window global trace stores; first_mma is before warpgroup_arrive; arrivals are not last-participant completion; store_tail_return is not asserted as global completion\"}";
}
double tail_cv(const std::vector<float>& times) {
  double mean = 0, variance = 0;
  for (size_t i = times.size() - 5; i < times.size(); ++i) mean += times[i];
  mean /= 5;
  if (!(mean > 0)) throw std::runtime_error("nonpositive warmup event window");
  for (size_t i = times.size() - 5; i < times.size(); ++i) variance += (times[i] - mean) * (times[i] - mean);
  return std::sqrt(variance / 4) / mean;
}
int run(const EOptions& options) {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 9 || properties.minor != 0 || properties.multiProcessorCount != 132 ||
      std::string(properties.name).find("GH200") == std::string::npos)
    throw std::runtime_error("this fixed probe requires GH200 SM90 with 132 SMs");
  DeviceBuffer<__half> a(size_t(M) * K), b(size_t(K) * N);
  DeviceBuffer<float> guarded(OutputElements + 2 * GuardElements);
  fill_input<<<256, 256>>>(a.pointer, M, K, options.seed, true, false);
  fill_input<<<256, 256>>>(b.pointer, K, N, options.seed, false, false);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  auto sa = make_stride(int64_t(K), Int<1>{}, int64_t(M) * K);
  auto sb = make_stride(Int<1>{}, int64_t(N), int64_t(K) * N);
  auto sd = make_stride(int64_t(N), Int<1>{}, int64_t(M) * N);
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<cutlass::half_t*>(a.pointer), sa, reinterpret_cast<cutlass::half_t*>(b.pointer), sb},
      {{1.f, 0.f}, nullptr, sd, guarded.pointer + GuardElements, sd}};
  arguments.hw_info.device_id = 0; arguments.hw_info.sm_count = 1;
  arguments.scheduler.max_swizzle_size = 1;
  arguments.scheduler.raster_order = decltype(arguments.scheduler.raster_order)::AlongM;
  Gemm gemm;
  check_cutlass(gemm.can_implement(arguments), "can_implement");
  DeviceBuffer<unsigned char> workspace(std::max<size_t>(1, Gemm::get_workspace_size(arguments)));
  check_cutlass(gemm.initialize(arguments, workspace.pointer), "initialize");
  dim3 grid = Gemm::get_grid_shape(gemm.params());
  if (grid.x * grid.y * grid.z != 1) throw std::runtime_error("scheduler did not produce one CTA");
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, cutlass::device_kernel<Kernel>));
  int occupancy = Gemm::maximum_active_blocks();
  if (occupancy != 1 || attributes.localSizeBytes != 0)
    throw std::runtime_error("required occupancy=1/local_size_bytes=0 was not observed");
#if CFG_B_PINGPONG_TRACE
  DeviceBuffer<EPingpongTrace> trace_buffer(1);
  CUDA_CHECK(cudaMemcpyToSymbol(e_trace_pointer, &trace_buffer.pointer, sizeof(trace_buffer.pointer)));
#endif
  std::vector<uint32_t> initial(OutputElements + 2 * GuardElements, GuardBits), copied(initial.size());
  auto expected = reference(options.seed);
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  auto trial = [&]() {
    Trial result;
    CUDA_CHECK(cudaMemcpy(guarded.pointer, initial.data(), initial.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
#if CFG_B_PINGPONG_TRACE
    CUDA_CHECK(cudaMemset(trace_buffer.pointer, 0, sizeof(EPingpongTrace)));
#endif
    CUDA_CHECK(cudaEventRecord(begin));
    check_cutlass(gemm.run(), "run");
    CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&result.event_ms, begin, end));
    CUDA_CHECK(cudaMemcpy(copied.data(), guarded.pointer, copied.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    result.output.resize(OutputElements);
    std::memcpy(result.output.data(), copied.data() + GuardElements, OutputElements * sizeof(float));
    for (int q = 0; q < OutputElements; ++q) {
      float observed = result.output[q];
      double error = std::isfinite(observed) ? std::abs(double(observed) - expected[q]) : std::numeric_limits<double>::infinity();
      if (observed != expected[q]) ++result.errors;
      result.max_abs_error = std::max(result.max_abs_error, error);
    }
    for (int q = 0; q < 2 * GuardElements; ++q) {
      uint32_t observed = copied[q < GuardElements ? q : OutputElements + q];
      result.guards[q] = observed;
      if (observed != GuardBits) ++result.guard_errors;
    }
#if CFG_B_PINGPONG_TRACE
    CUDA_CHECK(cudaMemcpy(&result.trace, trace_buffer.pointer, sizeof(result.trace), cudaMemcpyDeviceToHost));
#endif
    return result;
  };
  std::vector<float> warmup_ms;
  bool converged = false; double cv = 0;
  for (int i = 0; i < options.warmup_limit; ++i) {
    auto result = trial();
    if (result.errors || result.guard_errors) throw std::runtime_error("warmup full-output/guard mismatch");
    warmup_ms.push_back(result.event_ms);
    if (warmup_ms.size() >= 8) {
      cv = tail_cv(warmup_ms);
      if (cv <= 0.02) { converged = true; break; }
    }
  }
  Trial result = trial();
  CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
  std::ofstream output("output.f32", std::ios::binary);
  output.write(reinterpret_cast<const char*>(result.output.data()), OutputElements * sizeof(float));
  if (!output) throw std::runtime_error("cannot write output.f32");
  bool correct = !result.errors && !result.guard_errors;
  std::cout << std::setprecision(17) << "{\"schema\":\"gh200-cfg-b-pingpong-chain/1\",\"status\":"
      << json_string(correct ? "measured" : "error") << ",\"trace\":" << (CFG_B_PINGPONG_TRACE ? "true" : "false")
      << ",\"seed\":" << options.seed
      << ",\"conditions\":{\"m\":384,\"n\":128,\"k\":832,\"stages\":6,\"output_tiles\":3,\"ktiles_per_output\":13,\"scheduler_sm_count\":1}"
      << ",\"device\":{\"name\":" << json_string(properties.name) << ",\"uuid\":" << json_string(uuid_string(properties))
      << ",\"sms\":" << properties.multiProcessorCount << ",\"major\":" << properties.major << ",\"minor\":" << properties.minor << '}'
      << ",\"resources\":{\"threads\":" << Kernel::MaxThreadsPerBlock << ",\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z
      << "],\"dynamic_smem_bytes\":" << Kernel::SharedStorageSize << ",\"registers_per_thread\":" << attributes.numRegs
      << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes << ",\"local_size_bytes\":" << attributes.localSizeBytes
      << ",\"occupancy_ctas_per_sm\":" << occupancy << '}'
      << ",\"timing\":{\"cuda_event_ms\":" << result.event_ms
      << ",\"qualification\":\"complete actual CUTLASS kernel stream-event window including inputs, scheduling and epilogue; not an isolated output price\"}"
      << ",\"warmup\":{\"limit\":" << options.warmup_limit << ",\"elapsed_ms\":[";
  for (size_t i = 0; i < warmup_ms.size(); ++i) std::cout << (i ? "," : "") << warmup_ms[i];
  std::cout << "],\"cv\":" << cv << ",\"converged\":" << (converged ? "true" : "false")
      << ",\"sample_std_ddof\":1},\"correctness\":{\"checked_elements\":" << OutputElements
      << ",\"errors\":" << result.errors << ",\"max_abs_error\":";
  if (std::isfinite(result.max_abs_error)) std::cout << result.max_abs_error; else std::cout << "null";
  std::cout << ",\"guard_elements\":32,\"guard_errors\":" << result.guard_errors
      << "},\"output_file\":\"output.f32\",\"output_elements\":49152,\"output_bytes\":196608,\"guard_bits\":[";
  for (size_t i = 0; i < result.guards.size(); ++i) std::cout << (i ? "," : "") << result.guards[i];
  std::cout << "],\"trace_detail\":";
  if (CFG_B_PINGPONG_TRACE) emit_trace(result.trace); else std::cout << "null";
  std::cout << "}\n";
  return correct ? 0 : 2;
}
int main(int argc, char** argv) {
  try {
    EOptions options(argc, argv);
    if (options.inspect) { inspect_program(); return 0; }
    return run(options);
  } catch (const std::exception& error) {
    std::cout << "{\"schema\":\"gh200-cfg-b-pingpong-chain/1\",\"status\":\"error\",\"error\":"
        << json_string(error.what()) << "}\n";
    return 1;
  }
}
