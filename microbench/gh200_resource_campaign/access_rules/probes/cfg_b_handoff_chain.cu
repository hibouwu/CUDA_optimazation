// cfg_b resident-operand handoff: one producer WG and one consumer WG.
// Six physical operand slots, four or six active full(1)/empty(128) barriers.
// Ordinary arrival models control handoff; there is no TMA or operand refill.
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cerrno>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#ifndef GH200_NORMAL_WAIT
#define GH200_NORMAL_WAIT 1
#endif

namespace G = cute::SM90::GMMA;
using U64 = uint64_t;
using U32 = uint32_t;
constexpr int Threads = 256, ConsumerThreads = 128, Slots = 6, MaxTiles = 32, FragmentRegisters = 64;
using Barrier = cutlass::arch::ClusterBarrier;
constexpr int OperandElements = Slots * 128 * 64;
constexpr int SmemBytes = 2 * OperandElements * sizeof(__half);
constexpr int OutputElements = ConsumerThreads * 2 * FragmentRegisters;
using Mma = G::MMA_64x128x16_F32F16F16_SS<G::Major::K, G::Major::MN>;

struct RingIndex {
  int slot, generation;
  __host__ __device__ void advance(int stages) {
    if (++slot == stages) {
      slot = 0;
      ++generation;
    }
  }
};

struct ClockPair { U64 cycle = 0, ns = 0; };
struct ProducerTrace {
  ClockPair acquire_before, acquire_after, full_before, full_after;
};
struct ConsumerTrace {
  ClockPair wait_before, wait_after, issue_before, commit_after, wait1_after;
  ClockPair release_before, release_after;
  U32 first_try_success = 0, first_try_success_warp_masks[4]{};
};
struct ReleaseInterval { ClockPair before, after; };
struct TileTrace {
  ProducerTrace producer; ConsumerTrace consumer;
  ReleaseInterval release_warps[4];
  ClockPair release_last_before, release_last_after;
};
struct Record {
  ClockPair begin, end, tail_wait0_after, tail_release_before, tail_release_after;
  ClockPair tail_release_last_before, tail_release_last_after;
  U32 smid_start = 0, smid_end = 0;
  U64 generation_errors = 0;
};
__device__ __forceinline__ ClockPair clocks() {
  ClockPair value;
  asm volatile("mov.u64 %0, %%clock64; mov.u64 %1, %%globaltimer;"
      : "=l"(value.cycle), "=l"(value.ns) :: "memory");
  return value;
}
__device__ __forceinline__ U32 smid() {
  U32 value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}
__device__ __forceinline__ ClockPair consumer_end(const volatile float* last) {
  U32 word = __float_as_uint(*last);
  ClockPair value;
  // The CTA barrier precedes this real shared consumer. The predicate keeps the
  // clocks dependent on its result; a valid finite dyadic checksum never is NaN.
  asm volatile("{.reg .pred p; setp.ne.u32 p,%2,0xffffffff;"
      "@p mov.u64 %0,%%clock64; @p mov.u64 %1,%%globaltimer;"
      "@!p mov.u64 %0,0; @!p mov.u64 %1,0;}"
      : "=l"(value.cycle), "=l"(value.ns) : "r"(word) : "memory");
  return value;
}
__device__ __forceinline__ U64 warp_max(U64 value) {
#pragma unroll
  for (int offset = 16; offset; offset /= 2) {
    U64 other = __shfl_down_sync(0xffffffff, value, offset);
    value = value > other ? value : other;
  }
  return value;
}
__device__ __forceinline__ ClockPair warp_max_pair(ClockPair value) {
  return {warp_max(value.cycle), warp_max(value.ns)};
}
__host__ __device__ int a_numerator(int slot, int row, int k, U32 seed) {
  return (7 * row + 13 * k + 3 * slot + 3 * int(seed % 17)) % 17 - 8;
}
__host__ __device__ int b_numerator(int slot, int column, int k, U32 seed) {
  return (11 * column + 5 * k + 7 * slot + 5 * int(seed % 17)) % 17 - 8;
}
template <size_t... I>
__device__ __forceinline__ void call_mma(U64 a, U64 b, float* d, G::ScaleOut scale,
    std::index_sequence<I...>) {
  Mma::fma(a, b, d[I]..., scale);
}
template <class TensorA, class TensorB>
__device__ __forceinline__ void issue_tile(TensorA a, TensorB b, int slot, bool first,
    float (&d)[2][FragmentRegisters]) {
  using namespace cute;
  auto as = a(_, _, slot);
  auto bs = b(_, _, slot);
  // Match the frozen cfg_b SASS order: K16 first, then the two distinct M64 pieces.
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    auto bv = local_tile(bs, make_shape(_128{}, _16{}), make_coord(_0{}, q));
    U64 bd = G::make_gmma_desc<G::Major::MN>(bv);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      auto av = local_tile(as, make_shape(_64{}, _16{}), make_coord(m, q));
      U64 ad = G::make_gmma_desc<G::Major::K>(av);
      call_mma(ad, bd, d[m], first && q == 0 ? G::ScaleOut::Zero : G::ScaleOut::One,
          std::make_index_sequence<FragmentRegisters>{});
    }
  }
}
template <bool Trace, bool IncrementalIndex, bool ConsumerOnlyIndex>
__global__ __launch_bounds__(Threads, 1) void handoff_chain(int ktiles, int stages,
    U64 delay_cycles, U32 seed,
#ifdef GH200_CONSUMER_ENTRY_DELAY
    U64 consumer_entry_delay_cycles,
#endif
    Record* records, TileTrace* exported_trace, float* output) {
  using namespace cute;
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{},
      Shape<_128, _64, Int<Slots>>{}, Step<_1, _2, _3>{});
  auto lb = tile_to_shape(G::Layout_MN_SW128_Atom<__half>{},
      Shape<_128, _64, Int<Slots>>{}, Step<_2, _1, _3>{});
  extern __shared__ __align__(128) unsigned char storage[];
  auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)), la);
  auto b = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage) + OperandElements), lb);
  __shared__ __align__(8) U64 full[Slots], empty[Slots];
  __shared__ volatile U32 generation[Slots];
  __shared__ volatile float drain[ConsumerThreads];
  __shared__ U32 generation_errors[ConsumerThreads];
  __shared__ TileTrace trace[MaxTiles];
  __shared__ ReleaseInterval tail_release_warps[4];
  if (threadIdx.x == ConsumerThreads) {
    for (int slot = 0; slot < Slots; ++slot) {
      Barrier::init(full + slot, 1);
      Barrier::init(empty + slot, ConsumerThreads);
      generation[slot] = ~U32(0);
    }
    cutlass::arch::fence_barrier_init();
  }
  if constexpr (Trace) {
    for (int tile = threadIdx.x; tile < MaxTiles; tile += Threads) trace[tile] = TileTrace{};
  }
  for (int index = threadIdx.x; index < OperandElements; index += Threads) {
    int slot = index / (128 * 64), outer = (index / 64) % 128, k = index % 64;
    a(outer, k, slot) = __float2half_rn(a_numerator(slot, outer, k, seed) / 32.f);
    b(outer, k, slot) = __float2half_rn(b_numerator(slot, outer, k, seed) / 32.f);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  // Initial phase is zero. Producer empty wait phase one is already satisfied.
  // Establish the first active generation outside the measured rendezvous.
  if (threadIdx.x == ConsumerThreads) {
    for (int tile = 0; tile < stages && tile < ktiles; ++tile) {
      generation[tile] = 0;
      asm volatile("" ::: "memory");
      Barrier::arrive(full + tile, 0, 1);
    }
  }
  __syncthreads();
  Record record;
  U32 errors = 0;
  float d[2][FragmentRegisters];
  // Define accumulator registers on a common path before role divergence.
  // Producer lanes do not use these values; this avoids a conditional SSA merge
  // being interpreted by ptxas as a definition within the asynchronous pipeline.
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int j = 0; j < FragmentRegisters; ++j) {
      d[m][j] = (1 + m) / 64.f;
      warpgroup_fence_operand(d[m][j]);
    }
  }
  if (threadIdx.x == 0) {
    record.smid_start = smid();
    record.begin = clocks();
  }
  __syncthreads(); // only the start and final rendezvous involve the whole CTA
  if (threadIdx.x == ConsumerThreads) {
    RingIndex index{0, 1};
    for (int tile = stages; tile < ktiles; ++tile) {
      int slot, gen;
      if constexpr (IncrementalIndex) { slot = index.slot; gen = index.generation; }
      else { slot = tile % stages; gen = tile / stages; }
      ProducerTrace stamp;
      if constexpr (Trace) stamp.acquire_before = clocks();
      Barrier::wait(empty + slot, (gen + 1) & 1);
      asm volatile("" ::: "memory");
      if constexpr (Trace) stamp.acquire_after = clocks();
      if (tile == stages && delay_cycles) {
        U64 start = 0, now = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
        do {
          asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
        } while (now - start < delay_cycles);
      }
      generation[slot] = U32(gen);
      asm volatile("" ::: "memory");
      if constexpr (Trace) stamp.full_before = clocks();
      Barrier::arrive(full + slot, 0, 1);
      if constexpr (Trace) {
        stamp.full_after = clocks();
        trace[tile].producer = stamp;
      }
      if constexpr (IncrementalIndex) index.advance(stages);
    }
  } else if (threadIdx.x < ConsumerThreads) {
#ifdef GH200_CONSUMER_ENTRY_DELAY
    if (consumer_entry_delay_cycles) {
      U64 start = 0, now = 0;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
      do {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
      } while (now - start < consumer_entry_delay_cycles);
    }
#endif
    RingIndex index{0, 0};
    int previous_slot = 0;
    for (int tile = 0; tile < ktiles; ++tile) {
      int slot, gen;
      if constexpr (IncrementalIndex || ConsumerOnlyIndex) { slot = index.slot; gen = index.generation; }
      else { slot = tile % stages; gen = tile / stages; }
      ConsumerTrace stamp;
      if constexpr (Trace) if (threadIdx.x == 0) stamp.wait_before = clocks();
      bool first_try_success = Barrier::try_wait(full + slot, gen & 1);
      if (!first_try_success) Barrier::wait(full + slot, gen & 1);
      asm volatile("" ::: "memory");
      if constexpr (Trace) {
        if (threadIdx.x == 0) {
          stamp.wait_after = clocks();
          stamp.first_try_success = first_try_success;
        }
        // Ballot follows the wait; it does not gate the producer or precede waiting.
        U32 mask = __ballot_sync(0xffffffff, first_try_success);
        if ((threadIdx.x & 31) == 0)
          trace[tile].consumer.first_try_success_warp_masks[threadIdx.x / 32] = mask;
      }
      if (generation[slot] != U32(gen)) ++errors;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int j = 0; j < FragmentRegisters; ++j) warpgroup_fence_operand(d[m][j]);
      }
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      if constexpr (Trace) if (threadIdx.x == 0) stamp.issue_before = clocks();
      issue_tile(a, b, slot, tile == 0, d);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr (Trace) if (threadIdx.x == 0) stamp.commit_after = clocks();
      if (tile != 0) {
        asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(GH200_NORMAL_WAIT) : "memory");
        ReleaseInterval release;
        if constexpr (Trace) {
          if (threadIdx.x == 0) stamp.wait1_after = clocks();
          release.before = clocks();
        }
        // wait1 after committing tile proves the previous tile's operands consumed.
        if constexpr (IncrementalIndex || ConsumerOnlyIndex) Barrier::arrive(empty + previous_slot, 0, 1);
        else Barrier::arrive(empty + (tile - 1) % stages, 0, 1);
        if constexpr (Trace) {
          release.after = clocks();
          if (threadIdx.x == 0) {
            stamp.release_before = release.before;
            stamp.release_after = release.after;
          }
          release.before = warp_max_pair(release.before);
          release.after = warp_max_pair(release.after);
          if ((threadIdx.x & 31) == 0) trace[tile].release_warps[threadIdx.x / 32] = release;
        }
      }
      if constexpr (Trace) if (threadIdx.x == 0) {
        // Preserve the four independently written masks; copy the leader fields only.
        auto& destination = trace[tile].consumer;
        destination.wait_before = stamp.wait_before;
        destination.wait_after = stamp.wait_after;
        destination.issue_before = stamp.issue_before;
        destination.commit_after = stamp.commit_after;
        destination.wait1_after = stamp.wait1_after;
        destination.release_before = stamp.release_before;
        destination.release_after = stamp.release_after;
        destination.first_try_success = stamp.first_try_success;
      }
      if constexpr (IncrementalIndex || ConsumerOnlyIndex) {
        previous_slot = slot;
        index.advance(stages);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    ReleaseInterval tail_release;
    if constexpr (Trace) {
      if (threadIdx.x == 0) record.tail_wait0_after = clocks();
      tail_release.before = clocks();
    }
    // consumer-only retains the original tail modulo to isolate loop control.
    if constexpr (IncrementalIndex) Barrier::arrive(empty + previous_slot, 0, 1);
    else Barrier::arrive(empty + (ktiles - 1) % stages, 0, 1);
    if constexpr (Trace) {
      tail_release.after = clocks();
      if (threadIdx.x == 0) {
        record.tail_release_before = tail_release.before;
        record.tail_release_after = tail_release.after;
      }
      tail_release.before = warp_max_pair(tail_release.before);
      tail_release.after = warp_max_pair(tail_release.after);
      if ((threadIdx.x & 31) == 0) tail_release_warps[threadIdx.x / 32] = tail_release;
    }
    float checksum = 0;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int j = 0; j < FragmentRegisters; ++j) {
        warpgroup_fence_operand(d[m][j]);
        checksum += d[m][j];
      }
    }
    drain[threadIdx.x] = checksum;
    generation_errors[threadIdx.x] = errors;
  }
  // Producer threads do no repeated CTA barrier or polling while consumer computes.
  __syncthreads();
  if (threadIdx.x == 0) {
    record.end = consumer_end(drain + ConsumerThreads - 1);
    record.smid_end = smid();
  }
  __syncthreads(); // global export and final bookkeeping are outside the window
  if (threadIdx.x == 0) {
    for (int thread = 0; thread < ConsumerThreads; ++thread)
      record.generation_errors += generation_errors[thread];
    if constexpr (Trace) {
      for (int warp = 0; warp < 4; ++warp) {
        record.tail_release_last_before.cycle = max(record.tail_release_last_before.cycle, tail_release_warps[warp].before.cycle);
        record.tail_release_last_before.ns = max(record.tail_release_last_before.ns, tail_release_warps[warp].before.ns);
        record.tail_release_last_after.cycle = max(record.tail_release_last_after.cycle, tail_release_warps[warp].after.cycle);
        record.tail_release_last_after.ns = max(record.tail_release_last_after.ns, tail_release_warps[warp].after.ns);
      }
    }
    records[0] = record;
  }
  if constexpr (Trace) {
    for (int tile = threadIdx.x; tile < ktiles; tile += Threads) {
      TileTrace result = trace[tile];
      for (int warp = 0; warp < 4; ++warp) {
        result.release_last_before.cycle = max(result.release_last_before.cycle, result.release_warps[warp].before.cycle);
        result.release_last_before.ns = max(result.release_last_before.ns, result.release_warps[warp].before.ns);
        result.release_last_after.cycle = max(result.release_last_after.cycle, result.release_warps[warp].after.cycle);
        result.release_last_after.ns = max(result.release_last_after.ns, result.release_warps[warp].after.ns);
      }
      exported_trace[tile] = result;
    }
  }
  if (threadIdx.x < ConsumerThreads) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int j = 0; j < FragmentRegisters; ++j)
        output[(threadIdx.x * 2 + m) * FragmentRegisters + j] = d[m][j];
    }
  }
}

std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << c;
  }
  out << '"';
  return out.str();
}
U64 integer(const std::string& value, U64 maximum) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("invalid unsigned integer");
  errno = 0;
  char* end = nullptr;
  U64 result = std::strtoull(value.c_str(), &end, 10);
  if (errno || *end || result > maximum) throw std::runtime_error("integer out of range");
  return result;
}
struct ChainOptions {
  int stages = 6, ktiles = 4, warmup_limit = 30;
  U64 delay_cycles = 0;
#ifdef GH200_CONSUMER_ENTRY_DELAY
  U64 consumer_entry_delay_cycles = 0;
#endif
  U32 seed = 17;
  bool trace = false, incremental_index = false, consumer_only_index = false;
  ChainOptions(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing option value");
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--stages") stages = int(integer(value, 6));
      else if (key == "--delay-cycles") delay_cycles = integer(value, std::numeric_limits<U64>::max());
#ifdef GH200_CONSUMER_ENTRY_DELAY
      else if (key == "--consumer-entry-delay-cycles")
        consumer_entry_delay_cycles = integer(value, std::numeric_limits<U64>::max());
#endif
      else if (key == "--ktiles") ktiles = int(integer(value, 32));
      else if (key == "--seed") seed = U32(integer(value, std::numeric_limits<U32>::max()));
      else if (key == "--trace") trace = bool(integer(value, 1));
      else if (key == "--index-mode") {
        if (value != "runtime" && value != "incremental" && value != "consumer-only")
          throw std::runtime_error("index-mode must be runtime, incremental, or consumer-only");
        incremental_index = value == "incremental";
        consumer_only_index = value == "consumer-only";
      }
      else if (key == "--warmup-limit") warmup_limit = int(integer(value, 60));
      else throw std::runtime_error("unknown option: " + key);
    }
    if (stages != 4 && stages != 6) throw std::runtime_error("stages must be 4 or 6");
    if (ktiles != 4 && ktiles != 13 && ktiles != 32)
      throw std::runtime_error("formal ktiles must be 4, 13, or 32");
    if (warmup_limit < 8) throw std::runtime_error("warmup-limit must be 8..60");
  }
};
std::vector<float> reference(int ktiles, int stages, U32 seed) {
  std::vector<float> expected(OutputElements);
  // Independent scalar integer dot products; no CuTe layout or descriptors used.
  for (int thread = 0; thread < ConsumerThreads; ++thread)
    for (int m = 0; m < 2; ++m)
      for (int j = 0; j < FragmentRegisters; ++j) {
        int row = 64 * m + 16 * (thread / 32) + (thread % 32) / 4 + 8 * ((j / 2) % 2);
        int column = 2 * (thread % 4) + j % 2 + 8 * (j / 4);
        int64_t numerator = 0;
        for (int tile = 0; tile < ktiles; ++tile)
          for (int k = 0; k < 64; ++k)
            numerator += int64_t(a_numerator(tile % stages, row, k, seed)) *
                b_numerator(tile % stages, column, k, seed);
        expected[(thread * 2 + m) * FragmentRegisters + j] = float(numerator) / 1024.f;
      }
  return expected;
}
double tail_cv(const std::vector<U64>& windows) {
  double mean = 0, variance = 0;
  for (size_t i = windows.size() - 5; i < windows.size(); ++i) mean += double(windows[i]) / 5;
  if (mean <= 0) throw std::runtime_error("nonpositive warmup window");
  for (size_t i = windows.size() - 5; i < windows.size(); ++i) {
    double difference = double(windows[i]) - mean;
    variance += difference * difference / 4;  // sample std, ddof=1
  }
  return std::sqrt(variance) / mean;
}
struct Trial {
  Record record;
  float event_ms = 0;
  U64 errors = 0;
  double max_abs_error = 0;
  std::vector<float> values;
  std::vector<TileTrace> trace;
};
void print_pair(const ClockPair& value) {
  std::cout << "{\"cycle\":" << value.cycle << ",\"ns\":" << value.ns << '}';
}
void print_correctness(const Trial& trial) {
  std::cout << "{\"checked_elements\":" << trial.values.size() << ",\"errors\":" << trial.errors
      << ",\"max_abs_error\":" << trial.max_abs_error
      << ",\"generation_checked\":" << trial.trace.size() * ConsumerThreads
      << ",\"generation_errors\":" << trial.record.generation_errors << '}';
}
template <bool Trace, bool IncrementalIndex, bool ConsumerOnlyIndex>
int run_chain(const ChainOptions& options) {
  const U32 endian_check = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian_check) != 1)
    throw std::runtime_error("little endian output witness required");
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count != 1) throw std::runtime_error("exactly one visible GPU required");
  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  if (device.major != 9 || device.minor != 0 || std::string(device.name).find("GH200") == std::string::npos)
    throw std::runtime_error("GH200 CC9.0 required");
  auto kernel = handoff_chain<Trace, IncrementalIndex, ConsumerOnlyIndex>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, Threads, SmemBytes));
  if (occupancy != 1 || attributes.localSizeBytes != 0)
    throw std::runtime_error("one CTA/SM capacity and zero local memory required");
  int driver_version = 0, runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::ostringstream uuid;
  uuid << "GPU-";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
    uuid << std::hex << std::setw(2) << std::setfill('0')
        << unsigned(static_cast<unsigned char>(device.uuid.bytes[i]));
  }
  DeviceBuffer<Record> records(1);
  DeviceBuffer<TileTrace> trace_records(MaxTiles);
  DeviceBuffer<float> output(OutputElements);
  cudaEvent_t event_begin, event_end;
  CUDA_CHECK(cudaEventCreate(&event_begin));
  CUDA_CHECK(cudaEventCreate(&event_end));
  auto execute = [&](int ktiles, const std::vector<float>& expected) {
    CUDA_CHECK(cudaMemset(records.pointer, 0xff, sizeof(Record)));
    CUDA_CHECK(cudaMemset(output.pointer, 0xff, OutputElements * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(event_begin));
    kernel<<<1, Threads, SmemBytes>>>(ktiles, options.stages, options.delay_cycles, options.seed,
#ifdef GH200_CONSUMER_ENTRY_DELAY
        options.consumer_entry_delay_cycles,
#endif
        records.pointer, trace_records.pointer, output.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    Trial trial;
    trial.values.resize(OutputElements);
    trial.trace.resize(ktiles);
    CUDA_CHECK(cudaEventElapsedTime(&trial.event_ms, event_begin, event_end));
    CUDA_CHECK(cudaMemcpy(&trial.record, records.pointer, sizeof(Record), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trial.values.data(), output.pointer, OutputElements * sizeof(float), cudaMemcpyDeviceToHost));
    const auto& r = trial.record;
    if (r.smid_start != r.smid_end || r.smid_start == ~U32(0) ||
        r.end.cycle <= r.begin.cycle || r.end.ns <= r.begin.ns ||
        !std::isfinite(trial.event_ms) || trial.event_ms <= 0 ||
        double(r.end.ns - r.begin.ns) > double(trial.event_ms) * 1.05e6)
      throw std::runtime_error("invalid or migrated timing window");
    if constexpr (Trace) {
      CUDA_CHECK(cudaMemcpy(trial.trace.data(), trace_records.pointer, ktiles * sizeof(TileTrace), cudaMemcpyDeviceToHost));
      for (int tile = 0; tile < ktiles; ++tile) {
        const auto& c = trial.trace[tile].consumer;
        auto ordered = [](const ClockPair& a, const ClockPair& b) {
          return a.cycle <= b.cycle && a.ns <= b.ns;
        };
        if (!ordered(r.begin, c.wait_before) || !ordered(c.wait_before, c.wait_after) ||
            !ordered(c.wait_after, c.issue_before) || !ordered(c.issue_before, c.commit_after) ||
            !ordered(c.commit_after, r.tail_wait0_after) || !ordered(r.tail_release_after, r.end))
          throw std::runtime_error("invalid consumer trace order");
        if (tile && (!ordered(trial.trace[tile].release_last_before, trial.trace[tile].release_last_after) ||
            !ordered(c.commit_after, c.wait1_after) ||
            !ordered(c.wait1_after, c.release_before) || !ordered(c.release_before, c.release_after)))
          throw std::runtime_error("invalid release trace order");
        if (tile >= options.stages) {
          const auto& p = trial.trace[tile].producer;
          const auto& previous_release = trial.trace[tile - options.stages + 1];
          if (!ordered(previous_release.release_last_before, p.acquire_after) ||
              !ordered(r.begin, p.acquire_before) || !ordered(p.acquire_before, p.acquire_after) ||
              !ordered(p.acquire_after, p.full_before) || !ordered(p.full_before, p.full_after) ||
              !ordered(p.full_after, r.end)) throw std::runtime_error("invalid producer trace order");
        }
      }
    }
    for (int i = 0; i < OutputElements; ++i) {
      float value = trial.values[i];
      if (!std::isfinite(value)) {
        ++trial.errors;
        continue;
      }
      double error = std::abs(double(value) - double(expected[i]));
      trial.max_abs_error = std::max(trial.max_abs_error, error);
      if (error != 0) ++trial.errors;
    }
    return trial;
  };
  std::vector<Trial> short_checks;
  std::vector<int> short_lengths{1, 2, options.stages + 1};
  for (int ktiles : short_lengths) {
    short_checks.push_back(execute(ktiles, reference(ktiles, options.stages, options.seed)));
    if (short_checks.back().errors || short_checks.back().record.generation_errors) throw std::runtime_error("nonuniform short witness failed");
  }
  auto expected = reference(options.ktiles, options.stages, options.seed);
  std::vector<U64> warmup_ns, warmup_cycles;
  bool converged = false;
  double cv = 0;
  for (int i = 0; i < options.warmup_limit; ++i) {
    auto trial = execute(options.ktiles, expected);
    if (trial.errors || trial.record.generation_errors) throw std::runtime_error("warmup output mismatch");
    warmup_ns.push_back(trial.record.end.ns - trial.record.begin.ns);
    warmup_cycles.push_back(trial.record.end.cycle - trial.record.begin.cycle);
    if (i >= 7) {
      cv = tail_cv(warmup_ns);
      if (cv <= 0.02) { converged = true; break; }
    }
  }
  Trial trial = execute(options.ktiles, expected);
  std::ofstream file("output.f32", std::ios::binary);
  file.write(reinterpret_cast<const char*>(trial.values.data()), OutputElements * sizeof(float));
  file.close();
  if (!file) throw std::runtime_error("could not save output.f32");
  CUDA_CHECK(cudaEventDestroy(event_begin));
  CUDA_CHECK(cudaEventDestroy(event_end));
  const auto& r = trial.record;
  std::cout << std::setprecision(17)
      << "{\"schema\":\"gh200-cfg-b-handoff-chain/1\",\"status\":\""
      << ((trial.errors || trial.record.generation_errors) ? "correctness_error" : "measured") << "\",\"ktiles\":" << options.ktiles
      << ",\"stages\":" << options.stages << ",\"delay_cycles\":" << options.delay_cycles
      << ",\"seed\":" << options.seed << ",\"trace\":" << (Trace ? "true" : "false")
      << ",\"index_mode\":\"" << (ConsumerOnlyIndex ? "consumer-only" : IncrementalIndex ? "incremental" : "runtime") << '"'
      << ",\"samples\":1,\"work_flop\":" << U64(2) * 128 * 128 * 64 * options.ktiles
      << ",\"wgmma_count\":" << 8 * options.ktiles
      << ",\"performance_eligible\":" << (converged && !trial.errors && !trial.record.generation_errors ? "true" : "false")
      << ",\"device\":{\"uuid\":" << quote(uuid.str()) << ",\"name\":" << quote(device.name)
      << ",\"sms\":" << device.multiProcessorCount << ",\"major\":" << device.major
      << ",\"minor\":" << device.minor << ",\"driver_version\":" << driver_version
      << ",\"runtime_version\":" << runtime_version << ",\"compiler_version\":\""
      << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__ << "\"}"
      << ",\"resources\":{\"threads\":256,\"grid_ctas\":1,\"dynamic_smem_bytes\":" << SmemBytes
      << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
      << ",\"registers_per_thread\":" << attributes.numRegs
      << ",\"local_size_bytes\":" << attributes.localSizeBytes
      << ",\"occupancy_ctas_per_sm\":" << occupancy << '}'
      << ",\"conditions\":{\"slots\":6,\"active_slots\":" << options.stages
      << ",\"producer_threads\":128,\"consumer_threads\":128,\"full_arrivals\":1,\"empty_arrivals\":128,"
      << "\"initial_barrier_phase\":0,\"signal_kind\":\"normal_arrive\",\"transaction_bytes\":0,"
      << "\"prefilled_tiles\":" << std::min(options.stages, options.ktiles)
#ifdef GH200_CONSUMER_ENTRY_DELAY
      << ",\"consumer_entry_delay_cycles\":" << options.consumer_entry_delay_cycles
      << ",\"consumer_entry_delay_boundary\":\"one clock64 spin after start rendezvous and before first consumer full wait\""
#endif
      << ",\"delayed_tile\":" << (options.ktiles > options.stages ? std::to_string(options.stages) : "null")
      << ",\"software_tile\":[128,128,64],\"instruction_shape\":[64,128,16],"
      << "\"a_major\":\"K\",\"b_major\":\"MN\",\"smem_swizzle_bytes\":128,"
      << "\"commit_per_ktile\":1,\"wait_groups\":" << GH200_NORMAL_WAIT << ",\"tail_wait_groups\":0,"
      << "\"prologue_wait\":false,\"instruction_order\":\"q_then_m64\","
      << "\"input_profile\":\"slot_coordinate_dyadic\",\"mathematical_c0\":0,"
      << "\"ring_policy\":\"immutable slot=tile%stages; generation sideband only, no operand refill\","
      << "\"window\":\"post-prepare start rendezvous through wait0, last empty arrival, accumulator consumers and CTA rendezvous\","
      << "\"event_window\":\"whole kernel including preparation and global export\","
      << "\"trace_issue_boundary\":\"before eight WGMMA; local SASS ordering requires static audit\","
      << "\"trace_completion_boundary\":\"wait returns; physical group completion is not directly timestamped\","
      << "\"first_try_success_meaning\":\"first try_wait returned true; not proof ready at wait entry\","
      << "\"release_trace_scope\":\"leader release_before/after plus max of all 128 consumer release brackets; reduction after arrivals, cross-warp aggregation outside timing\","
      << "\"delay_boundary\":\"producer empty-wait return then one clock64 spin for tile=stages; requested delay is not response price\"}"
      << ",\"short_checks\":[";
  for (size_t i = 0; i < short_checks.size(); ++i) {
    std::cout << (i ? "," : "") << "{\"ktiles\":" << short_lengths[i] << ",\"correctness\":";
    print_correctness(short_checks[i]); std::cout << '}';
  }
  std::cout << "],\"warmup\":{\"converged\":" << (converged ? "true" : "false")
      << ",\"cv\":" << cv << ",\"sample_std_ddof\":1,\"limit\":" << options.warmup_limit
      << ",\"elapsed_ns\":[";
  for (size_t i = 0; i < warmup_ns.size(); ++i) std::cout << (i ? "," : "") << warmup_ns[i];
  std::cout << "],\"elapsed_cycles\":[";
  for (size_t i = 0; i < warmup_cycles.size(); ++i) std::cout << (i ? "," : "") << warmup_cycles[i];
  std::cout << "]},\"timing\":{\"begin_cycle\":" << r.begin.cycle << ",\"end_cycle\":" << r.end.cycle
      << ",\"begin_ns\":" << r.begin.ns << ",\"end_ns\":" << r.end.ns
      << ",\"smid_start\":" << r.smid_start << ",\"smid_end\":" << r.smid_end
      << ",\"elapsed_cycles\":" << r.end.cycle - r.begin.cycle
      << ",\"elapsed_ns\":" << r.end.ns - r.begin.ns << ",\"event_ms\":" << trial.event_ms
      << "},\"trace_detail\":";
  if constexpr (Trace) {
    std::cout << "{\"tiles\":[";
    for (int tile = 0; tile < options.ktiles; ++tile) {
      const auto& p = trial.trace[tile].producer;
      const auto& c = trial.trace[tile].consumer;
      bool prefilled = tile < options.stages;
      std::cout << (tile ? "," : "") << "{\"tile\":" << tile << ",\"slot\":" << tile % options.stages
          << ",\"generation\":" << tile / options.stages
          << ",\"full_wait_phase\":" << (tile / options.stages) % 2
          << ",\"empty_wait_phase\":" << (tile / options.stages + 1) % 2
          << ",\"prefilled\":" << (prefilled ? "true" : "false") << ",\"producer\":";
      if (prefilled) std::cout << "null";
      else {
        std::cout << "{\"acquire_before\":"; print_pair(p.acquire_before);
        std::cout << ",\"acquire_after\":"; print_pair(p.acquire_after);
        std::cout << ",\"full_before\":"; print_pair(p.full_before);
        std::cout << ",\"full_after\":"; print_pair(p.full_after); std::cout << '}';
      }
      std::cout << ",\"consumer\":{\"wait_before\":"; print_pair(c.wait_before);
      std::cout << ",\"wait_after\":"; print_pair(c.wait_after);
      std::cout << ",\"first_try_success\":" << (c.first_try_success ? "true" : "false")
          << ",\"first_try_success_warp_masks\":[";
      for (int warp = 0; warp < 4; ++warp)
        std::cout << (warp ? "," : "") << c.first_try_success_warp_masks[warp];
      std::cout << "],\"issue_before\":"; print_pair(c.issue_before);
      std::cout << ",\"commit_after\":"; print_pair(c.commit_after);
      std::cout << ",\"wait1_after\":";
      if (tile) print_pair(c.wait1_after); else std::cout << "null";
      std::cout << ",\"release_tile\":" << (tile ? std::to_string(tile - 1) : "null")
          << ",\"release_before\":";
      if (tile) print_pair(c.release_before); else std::cout << "null";
      std::cout << ",\"release_after\":";
      if (tile) print_pair(c.release_after); else std::cout << "null";
      std::cout << ",\"release_last_before\":";
      if (tile) print_pair(trial.trace[tile].release_last_before); else std::cout << "null";
      std::cout << ",\"release_last_after\":";
      if (tile) print_pair(trial.trace[tile].release_last_after); else std::cout << "null";
      std::cout << "}}";
    }
    std::cout << "],\"tail_wait0_after\":"; print_pair(r.tail_wait0_after);
    std::cout << ",\"tail_release_tile\":" << options.ktiles - 1;
    std::cout << ",\"tail_release_before\":"; print_pair(r.tail_release_before);
    std::cout << ",\"tail_release_after\":"; print_pair(r.tail_release_after);
    std::cout << ",\"tail_release_last_before\":"; print_pair(r.tail_release_last_before);
    std::cout << ",\"tail_release_last_after\":"; print_pair(r.tail_release_last_after); std::cout << '}';
  } else std::cout << "null";
  std::cout << ",\"correctness\":";
  print_correctness(trial);
  std::cout << ",\"output_file\":\"output.f32\",\"output_elements\":" << OutputElements
      << ",\"output_bytes\":" << OutputElements * sizeof(float)
      << ",\"output_layout\":\"flat=((thread*2+m64)*64+j); native little-endian float32\"}\n";
  return (trial.errors || trial.record.generation_errors) ? 2 : 0;
}
int main(int argc, char** argv) try {
  ChainOptions options(argc, argv);
  if (options.consumer_only_index)
    return options.trace ? run_chain<true, false, true>(options) : run_chain<false, false, true>(options);
  if (options.incremental_index)
    return options.trace ? run_chain<true, true, false>(options) : run_chain<false, true, false>(options);
  return options.trace ? run_chain<true, false, false>(options) : run_chain<false, false, false>(options);
} catch (const std::exception& error) {
  std::cout << "{\"schema\":\"gh200-cfg-b-handoff-chain/1\",\"status\":\"error\",\"error\":"
      << quote(error.what()) << "}\n";
  return 2;
}
