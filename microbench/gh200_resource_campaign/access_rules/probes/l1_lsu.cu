// One CTA, warp-contiguous 16 B pointer nodes, fixed total work across J.
// J counts independent software chains; SASS owns the actual issue schedule.
#include <cuda_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using U32 = unsigned;
using U64 = unsigned long long;
constexpr int Threads = 128;
constexpr int Rounds = 256;
#define CUDA_CHECK(call) do { cudaError_t status_ = (call); if (status_ != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); } while (0)

struct Record { U64 begin_cycle, end_cycle, begin_ns, end_ns; U32 smid_start, smid_end; };
__device__ __forceinline__ U64 cycle_stamp() {
  U64 v; asm volatile("mov.u64 %0, %%clock64;" : "=l"(v) :: "memory"); return v;
}
__device__ __forceinline__ U64 ns_stamp() {
  U64 v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v;
}
__device__ __forceinline__ U32 sm_stamp() {
  U32 v; asm volatile("mov.u32 %0, %%smid;" : "=r"(v) :: "memory"); return v;
}
template<bool CacheAll> __device__ __forceinline__ uint4 load_node(const uint4* p) {
  uint4 v;
  if constexpr (CacheAll)
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  else
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

template<int J, bool CacheAll>
__global__ void l1_target(const uint4* input, int batches, int rounds,
                         U32* warm_checksums, U32* timed_checksums, Record* record_out) {
  __shared__ volatile U32 drain[Threads];
  U32 index[J], sum[J]{};
  #pragma unroll
  for (int chain = 0; chain < J; ++chain) index[chain] = chain * Threads + threadIdx.x;
  Record record{};
  if (threadIdx.x == 0) record.smid_start = sm_stamp();
  U32 warm_sum = 0;
  // Each launch warms its own complete W; no cross-kernel L1 residency assumption.
  #pragma unroll 1
  for (int step = 0; step < batches / J; ++step) {
    uint4 values[J];
    #pragma unroll
    for (int chain = 0; chain < J; ++chain) values[chain] = load_node<CacheAll>(input + index[chain]);
    #pragma unroll
    for (int chain = 0; chain < J; ++chain) {
      index[chain] = values[chain].x;
      warm_sum += values[chain].x + values[chain].y + values[chain].z + values[chain].w;
    }
  }
  drain[threadIdx.x] = warm_sum;
  __syncthreads();
  if (threadIdx.x == 0) { record.begin_ns = ns_stamp(); record.begin_cycle = cycle_stamp(); }
  __syncthreads();
  #pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    #pragma unroll 1
    for (int step = 0; step < batches / J; ++step) {
      uint4 values[J];
      #pragma unroll
      for (int chain = 0; chain < J; ++chain) values[chain] = load_node<CacheAll>(input + index[chain]);
      #pragma unroll
      for (int chain = 0; chain < J; ++chain) {
        index[chain] = values[chain].x;
        sum[chain] += values[chain].x + values[chain].y + values[chain].z + values[chain].w;
      }
    }
  }
  U32 final_sum = 0;
  #pragma unroll
  for (int chain = 0; chain < J; ++chain) final_sum += sum[chain];
  // The volatile CTA drain and barrier precede both ending stamps. Global
  // diagnostic stores follow a second barrier, outside every timed warp.
  drain[threadIdx.x] = final_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    record.end_cycle = cycle_stamp(); record.end_ns = ns_stamp(); record.smid_end = sm_stamp();
  }
  __syncthreads();
  warm_checksums[threadIdx.x] = warm_sum;
  timed_checksums[threadIdx.x] = final_sum;
  if (threadIdx.x == 0) *record_out = record;
}

std::string quote(const std::string& s) {
  std::ostringstream out; out << '"';
  for (char c : s) { if (c == '"' || c == '\\') out << '\\'; out << c; }
  return out.str() + '"';
}
U64 integer(const std::string& text, U64 minimum, U64 maximum) {
  if (text.empty() || text[0] == '-') throw std::runtime_error("invalid unsigned argument");
  errno = 0; char* end = nullptr; U64 v = std::strtoull(text.c_str(), &end, 10);
  if (errno || *end || v < minimum || v > maximum) throw std::runtime_error("argument out of range");
  return v;
}
struct Options {
  int kib = 16, chains = 1, carveout = 100; U32 seed = 3; bool ca = true;
  Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing option value");
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--working-set-kib") kib = integer(value, 16, 256);
      else if (key == "--chains") chains = integer(value, 1, 4);
      else if (key == "--carveout") carveout = integer(value, 0, 100);
      else if (key == "--seed") seed = integer(value, 0, std::numeric_limits<U32>::max());
      else if (key == "--policy") {
        if (value != "ca" && value != "cg") throw std::runtime_error("policy must be ca or cg");
        ca = value == "ca";
      } else throw std::runtime_error("unknown option: " + key);
    }
    if ((kib != 16 && kib != 64 && kib != 128 && kib != 256) ||
        (chains != 1 && chains != 4) || (carveout != 0 && carveout != 100))
      throw std::runtime_error("outside finite probe condition domain");
  }
};
U32 expected_checksum(int batches, int thread, int rounds, U32 seed) {
  // x is a permutation of the thread's vector indices; y/z/w use affine words.
  U64 index_sum = U64(Threads) * batches * (batches - 1) / 2 + U64(thread) * batches;
  U64 per_sweep = index_sum + 17 * (12 * index_sum + 6 * U64(batches)) + 3 * U64(batches) * seed;
  return U32(per_sweep * U64(rounds));
}
template<int J> const void* function(bool ca) {
  return ca ? reinterpret_cast<const void*>(l1_target<J, true>) : reinterpret_cast<const void*>(l1_target<J, false>);
}
void print_values(const std::vector<U32>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
  std::cout << ']';
}
std::string uuid_string(const cudaUUID_t& uuid) {
  std::ostringstream out; out << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
    out << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
  }
  return out.str();
}
int main(int argc, char** argv) {
  try {
    Options options(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const void* kernel = options.chains == 1 ? function<1>(options.ca) : function<4>(options.ca);
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, options.carveout));
    cudaFuncAttributes attributes{}; CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int occupancy = 0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, Threads, 0));
    int bytes = options.kib * 1024, batches = bytes / (Threads * 16);
    std::vector<uint4> fixture(bytes / 16);
    for (std::size_t i = 0; i < fixture.size(); ++i) {
      U32 batch = i / Threads, thread = i % Threads;
      U32 next = ((batch + options.chains) % batches) * Threads + thread;
      fixture[i] = make_uint4(next, 17u * U32(4*i + 1) + options.seed,
          17u * U32(4*i + 2) + options.seed, 17u * U32(4*i + 3) + options.seed);
    }
    uint4* input = nullptr; U32 *warm = nullptr, *timed = nullptr; Record* record = nullptr;
    CUDA_CHECK(cudaMalloc(&input, bytes)); CUDA_CHECK(cudaMalloc(&warm, Threads * sizeof(U32)));
    CUDA_CHECK(cudaMalloc(&timed, Threads * sizeof(U32))); CUDA_CHECK(cudaMalloc(&record, sizeof(Record)));
    CUDA_CHECK(cudaMemcpy(input, fixture.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, end; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    std::cout << "{\"schema\":\"gh200-l1-lsu/1\",\"status\":\"measured\",\"working_set_bytes\":" << bytes
      << ",\"chains\":" << options.chains << ",\"policy\":" << quote(options.ca ? "ca" : "cg")
      << ",\"requested_carveout_percent\":" << options.carveout << ",\"actual_carveout\":null"
      << ",\"seed\":" << options.seed << ",\"rounds\":" << Rounds << ",\"logical_read_bytes\":" << U64(bytes)*Rounds
      << ",\"device\":{\"name\":" << quote(prop.name) << ",\"uuid\":" << quote(uuid_string(prop.uuid))
      << ",\"sms\":" << prop.multiProcessorCount << ",\"major\":" << prop.major << ",\"minor\":" << prop.minor
      << ",\"l2_cache_bytes\":" << prop.l2CacheSize << "},\"resources\":{\"grid_ctas\":1,\"threads\":128"
      << ",\"registers_per_thread\":" << attributes.numRegs << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
      << ",\"dynamic_smem_bytes\":0,\"local_size_bytes\":" << attributes.localSizeBytes
      << ",\"occupancy_ctas_per_sm\":" << occupancy << "},\"trials\":[";
    int rounds_values[] = {1, 2, Rounds};
    for (int trial = 0; trial < 3; ++trial) {
      int rounds = rounds_values[trial];
      void* args[] = {&input, &batches, &rounds, &warm, &timed, &record};
      CUDA_CHECK(cudaEventRecord(start));
      CUDA_CHECK(cudaLaunchKernel(kernel, dim3(1), dim3(Threads), args, 0, nullptr));
      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
      float event_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&event_ms, start, end));
      Record r{}; std::vector<U32> warm_values(Threads), timed_values(Threads);
      CUDA_CHECK(cudaMemcpy(&r, record, sizeof(r), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(warm_values.data(), warm, Threads * sizeof(U32), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(timed_values.data(), timed, Threads * sizeof(U32), cudaMemcpyDeviceToHost));
      int errors = 0;
      for (int thread = 0; thread < Threads; ++thread)
        errors += (warm_values[thread] != expected_checksum(batches, thread, 1, options.seed)) +
                  (timed_values[thread] != expected_checksum(batches, thread, rounds, options.seed));
      if (errors || r.smid_start != r.smid_end || r.end_cycle <= r.begin_cycle || r.end_ns <= r.begin_ns)
        throw std::runtime_error("checksum, SM residency, or timing qualification failed");
      std::cout << (trial ? "," : "") << "{\"rounds\":" << rounds << ",\"logical_read_bytes\":" << U64(bytes)*rounds
        << ",\"event_ms\":" << std::setprecision(9) << event_ms << ",\"begin_cycle\":" << r.begin_cycle
        << ",\"end_cycle\":" << r.end_cycle << ",\"begin_ns\":" << r.begin_ns << ",\"end_ns\":" << r.end_ns
        << ",\"smid_start\":" << r.smid_start << ",\"smid_end\":" << r.smid_end
        << ",\"correctness\":{\"checked_elements\":256,\"errors\":" << errors << "},\"warm_checksums\":";
      print_values(warm_values); std::cout << ",\"timed_checksums\":"; print_values(timed_values); std::cout << '}';
    }
    std::cout << "]}\n";
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(warm)); CUDA_CHECK(cudaFree(timed)); CUDA_CHECK(cudaFree(record));
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
